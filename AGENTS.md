# AGENTS.md — how to work in this repository

This file is the operating manual for an AI agent (or a new human) picking this repository up cold. It is
the compressed state of ~50 measurement sessions. Read it before touching anything. **Flags and environment
variables are in [`FLAGS.md`](FLAGS.md)** — this file tells you what is true, that one tells you what is
switchable.

## 1. Naming — get this right first

| name | what it is |
|---|---|
| **K2** (t16) | the weight format + runtime: 16×16 trellis tiles, `I16` payload, one custom op |
| **SantanaQAT** | the training stage: scales-only QAT on frozen trellis codes (0.09 % of parameters) |
| **SC → RCO → recipe** | the allocation stage that produces the per-tensor `K` map |
| **K2E** | optional trellis embedding (shrinks the file 348 MB; costs 1.2 coding points, measured) |

Do not call the trellis "K2-QAT". Do not call SantanaQAT a fine-tune — it trains scales only.

## 2. What K2 is, mechanically

Each 16×16 tile of a weight matrix is a circular `256·K`-bit stream. The payload is a plain `I16` 3-D
tensor `[16K, out/16, in/16]` with standard strides; the side tensors are `su`/`sv` (F16); a
block-Hadamard(128) rotation is applied on both sides at encode time. There is **no new `ggml_type`**, though one is *possible*: this fork already stores multi-row blocks
(`IQ3_KS_R16`/`IQ4_KS_R16`/`IQ5_KS_R16` = 16 rows x 32 columns = 512 quants in 204 B, `blck_size` 32,
`type_size` charged per row; the `*_R4`/`*_R8` types predate them). We kept the payload as a 3-D `I16`
tensor because its element width *is* the per-tensor rate `K`, and the side scales are per-matrix
tensors rather than a block header. A port to the R16 pattern is future work, not an impossibility.
The only runtime addition is one op (dense matmul, MoE id-matmul, embedding gather).

## 3. Invariants — violating any of these produces plausible garbage, not a crash

1. **Never re-apply `su`/`sv` or the Hadamard** to already-decoded weights.
2. `K` is **per tensor** (2..6). Read it from the file: `K = ne[0] / 16` for every `I16` 3-D tensor. The
   "body rate" excludes embedding and head.
3. The head may be trellised (`-hb 6`); the embedding stays F16 unless K2E (`-eb 6`) is explicitly enabled.
4. Decoding must be **bit-exact** against the Python codec (max abs diff 0.0). `loader_probe.py` validates
   197 trellis tensors per artifact (shapes, K range, `su`/`sv` pairing, byte accounting, orphans).
5. Encoder step order is fixed: **measure (SC) → allocate (RCO) → recipe → encode → assemble**. Changing the
   order silently changes the rate map.

## 4. Failure catalog — "it is broken" is usually one of these

Check this list before believing a regression. Every entry below has burned hours here.

| symptom | actual cause |
|---|---|
| Greedy-free coherence probe gives nonsense ("The capital of France is" → "B. London / C. Berlin") | **default sampling**, not a broken checkpoint. Any coherence probe must use `--temp 0`. |
| Perplexity reads ~6.3 where the paper says 10.59 | **protocol**: one chunk at `-c 512` gives ~6.34 for K2 *and* for F16 (6.3438 vs 6.3425). 20 chunks gives 10.5894. |
| `libcudart.so.12: cannot open shared object file` | host binaries need the driver libs: `LD_LIBRARY_PATH=/home/titan/cuda-libs` on this box. |
| `build_ext` fails with `CUDA_HOME is not set` | the host has **no** CUDA compiler; encoder builds must run inside the `cuda-box` distrobox (nvcc 12.4, matching torch). |
| `--max_module` run produces no `model.safetensors` | expected: a partial run writes `<work>/ckpt/state.safetensors`. |
| `make: No rule to make target 'test-backend-ops'` | `test-backend-ops.cpp` **does not compile** against this fork's API (`ggml_quantize_chunk`, `ggml_upscale`, `ggml_upscale_ext` take different arguments). It is not a target here. |
| Checkpointing looks catastrophic (3.1 GB per save) | measured: two whole-state saves cost **~5 s** at 1.5B (page cache absorbs them). It matters only where the cache cannot hold the state (35B: ~70 GB vs 31 GB RAM). |
| `rfn` improves when you cut the calibration data | **train/eval overlap**: `rfn` is measured on the same rows that build the Hessian. Not a quality gain. |
| Attention-sink / NaN Hessians on a new model | `hessians_reg` (output side) is a different knob from the input-side damping `K2_SIGMA_REG`; see `FLAGS.md`. |

## 5. Verify a published artifact in four steps

```bash
md5sum k2-1.5b.gguf                    # expect 584a86cb105e9771e6b30d3c7f08f52d (PTQ pack, 1,197,866,720 B)
llama-cli -m k2-1.5b.gguf -ngl 99 -p "The capital of France is" -n 24 --temp 0
llama-perplexity -m k2-1.5b.gguf -f wikitext-2-raw/wiki.test.raw -c 512 --chunks 20 -ngl 99   # ~10.59 for the PTQ pack
```

The **PTQ pack** (`k2_ptq.gguf`) and the **SantanaQAT pack** (`k2_15b_general.gguf`) are two different
files with identical sizes. Their numbers are *not* interchangeable:

| file | coding (246) | HellaSwag | PPL |
|---|---|---|---|
| `k2_ptq.gguf` (PTQ, md5 `584a86cb…`) | 73.6 % | not measured | 10.5894 |
| `k2_E_f16emb` vs `k2_E` (K2E control, f16 vs trellis embed) | 190 vs 187 of 246 | — | — |
| `k2_15b_general.gguf` (SantanaQAT, md5 `45d62417…`) | 79.7 % | 87.3 % | 10.2886 |

Quote the file you actually ran. This distinction is the single easiest mistake to make in this repository.

## 6. Encode a model

```bash
# 1. sensitivity, 2. allocation, 3. recipe, 4. encode
python pipeline/sc_measure.py -m <hf model> -t wikitext-2-raw/wiki.test.raw -o sc.json --rows 4 --length 256 \
       --levels 0.30,0.15 --device cuda
python pipeline/sc_alloc.py -m sc_body.json -o grid.json --bpw 3.34 --menu 3,4,5 --ratio 1.96 \
       --alpha 2.0 --anchor 3:0.155 --tie "q_proj+k_proj+v_proj,gate_proj+up_proj"
# grid.json -> recipe.yaml (format in pipeline/README.md), then:
python exllamav3/convert.py -i <hf model> -o out -w out_work -rcp recipe.yaml -hb 6 -d 0
```

Cost on one RTX 3060: 1.5B ≈ 19.4 min encode + 10 min SantanaQAT; 7B ≈ 1.5–2 h; 35B ≈ 9 h. The tile
search is 54 % of the encode and **has no lossless speedup available** (see §7).

## 7. Measured status — do not re-litigate without a new run

| lever | verdict | number |
|---|---|---|
| Assembly (GGUF rewrite) | **shipped, byte-identical** | 91 s → 4 s |
| TF32 calibration Hessian | **rejected** | layer-1 `rfn` +36 %, then Cholesky loses positive-definiteness |
| Block size 512/256 (occupancy) | **rejected** | 268 s → 271 s (0.99×), pack byte-identical |
| Beam search in the tile Viterbi | **withdrawn** | lossless width 8192 = full state count at K=3 → 1.0×; pruning costs +2.3 %/+7.9 %/+12.1 % |
| Woodbury `H⁻¹` | **inapplicable** | rank(X) = d at 512k calibration positions → no low-rank form |
| Calibration reduction (`-cr`/`-cc`) | **option, quality unmeasured** | −23 % encode; `rfn` moves up to 0.0036 (biased proxy) |
| Entropy/arithmetic coding | **dead for this codec** | 1.000× (transition symbols exactly uniform) |
| Proxy trimming | **moot** | `num_ref_states = 5` |
| `-hreg` as a σ control | **broken** | ppl 44,440,843 at 0 / 0.005 / 0.05; the real knob is `K2_SIGMA_REG` |
| K2E embedding quality at 3.34 | **measured** | -1.2 coding points (187 vs 190 of 246, scales-controlled); runtime load test still not run |

QATv2's open items and their costs: `K2_QATV2_PLAN.md` §§10–11 (encoder column closed, ~2 h σ sweep first).

## 8. What not to claim without a new run

Prefill parity (2.6× behind int8 types), a K2E *runtime* claim (its coding cost is measured at -1.2 points; the load test is not), the int8 `mma.sync` path (attempted, not
correct, not in this branch), entropy coding (1.000×), any single-sample comparison (item resolution is
±4 points), and any encode/checkpoint time extrapolated to a 35B without saying so.

## 9. Where the truth lives

`main.pdf` for claims · `evidence/` for per-item records · `pipeline/` for the allocation that produced the
released rate map · `FLAGS.md` for every switch · the companion progress repository
([`qwen25-q3-qat`](https://github.com/TitanMamo/qwen25-q3-qat)) for the failed arms and the full run record,
including the ones that look like they should work and do not.
