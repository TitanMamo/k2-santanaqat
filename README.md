# K2 + SantanaQAT — trellis-coded weights at 3.34 bpw

**DOI (concept, always the latest version): [10.5281/zenodo.23678902](https://doi.org/10.5281/zenodo.23678902)**
· this release: [10.5281/zenodo.23678903](https://doi.org/10.5281/zenodo.23678903)
· runtime PR: [#2762](https://github.com/ikawrakow/ik_llama.cpp/pull/2762)
· artifact: [TitanMamo10/Qwen2.5-1.5B-K2-3.34bpw](https://huggingface.co/TitanMamo10/Qwen2.5-1.5B-K2-3.34bpw)

**What it is.** K2 is a weight format for LLMs built on a 16×16 trellis code (EXL3/QTIP class), a
small native runtime for `ik_llama.cpp`, and an encoder that turns any dense or MoE HF model into it.
It exists to answer one question with measurements: *how far is 3-bit quantization from the
rate–distortion bound, and what does closing that gap cost in speed, memory and engineering?*

This repository is the publication: the paper, the per-item evidence behind every number, the
allocation pipeline that produced the released rate map, and the reference artifact.

The training half is **SantanaQAT**: a scales-only QAT that trains **0.09 %** of the parameters
(per-channel `su`/`sv` scales, trellis codes frozen) against a frozen teacher, and buys **+6.1 coding
points** (73.6 % → 79.7 %, McNemar p=0.004) in **10 minutes on one 12 GB card** — 1.4·10⁴ fewer tokens
than a published 10.5 B-token recipe. Everything beyond that (latent codes, block-wise, end-to-end) was
measured and did not improve on it.

**Status:** runtime + encoder shipped and measured; encoder speedups and QATv2 items are explicitly
listed as open (§7). Nothing in this repository is a projection presented as a measurement.

---

## 1. Why you would use it

At a **3.34 body bits-per-weight** budget, on one 12 GB RTX 3060, Qwen2.5-1.5B:

| axis | measured | comparison |
|---|---|---|
| **quality** — PTQ pack | perplexity **10.5894**, coding **73.6 %** | F16 10.3099 · `IQ3_KS` **13.2312** (3.44 bpw) · `IQ4_KS` 10.9485 (4.54 bpw) |
| **quality** — SantanaQAT pack | perplexity **10.2886**, coding **79.7 %**, HellaSwag **87.3 %**, ARC **87.0** (100 %), probes **88.8** (106 %), GSM8K **52.0 %** (104 %) | +6.1 coding points over the PTQ pack, for 0.09 % of trainable parameters |
| **decode** | **174–176 t/s** (three sessions; 5/5 pairs) | 6–9 % behind `IQ4_KS` (184.4) / `IQ3_KS` (189.6), reading ~21 % fewer body bits (3.34 vs ~4.25 bpw; on disk the file is 6.22 bpw) |
| **prefill** | **3.18–3.25k t/s** (3180–3248 across sessions) | **27–28×** the CUDA-core reference (116 t/s); **2.55× behind** `IQ4_KS` (8098–8160 paired; 8488 single-run) — see §4 |
| **footprint** | **1198 MB** file (849.6 MB with the trellis embedding) | runs fully offloaded in 12 GB VRAM; CPU path also works (AVX2 2.98 t/s vs scalar 0.75) |
| **encode cost** | **19.4 min** for a 1.5B | plus a **10 min** **SantanaQAT** run (1.31M trainable params = 0.09 %), i.e. a full artifact in ~30 min on one card |

**Two files ship, same size, different scales.** `…k2-3.34bpw.gguf` is the PTQ pack (md5
`584a86cb105e9771e6b30d3c7f08f52d`); `…k2-3.34bpw-santanaqat.gguf` is the SantanaQAT pack (md5
`45d624171dc4a6168aff7ce6dae2794f`). The only difference is the 394 trained scale tensors, and the two
carry different numbers — quote the file you ran.

Quality per byte is the whole argument: at a comparable rate K2 is **~20 % better in perplexity** than
the 3-bit k-quant, and versus `IQ4_KS` it needs ~21 % fewer body bits at equal quality.
The decoder is also **exactly invertible** against its Python codec (max abs diff 0.0) and passes
`racecheck` with 0 hazards.

## 2. Quick start (verify the published artifact in ~10 minutes)

```bash
# 1. get the artifact  [~1-3 min, network]
#    this is the PTQ pack: md5 584a86cb105e9771e6b30d3c7f08f52d, 1,197,866,720 B, PPL 10.5894, coding 73.6%
#    for the SantanaQAT pack (79.7% coding, PPL 10.2886) use the -santanaqat.gguf file, md5 45d624171dc4a6168aff7ce6dae2794f
curl -L -o k2-1.5b.gguf \
  https://huggingface.co/TitanMamo10/Qwen2.5-1.5B-K2-3.34bpw/resolve/main/qwen2.5-1.5b-k2-3.34bpw.gguf
md5sum k2-1.5b.gguf

# 2. build this branch (CUDA needs a toolkit; CPU-only works too)  [~5 min, 6 cores]
cmake -B build -DGGML_CUDA=ON -DCMAKE_CUDA_ARCHITECTURES=native
cmake --build build -j --target llama-cli llama-perplexity

# 3. run it (greedy -- see the gotchas in §6)
./build/bin/llama-cli -m k2-1.5b.gguf -ngl 99 -p "The capital of France is" -n 24 --temp 0
# expect: " Paris. The capital of France is Paris. ..."

# 4. check the numbers (20-chunk protocol, n_ctx 512)  [~5 min on a 3060]
./build/bin/llama-perplexity -m k2-1.5b.gguf -f wikitext-2-raw/wiki.test.raw -c 512 --chunks 20 -ngl 99
#   (the corpus is wikitext-2-raw/wiki.test.raw; the same file the paper uses)
# expect: ~10.59   (F16 reference 10.3099; a 1-chunk run reads ~6.34 for both -- see §6)
```

## 3. Turning a model into K2

The released rate map is **not** a uniform rate step. The pipeline is
`SC sensitivity → RCO allocation → per-tensor K recipe → trellis encode → GGUF assembly`:

```bash
python pipeline/sc_measure.py -m <hf model> -t wikitext-2-raw/wiki.test.raw -o sc.json \
    --rows 4 --length 256 --levels 0.30,0.15 --device cuda
python pipeline/sc_alloc.py -m sc_body.json -o grid.json --bpw 3.34 --menu 3,4,5 \
    --ratio 1.96 --alpha 2.0 --anchor 3:0.155 --tie "q_proj+k_proj+v_proj,gate_proj+up_proj"
# grid.json -> recipe.yaml (pipeline/README.md has the format), then:
python exllamav3/convert.py -i <hf model> -o out -w out_work -rcp recipe.yaml -hb 6 -d 0
```

The allocator's prediction for the released artifact was `predicted_kld = 0.155326`; the artifact
measured `rfn = 0.1554`. Predicted, then measured — that is the claim being made, not a fitted story.

`pipeline/qwen15_3.34bpw/` contains the exact `sc.json`, `grid.json` and `recipe.yaml` used for the
released file, so the map can be re-derived instead of trusted.

## 4. What this is honest about being weak

- **Prefill is 2.6× behind** same-model int8 types. It is the shipped kernel, not the format: replace
  the trellis decode with a constant and pp512 moves only 14 %. An int8 `mma.sync` path was written
  and is **not correct** (one staging bug after a 182× constant fix); it is parked behind an env flag,
  off by default, and never claimed.
- **Decode is 6–9 % behind** at ~21 % fewer body bits — a real trade, stated rather than buried.
- **Coding and multiple-choice margins** are what 3.34 bpw costs; the 5.00/6.00 variants hold 97–98 %.
- **Trellis embedding (K2E)** shrinks the file to 849.6 MB (from 1198 MB). Its quality at 3.34 bpw **is
  measured**: at 3.34 bpw the trellis embedding costs **1.2 coding points** (187 vs 190 of 246, controlled: same trained scales, only the embedding differs) -- the same cost as at 6.00 bpw (97.2 vs 98.4). What is *not* done is the runtime load test, and it must stay off for
  qwen35moe-class models until the MTP/draft graphs learn the I16 embedding branch.
- **Entropy coding** of the payload is worth exactly **1.000×** here (stored transitions are already
  uniform); the gap to the bound is the code construction, not the coder.

## 5. Repository map

| path | what |
|---|---|
| `main.pdf`, `main.tex` | the paper: *K2 and SantanaQAT: how far is 3-bit LLM quantization from the rate–distortion bound?* |
| `evidence/k2_evidence.tar.gz` | every per-item result behind every table (48 files) |
| `pipeline/` | SC sensitivity + RCO allocation + recipe format + the released map |
| `fig_rd.pdf`, `fig_ladder.pdf`, `fig_qat.pdf` | the three figures |
| `AGENTS.md` | **the operating manual** for an agent or a new maintainer: invariants, failure catalog, measured status |
| `FLAGS.md` | **every flag and environment variable**, what it does, and whether it is safe |
| `G5_pairs.log`, `G5_ppl_ladder.log` | the archived speed A/B and ladder-perplexity logs |
| `measurement-log.md` | the project's full measurement log (source of the analysis numbers) |
| `README.md` | this file |

Companion repositories and records:
- **Runtime PR** (the reference implementation): branch `k2-t16` of
  [TitanMamo/ik_llama.cpp](https://github.com/TitanMamo/ik_llama.cpp) — GGML op + CPU reference, CUDA
  GEMV / f16-MMA prefill / MoE kernels, loader, graph; see `PR_DESCRIPTION.md` in this repository.
- **Progress/lessons repository** (failures, wins, exact reproduction of the QAT series):
  [TitanMamo/qwen25-q3-qat](https://github.com/TitanMamo/qwen25-q3-qat).
- **Artifact**: [TitanMamo10/Qwen2.5-1.5B-K2-3.34bpw](https://huggingface.co/TitanMamo10/Qwen2.5-1.5B-K2-3.34bpw).

## 6. For agents

Read **[`AGENTS.md`](AGENTS.md)** before touching anything — it is the operating manual: naming, the five
invariants, a failure catalog ("it is broken" is usually one of nine known causes), how to verify an artifact
in four commands, the measured status of every lever, and what not to claim. Every switch and environment
variable is in **[`FLAGS.md`](FLAGS.md)**, including the ones that are traps.

## 7. Open items (tracked, not hidden)

- **Encoder speed:** assembly 91 s → 4 s shipped (byte-identical); TF32 Hessian and block-size
  occupancy measured and **rejected** (the latter byte-identical but 0.99×); a beam search in the tile
  Viterbi was **withdrawn as a lever** — its quality-preserving width equals the full state count at
  K=3 (8192), so it is the exact search (1.0×), and pruning widths trade quality (+2.3 % at 4096,
  +7.9 % at 1024) for work-count speedups the kernel has never demonstrated. What remains on the
  search half is a quality trade or a codec change, not a free win.
- **SantanaQAT v2** (QATv2): a format/training programme with pre-registered gates; see the
  progress repo. Its first item is the pipeline-quality sweep, not a better encoder (§10 of that plan).
- **Publication:** DOI/arXiv pending.

## 8. Attribution and license

Code: `ik_llama.cpp` (MIT, ikawrakow); the K2 format follows the EXL3/QTIP trellis tradition and the
scales-only QAT follows On-policy-Quantization-style distillation; the rate allocation implements
IST-DASLab's RCO (arXiv:2605.00649) formulation and the GSQ allocation line. The trellis encoder is
built on `exllamav3`'s (turboderp-org) converter. Paper, measurements, encoder tooling and released
artifact: Baran Mamo.
