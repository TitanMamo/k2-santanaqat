- [x] I have read the [contributing guidelines](https://github.com/ikawrakow/ik_llama.cpp/blob/master/CONTRIBUTING.md)
- Self-reported review complexity:
  - [ ] Low
  - [ ] Medium
  - [x] High

---

## ggml: K2 — trellis-coded weights as an I16 tensor plus one custom op (ETA 3.34 bpw)

**Numbers up front (one RTX 3060, Qwen2.5-1.5B, 3.34 bpw):**

| | measured | context |
|---|---|---|
| decode (tg128) | **174-176 t/s** across three sessions (5/5 pairs in the interleaved A/B) | 6-9% behind same-model `IQ4_KS`/`IQ3_KS`, at ~21% fewer body bits (3.34 vs 4.25 bpw) and better perplexity |
| **prefill (pp512)** | **3.18-3.25k t/s** (3180-3248 across sessions) | **27-28x** the CUDA-core reference; **2.55-2.63x behind** `IQ4_KS`/`IQ3_KS` in the paired A/B (8098-8160 / 8363-8403 t/s across sessions; 8488 single-run) -- this is the number the kernel limits are about |
| **recovery at 3.34 bpw** | **79.7 % coding, 87.3 % HellaSwag, 100 % ARC, 106 % probes, 104 % GSM8K** | each as a share of F16 on the same items (F16 = 100 % by construction on the solved set) |

**What this adds.** A trellis-coded quantization (TCQ) format for transformer weights, stored as a
plain `I16` 3-D tensor plus two half-precision side vectors per matrix and one custom matmul op. It
is the storage format that `exllamav3` already produces (`EXL3`/QTIP-class: block-Hadamard(128)
rotation, 16×16 tiles, bitshift trellis, procedural `mul1` codebook), brought into `ik_llama.cpp`
so those models run natively.

**Where to get an artifact.** The deployed 3.34 bpw 1.5B build is on the Hub —
[`TitanMamo10/Qwen2.5-1.5B-K2-3.34bpw`](https://huggingface.co/TitanMamo10/Qwen2.5-1.5B-K2-3.34bpw)
(md5 `584a86cb105e9771e6b30d3c7f08f52d`, 1198 MB) — so the kernels can be exercised without running the
encoder first. It needs this branch to load.

**Two packs ship, same size, different scales** — quote the one you ran. `…k2-3.34bpw.gguf` is the PTQ
pack: **10.5894** perplexity, **73.6 %** coding. `…k2-3.34bpw-santanaqat.gguf` (md5
`45d624171dc4a6168aff7ce6dae2794f`) is the same file plus the 394 trained scale tensors: **10.2886**
perplexity, **79.7 %** coding, **87.3 %** HellaSwag. Only the **3.34** rows below are SantanaQAT (4.06/5.00/6.00/6.00+K2E are PTQ arms); see the SantanaQAT
pack unless a row says PTQ.
The run logs behind those numbers are in
[`evidence/k2_evidence.tar.gz`](https://huggingface.co/TitanMamo10/Qwen2.5-1.5B-K2-3.34bpw/blob/main/evidence/k2_evidence.tar.gz)
(48 JSONs: per-item eval dumps, the codec-reference and pack round-trip checks, weight-error and damage
maps). The speed pair-test log ships separately as `G5_pairs.log`.

**How the rate map was chosen** — not a uniform step. A per-tensor sensitivity measurement (weight-noise
injection at two relative Frobenius norms, KL of the last-token logits) feeds an exact-budget allocator
(greedy + exchange repair, with tied q/k/v and gate/up), which emits the per-tensor K recipe the encoder
consumes. The deployed file realizes 61.1% of trellis weights at K=3, 18.8% at K=4, 5.0% at K=5 and
15.1% at K=6 (the head) — exactly 3.34 body bpw (target 3.34, achieved 3.3394, predicted KLD 0.1553).
The sensitivity file, the allocation and the recipe ship in the paper's deposit (`pipeline/`, not on the Hub),
so the map can be re-derived rather than taken on trust.

**What it is worth (measured, Qwen2.5-1.5B, RTX 3060, 246-item strict retention on the items F16
solves, Clopper-Pearson 95%):**

| body bpw | file | coding | HellaSwag | PPL |
|---|---|---|---|---|
| 3.34 | 1198 MB | 79.7% [74.1, 84.5] | 87.3% [84.6, 89.8] | 10.2886 |
| 4.06 | 1316 MB | 90.2% [85.8, 93.6] | — | 10.4966 |
| 5.00 | 1470 MB | 97.6% [94.8, 99.1] | 93.6% [91.5, 95.3] | 10.3444 |
| 6.00 | 1634 MB | 98.4% [95.9, 99.6] | 97.0% [95.8, 97.9] | 10.3150 |
| 6.00 + trellis embedding | **1285 MB** | 97.2% | 96.7% | 10.3090 |
| F16 | 3094 MB | 100% | 100% | 10.3099 |

The rest of the battery is **at or above the F16 reference**, which is worth stating because coding
and HellaSwag are the two axes where a 3-bit perturbation shows up at all (both are thin-margin
tasks: free-form generation and a first-token letter decision). The same artifacts on the same
items:

| arm | coding (246) | HellaSwag (1200) | ARC-Easy (200) | probes (98) | GSM8K (200) |
|---|---|---|---|---|---|
| F16 | 100% | 100% | 87.0 | 83.7 | 50.0% |
| **K2 3.34** | 79.7% | 87.3% | **87.0** (100%) | **88.8** (106%) | **52.0%** (104%) |
| K2 5.00 | 97.6% | 93.6% | 85.5 (98%) | 85.7 (102%) | — |
| K2 6.00 | 98.4% | 97.0% | 87.0 (100%) | 84.7 (101%) | — |

So at 3.34 bpw the model keeps **all** of its ARC capability (87.0 of 87.0) and actually scores
*higher* on the probe set (88.8 vs 83.7), while losing thin-margin coding and multiple-choice
items; it is a precision limit on margin, not a broken model.

**GSM8K: it was mis-scored, not degenerate -- and we say how we found out.** An earlier draft of
this text claimed the quantized arms "recovered" math (38.5% against F16's 15.0%). Both numbers were
an artifact of the *scorer*: neither arm emits the `<|im_end|>` stop token on this 4-shot prompt, so
both keep generating past the answer, and the harness took the **last** `####` marker -- i.e. a number
from the degenerate tail (F16 drifts into unrelated text, the quantized arm loops on its own answer:
17.5 markers per answer against 3.0). The tell was the direction: a precision change cannot make the
reference model 23.5 points *worse*. With the generation retained and the **first** complete answer
scored, the same 200 items read **F16 100/200, PTQ 91/200, deployed arm 104/200** -- 91% retention for
PTQ and 104% (a four-item tie) for the arm we ship. The harness now keeps every generation and records
which extraction rule fired. This is the one axis where the fix was in *our* measurement code rather
than the model, and it is stated here because the first version of this very paragraph got it wrong.

**The 3-4 bit comparison, measured on this card (20-chunk perplexity, same protocol).** The fork's
own `IQ3_KS` at **3.44 bpw on disk** reads PPL **13.2312**; `IQ4_KS` at **4.54 bpw** reads **10.9485**;
K2 at a **3.34 bpw body rate** reads **10.5894**. So at a comparable rate K2 is ~20% better in
perplexity than the 3-bit k-quant, and against `IQ4_KS` it needs ~21% fewer body bits at
equal quality. The caveat, stated rather than hidden: the *file* is not smaller until the trellis
embedding is used (K2E, 1285 MB at 6.00 bpw), because the F16 embedding alone is 466 MB of the 1198 MB file
(39 %; the head is already trellis at K=6, 175 MB) and ours
stay f16 — that is the 6.22 bpw on-disk figure beside the 3.34 body rate.

**Why not reuse an existing type? There is already a trellis type in this fork.** This is the
first question we would ask, and the honest answer is that trellis-coded weights *are* precedented
here: the `iq1_kt`-`iq4_kt` family (e.g. `block_iq3_kt`, 100 B per 256 weights = 3.125 bpw) is
decoded by `Trellis3<>` in `ggml/src/iqk/iqk_gemm_ktquants.cpp`. That precedent is part of why we
think K2 belongs. But it is a different codec, and three differences are load-bearing:

| | `IQ3_KT` (already here) | K2 |
|---|---|---|
| trellis domain | one 256-weight block, contiguous along `ne0` | a **2-D 16x16 tile** (input x output) |
| fits `ggml_type` | yes -- a block is a contiguous row run | **no** -- a tile spans 16 rows, so no payload row is contiguous with a tensor row |
| codebook | its own fixed L=16/N=8 | EXL3/QTIP `mul1`, 16-bit sliding window + per-tensor `su`/`sv` |
| rate | fixed 3.125 bpw | per-tensor K=2..6 (2.0-6.0 bpw), set by the rate allocator |

Sharing the `Trellis3` class would share the word, not the codec: different domain, different
codebook, different rate. (For completeness: in our own quantization experiments the KT codebook
reaches ~22% RMS weight error on this model, well above the ~13.5% cliff this build shows, so it is
not an alternative we could point users at either. That is a statement about our measurements, not a
request to change anything, and this PR does not touch the KT family -- the two coexist.)

Two further reasons the existing types do not substitute:

1. **Quality at equal rate.** Measured above: `IQ3_KS` 13.2312 at 3.44 bpw on disk, `IQ4_KS` 10.9485
   at 4.54 bpw, K2 10.5894 at a 3.34 body rate -- ~20% better than the 3-bit k-quant at a comparable
   rate, and ~21% fewer body bits than `IQ4_KS` at equal quality (the on-disk file is 6.22 bpw because
embedding and head sit at higher rates). The shipped
   types cost a bit more and give a bit less at this rate.
2. **The format already exists outside this repo.** EXL3-class packs (and QTIP checkpoints) are
   produced by `exllamav3`'s converter; without a runtime path those models cannot be run here at
   all. This PR is the missing half of a format that already exists, not a new invention competing
   with the shipped ones.

**Why it cannot be a `ggml_type` (and why `IQ3_KT` can).** A `ggml_type` assumes blocks laid out
along `ne0` inside a row, which is exactly why `IQ3_KT` fits: its trellis is one-dimensional,
running along one row's 256 weights. A K2 tile is 16x16 and spans 16 rows, and its 256 values are
*overlapping 16-bit windows of one circular `256·K`-bit stream*: a value depends on bits that are not
contiguous with it, and no payload row is contiguous with a tensor row. So the payload is a plain `I16` 3-D tensor `[16K, out/16, in/16]` with standard
strides (no custom `ggml_row_size`), the side vectors are ordinary F16 tensors, and the only new
runtime piece is one op. The precedent in this fork is **Prism** (metadata-driven Hadamard +
`ggml_hadamard`), whose rotation this reuses.

**Speed.** 176.4 t/s decode (tg128) on the 1.5B at 3.34 bpw — 722 MB of weights per token, i.e.
≈127 GB/s and 73% of this card's isolated GEMV ceiling — and **3.18-3.25k t/s prefill (pp512: 3180-3248 across sessions)** through
an f16-MMA kernel: `wmma` 16x16x16, 8 warps = 128 tokens per block, the weight tile decoded once per
block into shared memory, the activation tile staged token-major so the `col_major` fragment needs
no transpose. That is **28x** the CUDA-core kernel it replaces (116 t/s) at unchanged quality
(perplexity 10.5895 -> 10.5894 over 20 chunks; 8.7896 -> 8.7893 at `-ub 100`, which exercises the
warp tail). Decode never touches that path: it stays on the GEMV. For calibration on this card, 5
interleaved pairs (order alternating) of the same model give: **IQ3_KS** (rate match, 3.44 bpw on
disk) 8363 t/s prefill / 189.6 decode at PPL 13.2312, and **IQ4_KS** (quality match, 4.54 bpw)
8098 / 184.4 at PPL 10.9485, against our 3180 / 174.0 at 10.5894. So K2 is **2.55-2.63x behind on
prefill** (5/5 pairs) and **6-9% behind on decode** (5/5), while perplexity is 20% better than the rate-matched
IQ3_KS and 3.3% better than IQ4_KS. Clocks were not locked (`nvidia-smi -lgc` needs root) -- that is
the one caveat left on the speed gate. `K2_T16_SLOW=1` forces the reference GEMV and `K2_T16_NOMMA=1` the
CUDA-core prefill; both agree to four decimals of perplexity.

### Honest statement about these kernels: we ship one, and it is not good

The tensor-core prefill in this PR **is the default path** — every `n > 1` call runs it, and the
CUDA-core code is only the fallback (`K2_T16_NOMMA=1`). We ship it knowing it is not a good kernel,
and we would rather say so than dress it up:

| path | pp512 | what it is |
|---|---|---|
| CUDA-core reference (half2 FMA) | 116 t/s | the oracle and the escape hatch, 28x slower |
| **f16 wmma prefill — what this PR ships and runs** | **3.18-3.25k t/s** (3180-3248) | **not good**: 2.55-2.63x behind `IQ4_KS`/`IQ3_KS`, and only 66% of this card's f16-with-f32-accumulate ceiling (~4.9k t/s) |
| int8 MMA (MMQ-shaped, `mma.sync.s8`) | — | attempted, **not shipped**: output still uncorrelated with the reference (the 182x codebook-constant bug is fixed, one staging bug remains) |
| `IQ4_KS` (the fork's int8 MMQ) | 8488 t/s single-run, 8160 in the paired A/B | what a good kernel looks like here |

Why we ship it anyway: the alternative is prefill on CUDA cores at 116 t/s, which makes the format
unusable rather than merely uncompetitive. So this kernel is a **floor, not an achievement** — 28x the
fallback, 2.6x behind the state of the art.

**It is the kernel, not the trellis.** This matters for how the PR should be read, so here it is
measured rather than asserted, on the kernel this PR ships:

- **Removing the trellis decode entirely** (weights replaced by a constant, everything else identical)
  takes pp512 from **3207 to 3720 t/s** — the format's own decode is **14 %** of prefill time. With no
  decode at all we would *still* be 2.3x behind `IQ3_KS`, because what remains is an ordinary GEMM on
  an **f16 path with f32 accumulate**, whose hardware ceiling on this card is ~4.9k t/s: half the int8
  rate that `IQ3_KS`/`IQ4_KS` ride (their ceiling is ~9.8k).
- **Per byte the format is at parity with the rate-matched type**: effective bandwidth is 127 GB/s for both
  K2 (174 t/s x 722 MB) and `IQ3_KS` (190 t/s x 663 MB), while `IQ4_KS` reaches ~161 GB/s at its higher rate
  (184 t/s x 874 MB). K2 reads ~21 % fewer body bits than `IQ4_KS` (3.34 vs ~4.25 bpw; its on-disk file is
  6.22 bpw because the embedding stays F16).
- **The decoded values are int8-representable** (±3.45, one constant scale for the whole tensor), so
  nothing in the codec prevents an int8 `mma.sync` path — which is precisely what the parked kernel
  does, and why its arithmetic ceiling is double ours.

So the 2.6x is a property of *this implementation* (f16 arithmetic, one barrier per 16-input stage,
no `cp.async`, 64x128 tiles at 2 blocks/SM), not of trellis coding as a format. Whether a polished
int8 MMQ on these same bits actually reaches `IQ3_KS`'s class is **not demonstrated here** — that is
the honest state, and it is the next piece of work rather than a claim.

What a good one takes, since the number is not a mystery: **int8 MMA** (`mma.sync.m16n8k32.s8` runs at
twice the f16-with-f32-accumulate rate, i.e. a ~9.8k t/s ceiling instead of our ~4.9k), **`cp.async`
double buffering** instead of our load-convert-store staging, **128x128 tiles** with 3-4 blocks/SM
instead of 64x128 with 2, and the MoE path given the same treatment (it has no tensor-core variant at
all yet). That is roughly the 1.74x arithmetic gap plus the 1.5x scheduling gap, and it is ordinary
kernel work — not assembly, and not a format change.

**Gates.**

| gate | result |
|---|---|
| bit-exact decode vs the reference Python codec | max abs diff 0.0 (K=2/3/6) |
| pack↔GGUF round-trip (6 tensors, bytes + decode identical; 55/55 converter layout checks on the Qwen3.5-MoE layouts) | pass |
| perplexity class | 10.2886 vs F16 10.3099 |
| coherence (France probe, CPU + GPU) | pass |
| fast vs slow path | PPL identical, texts identical |
| memcheck / racecheck | **racecheck: 0 hazards, 0 errors, 0 warnings**; memcheck: 0 invalid accesses (its 2 reports are `cudaErrorGraphExecUpdateFailure`, i.e. the sanitizer's own instrumentation invalidating CUDA-graph updates — the run completed and produced timings) |
| codebook vs oracle, all 65536 windows | **65536/65536 bit-identical**, max diff 0 (`t16_probes/`) |
| template-instance split | instances build separately; perplexity unchanged (8.9471) |
| **speed A/B, 5 interleaved pairs** | prefill 3180 vs `IQ3_KS` 8363 / `IQ4_KS` 8098 t/s (2.55-2.63x behind, 5/5 pairs); decode 174.0 vs 189.6 / 184.4 (6-9% behind, 5/5); clocks not locked (no root) |
| loader contract | `loader_probe.py`: 197 trellis tensors over 3 artifacts -- shapes, K range, su/sv pairing, byte accounting, no orphans |
| **CPU MoE kernel vs its scalar reference** (`tests/test-k2-moe-cpu.cpp`) | K=2..6, the 35B MoE in/out shapes, E=4/6/8, 1/2/3/6 threads, with and without `su`/`sv` — **bit-exact in every case** (max abs **0.000e+00**), `all cases passed` |
| rebased-tree smoke (upstream `main`, 46 new commits) | `llama-cli` builds; greedy France probe answers "Paris"; 1-chunk PPL **6.3438** vs F16 **6.3425**; K2 6.00 bpw output **byte-identical to F16** |

**Diff, in the shape this fork uses for quant types.**

| file | lines added | role |
|---|---|---|
| `ggml-cuda/t16-gemv.cuh` | 616 | GEMV kernel templates + shared primitives, no instantiations |
| `ggml-cuda/t16-matmul.cu` | 528 | CUDA-core reference matmul, **f16-MMA prefill**, MoE path, dispatch, the two op entry points |
| `ggml-cuda/template-instances/mmvq-instance-t16-k{2..6}.cu` | 5 × 12-15 | one file per rate (the fork's per-type layout), picked up by the existing `mmvq-instance*.cu` glob — **no CMakeLists change**; body is a launcher, not a device `vec_dot_*` (standalone GEMV, not an MMVQ specialization) |
| `ggml-cuda/t16-embed.cu` | 96 | trellis embedding lookup |
| `iqk/iqk_cpu_ops.cpp` | — | CPU: MoE AVX2, dense AVX2 + scalar reference, embedding gather |
| `tests/test-k2-moe-cpu.cpp` (+1 line in `tests/CMakeLists.txt`) | 315 | the CPU MoE kernel against its scalar reference, bit-exact; wired into the build (it was not before, so it never compiled) |
| `ggml.h`, `ggml.c`, `ggml-cuda.cu`, `llama-model-loader.cpp`, `llama-load-tensors.cpp`, `llama-build-context.cpp` | — | op plumbing, loader, graph |

No new `ggml_type`, no gguf-py size table, no `llama-quantize` changes at all (the shape-eligibility check lives in the loader).
Compile cost, measured: **one instance rebuilds in 9 s on a single core** (the pre-split
monolithic TU: 21 s).

## Known limits — stated up front

1. **One historical multi-slot sampling abort, not reproducible since.** A single occurrence
   (2026-10-25) aborted in `llama-sampling` under `-np 4` with four concurrent generators, in a
   memory-contended window (a trainer had just exited). Three reproduction attempts on the current
   build — `-np 2` and `-np 4`, four concurrent 64- and 480-token generations, fast and reference
   paths — are **all clean**. The cause is **not established** (one occurrence, three clean reproduction attempts,
and a single-slot racecheck cannot clear a two-slot race); we do not claim a K2 race until shown
   otherwise. A `compute-sanitizer` pass now backs that up: **racecheck 0 hazards / 0 errors**,
   memcheck 0 invalid accesses (its two reports are the sanitizer breaking CUDA-graph updates, not
   memory faults).
2. **No MMQ, and the MoE prefill has no tensor-core path yet.** Dense prefill now runs on tensor
   cores (f16-MMA, above); the *MoE* path (`ggml_cuda_op_t16_mul_mat_id`) still decodes and multiplies
   on CUDA cores, so a sparse model keeps a slow prefill. An int8
   `mma.sync` path was written against `mmq.cuh`'s shape and is **not correct** (one staging bug remains
   after a 182x constant fix), so it is parked outside this PR and not claimed (it is not in the tree)
   here. Measured consequence today, same card/model/binary: prefill 3180 t/s against `IQ3_KS` 8363
   and `IQ4_KS` 8098 t/s (2.55-2.63x behind, 5/5 interleaved pairs), where the CUDA-core kernel was 27-28x
   behind. Decode is 6-9% behind (174-176 vs 189.6/184.4 paired, 191.1/185.1 earlier session) while reading ~21% fewer body
   weight at better perplexity (10.5894 at a 3.34 body rate vs 13.2312 at 3.44 on disk and 10.9485 at
   4.54), which is why we say the case is quality-per-byte, not throughput.
3. **The encoder is tooling-side** (`exllamav3` / our Python writer). This PR is the runtime.
   **CPU is a correctness path, not a performance path**, but the numbers are better than an earlier
   draft of this text claimed: interleaved on a quiet box (6-core Skylake, `-ngl 0`, r=2, two
   rounds), dense AVX2 reads **2.98 / 2.97 t/s** tg8 against the scalar reference's **0.75 / 0.75**,
   i.e. **3.97x**, bit-identical. (The earlier 0.23 t/s / 1.35x pair came from a contaminated window
   and is withdrawn.) It is still ~60x below the GPU.
4. **Speed is 2.55-2.63x behind on prefill and 6-9% on decode**, measured against both a rate-matched
   `IQ3_KS` and a quality-matched `IQ4_KS` (5 interleaved pairs each), and we are saying so rather
   than burying it.** K2's case is quality per byte at a *lower* rate (10.5894 at
   a 3.34 body rate against `IQ4_KS` 10.9485 at 4.54 bpw, and `IQ3_KS` 13.2312 at 3.44), not raw decode
   throughput. The format is at
   127 GB/s of a measured 175 GB/s ceiling on the dense GEMV, and the remaining gap is kernel
   efficiency and the absent MMQ path — work we have committed to as follow-up and can now measure
   against a working baseline. If the maintainer's position is that
   a type must not regress decode throughput at introduction, that is a fair reason to hold this
   back, and we would rather hear it than argue the point.
5. **Scope of the kernel work.** The GEMV is split into per-K files under `template-instances/` using
   the fork's file layout (shared template header + a file per rate, picked up by the existing glob).
   The instance body is deliberately different from the `iqk` instances: those supply a device
   `vec_dot_*` that the shared templates turn into a kernel, ours is a launcher that fixes K,
   because this kernel is a standalone GEMV rather than an MMVQ specialization. Beyond that the
   GEMV is still one design (one warp per k-split); prefill *does* use tensor cores (f16 wmma), and
   what is missing is int8 MMA, a `cp.async` pipeline and 128x128 tiling. That is where the throughput
   work goes next.

## The ask

Review the *format contract* first (§1 of `K2_SPEC.md`: the tensor layout, the side tensors, the
"never re-apply su/sv or the Hadamard" rule) and treat the kernels as replaceable. If the answer is
"this does not belong in the runtime", say so plainly — the packer and the study stand on their own,
and the type can be revisited when the speed work lands.
