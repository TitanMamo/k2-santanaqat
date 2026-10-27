- [x] I have read the [contributing guidelines](https://github.com/ikawrakow/ik_llama.cpp/blob/master/CONTRIBUTING.md)
- Self-reported review complexity:
  - [ ] Low
  - [ ] Medium
  - [x] High

---

## ggml: K2 — trellis-coded weights as an I16 tensor plus one custom op (ETA 3.34 bpw)

**What this adds.** A runtime for the trellis-coded weights that `exllamav3` already produces
(EXL3/QTIP class: block-Hadamard(128), 16×16 tiles, bitshift trellis, procedural `mul1` codebook). The
payload is a plain `I16` 3-D tensor `[16K, out/16, in/16]`, the side scales are ordinary F16 tensors, and
the runtime addition is **one op** with three entry points (dense matmul, MoE id-matmul, embedding gather).
It also adds a small CPU reference implementation so the op is testable without a GPU.

**Numbers** (RTX 3060; `IQ*_KS` are this fork's own types on the same F16 base; 5 interleaved pairs):

| model | | K2 | fork reference | note |
|---|---|---|---|---|
| **Qwen2.5-1.5B**, 3.34 body bpw | perplexity | **10.5894** | `IQ3_KS` 13.2312 (3.44 bpw) · `IQ4_KS` 10.9485 (4.54 bpw) | same 20-chunk protocol |
| | decode | **174–176 t/s** | `IQ3_KS` 189.6 · `IQ4_KS` 184.4 | 6–9 % behind, 5/5 pairs |
| | prefill | **3.18–3.25k t/s** | `IQ3_KS` 8363 · `IQ4_KS` 8098 | 2.55–2.63× behind, 5/5 pairs |
| | retention | **79.7 %** coding, 87.3 % HellaSwag | F16 = 100 % by construction | ARC 100 %, probes 106 %, GSM8K 104 % |
| **Qwen2.5-3B**, 3.46 body bpw | perplexity | **9.0830** | `IQ4_KS` 9.1932 (4.22 bpw on disk) · `IQ3_KS` 42.32 | F16 8.6550 |
| | prefill | **1781 t/s** | `IQ3_KS` 4686 · `IQ4_KS` 4656 | 2.62–2.63× behind, 5/5 pairs |
| | decode | **93.0 t/s** | `IQ3_KS` 123.6 · `IQ4_KS` 122.5 | **1.32× behind**, 5/5 pairs — see below |
| | retention | **84.8 %** (252/297) | F16 = 100 % | |

At a matched body rate the format is ~20 % better in perplexity than the 3-bit k-quant, and against
`IQ4_KS` it needs ~21 % fewer body bits at equal quality. On disk the file is *larger* (6.22 bpw at 1.5B)
because the embedding stays F16 — that is the honest half of the comparison and it is stated in the paper.

**Read the 3B row honestly.** Quality holds as the model grows — 9.083 vs `IQ4_KS` 9.193 while spending
~23 % fewer body bits — but **decode is 32 % behind**, not the 6–9 % of the 1.5B. Part of that is bytes
(K2's file is 1.86 GB against `IQ4_KS`'s 1.63 GB at this size, because the F16 embedding is now more than
half the file), and the rest is the per-K GEMV at these shapes. The trellis embedding (K2E) is exactly the
lever for the first part — 348 MB at 1.5B, ~0.9 GB here — and it costs 1.2 coding points, so it is a
decision, not a free win. `IQ3_KS` at 3B is 42.3 perplexity without an imatrix: that type needs an imatrix
at this size, and we report what we measured rather than quoting a tuned number we did not run.

**Why the payload is a tensor and not a `ggml_type`.** Not because a tile spans rows — this fork already
solves that: `IQ3_KS_R16`/`IQ4_KS_R16`/`IQ5_KS_R16` store **16 rows × 32 columns = 512 quants in 204 bytes**
with `blck_size = 32` and `type_size = 13` (204 B charged per 16 rows), and `*_R4`/`*_R8` have done the
same for longer. K2 *could* follow that pattern with `type_size = 2K` per row. What we did instead is
driven by three things: the payload's element width **is** the per-tensor rate `K`, so one type would have
to fix K (six types) or become metadata-driven like the Prism path; the per-matrix `su`/`sv` scales are
separate tensors rather than a block header; and the decode is one custom call rather than a `vec_dot`
specialisation. Porting it to the R16 pattern is future work, not a claim that it cannot be done.

**Honest limits.**

- **Prefill is 2.55–2.63× behind** same-model int8 types. It is the shipped kernel, not the format:
  removing the trellis decode entirely moves pp512 by 16 %. An int8 `mma.sync` path was written and is
  **not correct** (one staging bug left); it is not in this PR and not claimed.
- **Decode is 6–9 % behind** while reading ~21 % fewer body bits — a real trade.
- **3.34 bpw costs coding margin** (79.7 % vs F16): thin-margin items, not broken capability. The 5.00 and
  6.00 arms hold 97–98 %.
- The trellis embedding (K2E) saves 348 MB and costs **1.2 coding points**, measured.

**Gates** (all on this branch):

| gate | result |
|---|---|
| decode vs the reference Python codec | max abs diff **0.0** (K=2/3/6) |
| pack↔GGUF round-trip | 6 tensors, bytes and decode identical |
| perplexity class | 10.5894 vs F16 10.3099 |
| `compute-sanitizer` | racecheck **0 hazards**, memcheck 0 invalid accesses |
| CPU MoE test (`tests/test-k2-moe-cpu.cpp`) | K=2..6, 35B MoE shapes, E=4/6/8, 1/2/3/6 threads, ±`su`/`sv` — **bit-exact, max abs 0.000e+00** |
| CPU-only build (`-DGGML_CUDA=OFF`) | builds; inference answers; 2-chunk PPL 6.1862 |
| loader contract | 197 trellis tensors over 3 artifacts: shapes, K range, `su`/`sv` pairing, byte accounting |

**Diff, by role** (25 files, +2979/−97):

| role | lines | what |
|---|---|---|
| CUDA kernels | 1315 | `t16-gemv.cuh` (616) per-K GEMV templates, `t16-matmul.cu` (528) CUDA-core reference + f16-MMA prefill + MoE, `t16-embed.cu` (96), 5 `mmvq-instance-t16-k*.cu` (66) so the existing glob picks them up, 3 headers |
| CPU kernels | 694 | `iqk_cpu_ops.cpp`: scalar reference (the specification), AVX2 MoE, dense AVX2, embedding gather |
| loader + graph | 456 | tensor recognition, `create_tensor`/`create_embd` paths, op build, split-mode guard |
| tests | 316 | `test-k2-moe-cpu.cpp` + one CMake line |
| ggml core | 198 | three op enums, the op's `ggml.c` plumbing, RPC version bump |

The bulk is the two kernel implementations plus the CPU specification of the op; the plumbing is 456 lines
across five upstream files, and each of those hunks is a new `case`/branch rather than a changed path.

**The ask.** Review the *format contract* (tensor layout, side tensors, the "never re-apply `su`/`sv` or the
Hadamard" rule) and treat the kernels as replaceable — the prefill kernel in particular is a floor, not an
achievement. If the answer is "this does not belong in the runtime", say so plainly: the packer, the
encoder and the study stand on their own.

---

The measurements, per-item evidence and the full study are in the deposit
([10.5281/zenodo.23678902](https://doi.org/10.5281/zenodo.23678902)); the reference artifact is on the Hub
(`TitanMamo10/Qwen2.5-1.5B-K2-3.34bpw`, two packs: PTQ `584a86cb…` = 73.6 % coding, SantanaQAT
`45d62417…` = 79.7 % — quote the file you ran).
