# Runtime speedups shipped on `k2-speed` (measured)

The numbers in `README.md` and `PR_DESCRIPTION.md` up to 2026-10-27 describe the runtime as it stood
when PR #2762 was opened (`k2-t16`, head `36a6fc18`). Since then a **12-commit speed branch,
`k2-speed`**, has been measured on the same RTX 3060 and the same released artifact
(`…k2-3.34bpw.gguf`, md5 `584a86cb105e9771e6b30d3c7f08f52d` — verified before benchmarking).

**None of this is in the PR yet.** The branch is local/unpushed, and it also carries unrelated
Spark-architecture work, so a PR update would have to be the kernel commits alone, rebased onto
`k2-t16`.

## The kernels

| commit | what changed | measured |
|---|---|---|
| `983279b2` | affine-integer trellis decode (no fp16 codebook step) | basis for the rest |
| `eb3f3785` | `inv_perm` staged in shared memory | +10 % |
| `6c16c9c2` | band-staged trellis words + precomputed index table | +32 % total |
| `3c3f8161` | fuse `su`/`sv` into the Hadamard rotations — 5 kernels → 3 per matmul | — |
| `82e19a4e` | CPU path: fused Hadamard scales | — |
| `fe862a56` | JIT materialisation — decode each tensor once into Q8_0, then stock MMQ | — |
| `36d53965` | stage the four decode tiles in shared memory (768 B) | — |
| `9786fa6c` | byte sum via `dp4a` instead of 3 shifts + 3 ands + 3 adds | **+2 %** |
| `925b5790` | one warp per 128-point Hadamard, no smem, no barriers | **+1.6 %** |
| `b4ca6dea` | GEMV: back to 2 output tiles per block | **decode +12.5 % on the 3B** |
| `a15902c4` | GEMV: activation pair via one `float2` instead of two scalars | — |
| `791c1711` | drop the dead clamp in the decode kernel (bit-exact) | **+2.4 % prefill** |

All are bit-exact for the same pack (PPL unchanged to 4 dp where re-measured).

## Arm-level result

Protocol: `llama-bench -p 512 -n 128 -r 5 -ngl 99 -t 6`, one warmup pass first (the card idles at
210 MHz, so a cold first rep swings prefill by 20-30 %), same session, same binary, artifact md5
verified.

**Qwen2.5-1.5B, 3.34 body bpw** (1.11 GiB in VRAM)

| arm | prefill pp512 | decode tg128 |
|---|---:|---:|
| **K2** | **5,660 ± 475** | **202.7 ± 0.9** |
| `IQ3_KS` 3.19 bpw | 8,700 ± 1,054 | 191.2 ± 0.5 |
| `IQ4_KS` 4.25 bpw | 8,614 ± 1,323 | 177.0 ± 0.1 |

**Qwen2.5-3B, 3.46 body bpw** (1.81 GiB)

| arm | prefill pp512 | decode tg128 |
|---|---:|---:|
| **K2** | **3,132 ± 156** | **113.7 ± 0.4** |
| `IQ3_KS` | 4,797 ± 364 | 123.9 ± 0.4 |
| `IQ4_KS` | 4,799 ± 477 | 122.6 ± 0.2 |

## Protocol warning — the two numbers in print are not comparable

The figures published before this note (decode **174–176 t/s**, prefill **3.18–3.25k**) were read
from **`llama-server`'s `/slots` counters** (`predicted_per_second` / `prompt_per_second`, via
`qat_tools/reliable_ab.py`). Everything in this document is **`llama-bench`**. The two paths differ
by sampling/tokenisation overhead, so a `llama-bench` figure and a `/slots` figure must never be
subtracted from one another — the earlier draft of this file did exactly that and its "+76 % prefill"
was an artefact of the tool swap, not of the kernels.

**What is solid** (one tool, one session, same artifact):

| | K2 | `IQ3_KS` | `IQ4_KS` | verdict |
|---|---:|---:|---:|---|
| 1.5B prefill pp512 | 5,660 | 8,700 | 8,614 | 34.9 % / 34.3 % behind |
| 1.5B decode tg128 | 202.7 | 191.2 | 177.0 | 6.0 % / 15.0 % ahead |
| 3B prefill pp512 | 3,132 | 4,797 | 4,799 | 34.7 % behind |
| 3B decode tg128 | 113.7 | 123.9 | 122.6 | 8.2 % behind |

**What is pending**: a like-for-like before/after. The pre-`k2-speed` binary is still on disk
(`ik_llama-k2/build-cuda/bin/llama-bench`, built 2026-10-26), so the A/B is one command per arm once
the GPU is free:

```bash
# old build vs new, same tool, same models, same session
build-cuda/bin/llama-bench -m <pack> -p 512 -n 128 -r 5 -ngl 99 -t 6
build-spark/bin/llama-bench -m <pack> -p 512 -n 128 -r 5 -ngl 99 -t 6
```

## Where the remaining prefill gap comes from

Unchanged from the PR analysis: prefill materialises the trellis once per chunk and then runs the
stock int8 MMQ path, so it pays a construction cost per chunk that the native int8 types do not have.
The 12 commits above cut that cost by ~40 %, not to zero. Removing the trellis decode entirely moves
pp512 by 16 %, which bounds what any further kernel work can recover.
