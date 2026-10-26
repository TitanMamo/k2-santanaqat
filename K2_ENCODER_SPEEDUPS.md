# Encoder speedups shipped in this release (measured)

Scope: what actually got faster in the K2 encoder (`exllamav3` converter + K2 writer), what was
rejected by measurement, and what is deliberately deferred. Everything here is reproducible from the
scripts named, and none of it changes the released pack (md5 `584a86cb105e9771e6b30d3c7f08f52d`).

## 1. Assembly (GGUF rewrite): 91 s -> 4 s, byte-identical

The trellis payload + side vectors were assembled into the GGUF with a per-tensor Python loop; it is now
a batched rewrite. Verified byte-identical on the released artifact (md5 unchanged). Saves ~87 s of a
19.4 min encode (~7 %).

## 2. TF32 calibration Hessian: REJECTED (not shipped)

The Hessian is `H = X^T X`, the only pure matmul in the encoder, so an env-only TF32 A/B was the first
test (`bench_scripts/tf32_ab.sh`, 6 modules):

| layer | fp32 `rfn` | TF32 `rfn` | delta |
|---|---|---|---|
| 0 | 0.062791 | 0.062981 | +0.3 % |
| 1 | 0.019867 | 0.027072 | **+36 %** |
| 2 | 0.009972 | **Cholesky failed** (`input is not positive-definite`) | — |

A TF32-accumulated `X^T X` loses positive definiteness and LDLQ needs the Cholesky factor. **Not usable.**
(The logged "121 s vs 270 s" was two layers before the abort, not a speedup; the script's `rc=$?` was
expanded after a `date` subshell and always read 0, hiding the crash — fixed.)

## 3. Where the 46 % non-search time actually goes (profiled)

`bench_scripts/measure_hessian_split.py` (torch profiler, 3 modules, wall 144.3 s, CUDA op time 195.2 s
= 1.35x overlap):

| item | CUDA s | share of wall |
|---|---|---|
| tile search `quantize_tiles_frac_kernel<3,...>` | 61.1 | 42 % |
| **Hessian accumulation `addmm_ [8960,8960]`** (750 calls = 250 cal rows x 3 modules) | **24.7** | **17 %** |
| capture dtype conversions (`_to_copy`/`to`/`copy_`) | ~8 | 6 % |
| `Memcpy DtoH` (state -> host, pageable) | 6.1 | 4 % |
| calibration forwards (f16 GEMMs, paged attn) | ~12 | 8 % |
| capture elementwise (`isfinite`, `square`, `pow`, `mul`) | ~6 | 4 % |

Two structural facts, both actionable without touching the algorithm:
- **The Hessian build is 512k token positions** (250 rows x 2048 cols) pushed into an 8960x8960 matrix,
  i.e. 8.2 s/layer. `--cal_rows` / `--cal_cols` cut it linearly.
- **Checkpointing is synchronous and rewrites the whole state** (3.1 GB here, ~70 GB for a 35B) every
  `-cpi` seconds (default 120) through `save_tensor` + a ckpt_new/ckpt/ckpt_old rename dance. On ntfs3
  (~150 MB/s) one save is ~20 s *if* the write reaches the disk -- measured, the page cache absorbs it:
  2 saves cost ~5 s at 1.5B (section 4b).
- The rfn/cos/sqnr proxies are already cheap: `num_ref_states = 5`. That idea is dead.
- Hessian sharing is already done: **4 Hessians per layer**, not 7 (q/k/v share, up/gate share).

## 4. Occupancy (1 -> 2-3 blocks/SM): see the A/B result

The kernel runs 1024 threads with `__launch_bounds__(1024, 1)` and a `__syncthreads()` every trellis step,
so one block per SM cannot fill the barrier. The host already sizes its scratch from
`cudaOccupancyMaxActiveBlocksPerMultiprocessor`, so lowering `qt_optimized_threads` is one constant.
Script: `bench_scripts/encoder_occupancy_ab.sh` (builds in the `cuda-box` distrobox — there is no CUDA
compiler on the host). Quality guarantee is byte-identical: the run's `ckpt/state.safetensors` md5 plus
per-layer `rfn`, against the measured baseline (268 s, md5 `e152a21cf7a40656c3e113d4f3adfbe9`).

RESULT: **NO GAIN — REJECTED.** NT=1024 baseline 268 s vs NT=512/256 271 s (0.99x, i.e. 1% *slower*),
with a byte-identical pack (md5 `e152a21cf7a40656c3e113d4f3adfbe9`) and identical per-layer `rfn`. So
the tile search is throughput-bound, not barrier/occupancy-bound: halving the block size doubles the
work per thread and buys nothing. Header and `.so` reverted automatically. This retires the
"35B 9 h -> 6.6 h" occupancy projection; and the search has NO lossless lever left either -- the beam's quality-preserving width equals the full state count at K=3, so it is the exact search (1.0x, cf. section 5).

## 4b. Checkpoint I/O and calibration data: MEASURED (2026-10-26)

**Checkpointing is ~2%, not the 15% I estimated.** The converter checkpoints synchronously and rewrites the
whole state (3.1 GB at 1.5B) every `-cpi` seconds (default 120). Measured, 6 modules:

| arm | wall | checkpoints | pack md5 |
|---|---|---|---|
| A: default, work dir on /mnt/Data (ntfs3) | **269 s** | 2 | `e152a21c...` |
| B: `-cpi 100000` (no periodic saves) | **264 s** | 0 | (none written) |
| C: default interval, work dir on the SSD | **266 s** | 2 | `e152a21c...` (identical to A) |

Two whole-state saves cost **~5 s**: the page cache absorbs them, so the device does not matter at this
size (C is 3 s faster than A = noise). The scaling argument survives only where the cache cannot hold the
state -- a 35B is ~70 GB against 31 GB of RAM, which is where `-cpi` and an incremental save actually pay.
Conclusion: this is **quality-lossless but worth ~2%**, not a headline.

**Calibration data: -23% encode, quality unmeasured, pack changes.** `-cr/--cal_rows` and `-cc/--cal_cols`
halve the token positions the Hessian sees (250x2048 = 512k positions):

| arm | wall | pack md5 | max abs rfn delta vs A |
|---|---|---|---|
| A: 250 x 2048 | **271 s** | `e152a21c...` | -- |
| B: 125 x 2048 | **207 s** (-24%) | `fb192031...` | **0.003600** (layer 1, +18% rel.) |
| C: 250 x 1024 | **208 s** (-23%) | `2d98a800...` | **0.002844** (layer 3, -27% rel.) |

`rfn` moves in *both directions* and the reductions mostly make it look **better** -- which is a train/eval
overlap: the proxy is measured on the same rows that build the Hessian, so less data means a better fit to
the measured rows. The proxy therefore cannot be used to adopt this. Adopting it needs a full-artifact A/B
(full encode + 20-chunk PPL + 246-item retention, ~1-2 h per arm). Until then: a documented knob with a
measured speed and an **unmeasured** quality cost, and the released artifact stays at 250 x 2048.

## 5. Deferred to QATv2 (bigger than a knob)

- **Beam in the tile search: WITHDRAWN as a speed lever (2026-10-26).** The measured lossless width
  (8192) equals the full state count at K=3 (8192) and exceeds it at K=4/5/6, so the quality-preserving
  beam *is* the exact search: 1.0x. Widths that prune trade quality (+2.3 % at 4096, +7.9 % at 1024,
  +12.1 % at 256), and the pruned speedups are work-count projections on a kernel that has already
  refused to scale with work (occupancy: predicted 2-3x, measured 0.99x). No "35B 9 h -> 4.7 h" claim
  survives. See `K2_ENCODER_BEAM.md` for the corrected verdict.
- **Woodbury `H^-1` instead of the Cholesky factor**: only applies when `rank(X) = r << d`, i.e. when the
  calibration has at most `d` token positions. At the shipped calibration (250 rows x 2048 cols = 512k
  positions vs d = 8960) rank(X) = d, so `XX^T` would be 512k x 512k and there is no low-rank form --
  inapplicable (the `O(r^3)` identity is exact, on low-rank matrices only) (identity verified to 1.4e-14, ~2000x cheaper per solve). The
  blocker is that `ldlq` consumes Cholesky slices (`bb_L = L[bj:, bi:bj]`), so switching to the `H^-1`
  (GPTQ) recursion is an algorithm change needing the full battery.
- **Incremental checkpoints** (save only the module just quantized): the proper fix for the 35B, where a
  whole-state save is ~70 GB.
