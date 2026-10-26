# FLAGS.md — every switch, what it does, and whether it is safe

Companion to [`AGENTS.md`](AGENTS.md). Anything marked **⚠** has burned hours; anything marked **✗** was
measured and does not work. Defaults are the upstream defaults unless stated.

---

## 1. Runtime environment variables (K2 build)

| variable | effect | default | status |
|---|---|---|---|
| `K2_T16_SLOW=1` | forces the **reference GEMV** (CUDA cores) instead of the fast trellis GEMV | off | safe; used to prove the fast path: identical perplexity and identical text |
| `K2_T16_NOMMA=1` | forces the **CUDA-core prefill** instead of the f16 tensor-core (wmma) prefill | off | safe; same purpose, ~28× slower prefill |
| `K2_T16_CPU_SCALAR=1` | forces the **scalar** CPU reference instead of the AVX2 path | off | safe; CPU parity check (AVX2 is ~4× the scalar rate) |
| `K2_T16_INT8` | int8 `mma.sync` prefill | — | **✗ not in this branch.** The int8 kernel is an external patch (`qat_tools/k2_int8_mma_wip.patch`) and its output is still incorrect. Do not document it as shipped. |

## 2. Encoder (converter) environment

| variable | effect | default | status |
|---|---|---|---|
| `K2_SIGMA_REG` | **input-side LDLQ damping** (`σ`) — the knob that shapes the error the encoder minimises | `0.025` (upstream value) | the *real* σ knob; requires `patches/exllamav3-k2_sigma_reg.patch`. Unset **and empty** both give 0.025; a non-numeric value fails loudly. **Sweep not yet run.** |

## 3. Converter flags (`convert.py`, exllamav3)

### 3.1 What you set for a K2 artifact

| flag | meaning | our value | notes |
|---|---|---|---|
| `-i, --in_dir` | source HF model directory | — | |
| `-o, --out_dir` | output directory (created at the end; a partial run leaves only the work dir) | — | |
| `-w, --work_dir` | scratch + checkpoints | — | put it on the fastest device you have (see `-cpi`) |
| `-b, --bits` | **target body bits-per-weight** | `3.34` | a **budget**, not a per-tensor rate: the allocator distributes it (61/19/5 % at K=3/4/5 + a 6-bit head on the released file) |
| `-hb, --head_bits` | output-head rate, 1–8, `1.5/2.5/3.5`, or `16` = unquantized | `6` | the head *is* trellised; only the embedding stays F16 by default |
| `-eb, --embed_bits` | token-embedding rate, 1–8 or `16` | `16` (F16) | `-eb 6` enables **K2E** (trellis embedding): file −348 MB; the cost is **measured** at −1.2 coding points at both 3.34 and 6.00 bpw; keep off for qwen35moe-class models |
| `-mb, --mtp_bits` | MTP (draft) layers' rate | architecture default | |
| `-cb, --codebook` | `mul1` (default) \| `mcg` \| `3inst` | `mul1` | K2 uses `mul1`; the half-integer rates exist only for it |
| `-rcp, --recipe` | per-tensor `K` map (YAML) from the SC/RCO allocator | `pipeline/qwen15_3.34bpw/recipe.yaml` | **without `-rcp` the built-in heuristic allocator runs: the rate matches, the file does not** |
| `-d, --devices` | GPUs to shard the encode over | `0` | |
| `-dr, --device_ratios` | shard ratios, e.g. `2,2,4` | — | |
| `-ss, --scaled_hessian` | factor the Hessian per tensor with that tensor's input scales folded in | off | per-tensor LDLQ; changes the pack |
| `-hess, --hessians` | directory of precomputed Hessians (`<key>.safetensors`) | — | skips the capture pass |

### 3.2 Calibration and I/O

| flag | meaning | default | status |
|---|---|---|---|
| `-cr, --cal_rows` | calibration rows | `250` | **−24 % encode at 125; quality unmeasured** (`rfn` +0.0036, a train/eval-overlap-biased proxy) |
| `-cc, --cal_cols` | calibration columns (tokens per row) | `2048` | **−23 % encode at 1024; quality unmeasured** |
| `-cd, --cal_data` | pre-packed token file (from `util/sc_trace.py`) instead of the default corpus | — | gives deployment-trace calibration |
| `-cpi, --checkpoint_interval` | **synchronous** whole-state checkpoint interval, seconds | `120` | measured: two saves ≈ **5 s** at 1.5B (page cache absorbs them); matters at 7B+ where the state outgrows RAM |
| `-lcpi, --last_checkpoint_index` | stop checkpointing after module N | — | debug only |
| `--max_module N` | stop after N modules | — | partial run writes `<work>/ckpt/state.safetensors` (**not** `model.safetensors`) |
| `-r, --resume` | resume an interrupted job from the work dir | off | |
| `-sh, --shard_size` | output shard size, MB | `8192` | |
| `-pm, --parallel_mode` | deprecated no-op | off | parallel encode is the default |
| `-v, --verbose` | extra diagnostics | off | |

### 3.3 ⚠ The trap flag

| flag | meaning | why it is a trap |
|---|---|---|
| `-hreg, --hessians_reg` | diagonal regularisation of the **output-side** Hessian, relative to its mean diagonal | At `0 / 0.005 / 0.05` it produced the identical broken artifact (**ppl 44,440,843**) — any non-default value destroys the file. It is **not** the input-side σ. Use `K2_SIGMA_REG` (§2). |

### 3.4 Not a thing

`--dry-run` is **not** a converter flag (some early notes claimed it was). There is no dry-run mode; use
`--max_module 2` to smoke-test a conversion cheaply.

## 4. SC / RCO tools (`pipeline/`)

| flag | tool | meaning |
|---|---|---|
| `-m` | both | input model dir (`sc_measure`) / measurement JSON (`sc_alloc`) |
| `-t` | `sc_measure` | calibration text file |
| `-o` | both | output path (`sc.json`, `grid.json`) |
| `--rows`, `--length` | `sc_measure` | rows × tokens per forward | 
| `--levels` | `sc_measure` | the two injected relative-Frobenius norms (we use `0.30,0.15`); two levels fix the exponent |
| `--device` | `sc_measure` | `cuda` \| `cpu` |
| `--bpw` | `sc_alloc` | the budget to hit exactly |
| `--menu` | `sc_alloc` | rate options per tensor, e.g. `3,4,5` |
| `--ratio` | `sc_alloc` | distortion scaling per bit step (we use `1.96`) |
| `--alpha` | `sc_alloc` | exponent in `kld = S_t · rfn(K)^α` (we use `2.0`) |
| `--anchor` | `sc_alloc` | pin one known point, e.g. `3:0.155` |
| `--tie` | `sc_alloc` | groups that must share a rate (`q_proj+k_proj+v_proj,gate_proj+up_proj`) |
| `--selftest` | both | CPU-only self-check; both pass from the deposit |

## 5. Runtime CLI flags used in this repository

| flag | meaning | note |
|---|---|---|
| `-m` | model path | |
| `-ngl N` | layers offloaded to GPU | `99` = all; `0` = CPU-only |
| `-c N` | context size | perplexity protocol uses `512` |
| `--chunks N` | perplexity chunks | **20** reproduces the paper; `1` reads ~6.34 instead of ~10.59 |
| `-f` | text file for perplexity | `wikitext-2-raw/wiki.test.raw` (using another wikitext file changes PPL by ~2.5 -- F16 reads 7.86 on the quant\_test copy, 10.31 on this one) |
| `-p`, `-n` | prompt, tokens to generate | |
| `--temp 0` | greedy | **required for any coherence probe** (default sampling produces plausible nonsense) |
| `-t N` | CPU threads | `6` on this box |
| `-ub N` | micro-batch | only relevant for the CPU path experiments |
