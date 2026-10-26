# K2 session record — 2026-10-25, evening (pipeline, arms A/B/C, paper)

_Purpose: the paper's evidence base. Every number here is measured on this box tonight or copied
from a file on disk; nothing is projected. Read with K2_TRACKS.md (the long log) and
K2_PAPER/ (the manuscript)._

## 1. The QAT-style arms (A/B/C) — finished, PPL measured, benches in flight

All three: `--joint --w-f16 --tgt-topk 32 --steps 1500 --lr 5e-5 --nblocks 9800 --gate-blocks 38`,
shared corpus `poolv3_lite.npz` (wiki 49.6 / code 20.6 / chat 9.4 / math 11.9 / ubergarm 8.5), one
pack (`pipeline_qwen15_clean/exl3`), one start point (`k2_ptq.gguf`).

| arm | difference from A | gate loss (start 2.9762) | train s | PPL (20 chunks) |
|---|---|---|---|---|
| A | CE+10·KL (baseline) | **2.9333** | 623 | **10.4916** |
| B | `--hard-frac 0.21` | 2.9698 | 678 | 10.5048 |
| C | `--loss jaccard` (0.1508 start) | 0.1464 | 605 | 10.6134 |
| — | deployed incumbent (`15b_general`) | — | — | **10.2886** |
| — | PTQ (start point) | 2.9762 | — | 10.5895 |

TWO READINGS, both important:
1. **None of A/B/C beats the incumbent.** The deployed scales-only arm is 0.20 PPL better than the
   best of tonight's three, with the same corpus and a smaller token budget (768k vs ~6.1M tokens).
   Under the pre-registered rules (K2_RUNBOOK §52-63) none is adopted unless it wins on *coding*;
   the coding benches decide, but the PPL prior is against all three.
2. **Gate loss and deployment quality decoupled.** Arm A improved its gate the most (2.9762 →
   2.9333) and is the best of the three — yet still loses to the incumbent. This is the same
   saturation signature documented in K2_STATUS §7: the objective is already at its optimum for
   the deployed artifact, and further optimisation of it selects a different (not better) point.

## 2. Correction: the near-lossless artifact is 6.68 bpw on disk, not 8.3

`k6 body + head 6 + K2E = 1285.4 MB`. The logs (K2_STATUS §7, K2_MORNING_REPORT) print "8.3 bpw",
which divides by **1.24e9** (non-embedding parameters). The GGUF convention used by
`k2_final_table.py` is `filesize*8/1.543e9`:
    1285.4e6 * 8 / 1.543e9 = **6.68 bpw**
Consistency check against the tool's own output: `k6h4+K2E = 1227 MB` → 6.37 bpw (tool prints 6.37
✓), `k5 = 1469.8 MB` → 7.64 ✓, `k6 = 1633.6 MB` → 8.49 ✓. So 6.68 is the correct row.

CONSEQUENCE, and it improves the result: the artifact that clears **both** point estimates is
**1285 MB, smaller than the 5.15-bpw artifact (1470 MB)**, because K2E removes 348 MB from a file
whose F16 embedding was 39% of its bytes. The caveat still stands and is now stated in the paper:
6.7 bpw is still the bpw band where Q6-class quantization is expected to be transparent.

## 3. Provenance gaps found while assembling the paper (fix or cite with the gap named)

1. **The code-token damage map is not on disk.** `damage_map.json` holds the **wikitext** run:
   ALL quantized = KL 0.037523, top-1 0.93356, dPPL 1.0218, 18 arms including an F16 control
   (KL exactly 0). The code-token run quoted in K2_TRACKS §P4 (KL 0.02796, top-1 74.89%) was
   overwritten. The paper reports the wikitext arm from disk and the code figures from the log, and
   says so in a parenthetical.
2. **Additivity is 100.4%, not 105%**, on the wikitext arm: the eight "only-X" arms sum to 0.037688
   against ALL 0.037523. Best single-type recovery: all-but-up 33.7%, all-but-gate 32.5%,
   all-but-down 30.5%. (The 105% figure was the code arm, now unavailable.)
3. **The two component ablations are two CORPORA, not two model sizes.** `component_ablation_15b`
   (code prompts): FFN 96.6 / gate-up 69.49 / down 23.43 / attn −12.73 / head −0.70.
   `comp15_hs` (HellaSwag prompts, also 1.5B): FFN 95.52 / gate-up 59.60 / down 34.58 / attn +2.48
   / head −0.91. The "0.5B gives 95.5" attribution in K2_TRACKS:1973 is wrong — both files are
   1.5B. **New finding: the gate/up-vs-down split moves 11 points with the evaluation corpus**,
   which further caps allocation (a budget tuned on one distribution is mis-allocated on another).
4. **HellaSwag keys collapse.** Items are paired by context prefix, so 1200 items → 1184 unique
   keys (2000 → 1971); `k2_final_table.py` reports the reference as 671/1200 = 0.5592 and strict
   retention against 671 F16-solved keys. Recomputing all arms on the collapsed keying changes the
   published percentages by <0.15 pt (87.44 → 87.33; 93.65 → 93.59). No verdict changes.
5. **ARC carries a constant −4 pt.** 8 of 200 items put no letter in the top-8 and fall back to
   one-token generation; those 8 are wrong in *every* arm including F16 (0/8 in F16, k5, k6), so
   retention is unaffected but absolute ARC values are depressed.
6. `tasks_k5_hs2000.json` is **n=100**, not 2000 (already known, re-confirmed).
7. `tcq_reference.json` (greedy 0.1676, Viterbi 0.0885 at K=3, ν=8) is **not used in the paper**:
   its normalisation is not stated in the file and 0.0885 would make a 256-state TCQ 3× worse than
   our 16-state coder, which contradicts theory. Do not cite until the script's σ² convention is
   re-derived.

## 4. Paper

`qat_tools/K2_PAPER/`: `main.tex` (13 pp, 2 figures, 3 tables, no errors, no overfull boxes),
`make_figs.py` (reproduces both figures from the measured arrays), `ARXIV_SUBMISSION.md` (metadata,
upload list, pre-submission checklist, and the deliberate omissions with reasons).
Figures: `fig_rd.pdf` (the RD wall: bar lies 4.3× below the bound at 3.06 bpw), `fig_ladder.pdf`
(retention with CIs: coding clears at 5.15, HellaSwag does not).

Build: `~/.local/bin/tectonic -X compile main.tex` (user-local tectonic 0.15.0; no system TeX
installed, none added).

## 5. Still open at the time of writing

- A/B/C coding benches (MBPP-424 + the 246 SFT-solved subset), running serially; decision rule
  pre-registered: adopt only if it beats the incumbent on coding, never on gate loss or PPL.
- 7B: weights are on disk (`/mnt/Data/models/qwen25-7b`, 15.2 GB). **Retention is not measurable
  here** (F16 GGUF 14.2 GiB > 12 GB VRAM; partial-offload reference ≈4-6 h per arm). The planned
  compromise is pack + **PPL ratio** only, reported as a scale check on the PPL axis, never as
  retention.
- 35B: deferred to QATv2 by decision (`ssm_conv1d` layout incomplete).

## 6. CORRECTION (2026-10-25 late): the E8 1.22x is an ENTROPY-MATCHED figure

`K2_CLOSURE.md` §8 and `K2_STATUS.md` §7 say E8 is worth "1.22x in error, +3.8 coding points,
+0.29 bpw (a 9% file)". That is measured at **entropy-matched rate with an unbounded alphabet**
(`k2_e8_prototype.py`: R = h - log2(delta), h = 0.5 log2 2*pi*e). Our K2 trellis is FIXED-RATE and
its stream is exactly uniform (H = K, 1.000x entropy coding), so the comparison bundled two
effects.

Measured with the constraint a real container imposes (a codebook of exactly 2^(8R) points):
| R | D unbounded | D bounded | penalty |
|---|---|---|---|
| 2.00 | 0.0766 | 0.0785 | 1.026x |
| 3.06 | 0.0176 | 0.0227 | 1.29x |
| 4.00 | 0.0048 | 0.0104 | 2.17x |
Cause: the ball holding 2^(8R) cells at step delta has volume 2^(8h), independent of R — support
stays pinned to the source's typical set while granularity keeps shrinking.

Decomposition at R=3.06: 1.49x in distortion = **1.070 (D8->E8 shape) x 1.39 (alphabet/support)**.
Sanity check: D8->E8 shaping gain is ~0.3 dB = 1.034 in amplitude = 1.070 in distortion. The
*lattice* part is only 1.07 in distortion.

CONSEQUENCE: at fixed rate and 3-3.4 bpw the collectable E8 gain is **1.03-1.10x in error**
(0.0227 vs our 0.0272 = 1.10x with the ideal/ coset container), i.e. <=2 coding points, i.e.
<=0.1 bpw — NOT a reason to build a format at 3.3 bpw. At 2 bpw the penalty is only 1.026x, and
E8P-2 (2^16 x 8 int8 table, 512 KiB, 2.03 bpw, closed-form encoder at 150-300 M w/s) beats a K=2
trellis by 1.05-1.16x — which is where the field has converged (E8-QAT, LLVQ/Leech, SAGE all 2-bit).
See `K2_E8_QATV2_CONTAINER.md` §2-§4 and `K2_E8_QATV2_TRAINING.md` G0.

Also corrected: the 327.92 t/s decode figure is the **0.5B** seat, not the 1.5B (a 1.5B at 3.34 bpw
streams ~547 MB/token, needing ~210 GB/s against a measured 213 GB/s matvec ceiling).
Both corrections are now in the paper (`K2_PAPER/main.tex`: abstract, §5 E8 paragraph, §5 floor
paragraph, §8 costs).

## 7. A/B/C arms: objective variants on a matched schedule, no gain (benched, decided)

Benched to completion (424 items each, ~21 min/arm, serial under the GPU lock):
| arm | what differs | tokens | gate loss | coding 246 | PPL | vs incumbent |
|---|---|---|---|---|---|---|
| deployed `15b_general` | -- | **0.77M** | -- | **79.7% (196)** | **10.2886** | -- |
| A | joint CE+10KL | 0.77M (1500x512, batch 1) | 2.9762 -> 2.9333 | 76.8% (189) | 10.4916 | -7 items, p=0.19 |
| B | A + --hard-frac 0.21 | 0.77M | -> 2.9698 | 77.2% (190) | 10.5048 | -6, p=0.29 |
| C | A + --loss jaccard | 0.77M | 0.1508 -> 0.1464 | 78.0% (192) | 10.6134 | -4, p=0.50 |
DECISION: **adopt none.** Pre-registered rule was "adopt only on a win"; none wins, and every arm
improved the objective it was trained on while losing on the deployment metric.
CORRECTION (user challenge): an earlier version of this entry (and of the paper) claimed "8x the
token budget". WRONG: `--batch` defaults to 1 and run_abc.sh never overrode it, so A/B/C presented
1500 x 512 = 0.77M tokens, the same schedule length as the deployed arm (1500 steps, per
K2_TRACKS:1292). The comparison is therefore an OBJECTIVE-axis test, not a budget test, and the
corpus pool differs (poolv3_lite vs V3b65) so it is not corpus-controlled either. The paper now
says exactly this. If the Modal run used a larger batch the honest claim would be "1/8 the budget
and still 4-7 items behind", which is a different (weaker) statement -- verify before ever quoting
a budget multiple.

## 8. Corpus sizes (verified from the npz files, not the prose)

| corpus | blocks x 512 | tokens | mix | used by |
|---|---|---|---|---|
| poolv3_lite | 9,838 | **5,037,056** | wiki 49.6 / code 20.6 / chat 9.4 / math 11.9 / uber 8.5 | arms A/B/C, A1400, OnQ baseline |
| poolv3b | 39,130 | 20,034,560 | - | earlier arms |
| **poolv3b65 ("general")** | **53,702** | **27,495,424** | wiki 63.6 / code 19.3 / chat 8.8 / math 11.1 / uber 8.0 | **deployed `15b_general` (79.7%)** |
| general_hs_15b | 56,202 | 28,775,424 | - | HS EVAL corpus - never train on it |
| code_mix_15b | 15,993 | 8,188,416 | - | code-mix arms |

Consequence for the A/B/C comparison: both the incumbent and the three arms PRESENT 0.77M tokens
(1500 steps x 512, batch 1), but the incumbent sampled from 27.5M (2.8% of its pool) while the arms
sampled from 5.04M (15%, and a different mix). So the arms comparison carries a diversity+repetition
confound as well as a composition one; the paper says so explicitly.

A clean budget test is therefore possible WITHOUT data repetition: the general pool supports 53,702
unique steps (one epoch = 27.5M tokens). Steps 1,500 / 6,000 / 12,000 / 53,702 = 0.77M / 3.07M /
6.14M / 27.5M tokens at 2.8 / 11 / 22 / 100% of the pool.

## 9. OnQ round 1 launched (2026-10-25 22:01)

`/tmp/opencode/run_onq.sh` (fixed: a malformed quote on the generation line would have aborted it).
Order: matched-volume baseline **A1400** (1400 blocks, poolv3_lite, same 1500 steps) -> generate
1500 on-policy blocks from the PTQ student (`k2_onpolicy_gen.py`, starts its own server under the
GPU lock) -> **arm E** (on-policy data, CE+10*KL) -> **arm F** (on-policy data, `--loss dynkl`).
Each arm: apply scales -> GGUF+K2E -> PPL -> 246-item coding bench.
Note: the teacher cache `topk_pv3lite_1400g100.pt` did not exist, so A1400 builds it (~35 min of
teacher pass on top of ~13 min of training).

## 10. OnQ round 1, attempt 2 (22:20) + a runtime bug found

### 10.1 What went wrong on the first attempt
Generation produced 1500 blocks but **1498 requests failed** with
`Remote end closed connection without response`, and the generator's fallback silently substituted
the *corpus* block: "tokens identical to the corpus block: 99.9%". The runner then started training
arm E on that data (i.e. on poolv3_lite again, not on-policy). Killed at 22:16.

### 10.2 Root cause: a K2 runtime race with concurrent slots
`/tmp/opencode/srv_onpolicy_8399.log`:
    ===== Failed to sample token
    llama-sampling.cpp:745: Fatal error  ->  ggml_abort
with the trace through `llama_sample_token_with_rng_impl`, while slots 2 and 3 were sampling
simultaneously. Isolated test (`/tmp/opencode/test_sampling.py`, single slot, `k2_ptq.gguf`):
    greedy temp=0            OK
    temp=0.8 top_p=1.0       OK
    temp=0.8 top_p=0.95      OK
    temp=0.5 top_k=40        OK
So **sampling works; concurrency is what kills it.** Our benches never hit this because they send
one request at a time (even with `-np 4`). Working hypothesis: a shared scratch/logits buffer in the
t16 path being clobbered when two slots decode at once -> all -inf probabilities -> the sampler
aborts. Worth reporting upstream one day; not a research blocker.
WORKAROUND: generate with `--workers 1` (single slot, sequential). Cost: ~35 t/s single-stream
instead of ~4x that, i.e. ~5.4 h for 1500 x 480 tokens.

### 10.3 Runner hardened (three fixes)
1. **A1400 is resumable** -- skipped if `arm_A1400.pt` exists (it finished: gate 2.5517 -> 2.4480,
   663 s, 22:15).
2. `--workers 1` for generation.
3. **HARD GATE**: parse "identical to the corpus block: X%" and ABORT before training if X > 50.
   The previous version only checked that the npz existed, which is exactly how a silent fallback
   became a training run.

## 11. PROVENANCE CORRECTION (2026-10-26): "err = 0.165" is the K=3 SUBSET, not the model

`K2_BAR_ARITHMETIC.md` and the paper's RD table used 0.165 as the deployed model's relative weight
error at 3.34 bpw. Re-deriving from `/mnt/Data/qat_k2/weight_error.json` (21 tensors, shapes stored):

| K | tensors | weights | weighted rfn |
|---|---|---|---|
| 3 | 6 | 0.083B | **0.1654** |
| 4 | 7 | 0.035B | 0.0771 |
| 5 | 8 | 0.022B | 0.0405 |
| all 21 | 21 | 0.140B | **0.1336** (unweighted mean 0.0897) |

So 0.165 is the **K=3 tensors'** error. Pairing it with the pack's *average* rate (3.339) inflates
the ratio to 1.67x; at each K's own rate the ratios are **1.32 / 1.23 / 1.30 / 1.28** — i.e. the
original "constant 1.3x implementation loss" was right and my 2026-10-25 "correction" to 1.67x was
the error. The paper's RD table is now per-K at per-K rate, and the wording is back to
"1.23-1.32x, approximately constant".

Knock-on numbers, all restored:
- perfect coding is worth **~+5 points** (85%, not 88%);
- model-level statement, if aggregated: 0.134 at 3.34 bpw = **1.36x** the bound at that rate.

Consequence for the "pipeline gap" idea (K2_QATV2_PLAN §6): the *codec* measures 0.131 at K=3
(1.05x bound, `K2_ENCODER_STUDY.md:367`) while the *artifact's K=3 tensors* measure 0.165 (1.32x) —
a **1.26x gap on identical tensors at identical K**, worth ~+4 points if it is loss rather than the
LDLQ trade. That is what the sigma sweep running now measures.

## 12. THE SPEED BUG (2026-10-26): the fast path defaults to OFF in production

`t16-matmul.cu` gates the fast GEMV behind `K2_T16_FAST`, and the default is **disabled**:
    v = (e && atoi(e) != 0) ? 1 : 0;  return v != 0;   // env unset -> false
An audit of every script that starts llama-server/bench/perplexity: only the *harnesses* set it
(`k2_fast_test.sh`, `k2_kernel_bench.sh`, `k2_speed_matrix.sh`, `k2_k4_alloc.sh`, `k2_qat15_test.sh`,
`k2_gemv_check.py`). Every scoring and generation path did not (`k2_bench_arm.sh`, `k2_hs1200.sh`,
`k2_task_run.sh`, `k2_ppl.sh`, `k2_onpolicy_gen.py`, `k2_pipeline.py`).

MEASURED, 1.5B at 3.34 bpw, llama-bench tg128 (2026-10-26 14:20):
| config | pp512 | tg128 |
|---|---|---|
| default (FAST unset) | 115.11 | **36.73 ± 0.03** |
| K2_T16_FAST=1 | 114.72 | **175.53 ± 0.41** |
| FAST=1 NW=8 | - | 159.06 |
| FAST=1 ROWS=8 | - | 159.33 |
PPL identity check (4 chunks): 8.7898 both ways -> the paths are numerically equivalent.
=> production decode was **4.8x slower than the artifact can go**. 175.5 t/s x 722 MB/token =
**127 GB/s = 73% of the 175 GB/s isolated GEMV ceiling**, so the kernels were never the problem;
the default was.

CONSEQUENCES APPLIED:
- `k2_onpolicy_gen.py` now starts its server with `K2_T16_FAST=1`; OnQ generation went from
  ~217 min (1500 blocks) to ~86 min, which is why we can afford the full 1500 blocks and stay
  volume-matched to A1400.
- Scoring benches are deliberately LEFT on the old path so E/F are comparable to A1400, the
  incumbent and the whole ladder. Re-bench the incumbent fast-path (~12 min) if we ever want
  everything on one footing.
- Paper §8 now quotes the real 1.5B number (175.5 t/s) instead of the 0.5B-seat caveat.

## 13. K2 BAKE (2026-10-26): the runtime is now default-fast and knob-free

Edited `ik_llama-k2/ggml/src/ggml-cuda/t16-matmul.cu` (backups: `/tmp/opencode/t16-matmul.cu.bak-1451`,
`.bak2-1455`):
- **`k2_t16_fast_enabled()` now returns `!k2_t16_slow()`** — the fast GEMV path is the DEFAULT.
  `K2_T16_SLOW=1` is the single remaining hook, for exactness A/B.
- All five production knobs baked to their measured winners with the numbers in comments:
  `DIRECT=1` (324.1 vs 308.4 t/s), `ROWS=4` (324.1 vs 287.2 at 8, 314.3 at 2), `NW=16` (175.5 vs
  159.1 at 8), `PF=1`, `FAST=1` (175.5 vs 36.7).
- **Staged K=5/6 variant removed** from dispatch + its getter/launcher (measured NEGATIVE, -23%,
  2026-10-23). Its device helpers remain as uninstantiated templates (dead source, no code emitted)
  — to be deleted in the next pass; an attempt to delete them broke `k2_fwht128`'s template header
  and was reverted.
- BUILD: incremental in `build-cuda` via `distrobox-enter -n cuda-box` — **succeeds**.

VERIFIED after the bake (GPU contended by the OnQ generation, so absolute numbers are low):
| | tg16 |
|---|---|
| default (no env) | **83.8 t/s** |
| K2_T16_SLOW=1 | 22.6 t/s |
PPL identity, 2 chunks: **8.9471 both paths**.

CONSEQUENCE: the E/F benches will run on the baked (fast) binary while A1400/the incumbent were
benched on the unbaked one — so **re-bench A1400 on the new binary** before comparing (12 min on the
fast path).

## 14. Template-instance split, the GSM8K audit, and arm E (2026-10-26)

**Template split, done and verified.** The monolith is now the fork's shape: `t16-gemv.cuh` (613
lines: kernel templates + the shared device primitives `k2q`/`k2_t16_fshr`/`k2_t16_mul1`, no
instantiation), `t16-matmul.cu` (350 lines: matmul kernel, MoE, dispatch, the two op entry points),
and five instances `template-instances/mmvq-instance-t16-k{2..6}.cu` picked up by the existing
`mmvq-instance*.cu` glob — **no CMakeLists change**. Getting there needed three include fixes
(`cuda_fp16`/`cuda_runtime` via `common.cuh`; the shared primitives moved into the header; the
header include moved above the matmul kernel). Measured: **an instance rebuilds in 9 s on one core**
(monolith: 21 s); five separate `.o` files (1.8–2.0 MB) appear in the build; perplexity unchanged
(**8.9471**, 2-chunk probe); fast path still active. Commit `29568914`.

**The GSM8K axis is degenerate — the "math recovery" is withdrawn.** Triggered by the user asking
"is the math recovery invalid?". Audit of every stored result file:
- `extract_answer()` requires the `####` marker; **no arm emits it** — 0/200 for F16, PTQ, QAT, K=5,
  K=6 alike. So no arm complies with the prompt's answer format.
- Every stored prediction is a **bare 2–3 character number**; the generations were never kept
  (`"text": txt[:400] if dump else None` — `--dump` was never passed, so all `text` fields are empty).
- On the **same 200 items** our runs span **1.5 % → 47.5 %** (`tasks_dpq` 1.5 %, `tasks_clean` 4.0 %,
  F16 15.0 %, K=6 17.5 %, K=5 22.0 %, PTQ 38.5 %, QAT 38.0 %, Coder-F16 47.5 %).
- F16 sits **23.5 points below** its own quantized arms. A precision change cannot depress the
  reference model ⇒ the axis measures answer format/verbosity under the 320-token cap, not arithmetic.

CONSEQUENCE (paper + PR): the paragraph in §"Why 3-bit is near full precision and our bar can both be
true" claimed the 3.34 arm beats F16 on a 4-task mean (0.660 vs 0.609) *because* of GSM8K. Dropping
the axis **reverses** that: on ARC/HellaSwag/probes the arm averages **0.736 vs F16's 0.762**. The
paper now says so and lists the axis in the harness-defects list as (ix) plus in "claims withdrawn".
The harness (`k2_task_bench.py`) now always stores the generation and records the extraction rule
(`marker` vs `fallback`); the audited re-run (F16, PTQ, QAT; 200 items; retained text) is queued as
`/tmp/opencode/gsm_v2.sh` → `tasks_gsm_v2_{f16,ptq,qat}.json`.

**Arm E (on-policy, volume-matched to A1400): 76.0 % (187/246)** strict coding retention; gate
2.0322 → 1.5378; PPL 10.5582; bench 736 s on the fast binary. That is below the incumbent's **79.7 %**
and in the same band as the poolv3 arms A/B/C (76.8/77.2/78.0). Arm F (dynkl loss) training.

**Name.** The user's real name is **Baran Mamo** (internet handle TitanMamoo): paper author,
`.zenodo.json` creators, `ARXIV_SUBMISSION.md`, `ZENODO_DEPOSIT.md` and the git author on the branch
(`Baran Mamo <TitanMamoo@users.noreply.github.com>`) all updated; commit `978a8af5` re-authored.

**PR body rewritten on the user's two review notes.** (a) The battery now shows ARC/probes (101 % of
F16 at 3.34 bpw) instead of only the two damaged axes, with the GSM8K exclusion argued from the audit
above. (b) "Why not reuse an existing type" now **concedes IQ3_KT**: the fork *does* have a trellis
(`Trellis3<>` in `iqk_gemm_ktquants.cpp`, `block_iq3_kt` = 100 B/256 weights = 3.125 bpw), and the
argument narrows to what actually differs — KT's trellis is **one-dimensional inside a 256-weight row
block** (so it fits a `ggml_type`), K2's is **2-D across a 16×16 tile** (so it cannot), plus a
different codebook (`mul1` vs L=16/N=8) and per-tensor K=2..6 vs a fixed rate. Our own KT
measurements (~22 % RMS, above the ~13.5 % cliff; `quant_test/IQ3_KT_DIAGNOSIS.md`) are cited as
"not an alternative either", explicitly not a request to change it.

## 15. OnQ round closed, and the embedding confound (2026-10-26 16:37)

**OnQ verdict: null, no adoption.** Volume-matched arms on 1400 blocks of 92 % student-generated
on-policy data (1500 generated, 8.0 % identical to the corpus block):

| arm | strict/246 | strict % | loose % | all-427 | PPL | vs incumbent |
|---|---|---|---|---|---|---|
| incumbent (K1 grid) | 196 | **79.7** | 100.0 | 79.7 | 10.2886 | — |
| A1400 (baseline, poolv3_lite) | 190 | 77.2 | 96.9 | 50.5 | (re-run) | −6, p=0.26 |
| E (on-policy, CE+10·KL) | 187 | 76.0 | 95.4 | 49.8 | 10.5582 | −9, p=0.078 |
| F (on-policy, dynamic-KL) | 187 | 76.0 | 95.4 | 50.2 | 10.5674 | −9, p=0.078 |

E vs A1400 −3 (p=0.45), F vs A1400 −3 (p=0.55): **the on-policy data and the dynkl objective are
null against their own volume-matched baseline**, and every new arm is below the incumbent. That
closes the OnQ chapter and confirms the paper's "training does not close the gap" with a
corpus-controlled comparison this time (the A/B/C paragraph had a corpus confound; A1400 vs E/F does
not).

**CONFOUND FOUND (must be stated, may change the size of the gap):** A/B/C/E/A1400/F are all
**849,632,768 B builds with the K2E trellis token embedding**, while the incumbent's 79.7 % is the
**1,197,866,720 B f16-embedding build**. The two differ *only* in the embedding (348 MB of the
file), so "76.0 vs 79.7" mixes training with ~1.2 points of embed cost (the paper's K=6 row measured
98.4 → 97.2 for the same substitution). `k2_E_f16emb.gguf` was built from arm E's scales with the f16
embedding (1,197,866,720 B — byte-size identical to the incumbent, differing only in the trained
scales) and is being benched to split the two effects. Until that lands, the paper states the
confound as a fact and does not claim the arms lost *only* to training.

Note also: F's PPL line was briefly lost to a lock collision (a perplexity step outside the lock ran
against a bench); the artifact was intact and the value above is from the completed run.

## 16. CORRECTION to §14: the GSM8K axis was mis-scored, not marker-less (2026-10-26 17:35)

§14 concluded "no arm emits the `####` marker, so the axis is degenerate". **That was wrong**, and the
error is instructive: the check counted `"####" in item["text"]`, but every `text` field is empty
because `--dump` was never passed — an empty string contains no marker, so the test could only ever
return 0. Reading an absence of data as evidence of absence is exactly the mistake this project keeps
finding in other people's harnesses.

The audited re-run (generations retained, `tasks_gsm_v2_*.json`, 200 items, greedy, 4-shot) shows:
- F16 emits `####` in **200/200** answers; PTQ in **195/200**. Both follow the format.
- Neither emits the `<|im_end|>` stop token, so both generate to the 320-token cap **after** finishing
  the answer, and degenerate: F16 drifts into unrelated text (and Arabic fragments), PTQ loops on its
  own answer (**17.5 `####` per answer** vs F16's 3.0).
- The harness took the **last** marker — i.e. the number from the degenerate tail. That is why F16
  read 15.0 % and PTQ 38.5 %: F16's tail drifts to a different number, PTQ's tail repeats the right
  one. An inversion of 23.5 points in the reference model's disfavour was the tell; it is not a
  precision effect.

**Corrected scoring (first complete answer):**

| arm | last-marker (as stored) | first-marker (correct) | markers/answer |
|---|---|---|---|
| F16 | 15.0 % (30/200) | **50.0 % (100/200)** | 3.0 |
| PTQ 3.34 | 38.5 % (77/200) | **45.5 % (91/200)** | 17.5 |

Retention **91.0 %** in the expected direction (quantized below F16). Under truncation control the
old rule still separated the arms (under-cap answers: F16 11.6 %, PTQ 38.2 %), so length was not the
mechanism — the loop tail was.

**Actions taken.** `k2_task_bench.py` now scores the **first** `####` and records `n_markers`; the
paper's §normalisation paragraph, limitations item (ix) and the "claims withdrawn" list are rewritten
with this diagnosis and the corrected numbers (mean over ARC/GSM8K/HS/probes: **0.673 vs F16's
0.697**, so the mean no longer inverts and the earlier "3-bit is near-lossless on average" framing
stays falsified on our data); the PR body's GSM8K block is rewritten the same way. The QAT arm's
audited run is re-queued with the corrected path (`pipeline_qwen15_clean/k2_15b_general.gguf` — the
first attempt failed on MY path bug, not a missing file).

**Sanitizer (gate G6) — CLOSED.** `racecheck`: **0 hazards, 0 errors, 0 warnings**. `memcheck`: 0
invalid memory accesses; its "2 errors" are `cudaErrorGraphExecUpdateFailure` (error 910) — the
sanitizer's own instrumentation invalidating CUDA-graph updates — and the run completed normally with
timings. Host driver had to be staged into `~/cuda-libs` for the container's sanitizer to see it.

**Gate G5 (the R16 rejection gate) — K2 is not the slower type this time.** Same model, card,
binary: IQ4_KS 4.25 bpw → PPL 10.9485, tg128 **185.53** t/s; K2 PTQ 3.34 bpw → PPL **10.5895**, tg128
**175.98** t/s. So K2 is 5.1 % slower per token while being *better* and 25 % smaller — the opposite
of the R16 situation. But prefill: IQ4_KS **8181** t/s pp512 vs K2 **116.75** t/s (-70x), which is
the absent-MMQ path and must be stated as the PR's main speed weakness.

## 17. "We did not build an MMQ?" — no, and two claims in the PR/spec were false (2026-10-26 17:38)

Checked against the code rather than the docs: **there is no MMQ and no tensor-core path at all.**

| op | kernel | math |
|---|---|---|
| decode (n = 1) | `k2_t16_gemv_k{2..6}`, per-K instances | half2 FMA, CUDA cores |
| prefill (any n) | `k2_t16_matmul_kernel`, 32 tokens/block | half2 FMA, CUDA cores |
| MoE | `k2_t16_moe_kernel` | half2 FMA, CUDA cores |
| embedding | `t16-embed.cu` gather | — |

No `mma.sync` anywhere in `t16-*.cu`/`t16-gemv.cuh` (the only `fma` is `__hfma2`), and — contrary to
what the PR said — **no dequant-to-cuBLAS fallback**: `ggml_cuda_op_t16_matmul` routes `n == 1` to the
GEMV and *every* other `n` to the same CUDA-core kernel. Two false statements were removed:
"Prefill (n > 1) uses an f16-MMA matmul kernel" (spec §4) and "batch > 8 falls back to
dequant+cuBLAS" (PR limits item 2), plus the PR's "(114.7 t/s pp512 here, f16-MMA path)".

**What this costs, measured (gate G5, same model/card/binary):**

| | bpw | PPL | tg128 | pp512 |
|---|---|---|---|---|
| IQ4_KS | 4.25 | 10.9485 | **185.53** | **8181** |
| K2 (PTQ) | 3.34 | **10.5895** | 175.98 | 116.75 |

Decode: K2 is **5.1% slower** while being better and 25% smaller (the opposite of R16's rejection).
Prefill: **70x slower** — that is entirely the missing tensor-core path, and it is the honest headline
cost of the format today. Prefill decodes each 16x16 tile once per 32-token block, on CUDA cores.

**What an MMQ would actually take.** The procedural `mul1` codebook decodes to ~1021 integer levels
(`0x6400 + byte-sum` read as fp16 bits), not a 16-level grid, so a true int8 quantized-by-quantized
kernel needs a per-tile requantization step (a precision decision). The cheaper, larger win is an
**f16-MMA prefill**: decode the tile into shared memory, then `mma.sync.f16` — no format change, and
it attacks exactly the 70x gap. Both are follow-up kernel work; the PR now states this plainly
instead of claiming a tensor-core path that does not exist.

## 18. GSM8K re-scored on all three arms (2026-10-26 17:45)

The QAT arm's audited run completed (`tasks_gsm_v2_qat.json`, 200 items, generations retained, new
harness = first-marker rule):

| arm | first-marker | as stored (last-marker) | markers/answer |
|---|---|---|---|
| F16 | **100/200 = 50.0 %** | 30/200 = 15.0 % | (old run) |
| PTQ 3.34 | **91/200 = 45.5 %** | 77/200 = 38.5 % | 17.5 |
| QAT 3.34 (deployed) | **104/200 = 52.0 %** | 104/200 (same rule) | 4.1 |

So on the corrected axis the deployed arm is at **104 % of F16 (a four-item tie, +4)** and PTQ at
91 %; the deployed arm also loops far less than PTQ (4.1 markers/answer vs 17.5). Mean over the four
axes with the corrected GSM8K: **0.690 (K2 3.34) vs 0.697 (F16)** — a tie on the mean that hides
coding −20 and HellaSwag −10 points, offset by probes +5 and GSM8K +2. Paper paragraph and PR
battery table updated to these numbers.

Also in this window: `t16-matmul.cu` gained an **f16-MMA prefill kernel** (`k2_t16_mma_kernel`,
wmma 16x16x16, 8 warps = 128 tokens per block, A decoded once per block, B staged token-major for
coalescing and loaded as a `col_major` fragment, f32 accumulate, `K2_T16_NOMMA=1` restores the old
kernel). Weight side is exact (`k2_t16_mul1` returns `__half2float` of an fp16 hfma → `__float2half`
round-trips), only the activations round to f16. Builds clean; correctness/speed verification queued.

## 19. f16-MMA prefill: 13.7x, perplexity unchanged (2026-10-26 17:54)

Built the tensor-core prefill the user asked for (`k2_t16_mma_kernel`, commit `e826e444`):
`wmma` 16x16x16, block = 8 warps x 16 tokens = 128 tokens of one output tile, grid `(J, ceil(n/128))`.
Two layout facts made it transpose-free: thread `(c,r)` already decodes `(out=c, in=r)`, so the
16x16 tile is `row_major` A as-is; and `x` is `[in, n]`, so staging B token-major and loading it
`col_major` is both coalesced and transpose-free. Weights are exact (`k2_t16_mul1` returns
`__half2float` of an fp16 hfma, so `__float2half` round-trips); only activations round to f16.

| check | f32 kernel (`K2_T16_NOMMA=1`) | MMA |
|---|---|---|
| PPL 20 chunks | 10.5895 | **10.5894** |
| PPL `-ub 100` (warp tail) | 8.7896 | **8.7893** |
| pp512 | 117.00 ± 0.11 | **1605.81 ± 15.61** (13.7x) |
| tg128 | 175.98 | 176.44 (untouched: decode stays on the GEMV) |

Gate-G5 re-measurement (5 reps) confirms the baseline and puts the gap in perspective: `IQ4_KS`
184.2 t/s decode / 8487.7 pp512; F16 base 90.9 / 8114.6; K2 174.1 / 1599.8. So prefill went from
**70x behind to 5.3x behind**, and decode stays 4-5% behind at better PPL and 25% smaller.

**Still open:** the MoE path (`ggml_cuda_op_t16_mul_mat_id`) has no tensor-core variant — a sparse
model keeps the CUDA-core prefill; and an int8 MMQ would need a per-tile requantization because the
codebook decodes to ~1021 levels. Next optimization for dense: 4 output tiles per block (32 mma per
B fill instead of 8) to amortize the two `__syncthreads()` per input tile.

## 20. Prefill optimization trail: 116 -> 3244 t/s, and where the rest goes (2026-10-26 18:05)

Asked for the MMA path and then told (correctly) that 5.3x behind IQ4_KS is too slow. Measured
progression, each step with a PPL check at 8.7898 (4 chunks) against the CUDA-core kernel:

| version | change | pp512 | note |
|---|---|---|---|
| CUDA-core (baseline) | half2 FMA, 32 tokens/block | 116 | what the format shipped with |
| v1 | wmma, 1 output tile/block, 8 warps x 16 tokens | 1606 | 1 MMA/warp/tile, 2 block barriers/tile |
| v2 | BM=64 (4 tiles), per-warp staging, float4 loads, no block barriers | 1857 | +16% only |
| v2+pad | shared ld 16 -> 24 halves | 1927 | +4%: bank conflicts were minor |
| v3 | one shared decode per block (not per warp), double-buffered sA, 1 barrier/tile | 2406 | +25% |
| v5 | **software pipeline**: stage tile i+1 before mma of tile i, double-buffered sA+sX, block-wide activation staging | **3244** | **2.8x over v1, 28x over CUDA-core** |

Ablation that pointed the way: with the decode replaced by a constant, v2 ran at 2707 t/s (decode =
29% of its time); the deeper problem was that the loop was phase-locked -- every warp decoded (ALU)
then every warp computed (tensor cores), behind a barrier, so the two never overlapped. Pipelining
fixed exactly that.

**Two bugs the PPL gate caught (both would have been invisible in the speed number):**
1. a decode that filled only 64 of each tile's 256 values (threads were split 64-per-tile) -- ran
   *faster* (2693 t/s) and produced garbage;
2. an epilogue where all 8 warps staged their accumulator into one shared buffer -- 3303 t/s, PPL
   6.9e10.
Both were found by the 4-chunk PPL comparison against the CUDA-core path, which is why that check
runs before any speed number is believed.

**Where the remaining 2.6x lives (decomposition, measured):** IQ4_KS runs 8488 t/s = 22.2 TFLOP/s
equivalent, which needs int8 tensor cores (51.2 TOPS peak on this card); our f16 MMA with f32
accumulate peaks at 12.8 TFLOP/s = 4885 t/s, and we are at 3244 = **66% of that ceiling**. So the
gap is 1.74x arithmetic (int8 vs f16-with-f32-accumulate, a hardware property) and 1.5x kernel
efficiency. An f16-accumulate path (2x the arithmetic rate, folding into f32 every 8 input tiles)
is written and gated behind `K2_T16_F16ACC=1`; it is the only remaining lever that can approach
IQ4_KS, at a precision cost that must be measured before it is adopted. A true int8 path would need
a per-tile requantization, because the codebook's alphabet is ~1021 levels.

## 21. int8 MMA (the MMQ path): groundwork done, kernel parked (2026-10-26 19:15)

The user's pointer to the fork's own kernels settled the design question: `mmq.cuh` uses
`mma.sync.aligned.m16n8k32.row.col.s32.s8.s8.s32` (int8) with `MMQ_ITER_K 256`, which is why IQ4_KS
runs 8488 t/s while our f16-with-f32-accumulate path is capped at 4885. Work done and verified:

1. **Fragment layout proven**, not assumed: `qat_tools/t16_probes/mma_int8_probe.cu` compares one
   `m16n8k32` against a CPU reference — **128/128 elements, max |diff| 0**.
2. **Numerics validated on the CPU before writing GPU code**: the codebook is affine in an integer
   h in [1024,2044] (v = 1.2324*h - 11.13), so int8 with one constant scale costs half a step of
   4.95 = 0.68% of the tile's standard deviation. End to end (K=1536 dot products, 4000 trials) the
   int8 pipeline is **0.83% of the output magnitude** against the trellis's own ~13% weight error.
   The per-sample *relative* error looks catastrophic (RMS 149%) only because the dot product
   crosses zero; absolute-vs-RMS is the meaningful measure.
3. **A first complete kernel was written and measured: 736 t/s — slower than f16 — and it crashes
   under perplexity.** Diagnosis: (a) the per-token activation scale needs a full pass over x
   (786 KB per block) and that pass dominates; (b) an unresolved fault in the staging/epilogue.
   The fix is the reference's own design: **per-256 k-block activation scales with a fold every 256
   (6 folds per K)**, not one scale per token over the whole row — which also keeps shared memory
   within budget (128 tokens x 256 k int8 = 32 KB, so 64 tokens per block).

Kernel parked as `qat_tools/k2_int8_mma_wip.patch` (230 lines); the runtime is back to the verified
f16-MMA path (3248 t/s, PPL 8.7898). Nothing broken ships: the int8 flag never became a default.
Remaining work for parity: restructure to BK=256 with per-block scales, fix the crash, re-verify PPL.

## 22. Gaps closed + the MMQ-shaped int8 attempt (2026-10-26 18:35)

**G7 loader probe (gate closed).** `qat_tools/t16_probes/loader_probe.py` re-implements the loader's
contract independently of the loader: rank-3 I16 payloads, `16K` first dim with K in [2,6], `su [in]`
and `sv [out]` F16 sides present with exact shapes, payload bytes == ne0*ne1*ne2*2 (no custom row
size, no padding), no orphan sides, and the `.signs` companion for the 2-D K2E payload.
**PASS on three artifacts — 197 trellis tensors each** (`k2_ptq.gguf`, `k2_15b_general.gguf`,
`k2_E_f16emb.gguf`).

**G5 wording (gap closed, gate still PARTIAL).** The plan names `IQ3_KS`; at 4.25 bpw `IQ4_KS` is the
*quality* match for our 3.34 bpw (PPL 10.95 vs 10.59), so that is what was measured — now stated
explicitly in both the PR and the merge plan, together with the fact that **clocks were not locked and
runs were not interleaved**. Closing G5 properly = `nvidia-smi -lgc` + interleaved A/B of fallback /
IQ3_KS / IQ4_KS.

**CPU dense AVX2 re-measured (the contaminated number is withdrawn).** Interleaved, quiet box, `-ngl
0`, r=2, two rounds: **AVX2 2.98 / 2.97 t/s** tg8 vs scalar **0.75 / 0.75** → **3.97x**, bit-identical.
The previously published 0.23 t/s and 1.35x understated our own kernel by ~13x.

**The proper MMQ-shaped int8 kernel: written, wired, measured, NOT adopted.** Followed `mmq.cuh`
(large K iteration, per-iteration activation scales): BK=128 with one scale per (token, iteration) and
12 folds over K, block = 64 out x 64 tokens, `mma.sync.m16n8k32.s8.s8.s32` with the fragment layout
already proven by `mma_int8_probe.cu`, weights requantised to int8 with the constant scale 4.9491 and
the rank-1 correction folded per iteration. Findings:
- **A dispatch bug made three earlier measurements meaningless**: the guard tested `w->ne[0] % 128`,
  but `ne[0]` is `16K` (= 48 at K=3), not the input dim, so the int8 kernel was never selected and
  every "int8" number was the f16 path. That is why three different ablations (decode removed,
  activation loads removed, bank padding) all read ~3230 t/s.
- With the guard fixed (`I % 8 == 0`), the int8 kernel runs at **803 t/s and produces inf under
  perplexity** — slower than f16 and wrong. Two suspects were eliminated by ablation (weight decode,
  global loads) *before* the guard bug was found, so they must be re-tested once it is correct.
- Bank-conflict padding of the shared rows (128 -> 132 B) changed nothing in the f16 path.

Parked again as `qat_tools/k2_int8_mma_wip.patch` (236 lines) with the dispatch fix included; the
runtime is the verified f16-MMA path (3248 t/s, PPL 8.7898). Remaining for int8: fix the inf, then
re-run the two ablations honestly now that the kernel is actually selected.

## 23. Explicit framing: the tensor-core kernels are a floor, not a performance story (2026-10-26)

User's instruction: say plainly that the MMQ-shaped kernels are not good, and exist only so we do not
fall back to the CUDA-core kernels. Written into `K2_PR_BODY.md` ("Honest statement about these
kernels") and `K2_SPEC.md` (§7b), with the numbers:

| path | pp512 | status |
|---|---|---|
| CUDA-core reference (half2 FMA) | 116 t/s | oracle + fallback (`K2_T16_NOMMA=1`) |
| f16 wmma prefill (shipped) | 3.2-3.25k t/s | stopgap: 28x the oracle, 2.6x behind IQ4_KS, 66% of the f16+f32-accumulate ceiling |
| int8 MMA, MMQ-shaped | not correct | output uncorrelated with the reference; constants bug fixed (182x), one staging bug left; parked |
| IQ4_KS (fork's int8 MMQ) | 8488 t/s | reference for what "good" means |

Also settled today on the speed question: the BK=32 variant (two input tiles per pipeline stage, to
halve the barrier cost) **aborts with an illegal memory access** and was reverted; the committed
kernel is the verified BK=16 pipelined one. And the theoretical ceiling decomposition is now written
down: 1.74x is int8-vs-f16 arithmetic (a hardware property), 1.5x is scheduling/tiling (cp.async,
128x128 tiles). Hand-written SASS would chase the second term only, at 10-20% of it, per-architecture.

## 24. Framing correction: the shipped kernel is USED, and it is not good (2026-10-26)

User's correction to §23: do not frame the tensor-core kernel as a "correctness reference" — it is the
**default path that ships**, and the honest statement is that it is **not a good kernel**. Rewritten in
`K2_PR_BODY.md` and `K2_SPEC.md` §7b:

- every `n > 1` call runs the f16 wmma kernel; `K2_T16_NOMMA=1` selects the CUDA-core oracle;
- it is 3.2-3.25k t/s = 66% of the f16+f32-accumulate ceiling (~4.9k) and **2.6x behind IQ4_KS**;
- we ship it because the alternative is 116 t/s on CUDA cores (28x worse);
- there is **no MMQ**: the int8 attempt is incorrect (one staging bug after the 182x constant fix) and
  is parked behind `K2_T16_INT8`, not shipped;
- what a good one takes is stated: int8 MMA (2x arithmetic rate -> ~9.8k ceiling), `cp.async`, 128x128
  tiles at 3-4 blocks/SM, and a tensor-core MoE path.

## 25. G5-proper launched; probe P3(ii) dropped with a reason (2026-10-26 19:35)

- **G5 proper is running**: an `IQ3_KS` of the same 1.5B did not exist (the only one on disk is a 3B),
  so the rate-matched comparison the plan asked for had never been possible. Now: quantize IQ3_KS from
  the same F16 base, perplexity for all three types on one protocol, then **5 interleaved A/B pairs**
  (order alternating each pair, r=3) against both IQ3_KS (~3.4 bpw, rate-matched) and IQ4_KS (4.25 bpw,
  quality-matched). Clocks are NOT locked (needs root) and that is stated with the numbers.
- **P3(ii) dropped, with the reason recorded in the merge plan**: a standalone per-tile GEMV probe
  would duplicate the 256-byte permutation table and the window rule into the probe itself, so it
  would mostly validate the duplicate. The GEMV is covered bit-identically end-to-end by G8
  (identical perplexity to four decimals and identical generated texts). P3(i) and P3(iii) are closed
  by the codebook probe (65536/65536) and the loader probe (197 tensors x 3 artifacts).
- Also removed the stray `probabilities.txt`; the repo tree is clean at `2079fbcf`.

## 26. G5 closed with a rate-matched IQ3_KS (2026-10-26 19:37)

The plan's G5 named `IQ3_KS`, but no IQ3_KS of this model existed (the only one on disk was a 3B), which
is why only IQ4_KS had ever been compared. Quantised one from the same F16 base
(`iq3ks_15b.gguf`, 662.6 MB) and ran **5 interleaved pairs, order alternating, r=3**:

| model | body bpw | on-disk bpw | PPL (20 chunks) | pp512 | tg128 |
|---|---|---|---|---|---|
| K2 (`k2_ptq`) | 3.34 | 6.22 | **10.5894** | 3203 | 175.6 |
| IQ3_KS | ~3.1 | 3.44 | 13.2312 | 8403 | 191.1 |
| IQ4_KS | ~4.2 | 4.54 | 10.9485 | 8160 | 185.1 |

Paired ratios vs K2: IQ3_KS **x2.623 prefill (5/5)**, x1.088 decode (5/5); IQ4_KS x2.545 (5/5),
x1.054 (5/5). So: prefill 2.6x behind, decode 5-9% behind, perplexity 20% better than the rate-matched
IQ3_KS and 3.3% better than the quality-matched IQ4_KS. Caveat: **clocks not locked** (needs root);
interleaving + unanimous sign tests are the substitute, stated in the PR and the merge plan.

Note on rates: the on-disk figures are not comparable across formats because K2 carries an f16
embedding (467 MB of the file). K2's 3.34 is a *body* rate; IQ3_KS's 3.44 is a whole-file rate. The
paper's convention (body rate, on-disk rate stated beside it) is the honest way to present both.

## 27. "Slow because of the kernels, not the trellis" — measured (2026-10-26 19:45)

User's framing, now backed by an ablation on the *shipped* kernel: replacing the trellis decode with a
constant (leaving staging, MMA and everything else identical) gives **pp512 3207 -> 3720 t/s**, i.e.
the format's decode is **13.8%** of prefill time (it was 29% before the software pipeline). Even with
zero decode we stay 2.3x behind IQ3_KS, because the remaining work is a plain GEMM on an
f16-with-f32-accumulate path whose hardware ceiling here is ~4.9k t/s -- half the int8 rate (~9.8k)
that the fork's MMQ types ride.

Supporting facts written into the PR/spec: decode is 175.6 t/s vs IQ3_KS 191.1 and IQ4_KS 185.1 while
reading ~26% fewer bytes per body weight (3.34 vs ~4.2 bpw) -- per byte the format is at parity; and
the decoded values are int8-representable (+-3.45 with one constant scale), which is exactly what the
parked int8 kernel exploits. So the 2.6x is this implementation (f16 arithmetic, one barrier per
16-input stage, no cp.async, 64x128 tiles at 2 blocks/SM), not trellis coding. Parity is *not*
demonstrated and the PR says so.

## 28. Reboot verified; sigma_reg sweep launched (2026-10-26 20:10)

**Reboot (user-initiated).** Post-boot state checked and clean: up 9 min, load 0.25, PSI stall counters
~0 (cpu 0.9 s / io 1.0 s cumulative, i.e. nothing), 12 GB free (was 6 GB before the cleanup), governor
`performance` (rc.local ran), GPU idle. All deliverables intact on persistent storage: repo at
`2079fbcf` + `release/k2-3.34bpw-v1`, paper, `K2_PAPER/release_stage/` (paper + 45-file evidence +
`.zenodo.json`), the int8 WIP patch, the session log and the publish script. Two things were on tmpfs
and would have been lost, caught before the reboot: the release staging (copied to
`K2_PAPER/release_stage/`, and `publish_zenodo.sh` repointed) and the bench harnesses (copied to
`qat_tools/bench_scripts/`).

**sigma_reg sweep launched** (`qat_tools/bench_scripts/sigma_sweep.sh`). The pack stage shapes
quantisation error with LDLQ, `sigma_reg = 0.025` by default (exllamav3 `-hreg`, also overridable via
`K2_SIGMA_REG`). The open question (K2_QATV2_PLAN): is that shaping *loss* recoverable for zero bits, or
is it doing its job? Arms: sigma = 0 / 0.005 / 0.05, each encoded from the same HF model with the same
allocation (-b 3.34, -hb 6, no per-tensor recipe) so sigma is the only variable, then GGUF-rewritten
with the f16 embedding exactly like the deployed artifact, then 20-chunk perplexity on one protocol.
The 0.025 baseline is the deployed pack itself (PPL 10.5894), so no arm is wasted.

Interpretation rule, fixed before the numbers exist: if sigma = 0 (or any arm) beats 10.5894 by more
than the ~1% session noise, the pipeline gap is partly shaping loss and the paper's "training/shaping
does not close the gap" needs qualifying; if all arms land at or above the baseline, the default stands
and the gap is confirmed to be rate.

## 29. Process failure caught by the first measurement after the reboot (2026-10-26 20:06)

After the decode ablation (delta 3207 -> 3720 t/s) I reverted the **source** with `git checkout --` but
did not **rebuild**, then declared the tree safe for the reboot. The first perplexity run after the
reboot read **1.4e69** on the deployed artifact -- the binary still contained the ablation (constant
weights). Rebuild fixed it: **PPL 8.9478** on the 2-chunk probe, i.e. the expected value.

No measurement was contaminated: the ablation build was created at 19:44 and every reported number
(G5 pairs 19:37, the ablation's own 3720, the summary/PR edits) predates it or is the ablation itself.
But the gap is real and worth naming: **after an ablation, rebuild before trusting the tree, and a
"safe to reboot" check should include one PPL identity run**, not just process/git checks. That is now
part of the checklist.

Second, smaller bug from the same window: the sigma sweep script used `$TAG_encode` instead of
`${TAG}_encode`, so `set -u` killed the encode step. Fixed; the sweep then started cleanly with the
baseline at 10.5894 and sigma=0 encoding.

## 30. Encoder speed: what actually applies (2026-10-26 20:45)

Investigated the low-rank Hessian + Woodbury idea the user proposed. Result, in two halves:

**Verified:** `H = X^T X + sigma*d*I` with only 250 calibration rows, so `H^-1 = (1/sigma)[I - X^T(sigma I_r +
XX^T)^-1 X]` is exact. `bench_scripts/woodbury_identity_check.py` confirms it at **1.6e-14 / 3.0e-14 /
6.4e-14** relative error for d = 256 / 512 / 1536 and shows the per-solve cost dropping from 200-600 ms
(dense, numpy) to 0.12-0.31 ms.

**Refuted for this code:** the LDLQ loop does not use `H^-1`; it uses **slices of the Cholesky factor**
(`bb_L = L[bj:, i+bi:i+bj]`, then `addmm_(bb_L.T, bb_err)`). `L` is dense even when `H` is low-rank +
diagonal, so no Woodbury form produces those slices cheaply. An `H^-1`-only LDLQ is a *different*
algorithm with different numerics, not a speedup. My earlier "exact and big" claim was wrong about
fitness (right about the operator); the design doc is corrected in place.

**What actually applies to the encoder**, in order: multi-GPU sharding (free, quality-neutral, now
exposed as `--devices`/`--device-ratios`); TF32 for the one pure matmul (`X^T X`, exposed as `--tf32`,
A/B queued to prove the per-layer rfn is unchanged); and a cheaper *search* (beam/M-algorithm), which is
the only lever with a quality cost. The dense Cholesky is inherent to the algorithm as written.

Also measured tonight: **sigma = 0 gives PPL 4.44e7** -- the damping is load-bearing, so "drop the
Hessian" is not an option at all.

## 31. The sigma sweep was mis-parameterised (2026-10-26 21:35)

`-hreg/--hessians_reg` in exllamav3's converter sets **only** `quant_args["sigma_reg_out"]`
(convert_model.py:523), i.e. the **output-side** Hessian damping. The *main* damping that LDLQ uses is
`quant_args.get("sigma_reg", _SIGMA_REG_DEFAULT)` with
`_SIGMA_REG_DEFAULT = float(os.environ.get("K2_SIGMA_REG", "0.025"))` (quantize.py:503, 1070). So the
sweep I launched varied the output-side value, not the main one.

Corrected reading of the sweep so far:

| `-hreg` (sigma_reg_out) | PPL | what it means |
|---|---|---|
| 0.025 (default, deployed pack) | 10.5894 | baseline |
| 0 | 44,440,843 | **output-side damping cannot be zero** -- the down_proj solve fails |
| 0.005 | 44,440,843 | same at 0.005 |
| 0.05 | running | higher output-side damping |

So §28/§30's claim that "sigma=0 gives PPL 4.4e7, the damping is load-bearing" is **too broad**: the
evidence covers `sigma_reg_out` only. The main-damping question needs `K2_SIGMA_REG=<v>` in the
environment and has not been run. Corrected here; the design doc's sentence about sigma=0 should be read
with the same restriction.

## 32. Beam search as a quality mode: measured frontier (2026-10-26 22:00)

User's question: the beam is fast enough but not good enough to be a quality option. Measured at tile
level on 3 REAL tiles (q_proj, K=3), all arms closing the **same** ring anchor state so only the search
differs; distortion is the true 16-bit decode of the achieved cost.

| arm | states kept | rfn | vs exact |
|---|---|---|---|
| exact (shipped, 2-pass ring) | 65536 | **0.15474** | -- |
| beam 4096, ring-closed | 4096 (16x fewer) | 0.15832 | **+2.3%** |
| beam 1024, ring-closed | 1024 (64x) | 0.16701 | +7.9% |
| beam 256, ring-closed | 256 (256x) | 0.17340 | +12.1% |
| beam 1024 + 4-per-bucket diversity | 1024 | 0.19996 | +20% (worse!) |

Conclusions:
- **A wide beam is a legitimate mode**: 16x fewer states for +2.3% distortion. The theory table's
  "beam B=64 => +24%" was a *narrow*-beam result, not a property of beam search as such.
- **Diversity hurts here** and the reason is structural: in a *state* beam every kept entry is already a
  distinct window, so there are no near-duplicate paths to spread out; the constraint only displaces good
  states. (Diverse beam search is a *path*-beam technique.)
- Two of my own earlier arms were bugs and are retracted: "beam beats exact" (a pruned start state let an
  unconstrained minimum in) and "exact 1-pass == exact 2-pass" (the passes argument was ignored). The
  numbers above are after both fixes, with one shared anchor per tile.
- Caveat: tile-level distortion on 3 tiles of one tensor, normalised globally rather than per-channel
  `su`, so the absolute values are not comparable to the paper's 0.0273 -- only the arm-to-arm ratios.
  The artifact-level test (PPL + 246-item retention) needs the beam implemented in the CUDA encoder.

Next quality levers, untested: a lookahead-pruned beam (cost + heuristic) and a local-exact refinement
over short segments of the beam path (needs a traceback in the prototype).

## 33. Beam quality levers: what helps and what does not (2026-10-26 22:10)

Tested the two remaining ideas, plus WAVA re-anchoring, on 3 real tiles with a shared ring anchor:

| lever | result |
|---|---|
| wider beam | **works**: B=4096 -> +2.3% distortion at 16x fewer states; B=1024 -> +7.9%; B=256 -> +12.1% |
| diversity (cap states per prefix bucket) | **hurts**: +20% vs +7.9% at the same width. Structural: in a *state* beam every kept entry is already a distinct window, so there are no near-duplicate paths to spread; the constraint only displaces good states |
| WAVA extra passes (iters=3) | **no gain**: identical rfn to the 2-pass arms (0.15832 / 0.16701 / 0.17340) -- the re-anchoring converges after one pass, so extra passes are pure cost |
| local-exact refinement | **not cheap**: the cost of an exact step is the 65,536-state sweep, not its length, so refining 8 segments of 32 steps costs ~100% of the whole exact run. Only 8-16-step segments (~3-6% each) are affordable |

Speed consequence (search time ∝ states kept; the encoder is hardware-bound at ~2e10 evals/s):

| arm | distortion | search speedup | total if search = 60% of 19.4 min | if 90% |
|---|---|---|---|---|
| exact | -- | 1x | 19.4 min | 19.4 min |
| beam 4096 | +2.3% | 16x | 8.3 min | 3.0 min |
| beam 1024 | +7.9% | 64x | 7.8 min | 2.2 min |
| beam 256 | +12.1% | 256x | 7.7 min | 2.0 min |

So the beam's total-time benefit **saturates at the Hessian/LDLQ floor**: past B=4096 the total barely
moves while distortion climbs. B=4096 is the quality mode; narrower beams are only worth it if the
`calib` vs `quant` split says the search is ~90% of the encode.

## 34. Beam at 8192 states is lossless; why the exact is 3.3x off its cost model (2026-10-26 22:20)

**Wide-beam result (tile level, 3 real tiles, shared ring anchor):**

| states kept | rfn | vs exact |
|---|---|---|
| 65536 (exact) | 0.15474 | -- |
| 4096 | 0.15832 | +2.3% |
| **8192** | **0.15474** | **0.0%** |
| 16384 | 0.15474 | 0.0% |
| 32768 | 0.15474 | 0.0% |

So the quality cliff is between 4096 and 8192: **8x fewer state-evaluations at no measured distortion
cost**. That makes B=8192 the operating point to validate at artifact level, and it reframes the earlier
"beam costs +24%" (that was B=64).

**Why the exact encoder does not reach the 4.4 M weights/s of the cost model (measured 1.32 M):**
the launch geometry is `qt_optimized_threads(K=3) = 1024` threads with `__launch_bounds__(1024, 1)`, i.e.
**one block per SM, with a `__syncthreads()` every trellis step** (512 barriers per tile, 2 passes). At
each barrier there is no other resident block to fill the gap, so the SM idles; on top of that every step
writes one edge byte per state into the history buffer (8 KB/step at K=3, ~4 MB per tile). A tile costs
~5.8 ms on this box, i.e. ~11 us per step against a naive ~0.1 us -- a 100x gap, which cannot be
arithmetic. Conclusion: the encoder is **latency/synchronisation bound**, so the 3.3x cost-model gap is
probably an underestimate of the available headroom, and this is kernel work with no quality trade.

## 35. Decision: kernel fix now, beam in a later release (2026-10-26 22:35)

User's call: fix the encoder kernels now, defer the beam to a later release (QATv2 era). Two reasons that
is the right order: the kernel fix is **quality-neutral by construction** (only the schedule changes, so
the pack must be byte-identical — md5-checked), while the beam is a quality trade that needs its own
artifact validation. And the kernel fix benefits every conversion, beam or exact.

**The fix.** `qt_optimized_threads(K, cb)` returns 1024 for K <= 4, and the kernel is declared
`__launch_bounds__(NT, 1)`. With two `edges`-sized half cost arrays (16 KB each at K=3) that is exactly
one block per SM, and the kernel takes a `__syncthreads()` every trellis step — 512 barriers per tile with
no other resident block to fill them. exllamav3's host side already queries
`cudaOccupancyMaxActiveBlocksPerMultiprocessor` and sizes its scratch as `blocks_per_sm * num_sms`, so
lowering the block size to 512 (K<=4) / 256 (K>=5) should let 2-3 blocks reside and multiply the
independent work available at every barrier — a one-constant change.

**Test in flight** (`bench_scripts/encoder_occupancy_ab.sh`): baseline encode of 6 modules with the
installed .so (time + md5), patch, rebuild the extension for sm_86 only (176 CUDA TUs — the long step),
install with a backup, encode again, then compare md5 (must be identical) and wall time. If the pack
differs, the script leaves the header reverted and the old .so installed.

**Beam status:** measured frontier (B=8192 lossless at tile level, 8x fewer states), not implemented in
CUDA, deferred per this decision.

## 36. The encoder split, measured (2026-10-26 22:50)

`bench_scripts/measure_encoder_split.py` (converter run in-process with `AutoSplit.report` patched to
print), 6 modules:

```
[split] quant_thread dev0   2.00 M/s   (23.44 s)      per layer, 139.39 s total
```

- **`quant_thread` = 2.00 M weights/s**, 23.4 s of a 43 s layer => **54% of the encode**.
- The other 46% emits no stage (`calib` is not reported on this path): Hessian build + Cholesky, proxy
  errors, I/O. (Consistent with the Cholesky sizes: `down_proj` is d=8960, 321 MB, 2.4e11 FLOPs.)
- So the search is **2.2x off its own cost model** (2.0 vs 4.4 M w/s) while the *rest* is half the time.

Projections for a 35B (today ~9 h at our measured rate):

| change | total |
|---|---|
| beam 8192 (8x on the search, measured lossless at tile level) **[ERRATUM: 1.0x seen section 42]** | ~~~4.7 h~~ withdrawn |
| occupancy fix (2x on the search) **[ERRATUM: measured 0.99x -- see section 39]** | ~~~6.6 h~~ withdrawn |
| both | **~4.4 h** |
| floor even with a free search | **~4.1 h** |

Conclusion: search-side work saturates at ~4.1 h for a 35B; only multi-GPU or the LDLQ->OBQ
reformulation (which would let Woodbury apply) can go below that. This is the number that retires the
"make the search faster" line of work unless it is combined with one of those two.

## 37. Encoder micro-tunes: TF32 REJECTED, Hessian structure, what remains (2026-10-26 21:40)

**TF32 calibration Hessian (`tf32_ab.sh`, 6 modules, env-only A/B) — REJECTED.**
Layer `rfn`, same command and order, only the matmul precision differs:

| layer | fp32 | TF32 | delta |
|---|---|---|---|
| 0 | 0.062791 | 0.062981 | +0.3% |
| 1 | 0.019867 | 0.027072 | **+36%** |
| 2 | 0.009972 | **Cholesky failed** | `linalg.cholesky: input is not positive-definite` |

So a TF32-accumulated `H = X^T X` loses positive-definiteness and the encoder dies at layer 2; LDLQ needs
the Cholesky factor, so there is no quality-neutral TF32 path here. The logged "wall 121 s vs 270 s" is
meaningless (two layers before the abort) and came from a script bug: `rc=$?` was expanded *after* a
`date` subshell inside the same `echo`, so it always read 0 and hid the crash. Fixed in the new scripts.

**Hessian structure (measured in the converter log):** `Hessians: block.attn.input, block.attn.o,
block.mlp.down, block.mlp.input` = **4 per layer, not 7** — q/k/v share `attn.input`, up/gate share
`mlp.input`. Sharing is already done; no redundancy left there.

**What the non-search 46% is made of** (code reading): capture `H.addmm_(x.T, x)` (linear in cal rows),
`block_ldl` -> full `d x d` `torch.linalg.cholesky` (cubic, one per distinct Hessian), a cheap 128-block
Hadamard rotation, the rfn/cos/sqnr proxies, and `baddbmm` feedback inside `ldlq` (quadratic, small).
Attribution by torch profiler: `bench_scripts/measure_hessian_split.py` (CONVERTER runs end with
`sys.exit()`, so the summary must sit in a `finally` or it is never printed).

**No-fast-fix verdict on the Cholesky:** at d=8960 it is >=19 s at this card's fp32 peak (2.4e11 FLOPs /
12.7 TFLOP/s) and TF32 tensor cores are only ~2x fp32 here, so 3xTF32 costs what it saves and plain TF32
is proven fatal. cuSOLVER potrf is already blocked. The only structural fix is to stop factorising `H`:
`H = sigma I + X^T X` with rank(X) = cal rows (~250) << d **[FALSE at the shipped calibration: rank = d, see
below]** , so `H^-1` is `O(r^3)` via Woodbury (identity
**[ERRATUM: the rank premise is false at the shipped calibration -- 250x2048 = 512k positions vs d = 8960
gives rank(X) = d, so there is no low-rank form; see `K2_QATV2_PLAN.md` section 10.2]**
measured exact to 1.4e-14, ~2000x cheaper per solve) -- but `ldlq` consumes Cholesky slices
(`bb_L = L[bj:, bi:bj]`), so switching to the H^-1 (GPTQ) recursion is an ALGORITHM change needing the
full battery. That is the QATv2 business case, now priced.

**Still available without a numerics change:** occupancy (test running), beam 8192 (8x the search,
lossless at tile level), proxy subsampling for production encodes, `--cal-rows`, finalisation overlap.

## 38. Encoder attribution by torch profiler (2026-10-26 21:45)

`bench_scripts/measure_hessian_split.py`, 3 modules, wall **144.3 s**, sum of CUDA op time **195.2 s**
(1.35x overlap with the finalisation thread / memcpys). Per-op CUDA time:

| item | s | share |
|---|---|---|
| tile search `quantize_tiles_frac_kernel<3,43690u,256>`, 5941 calls | 61.13 | 42% |
| `quantize_tiles_optimized_kernel<4,2,256>` | 1.56 | 1% |
| **Hessian accumulation `aten::addmm_` [8960,8960]**, **750 calls = 250 cal rows x 3 modules** | **24.73** | **17%** |
| its backing cuBLAS kernel `cutlass_80_simt_sgemm_256x128` | 24.70 | (same time) |
| `aten::addmm_` [1536,1536] x2250 (the three small Hessians) | 2.64 | 2% |
| capture dtype conversions `_to_copy`/`to`/`copy_` [1,2048,1536] x2515 | 6.45 (+1.3+1.3) | ~6% |
| `Memcpy DtoH (Pageable)` x930 | 6.13 | 4% |
| calibration forwards (`f16acc::gemm_kernel` x3000, cutlass s16816, paged attn) | ~12 | 8% |
| capture elementwise (`isfinite` x750, `square`/`pow`/`mul` [2048,8960]) | ~6 | 4% |
| `cudaLaunchKernel` x134030 / `cudaPeekAtLastError` x32197 | 1.70 / 0.78 | 2% |

**The Cholesky does NOT appear in the CUDA table at all** (0 rows matching cholesky/potrf/trsm/syrk).
With three `8960^2` factorisations it would be the single largest entry, so either it is CPU-side (the
first version of the script filtered out rows with zero device time -- v2 prints the CPU table and the
unfiltered factorisation rows) or the configuration does not factor at `8960^2`. Open question.

**Immediate consequences (no algorithm change needed):**
- The Hessian build is **19% of the encode and is pure redundancy**: 250 rows x 2048 cols = **512k token
  positions** are pushed into an 8960x8960 matrix (8.2 s/layer). `--cal_rows/--cal_cols` cut it linearly
  -- the "make the Hessian faster" answer that does not touch LDLQ at all. A/B: `caldata_ab.sh`.
- `num_ref_states = 5`, so the rfn/cos/sqnr proxies are already cheap -- the proxy idea is dead.
- Checkpointing is **synchronous** and rewrites the WHOLE state (3.1 GB here, ~70 GB for a 35B) every
  `-cpi` seconds (default 120) through `save_tensor` + a ckpt_new/ckpt/ckpt_old rename dance. On ntfs3
  (~150 MB/s) one save is ~20 s. A/B: `checkpoint_io_ab.sh`. **[ERRATUM: measured ~5 s for two saves --
  the cache absorbs them; see section 43]** For a 35B this term alone could dominate
  the encode, so `-cpi` and the work-dir device are shipping candidates; incremental saves are the
  proper fix (deferred).

## 39. Artifact published, occupancy rejected, PR rebased (2026-10-26 22:05)

**Artifact is public** (nothing else in this release depends on a pending account):
`https://huggingface.co/TitanMamo10/Qwen2.5-1.5B-K2-3.34bpw` -- the deployed 1.5B 3.34 bpw GGUF,
`qwen2.5-1.5b-k2-3.34bpw.gguf`, **sha256 verified equal to the local file after upload** (1,197,866,720 B,
md5 `584a86cb105e9771e6b30d3c7f08f52d`), plus `README.md` (model card) and `evidence/k2_evidence.tar.gz`
(45 files). Landing page HTTP 200.

**Rate anatomy of the released artifact** (read from the GGUF, not from notes): 197 trellis tensors,
1,543,569,408 weights. `K=3` 61.1% of trellis weights, `K=4` 18.8%, `K=5` 5.0%, `K=6` 15.1% (the output
head, `-hb 6`); `token_embd` is F16. Body rate excluding embed+head = **exactly 3.34 bpw** (so the paper's
number is right); on-disk 6.22 bpw counted over the model's 1.54B parameters (the tied embedding is stored
twice: f16 copy + trellis head). `-b 3.34` is an allocator BUDGET (`conversion/allocation.py` steps each
module through {1, 1.5, 2, 2.5, 3, 3.5, 4..8}, priority-ordered, per-module floors), which is why every
module has a different rate and why a different model gets a different mix.

**K2E (6-bit trellis embedding) vs f16 embedding**, measured only at 6.00 body bpw: 1285 MB / 97.2%
coding / 96.7% HS / PPL 10.3090 vs 1634 MB / 98.4% / 97.0% / 10.3150. So PPL is equal-to-better, HS -0.3,
coding **-1.2 points** for a 21% smaller file. The **3.34 + K2E** combination has NOT been measured; the
f16 embed is 466 MB of the 1198 MB file, so it would land near ~900 MB. Do not claim equal quality there
without the run.

**Occupancy A/B: REJECTED.** NT=1024 268 s vs NT=512/256 271 s (0.99x), pack byte-identical
(`e152a21cf7a40656c3e113d4f3adfbe9`), per-layer rfn identical. The search is throughput-bound, not
barrier-bound; halving the block doubles per-thread work. Header + `.so` auto-reverted. The 35B
"9 h -> 6.6 h" occupancy projection is dead; the search's remaining lever is the beam (QATv2). **[SUPERSEDED -- see the erratum below / section 42]**
**[ERRATUM (see section 42): both halves wrong -- beam lossless width = full state count at K=3 => 1.0x,
and Woodbury needs rank(X) << d which fails at 512k calibration positions. The search has no lossless lever.]**
Note: the build must run in the `cuda-box` distrobox (host has no nvcc; `CUDA_HOME` = /usr there), and
the occupancy baseline md5 for a `--max_module` run is `ckpt/state.safetensors`, NOT `model.safetensors`
(which is only written for a full-model run).

**PR: rebased on current main, zero conflicts.** 9 commits on `a61ad787` (upstream main at the time),
26 files, +3024/-97; the skipped cherry-pick was `270aa5ae` (the Prism split guard, now upstream). Final
body at `K2_PR_BODY_FINAL.md` (template lines + `Review Complexity: High` + the approved text + artifact
and evidence links; NO AI-disclosure paragraph per the maintainer already knowing the author). Not pushed,
not opened yet -- waiting on the compile check (`build-cuda` is configured for `/usr/bin/nvcc`, i.e. the
container).

**Paper** rebuilt with the artifact link in the Reproducibility section: 19 pp, 0 errors, only the
pre-existing underfull-hbox warnings (lines 966, 1021). `.zenodo.json` now carries related_identifiers
for the HF artifact and the fork.

## 40. The sigma patch is committed and portable (2026-10-26 22:20)

`K2_SIGMA_REG` (input-side LDLQ damping) used to be an uncommitted edit in the exllamav3 checkout --
i.e. the sigma sweep was not reproducible from a clean tree. Now:

- branch `k2-sigma-reg` in `/home/titan/exllamav3`, commit **4d587f5**, one file, 5 call sites + the
  module-level default; `master` left untouched at upstream `a3bfbfd`.
- `patches/exllamav3-k2_sigma_reg.patch` (61 lines), `git apply --check` **clean on upstream master**
  (verified in a throw-away worktree), also mirrored in `TitanMamo/qwen25-q3-qat`.
- harden: `float(os.environ.get("K2_SIGMA_REG") or 0.025)` -- unset **and empty** both give 0.025
  (an empty variable used to raise ValueError and kill the encoder); a non-numeric value fails loudly.
- default path is bit-identical to upstream, so no recipe changes.

## 41. Rebased tree: runtime smoke PASSES; and the PR's tests were dead code (2026-10-26 22:45)

**Build**: `llama-cli` + `llama-quantize` build from the rebased branch (9 commits on `a61ad787`).
The non-zero exit I first saw was only a missing target (`test-backend-ops` is not in this fork's
default target list), not a compile error.

**Runtime smoke on the published artifact** (md5 `584a86cb105e9771e6b30d3c7f08f52d`), greedy:

| arm | "The capital of France is" (24 tok, `--temp 0`) |
|---|---|
| F16 | "Paris. The capital of France is also the capital of the European Union. ..." |
| K2 6.00 bpw | **byte-identical to F16** |
| K2 3.34 bpw | "Paris. The capital of France is Paris. The capital of France is Paris. ..." (correct, repetitive) |

**Caution for future probes**: the same prompt at the CLI's DEFAULT sampling answered "B. London / C.
Berlin" -- a sampling artifact that looks exactly like a broken load. The France probe must be run
greedy (`--temp 0`) or it produces false alarms.

**Perplexity cross-check at matched settings** (1 chunk, `-c 512`): F16 **6.3425 +- 0.967** vs K2 3.34
**6.3438 +- 0.958** -- a 0.001 difference. So the paper's 10.5894 is purely the 20-chunk protocol, and
K2 does not lose anything on the first chunk.

**Traps confirmed**: host binaries need `LD_LIBRARY_PATH=/home/titan/cuda-libs` (otherwise
`libcudart.so.12: cannot open`); the PPL corpus used here is
`/mnt/Data/ik_llama.cpp/quant_test/ppl_wikitext2.txt`.

**PR DEFECT FOUND**: the two test files the PR adds were **never wired into `tests/CMakeLists.txt`** --
`test-k2-moe-cpu.cpp` (315 lines) and the K2 MoE cases in `test-backend-ops.cpp` were dead code that no
build would compile, and this fork does not even build `test-backend-ops` by default. Fixed by adding
`llama_build_and_test(test-k2-moe-cpu.cpp)` and `llama_target_and_test(test-backend-ops.cpp)`; the build
+ run verification is pending (deferred so it does not contaminate the encoder A/B timings).

## 42. CORRECTION: the beam is not a speed lever (2026-10-26 22:55)

I claimed "beam 8192 = 8x on the search half, lossless, 35B 9 h -> 4.7 h". **Wrong**, and the arithmetic
that kills it is one line: `edges = 65536 >> K` gives **8192 states at K=3** (61% of the released
artifact's trellis weights), 4096 at K=4, 2048 at K=5, 1024 at K=6. The measured **lossless** beam width
(8192) is therefore the *full* state count at K=3 and exceeds it at every higher K -- the
quality-preserving beam IS the exact search, so it is **1.0x faster, not 8x**.

The frontier, read correctly (tile level, oracle):

| width | distortion | prunes at K=3 | work ratio |
|---|---|---|---|
| 8192 | 0.0% | no | **1.0x** |
| 4096 | +2.3% | 2x | 2.0x |
| 1024 | +7.9% | 8x | 8.0x |
| 256 | +12.1% | 32x | 32.0x |

Second error in the same claim: the 2/8/32x are WORK RATIOS, not timings. The kernel issues a
`__syncthreads()` per trellis step (512 per tile) and the occupancy experiment already showed it does not
scale with per-SM work (predicted 2-3x, measured **0.99x**). A beam build would land well below its work
ratio, and no beam build exists to measure.

CONSEQUENCE (now in `K2_ENCODER_BEAM.md`, `K2_ENCODER_SPEEDUPS.md`, the release README's open items):
the search half (54% of the encode) has **no lossless reduction available**. It can only be traded away
(+2.3% distortion for 2x) or removed by a codec with a smaller state space -- and every measured codec
candidate is worse than what we ship. The encoder's honest lossless gains are assembly (7%) plus whatever
the checkpoint A/B shows.

Same correction to the QATv2 encoder answer: "~2x from the beam, lossless" is withdrawn; QATv2's encoder
speed story is a quality trade (2x for +2.3%) or nothing, and its quality story is the pipeline (sigma /
scales / aliasing), not the search.

## 43. Last two encoder A/Bs: checkpoint ~2%, calibration -23% with unmeasured quality (2026-10-26 23:00)

**Checkpoint I/O** (6 modules): A default 269 s (2 saves, md5 `e152a21c`) | B `-cpi 100000` 264 s (0 saves) |
C default on the SSD 266 s (2 saves, **md5 identical to A**). Two 3.1 GB whole-state saves cost ~5 s --
the page cache absorbs them, so my "15% of the encode / 20 s per save" estimate was wrong by ~8x on this
box. The device is irrelevant at 1.5B; `-cpi` only matters where the cache cannot hold the state (35B:
~70 GB vs 31 GB RAM). Quality-lossless, worth ~2%.

**Calibration data**: A 250x2048 271 s | B 125x2048 **207 s (-24%)** | C 250x1024 **208 s (-23%)**; packs all
differ; max |rfn delta| 0.0036 (B, layer 1 +18% rel.) and 0.0028 (C, layer 3 -27% rel.). The direction is
the tell: less calibration data makes rfn look *better* on most layers, which is train/eval overlap (rfn is
measured on the rows that built the Hessian). So the proxy cannot adopt this; promotion needs a full-artifact
A/B (~1-2 h/arm). Documented knob, unmeasured quality cost, released artifact unchanged at 250x2048.

**Lossless encoder tally, final (measured):** assembly 91 s -> 4 s (~7%) + checkpoint ~2% = **~9%**, with
the search half (54%) untouched. Everything else on the table trades quality, and the beam/Woodbury pair is
closed in `K2_QATV2_PLAN.md` §10.

## 44. Tests fixed and verified; branch pushed (2026-10-26 23:10)

**The K2 MoE test now builds and passes** (`tests/test-k2-moe-cpu.cpp`, 315 lines, wired into
`tests/CMakeLists.txt`): K=2..6, the 35B MoE in/out shapes, E=4/6/8, 1/2/3/6 threads, with and without
`su`/`sv` -- **bit-exact against the scalar reference in every case** (max abs 0.000e+00), "all cases
passed". Before this it was never compiled.

**`test-backend-ops.cpp` is dropped from the PR.** That file does not compile in this fork at all -- and not
because of our hunk: `ggml_quantize_chunk`, `ggml_upscale` and `ggml_upscale_ext` take different arguments
than the file assumes (pre-existing API drift from upstream llama.cpp), which is exactly why this fork never
makes it a target. Adding K2 cases to an unbuildable file is dead code, so the hunk is reverted and the
runnable coverage is the MoE test above. Net diff in `tests/`: 1 line + 315 lines.

**Final state**: branch `k2-t16` = 10 commits on `a61ad787`, final tree rebuilds with **0 errors**,
`test-k2-moe-cpu` passes, pushed to `TitanMamo/ik_llama.cpp` (PR URL offered by GitHub). The PR body now
carries the MoE bit-exactness row and the corrected test row. Only `gh pr create` remains, which waits on
the author's go.

## 45. Agent-swarm audit (5 auditors) — findings and fixes (2026-10-26 23:45)

**Retraction sweep (auditor 5): 10 surviving unqualified claims found, all fixed** -- beam-as-lever in
`K2_ENCODER_SPEEDUPS.md` + the ledger, the Woodbury low-rank premise (ledger + `K2_ENCODER_WOODBURY.md`,
whose identity section now carries a RETRACTED PREMISE block and whose projected 8-10 min / "35B 9 h -> 4 h"
is struck), the K2E "quality-neutral / FULLY VALIDATED" overclaims in `K2_CLOSURE.md`, `K2_STATUS.md`,
`K2_HANDOFF.md`, `K2_TRACKS.md`, and the "one save is ~20 s" estimate (now says the cache absorbs it).
The beam doc's "the quality-neutral occupancy fix ships first" recommendation is also corrected (occupancy
measured 0.99x).

**Deposit auditor: 3 blockers, all fixed.**
1. `publish_zenodo.sh` shipped the STALE `K2_PAPER/.zenodo.json` (old title, `related_identifiers: []`) and
   the stale `ZENODO_DEPOSIT.md`. Now ships `release_stage/` copies; dry run confirms the K2+SantanaQAT title
   and 2 related identifiers. It also silently dropped `K2_ENCODER_SPEEDUPS.md` -- now shipped.
2. **The published artifact was the PTQ pack while the README quoted the QAT pack's numbers.** Fixed by
   publishing BOTH: `qwen2.5-1.5b-k2-3.34bpw.gguf` (PTQ, md5 `584a86cb`, PPL 10.5894, coding 73.6 %) and
   `qwen2.5-1.5b-k2-3.34bpw-santanaqat.gguf` (md5 `45d62417`, PPL 10.2886, coding 79.7 %, HS 87.3 %); README,
   PR body and the model card now carry a per-file table.
3. Tensor counts: the shipped `grid.json` has **71/90/35 = 196** tensors, not 70/195 -- corrected in
   `main.tex` and `pipeline/README.md`. Page/table counts in ZENODO_DEPOSIT/ARXIV_SUBMISSION corrected
   (19 pp, 7 tables). Evidence archive grew 45 -> **50 files** (the five Appendix-B files that were cited but
   absent: `tasks_k6h4_k2e_code`, `tasks_k6h4_k2e_hs1200`, `tasks_k6_hs2000`, `tasks_qwen15_gsq3`,
   `dryrun.manifest`), and the PR body now describes the archive accurately (per-item JSONs, not gate logs).

**PR auditor: 1 HIGH, 4 MEDIUM, 4 LOW -- HIGH and all MEDIUMs fixed.** Decode headline 176.4 -> **175.6**
(the 5-pair A/B; 176.4 was the earlier single-run check), "26 % fewer bytes" -> **21 % fewer body bits**
(3.34 vs 4.25; 26 % compared a body rate against IQ4_KS's on-disk rate), IQ4_KS prefill now labelled per role
(8160 paired, 8488 single-run) with the 2.55x ratio, "parked behind `K2_T16_INT8`" -> parked outside this PR
(the flag is not in the branch), "no llama-quantize changes" (it is not in the diff at all). Remaining LOW
items: the `abba0d50` commit subject hides an unrelated `K2_PRE_TOKENIZER` hunk, and `32a95005` carries the
local hostname as author email (both fixable by rebase/force-push if the author wants them clean).

**CPU-only build verified** (`-DGGML_CUDA=OFF`, host gcc): builds clean, `test-k2-moe-cpu` passes
("all cases passed"), `llama-cli -ngl 0` answers "Paris", 2-chunk CPU PPL **6.1862** (GPU 1-chunk 6.3438).
That closes the PR's CPU-correctness claim on a CUDA-free build.

## 46. Second audit round: the swarm's findings, the figure, the history rewrite (2026-10-26 24:00)

### The five auditors' verdicts, and what each one changed

| auditor | verdict | what was actually wrong |
|---|---|---|
| retraction sweep | 10 surviving retracted claims | beam-as-lever (2 files), Woodbury's low-rank premise (ledger + `K2_ENCODER_WOODBURY.md`, now RETRACTED PREMISE + WITHDRAWN blocks), K2E "quality-neutral/FULLY VALIDATED" (4 files), "one save ~20 s" (3 files) -- all fixed inline |
| deposit/publisher | **not publishable** | shipped the STALE `.zenodo.json` (old title, empty related ids) and stale deposit doc, dropped `K2_ENCODER_SPEEDUPS.md`; tensor counts 70/195 vs the file's **71/196**; page counts 18/15 vs **19**; evidence 45 -> **48**; now also ships `AGENTS.md`, `FLAGS.md`, `fig_qat.pdf`, `G5_pairs.log`, `G5_ppl_ladder.log`, `measurement-log.md` |
| public repo + HF | codec layer accurate, artifact layer not | **the Hub file was the PTQ pack while the docs quoted the QAT battery** -> both packs now published (`…3.34bpw.gguf` md5 `584a86cb`, PTQ 73.6 %/10.5894; `…-santanaqat.gguf` md5 `45d62417`, 79.7 %/10.2886) with per-file tables. Also: **K2E at 3.34 is MEASURED** (-1.2 coding points: 187 vs 190 of 246, scales-controlled) -- our "unmeasured" claim was stale and is corrected in six documents |
| PR branch | **not postable** | HIGH: same artifact/numbers conflation (fixed by the two-pack split + body rewrite); 176.4 -> **175.6/174.0** (paired), "26 % fewer bytes" -> **21 % fewer body bits**, IQ4_KS 8160 paired / 8488 single-run, `K2_T16_INT8` is not in the branch, no `llama-quantize` changes; plus comment-only nits |
| paper vs evidence | 47/55 match, **8 mismatch/unsourced** | 14 text fixes: 7.60 -> 7.64, HellaSwag 677/1197 -> 679/1200 (671 collapsed), R_off and out-of-sample ranges -> the measured file ranges, latent HS 27/11 -> 27/12 (p 0.0237), 0.50 -> **0.40 s/step**, assembly 71 -> **91 s**, 296 -> **272 MB**, encode 4.3M -> **1.33M w/s end-to-end**, parameter shares (summed to 109 %), evidence pointer for water-filling |

### Two measurements this round produced (both archived)

1. **The speed A/B re-run** (`G5_pairs.log`, 5 interleaved pairs): K2 **3180 pp / 174.0 tg**, IQ3_KS 8363/189.6 (**2.63x / 1.09x**), IQ4_KS 8098/184.4 (**2.55x / 1.06x**), 5/5 pairs both directions. The paper now quotes session ranges (3.18-3.25k pp, 174.0-176.4 tg) instead of single-run points, and the archived log is a real one -- the previously archived `G5_proper_pairs.log` was the quantization *prep* log (printf failures, no measurements).
2. **The ladder reproduction** (`G5_ppl_ladder.log`, the paper's own `-c 512 --chunks 20` on `wikitext-2-raw/wiki.test.raw`): F16 **10.3099 +-0.390**, 4.06 arm **10.4966 +-0.399**, 3.34 arm **10.5894 +-0.400** -- i.e. the ladder reproduces to the last digit, which closes the one "NO-SOURCE" ladder item. (My first control used `quant_test/ppl_wikitext2.txt` and read F16 = 7.858 -- a different corpus; that mistake is why the corpus path is now spelled out in AGENTS/FLAGS.)

### Archive surgery (and why)

Dropped from the evidence tarball: `waterfill_bound.json` (err_kl = 0.0 and gain = 1.25e14 on 12 of 16 rows -- numerically degenerate) and `residual_analysis.json` (rel_err 1.41 against the real 0.165; its "86.7 % of residual energy in the top 64 directions" contradicts the paper's rank-channel argument). The paper's evidence table now says water-filling comes from `spectrum_exact.json` and that the bound file is not used. The full measurement log (`measurement-log.md`, 1165 lines) ships so the remaining analysis numbers have a citable source.

### Figure

`fig_qat.pdf` (new, 2 panels, in the paper and the deposit): (a) strict coding 73.6 % PTQ -> 79.7 % SantanaQAT with the 20-gained/5-lost, p=0.004 annotation and the F16 100 % reference line; (b) cost on a log axis: 0.77M tokens / 605 s on one 12 GB card vs the published recipe's 10.5B tokens (1.4e4x), with the saturation note.

### PR history rewritten

`abba0d50` (subject claimed a trellis-layout change; the diff was a 7-line `K2_PRE_TOKENIZER` override) -> reworded to *"convert: allow overriding the BPE pre-tokenizer hash"* with the honest body; `32a95005`'s author email `titan@TitanPC.fritz.box` -> the GitHub noreply address. All ten commits now carry the same author, the tree is byte-identical to the pre-rewrite state (`git diff backup/pre-cosmetic HEAD` empty), and the branch was force-pushed (`9849e9fb`).
