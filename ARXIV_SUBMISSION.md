# arXiv submission — K2 measurement study

Status: **draft complete, build verification in progress**. Do not submit until the checklist at the
bottom is green.

## Metadata to paste into the arXiv form

**Title:** K2 and SantanaQAT: how far is 3-bit LLM quantization from the rate-distortion bound? A measurement study, and why the last points are unavailable

**Primary category:** cs.LG
**Cross-list:** cs.CL
**ACM/other:** none
**Comments:** 20 pages, 3 figures, 7 tables. Technical report; measurements and per-item evidence
released with the paper.

**License:** CC BY 4.0 (or the author's preference)

### Abstract (plain text, the arXiv field)

We ask a narrow, falsifiable question: can a ~3.3 bit-per-weight (bpw) trellis codec be made
near-lossless — retaining >= 96.5% of the full-precision (F16) model's item-level benchmark
performance — by better mathematics or better training, on consumer hardware? We answer it with
measurements rather than projections, using a QTIP/EXL3-class trellis-coded quantizer on
Qwen2.5-1.5B (1.54B deployed parameters) on a single RTX 3060 with 12 GB of VRAM, cross-checked at
Qwen2.5-3B. (i) The ceiling is arithmetic: the coder sits 1.23–1.32x above the Shannon bound at every rate
measured (K=3,4,5,6) — a constant implementation loss — so perfect coding is worth about +5
retention points, not the 17 the bar demands, and the bar requires an effective weight error 3.8x
below the bound at 3.34 bpw. (ii) The source offers
no structure: post-rotation weights are white (kurtosis 3.07; spectral AM/GM 1.13–1.24; residual
AM/GM 1.22–1.38), and an activation-side anisotropy gain of 1.47x in sample collapses to
0.98–1.02x out of sample. (iii) The codec column closes by identification: the trellis is
numerically the D8 lattice (0.02723 vs 0.02735); E8 is 1.22x better but every lattice is bounded by
G*2*pi*e = 1.221x and E8 sits on that floor, so E8 is the whole lattice headroom (+0.29 bpw, a 9%
larger file); entropy coding of the stored stream is exactly 1.000x (H = K to three decimals).
Training saturates: scales-only QAT buys +6.1 points (p = 0.004) and then drives perplexity below
F16 (10.2886 vs 10.3099) while coding stays at 79.7%; latent training is significantly worse on
HellaSwag (p = 0.0139); end-to-end and wider joint windows are monotonically worse. Mechanism: at
K=3 only 0.03% of trellis words flip, because the AdamW walk is ~2% of a Voronoi cell radius, and
trellis codes move 0% or ~100% of their symbols — there is no sparse-nudge regime. The residue is
additive (eight "only-X" arms sum to 100.4%) and thin-margin (KL 0.028 nats, 74.9% top-1 on code vs
93.4% on prose). What moves the number is rate and scale: 5.00 bpw (block rate) reaches 97.6% coding but 93.6%
HellaSwag (an established miss), 6.00 bpw clears both point estimates at 8.49 bpw on disk, and each
parameter doubling is worth +5.1 points (measured 1.5B->3B). We also show that retention
normalisation (conditional on what F16 solves vs a ratio of raw accuracies) can flip a published
verdict, and we list eleven threats to validity found in our own harness.

## Files to upload

| file | role |
|---|---|
| `main.tex` | the paper (single file, no \input) |
| `fig_rd.pdf` | Figure 1 (rate–distortion wall) |
| `fig_ladder.pdf` | Figure 2 (retention ladder with CIs) |
| `fig_qat.pdf` | Figure 3 (SantanaQAT: what it buys, what it costs) |

No bibliography file: citations are in a `thebibliography` block inside `main.tex`.
No external packages beyond TeX Live's standard set (amsmath, booktabs, graphicx, hyperref,
microtype, caption, enumitem, xcolor).

## Build

```bash
# arXiv compiles with pdflatex + its standard bundle
pdflatex main.tex && pdflatex main.tex
# or locally, with no system install:
~/.local/bin/tectonic -X compile main.tex
```

## Pre-submission checklist

- [ ] `main.pdf` builds with **zero errors** (warnings about overfull boxes are acceptable).
- [ ] All three figures present and legible at 100% zoom.
- [ ] Every number in Table 1 (normalisation), Table 2 (RD), Table 3 (ladder) matches the value in
      the corresponding result file listed in Appendix B.
- [ ] The abstract's numbers agree with the body (they were reconciled after the residue-section
      correction: 100.4%, 34%).
- [ ] Claims that were withdrawn during the campaign are labelled as such (Section 9, "Claims
      withdrawn during the campaign") — do not remove that paragraph.
- [ ] Author name/affiliation and any acknowledgements added.
- [ ] Decide whether to release the result JSONs as an ancillary archive; if so, strip absolute
      paths and any keys from the logs (none of the result files contain credentials, but check).

## Deliberate omissions (and why)

- **No 7B retention point.** The 3.34 bpw 7B pack is only ~3.2 GB and runs on the card, but the F16
  *reference* GGUF is 14.2 GiB against 12 GB of VRAM, so the reference arm needs partial offload
  (est. 4–6 h per evaluation). The paper says so in Limitations rather than quoting a
  PPL-only number as if it were retention.
- **No 35B measurement.** Deferred by decision; the paper reports it as an extrapolation from two
  measured doublings and flags it.
- **No QAD/on-policy arm result.** The objective and data path were implemented and verified, but
  the arm was not run to completion; the paper gives the mechanism argument and the measured
  saturation instead of an unrun claim.
- **The code-token damage map.** Its per-arm JSON was overwritten by the wikitext run; the paper
  reports the wikitext arm from disk and the code figures from the log, and says so explicitly.

## Rate-label correction (2026-10-25, after the draft)

Two rate labels inherited from the run tables were wrong, and the paper now corrects them in the
Table 3 caption rather than silently adopting them:

| arm | old label | measured block rate | why the old label was wrong |
|---|---|---|---|
| gate/up | 3.52 | **3.7005** | 3.52 was the allocator's *target*; the greedy solution overshot |
| K>=5 | 5.15 | **5.0000** | 5.15 counted the 6-bit output head; the block rate is exactly 5.00 |

Consequence: the coding exchange rate is 19 points/bpw for the gate/up arm and 15 for a uniform
K>=4 step (not the 2.5x ratio the wrong labels implied), and the interesting finding is the
monotone decline 19 -> 15 -> 12 -> 11 -> 7 points/bpw from 3.70 to 6.00 bpw.

A third correction, found 2026-10-26: the value "err = 0.165" used as the deployed model's weight
error is the **K=3 subset's** error, not the pack's. Re-derived from the stored per-tensor file,
each K measured at its own rate gives 1.32 / 1.23 / 1.30 / 1.28x the bound — a constant ~1.3x
implementation loss, which restores the paper's original thesis. (An intermediate version of the
draft paired the K=3 error with the pack's average rate and claimed 1.67x; that was wrong.) The
model-level figure, if aggregated, is 0.134 at 3.34 bpw = 1.36x the bound.

## Prerequisites only the author can complete

1. arXiv account: https://arxiv.org/user/register
2. **Endorsement** — first-time submitters in cs.LG normally need an endorsement code from an
   existing arXiv author in that category (an academic/institutional email can auto-endorse).
   Without it the submission cannot be started.
3. Confirm the author name and affiliation as they should be cited (currently "Baran Mamo, Independent research").
4. Choose the license (CC BY 4.0 suggested) and accept arXiv's submission terms.
