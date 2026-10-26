# Zenodo deposit — K2 measurement study

**Status:** metadata ready; deposit not yet made (needs the author's Zenodo account + ORCID).
**Why Zenodo:** a DOI without arXiv endorsement, indexed by OpenAlex and Google Scholar, timestamped.

## Metadata to paste

**Resource type:** Publication → Preprint
**Title:** K2 and SantanaQAT: how far is 3-bit LLM quantization from the rate-distortion bound? A measurement study, and why the last points are unavailable

**Authors:** Baran Mamo (Independent research; GitHub/HF handle TitanMamo10) — add ORCID if you have one.

**Publication date:** 2026-10-26
**Version:** 1.0
**Language:** English
**License:** CC BY 4.0
**Keywords:** quantization; quantization-aware training; trellis-coded quantization; lattice
quantization; rate–distortion; LLM compression; negative results; reproducibility

**Description (plain text):**

We ask a narrow, falsifiable question: can a ~3.3 bit-per-weight (bpw) trellis codec be made
near-lossless — retaining at least 96.5% of the full-precision model's item-level benchmark
performance — by better mathematics or better training, on consumer hardware? We answer with
measurements on Qwen2.5-1.5B (1.54B deployed parameters) on a single 12 GB GPU, cross-checked at
3B. (i) The ceiling is arithmetic: the coder sits 1.23–1.32x above the Shannon bound at K=3,4,5,6,
approximately constant, so perfect coding is worth about +5 retention points, not the 17 the bar
demands; at 3.34 bpw the bar requires an effective weight error 3.8x below the bound, which no
coder reaches. (ii) The source offers no exploitable structure: post-rotation weights are white
(kurtosis 3.07; spectral AM/GM 1.13–1.24; residual AM/GM 1.22–1.38), and an activation-side
anisotropy gain of 1.47x in sample collapses to 0.98–1.02x out of sample. (iii) The codec column
closes by identification: the trellis is numerically the D8 lattice (0.02723 vs 0.02735); E8 is
1.22x better at entropy-matched rate but only 1.03–1.10x once a fixed-rate 2^(8R) container is
imposed; entropy coding of the stored stream is exactly 1.000x (H = K to three decimals).
Training saturates: scales-only QAT buys +6.1 points (p = 0.004) for 0.09% of the parameters and
0.77M tokens, then drives perplexity below the full-precision model while coding stays flat;
latent training is significantly worse on HellaSwag (p = 0.0139); end-to-end and wider joint
windows are monotonically worse. Mechanism: at K=3 only 0.03% of trellis words flip, because the
optimizer walk is about 2% of a Voronoi cell radius, and trellis codes move 0% or about 100% of
their symbols — there is no sparse-nudge regime. The residual damage is additive and
unlocalisable (eight substitution arms sum to 100.4% of the whole; best single-type recovery 34%)
and consists of thin-margin flips. What moves the number is rate and scale: 5.00 bpw reaches 97.6%
coding but 93.6% HellaSwag, 6.00 bpw clears both point estimates at 8.49 bpw on disk, and each
parameter doubling is worth +5.1 points (measured 1.5B to 3B). We also show that the choice of
retention normalisation can flip a published verdict, and we list the threats to validity found in
our own harness.

## Files to upload

| file | why |
|---|---|
| `K2_PAPER/main.pdf` | the paper (20 pp, 3 figures, 7 tables) |
| `K2_PAPER/main.tex` + `fig_rd.pdf`, `fig_ladder.pdf`, `fig_qat.pdf` | LaTeX source, so the result is rebuildable |
| `k2_evidence.tar.gz` | per-item result JSONs (see below) — the traceability claim |

Build the evidence archive (strip paths, no credentials — checked; result files contain only
task ids, generated text, and ok-flags):

```bash
cd /mnt/Data/qat_k2 && tar czf /tmp/k2_evidence.tar.gz \
  tasks_f16_15b.json tasks_ptq_15b.json tasks_15b_general.json tasks_15b_cleared.json \
  tasks_qatv6_code.json tasks_gateup_code.json tasks_attr_down3_code.json \
  tasks_15b_ptq_k4min.json tasks_k5_code.json tasks_k6_code.json \
  tasks_hs1200_f16.json tasks_hs1200_qat334.json tasks_hs1200_k5.json \
  tasks_hs2000_f16.json tasks_qwen3b_f16_code.json tasks_qwen3b_k2_code.json \
  tasks_3b_f16_hs1200.json tasks_qwen15_gsq3_code.json \
  tasks_k6h6_k2e_code.json tasks_k6h6_k2e_hs1200.json \
  codec_reference.json e8_prototype.json symbol_entropy.json symbol_locality.json \
  damage_map.json component_ablation_15b.json comp15_hs.json weight_error.json \
  kl_scores_400.json trace_monitor.json act_cov.json act_spec.json act_oos.json \
  spectrum_exact.json the five Appendix-B files present in the 48-file archive pack_roundtrip.json \
  code_eval/ids_f16_15b.json \
  tasks_A1400_code.json tasks_E_code.json tasks_F_code.json tasks_E_f16emb_code.json \
  tasks_gsm_v2_f16.json tasks_gsm_v2_ptq.json tasks_gsm_v2_qat.json
```

## Checklist

- [ ] Zenodo account + ORCID linked (ORCID makes the author identity unambiguous)
- [ ] Abstract pasted exactly as above (Zenodo strips light HTML; the text is plain)
- [ ] Upload the 5 files, set the version, publish, then **record the DOI in `main.tex`** (a
      footnote "artifact: DOI …") before any arXiv submission — it makes the two mutually citable
- [ ] After publishing: check the OpenAlex entry appears (typically within a day) by searching
      the title; that is the indexing the user asked about
