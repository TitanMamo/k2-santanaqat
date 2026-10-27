# Allocation stage: SC -> RCO -> recipe -> encode

_Provenance: `sc_measure.py` is our implementation of the SC sensitivity idea (after exllamav3's
`sc_measure.py`); `sc_alloc.py` implements RCO-style rate-constrained assignment -- one option per
tensor, exact total budget, greedy with exchange repair (the method is IST-DASLab's, arXiv:2605.00649,
listed as the GSQ/RCO entry in the paper's bibliography). Neither file is copied from those projects._

The deployed artifact's per-tensor rate map is **not** a uniform rate step. It comes from the chain
below, and every intermediate file is included so the map can be re-derived rather than trusted.

1. **`sc_measure.py`** — per-tensor sensitivity. For every target Linear it injects weight noise at
   two relative Frobenius norms, runs the calibration rows, and takes the KL of the last-token logits
   against the clean run; the sensitivity is `S_t = geomean(kld / rfn^2)`. Output for the released
   artifact: `qwen15_3.34bpw/sc.json` (57.8 KB). Calibration text: wikitext-2 raw test
   (`ppl_wikitext2.txt`), 4 rows x 256 tokens, noise levels 0.30 and 0.15.
2. **`sc_alloc.py`** — exact-budget allocation over the trellis rates (greedy + exchange repair, with
   tie groups q/k/v and gate/up so tied tensors share a rate). Output:
   `qwen15_3.34bpw/grid.json` — target 3.34 bpw, achieved 3.3394, predicted KLD 0.1553, head 6 bits.
3. **recipe generation** — `qwen15_3.34bpw/recipe.yaml`: 196 body tensors, **71 at K=3, 90 at K=4,
   35 at K=5**, plus `head_bits: 6`.
4. **the encoder consumes it** — `convert.py ... -rcp recipe.yaml`. The per-tensor K map inside the
   released GGUF is exactly this file; re-deriving from a uniform assumption reproduces the *rate*
   but not the *file*.

`k2_pipeline.sh` is the end-to-end driver (measure -> allocate -> recipe -> encode -> assemble);
`sc_rco_pipeline.sh` is the earlier stage-order skeleton. Both tools have `--selftest`.

Sanity check of the delivered map (read out of the GGUF, i.e. what a user actually gets):

| K | share of trellis weights |
|---|---|
| 3 | 61.1% |
| 4 | 18.8% |
| 5 | 5.0% |
| 6 (output head) | 15.1% |

Excluding the embedding (f16) and the head, that is exactly **3.34 body bpw**.

## Recipe format (what the encoder consumes)

```yaml
model: <hf model path>
measurement: <path to sc.json>
target_bpw: 3.34          # allocator budget
achieved_bpw: 3.3394      # exact, from the allocation
predicted_kld: 0.155326   # model of kld(t,K) = S_t * rfn(K)^alpha
head_bits: 6              # head is exempt from the budget
tie: q_proj+k_proj+v_proj,gate_proj+up_proj   # tied tensors share one rate
tensors:                  # one K per tensor (196 entries here)
  model.layers.0.mlp.down_proj: 3
  ...
```

The encoder is invoked as `convert.py ... -rcp recipe.yaml`; without `-rcp` it falls back to the
built-in heuristic allocator, which reproduces the *rate* but not the per-tensor map (and therefore
not the file).

## Caveat on the driver

`k2_pipeline.sh` is the driver **as it was run**, with local absolute paths (`/home/titan/...`,
`/mnt/Data/...`) baked in. It is included as the exact reproduction record; the generic per-stage
commands are the ones above. Both Python tools carry `--selftest` (re-run at deposit time: OK).

## Streaming sensitivity measurement (2.2x, bit-identical)

`sc_measure_stream.py` is a rewritten `sc_measure.py` built around suffix replay. It produces the same
numbers (rho 1.000000, max deviation 0.0000 % on every tensor, identical rate allocation) and runs 2.1-2.2x
faster; see `SC_STREAM.md` for the implementation, the measurements and the limits.

```bash
python sc_measure_stream.py -m <model> -t <calibration.txt> -o sc.json \
    --rows 4 --length 256 --levels 0.30,0.15 --device cuda
python parity.py reference.json new.json      # verify the claim on your own model
```

`sc_measure.py` stays the portable reference and the fallback for architectures the streaming version
does not support (it raises `NotImplementedError` rather than silently measuring something else).
