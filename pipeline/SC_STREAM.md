# SC streaming — 2.2× faster, bit-identical

The sensitivity-measurement stage (`sc_measure`) has been rewritten around suffix replay. It produces
**exactly the same numbers** as before — rho = 1.000000, max deviation **0.0000 %** on every tensor, identical
`predicted_kld` and therefore an identical rate allocation — while running **2.1–2.2× faster**.

This is a drop-in: same CLI arguments, same JSON schema (the allocator reads it unchanged), same seed, same
noise construction, same definition of the measured quantity.

## What the implementation does

Four changes, each of which preserves the mathematics:

1. **Suffix replay.** A perturbation of a tensor in layer *L* can only affect layers ≥ *L*. The clean pass
   captures the input hidden state to every target layer once; each perturbed variant then replays **only
   layers L…N, the final norm and the head**, instead of the whole model. This is the FLOP win.
2. **Head slicing.** The measured quantity is a KL over the **last 16 logit positions**, so the LM head is
   only evaluated there rather than for all 256 positions. The LM head is 9–15 % of parameters in these
   models, so this is not cosmetic.
3. **Noise caching.** The reference calls `torch.manual_seed(1234)` *inside* the level loop, so both levels
   use the identical random direction at different scales. The direction is therefore a pure function of
   the weight shape and is cached per shape instead of regenerated — mathematically the same draw.
4. **Asynchronous KL.** The KL stays on the host (where the reference computes it) but runs on a worker
   thread, overlapping the next forward. This is why the default is bit-identical *and* fast: forcing the
   same placement synchronously costs ~2×, and moving the KL to the GPU costs ~0.1 % accuracy.

## Measured wins

Measure phase only (model load excluded), GPU lock held, box otherwise idle:

| model | reference | **new default (bit-identical)** | × | `--kl-device gpu` | × |
|---|---|---|---|---|---|
| Qwen2.5-1.5B (197 tensors) | 98.84 s | **44.5 s** | **2.22×** | 36.7 s | 2.69× |
| Qwen2.5-3B (253 tensors) | 207.27 s | **98.4 s** | **2.11×** | 89.8 s | 2.31× |

End-to-end wall including load: 1.5B 105.7 s → ~50 s; 3B 276.1 s → ~106 s.

Contention check: the reference measured 97.03 s with a compile running versus 98.84 s idle (1.9 %), so a
shared machine does not invalidate the comparison.

## Parity evidence

| model | mode | rho | max dev S | tensors > 1 % |
|---|---|---|---|---|
| Qwen2.5-1.5B | **default** | **1.000000** | **0.0000 %** | 0 |
| Qwen2.5-1.5B | `--kl-device gpu` | 0.999994 | 0.1004 % | 0 |
| Qwen2.5-1.5B | `--row-batch all` | 0.999832 | 3.8335 % | 58 |
| Qwen2.5-3B | **default** | **1.000000** | **0.0000 %** | 0 |
| Qwen2.5-3B | `--kl-device gpu` | 0.999999 | 0.4785 % | 0 |
| Qwen2.5-0.5B (untied head) | **default** | **1.000000** | **0.0000 %** | 0 |

In the default mode the replayed logits match the reference at `max|dlogit| = 0.000e+00`. Allocation
(`--bpw 3.34 --menu 3,4,5`): **197/197** and **253/253** K assignments identical, `predicted_kld` identical to
the digit (0.205041 / 0.127149).

## Why there is no 3× mode at parity

The exact budget is bounded by replay alone: `N / (mean(N − L) + 1)` ≈ 1.93–1.95× FLOPs, plus head slicing.
The reference also spends ~25 % of its 1.5B time in host-side KL, which asynchronous placement recovers. The
remaining third is GPU efficiency at batch 1 — the replays are already compute-bound. Larger GEMMs require
row batching or per-sample weights, and both change cuBLAS kernel selection, which drifts 1 ulp in fp16 and
shows up as 0.5–3.8 % in S.

`--row-batch all` peaks at **3.10×** with 3.83 % max deviation (58 of 197 tensors above 1 %). It stays
opt-in. On the models measured here it happened to produce the same K map, but that is luck, not a guarantee.

## Usage

```bash
python sc_measure_stream.py -m <model> -t <calibration.txt> -o sc.json \
    --rows 4 --length 256 --levels 0.30,0.15 --device cuda
```

Flags: `--row-batch 1|all`, `--kl-device cpuasync|gpu|cpu`, `--max-tensors N`, `--suffix 16`, `--quiet`,
`--selftest`.

Recommended: **the defaults** (bit-identical, 2.1–2.2×). Use `--kl-device gpu` only when the SC stage is the
bottleneck (4B and above), where it buys another ~20 % for 0.1–0.5 % deviation and an unchanged allocation.
`--row-batch all` only for a one-off where the resulting allocation is checked.

## Limits

Verified on Qwen2/Qwen3-style layouts: `model.model.layers` as a ModuleList, a model-level rotary embedding,
`model.model.norm`, a Linear `lm_head`, and layers taking `position_embeddings`. Tied and untied heads are
both exact (the untied case exercises head-only replay). Weight sharing is followed via `data_ptr`.

`NotImplementedError` with an explicit message — never a silent fallback — for: no `model.model.layers`,
per-layer sliding-window masks, pre-4.45 layer signatures, a non-Linear head, or a Linear target outside
`model.layers` that is not the head. Only CUDA was parity-tested.

## Reproducing the parity claim

`parity.py` compares two sensitivity files and reports rho (tie-averaged ranks), max/median relative
deviation and whether the K assignment matches:

```bash
python parity.py reference.json new.json
```

The reference implementation (`sc_measure.py`) is unchanged and remains the portable path for any
architecture the streaming version does not support.
