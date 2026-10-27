#!/usr/bin/env python
"""Streaming SC sensitivity measurement - same numbers as sc_measure.py, much faster.

Why it is mathematically identical
----------------------------------
A perturbation of a weight in decoder layer L can only change the outputs of
layers >= L.  So instead of a full forward per (tensor, level, row), we do ONE
clean pass per row that captures the input hidden state of every layer and then
replay only layers L..N per variant.  The final norm and the LM head are
position-wise, so only the last `suffix` positions are pushed through them.

Everything else is reused verbatim from sc_measure.py: collect_targets,
build_rows, kl_last, _logits_suffix, ALPHA and the noise construction
(manual_seed(1234) -> randn(shape, float32, device) -> /norm * (rfn*||W||)).
The noise is cached per shape: the reference re-seeds before every draw, so the
draw is a pure function of the shape and only the (rfn*||W||) scaling is
per tensor - bit-identical cached copy.

Exact speed levers (row_batch=1 is the default; bit-exact):
  * suffix replay   - layers 0..L-1 are never recomputed (layer-average ~1.9x)
  * head slicing    - the LM head sees `suffix` positions, not `length`
  * KL overlap      - the reference's per-row CPU log_softmax is ~25% of its
                      1.5B runtime; the default keeps those exact numbers but
                      evaluates them on a worker thread, hidden under the next
                      GPU replay (see --kl-device below)
  * noise caching   - one seeded randn per distinct weight shape
  * no HF forward wrapper for replays (no embed/rotary/mask rebuild per call)

--kl-device picks where the KL divergence is evaluated.  The reference keeps
the last-16 logits on the CPU and evaluates there; 'cpuasync' (default) does the
same on a worker thread, so its cost hides under the next GPU replay and the
output is BIT-IDENTICAL to sc_measure.py (measured: max deviation 0.0000% on
197/253 tensors, two models).  'gpu' evaluates it on the GPU instead (fp32
reduction order over the 151k vocabulary differs by ~1e-7, which shows up as
0.10% / 0.48% max deviation in S on the two models - inside the 1% gate) and is
~20% faster.  'cpu' is the serial exact path, for debugging.

--row-batch all additionally stacks the rows into one batch.  It is FASTER but
NOT parity-safe: batching changes cuBLAS kernel selection, which flips single
fp16 ulps, and fp16 chaos amplifies that to ~2e-3 relative on the last logits
(measured: max 3.8% deviation in S vs the reference, rho 0.9998).  The default
is 1 because "the numbers must be preserved".

Usage:
  python sc_measure_stream.py -m <hf model> -t <calib.txt> -o out.json \
      --rows 4 --length 256 --levels 0.30,0.15 --device cuda [--max-tensors N]
  python sc_measure_stream.py --selftest

Unsupported architectures raise NotImplementedError - never a silent fallback.
"""
import argparse
import inspect
import json
import math
import sys
import time
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor

import torch
import torch.nn as nn
import torch.nn.functional as F

# Reuse the reference implementation's helpers - do not re-derive them.
sys.path.insert(0, __file__.rsplit("/", 1)[0])
from sc_measure import (  # noqa: E402
    ALPHA,
    _logits_suffix,
    build_rows,
    collect_targets,
    kl_last,
)


def kl_last_batch(clean, pert):
    """Batched form of sc_measure.kl_last for [B, T, V] float32 tensors.

    The reference computes (p * (log_softmax(clean) - lq)).sum(-1).mean() on a
    [1,T,V] tensor = mean over the T positions.  For B rows of equal T that is
    the mean over rows of the per-row value; we compute exactly that, so
    batching never re-weights the result.
    """
    lp = F.log_softmax(clean, dim=-1)
    lq = F.log_softmax(pert, dim=-1)
    per_pos = (lp.exp() * (lp - lq)).sum(-1)            # [B, T]
    return per_pos.mean(-1).mean()


class ReplayAdapter:
    """Suffix replay for Qwen2/Qwen3-style decoder-only models.

    Requires model.model.layers (ModuleList) whose forward takes
    (hidden_states, attention_mask, position_embeddings), a model-level
    rotary_emb, model.model.norm and a Linear lm_head.  Everything else is
    rejected: a fallback would change semantics.
    """

    def __init__(self, model, suffix=16):
        core = getattr(model, "model", None)
        if core is None or not isinstance(getattr(core, "layers", None), nn.ModuleList):
            raise NotImplementedError(
                f"unsupported architecture {type(model).__name__}: expected a decoder "
                "with model.model.layers as nn.ModuleList (Qwen2/Qwen3/Llama-style); "
                "refusing a semantics-changing fallback")
        for attr in ("norm", "rotary_emb", "embed_tokens"):
            if not hasattr(core, attr):
                raise NotImplementedError(
                    f"unsupported architecture {type(model).__name__}: model.model.{attr} missing")
        if not isinstance(getattr(model, "lm_head", None), nn.Linear):
            raise NotImplementedError(
                f"unsupported architecture {type(model).__name__}: model.lm_head is not nn.Linear")
        params = set(inspect.signature(core.layers[0].forward).parameters)
        if not {"hidden_states", "attention_mask", "position_embeddings"} <= params:
            raise NotImplementedError(
                "unsupported decoder layer signature "
                f"{sorted(params)}: need hidden_states/attention_mask/position_embeddings "
                "(transformers >= 4.45 style model-level rotary)")
        if getattr(core, "has_sliding_layers", False):
            raise NotImplementedError(
                "model has per-layer sliding-window masks; a single captured mask "
                "would be wrong - refusing")
        self.model = model
        self.core = core
        self.layers = list(core.layers)
        self.suffix = suffix

    # ---------------- clean pass ----------------
    def clean(self, x):
        """Full forward of one row, exactly as sc_measure does it, capturing the
        per-layer input state and the kwargs the model passes to its layers."""
        cap_in = [None] * len(self.layers)
        cap_kw = {}
        cap_norm = {}
        counters = {"i": 0}
        handles = []

        def pre(mod, args, kwargs):
            cap_in[counters["i"]] = args[0] if args else kwargs["hidden_states"]
            counters["i"] += 1
            if not cap_kw:
                cap_kw.update({k: kwargs.get(k) for k in
                               ("attention_mask", "position_embeddings", "position_ids")})

        def post(mod, args, out):
            cap_norm["h"] = out

        for layer in self.layers:
            handles.append(layer.register_forward_pre_hook(pre, with_kwargs=True))
        handles.append(self.core.norm.register_forward_hook(post))
        try:
            with torch.inference_mode():
                logits = self.model(input_ids=x).logits
                suffix = _logits_suffix(logits, self.suffix)
        finally:
            for h in handles:
                h.remove()
        assert counters["i"] == len(self.layers), "layer hook count mismatch"
        return {"in": cap_in, "kw": cap_kw, "ids": x,
                "norm_suffix": cap_norm["h"][:, -self.suffix:, :],
                "logits": suffix, "logits_cpu": suffix.cpu()}

    # ---------------- replay ----------------
    def replay(self, start, batch_in=None, from_embed_ids=None, kw=None, norm_suffix=None):
        """Run layers[start:] + final norm (suffix positions) + LM head."""
        am, pe, pi = kw.get("attention_mask"), kw.get("position_embeddings"), kw.get("position_ids")
        with torch.inference_mode():
            if from_embed_ids is not None:
                h = F.embedding(from_embed_ids, self.core.embed_tokens.weight)
                start = 0
            elif start == len(self.layers):
                return self.model.lm_head(norm_suffix).float()      # head only
            else:
                h = batch_in
            for layer in self.layers[start:]:
                h = layer(h, attention_mask=am, position_embeddings=pe,
                          position_ids=pi, past_key_values=None, use_cache=False)
            h = self.core.norm(h[:, -self.suffix:, :])
            return self.model.lm_head(h).float()


def _kl_cpu(clean_cpu, lg_gpu):
    """Worker-thread KL: pull the variant's logits to the CPU (the reference's
    placement, bit exact) and evaluate there."""
    return kl_last(clean_cpu, lg_gpu.cpu())


def _kl_cpu_batch(clean_cpu, lg_gpu):
    return kl_last_batch(clean_cpu, lg_gpu.cpu())


def unit_noise(shape, device, cache):
    """noise/||noise|| for the reference's seeded draw of this shape.

    sc_measure does: manual_seed(1234); randn(shape, float32, device);
    noise/noise.norm().clamp_min(1e-30) * (rfn*w64.norm()) - so the draw only
    depends on the shape, and the per-tensor part is a scalar multiply.
    Small shapes are cached (bit-identical: the reference re-seeds every draw);
    large ones are not, so the fp32 buffers of a 300M-element tied LM head never
    pile up (returned with cached=False - the caller may then mutate in place).
    """
    key = (tuple(shape), str(device))
    if key in cache:
        return cache[key], True
    torch.manual_seed(1234)
    n = torch.randn(shape, dtype=torch.float32, device=device)
    n.div_(n.norm().clamp_min(1e-30))       # same elementwise division as noise/norm
    keep = n.numel() <= 32 * 1024 * 1024
    if keep:
        cache[key] = n
    return n, keep


def set_perturbed_weight(w, w64, norm, rfn, noise_cache):
    """w <- float16(w64 + noise), with noise = q*(rfn*||w||) exactly as the
    reference builds it.  In-place accumulation only (IEEE addition is
    commutative), so no extra full-size temporaries exist."""
    q, cached = unit_noise(w64.shape, w64.device, noise_cache)
    if cached:
        t = q * (rfn * norm)
        t.add_(w64)                         # == w64 + noise
        with torch.no_grad():
            w.copy_(t.to(w.dtype))
        del t
    else:
        q.mul_(rfn * norm)                  # fresh tensor, safe to mutate
        q.add_(w64)
        with torch.no_grad():
            w.copy_(q.to(w.dtype))
        del q


def plan_targets(model):
    """Replay entry point for every Linear target.

    start = index of the first decoder layer whose output can change;
    len(layers) means 'head only'.  from_embed = the weight is the input
    embedding matrix (tied LM head), so the clean prefix is invalid and the
    replay has to start at the embedding lookup - exactly what the reference's
    full forward does implicitly.
    """
    adapter = ReplayAdapter(model)
    n = len(adapter.layers)
    embed_w = adapter.core.embed_tokens.weight
    embed_ptr, embed_shape = embed_w.data_ptr(), tuple(embed_w.shape)
    owner = defaultdict(list)
    for i, layer in enumerate(adapter.layers):
        for _sub, m in layer.named_modules():
            if isinstance(m, nn.Linear):
                owner[m.weight.data_ptr()].append(i)
    plan = []
    for name, mod in collect_targets(model):
        ptr = mod.weight.data_ptr()
        from_embed = ptr == embed_ptr and tuple(mod.weight.shape) == embed_shape
        if name == "lm_head":
            start = n
        else:
            start = next((i for i in range(n) if name.startswith(f"model.layers.{i}.")), None)
            if start is None:
                raise NotImplementedError(
                    f"target {name!r} is not inside model.layers and is not lm_head; "
                    "unknown replay entry point - refusing a fallback")
            if owner.get(ptr):
                start = min(start, min(owner[ptr]))
        if from_embed:
            start = n
        plan.append((name, mod, start, from_embed))
    return adapter, plan


def describe_plan(plan):
    n_emb = sum(1 for _n, _m, _s, e in plan if e)
    n_head = sum(1 for _n, _m, s, e in plan if s == len(plan) and not e)
    return (f"targets={len(plan)} replays: {n_emb} from-embedding (tied LM head), "
            f"{n_head} head-only, {len(plan) - n_emb - n_head} from a layer")


def measure_stream(model, rows, targets, levels, device="cpu", suffix=16,
                   row_batch=1, kl_device="cpuasync", verbose=True):
    """Same return value / schema as sc_measure.measure().

    row_batch=1   -> one replay per row (bit-faithful to the reference)
    row_batch=all -> all rows stacked in one replay (faster, ~1e-3 logit drift)
    kl_device     -> 'gpu'  : KL on the GPU (fast, fp32 reorder ~1e-7)
                     'cpu'  : exact reference placement, serial
                     'cpuasync': exact reference placement on a worker thread,
                                 overlapped with the next GPU replay
    """
    kl_on_gpu = (kl_device == "gpu")
    pool = ThreadPoolExecutor(max_workers=2) if kl_device == "cpuasync" else None
    model.eval()
    adapter, plan = plan_targets(model)
    entry = {n: (start, emb) for n, _m, start, emb in plan}
    n_layers = len(adapter.layers)

    rows = [x.to(device) for x in rows]          # the reference moves each row per call
    groups = defaultdict(list)
    for x in rows:
        groups[int(x.shape[1])].append(x)
    n_rows = len(rows)
    batch_all = (row_batch in (0, "all", "auto") or row_batch >= n_rows)

    t0 = time.perf_counter()
    clean = []
    for _length, xs in groups.items():
        st = [adapter.clean(x) for x in xs]
        g = {"B": len(xs), "rows": st, "kw": st[0]["kw"]}
        if batch_all:
            g["in"] = [torch.cat([s["in"][i] for s in st], 0) for i in range(n_layers)]
            g["norm_suffix"] = torch.cat([s["norm_suffix"] for s in st], 0)
            g["logits"] = torch.cat([s["logits"] for s in st], 0)
            g["logits_cpu"] = torch.cat([s["logits_cpu"] for s in st], 0)
            g["ids"] = torch.cat(xs, 0)
            for s in st:
                s["in"] = None
        clean.append(g)
    t_clean = time.perf_counter() - t0

    noise_cache = {}
    out = {}
    t_replay = t_book = 0.0
    for name, mod in targets:
        if not hasattr(mod, "weight"):
            continue
        if name not in entry:
            raise NotImplementedError(f"target {name!r} missing from the replay plan")
        start, from_embed = entry[name]
        tb = time.perf_counter()
        w = mod.weight.data
        w64 = w.detach().float()
        norm = w64.norm()
        rec = {"numel": w.numel(), "levels": []}
        t_replay_target = 0.0
        for rfn in levels:
            set_perturbed_weight(w, w64, norm, rfn, noise_cache)
            kl_rows = []
            try:
                for g in clean:
                    if batch_all:
                        tr = time.perf_counter()
                        if from_embed:
                            lg = adapter.replay(None, from_embed_ids=g["ids"], kw=g["kw"])
                        elif start == n_layers:
                            lg = adapter.replay(start, norm_suffix=g["norm_suffix"], kw=g["kw"])
                        else:
                            lg = adapter.replay(start, batch_in=g["in"][start], kw=g["kw"])
                        t_replay_target += time.perf_counter() - tr
                        if kl_on_gpu:
                            kl_rows.append(kl_last_batch(g["logits"], lg) * g["B"])
                        elif pool is not None:
                            kl_rows.append(pool.submit(_kl_cpu_batch, g["logits_cpu"], lg) * g["B"])
                        else:
                            kl_rows.append(kl_last_batch(g["logits_cpu"], lg.cpu()) * g["B"])
                    else:
                        for s in g["rows"]:
                            tr = time.perf_counter()
                            if from_embed:
                                lg = adapter.replay(None, from_embed_ids=s["ids"], kw=g["kw"])
                            elif start == n_layers:
                                lg = adapter.replay(start, norm_suffix=s["norm_suffix"], kw=g["kw"])
                            else:
                                lg = adapter.replay(start, batch_in=s["in"][start], kw=g["kw"])
                            t_replay_target += time.perf_counter() - tr
                            if kl_on_gpu:
                                kl_rows.append(kl_last(s["logits"], lg))
                            elif pool is not None:
                                kl_rows.append(pool.submit(_kl_cpu, s["logits_cpu"], lg))
                            else:
                                kl_rows.append(kl_last(s["logits_cpu"], lg.cpu()))
            finally:
                with torch.no_grad():
                    w.copy_(w64.to(w.dtype))     # restore the exact fp16 original
            vals = [r.result() if hasattr(r, "result") else r for r in kl_rows]
            rec["levels"].append({"rfn": rfn, "kld": (torch.stack(vals).sum() / n_rows).item()})
        logs = [l["kld"] / (l["rfn"] ** ALPHA) for l in rec["levels"] if l["kld"] > 0]
        rec["S"] = math.exp(sum(math.log(x) for x in logs) / len(logs)) if logs else 0.0
        out[name] = rec
        t_replay += t_replay_target
        t_book += (time.perf_counter() - tb) - t_replay_target
        if verbose:
            print(f"[sc] {name:60s} S={rec['S']:.3e} "
                  + " ".join(f"rfn={l['rfn']:.2f}->kld={l['kld']:.4f}" for l in rec["levels"]),
                  flush=True)
    if pool is not None:
        pool.shutdown(wait=True)
    if verbose:
        print(f"[sc-stream] clean {t_clean:.2f}s  replay {t_replay:.2f}s  "
              f"bookkeeping(noise/weight/kl-sync) {t_book:.2f}s", flush=True)
    return out


def selftest():
    """Architecture guard: a non-decoder module must be rejected, not silently
    handled by a slower/different path."""
    class NotADecoder(nn.Module):
        def __init__(self):
            super().__init__()
            self.fc = nn.Linear(8, 8)

        def forward(self, input_ids=None):
            return self.fc(torch.zeros(1, 4, 8))

    try:
        ReplayAdapter(NotADecoder())
    except NotImplementedError as e:
        print("SELFTEST OK (unsupported arch rejected):", str(e)[:70])
        return
    raise AssertionError("expected NotImplementedError")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("-m", "--model")
    ap.add_argument("-t", "--text")
    ap.add_argument("-o", "--out")
    ap.add_argument("--rows", type=int, default=4)
    ap.add_argument("--length", type=int, default=256)
    ap.add_argument("--levels", default="0.30,0.15")
    ap.add_argument("--device", default="cpu")
    ap.add_argument("--max-tensors", type=int, default=0)
    ap.add_argument("--suffix", type=int, default=16)
    ap.add_argument("--row-batch", default="1",
                    help="1 = one replay per row (exact, default); 'all' = stack the "
                         "rows (faster, ~1e-3 logit drift from batch-dependent fp16 "
                         "kernels - NOT parity-safe)")
    ap.add_argument("--kl-device", choices=["gpu", "cpu", "cpuasync"], default="cpuasync",
                    help="where the KL is evaluated: gpu (fast, ~1e-7 fp32 reorder), "
                         "cpu (reference placement, bit exact, serial) or cpuasync "
                         "(same numbers as cpu, overlapped with the next replay)")
    ap.add_argument("--quiet", action="store_true")
    ap.add_argument("--selftest", action="store_true")
    args = ap.parse_args()
    if args.selftest:
        selftest()
        return
    t_start = time.perf_counter()
    from transformers import AutoModelForCausalLM, AutoTokenizer
    tok = AutoTokenizer.from_pretrained(args.model, trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(args.model, torch_dtype=torch.float16,
                                                 trust_remote_code=True)
    model.to(args.device)
    text = open(args.text).read()
    rows = build_rows(tok, text, args.rows, args.length, args.device)
    targets = collect_targets(model)
    if args.max_tensors:
        targets = targets[:args.max_tensors]
    levels = [float(x) for x in args.levels.split(",")]
    t_loaded = time.perf_counter()

    rb = args.row_batch if args.row_batch in ("all", "auto") else int(args.row_batch)
    if not args.quiet:
        _ad, plan = plan_targets(model)
        print("[sc-stream] " + describe_plan(plan)
              + f"; row_batch={rb} kl={args.kl_device}", flush=True)
    res = measure_stream(model, rows, targets, levels, device=args.device,
                         suffix=args.suffix, row_batch=rb,
                         kl_device=args.kl_device, verbose=not args.quiet)
    t_done = time.perf_counter()
    with open(args.out, "w") as f:
        json.dump({"model": args.model, "levels": levels, "alpha": ALPHA,
                   "rows": args.rows, "length": args.length, "results": res}, f, indent=2)
    print(f"[sc-stream] load {t_loaded - t_start:.1f}s  measure {t_done - t_loaded:.1f}s  "
          f"total {t_done - t_start:.1f}s")
    print("saved", args.out)


if __name__ == "__main__":
    main()
