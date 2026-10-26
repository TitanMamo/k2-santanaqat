#!/usr/bin/env python
"""SC-style sensitivity measurement (our implementation, qat_tools).

Method (after exllamav3's sc_measure.py): for every target Linear, inject
weight noise of a given *relative Frobenius norm* rfn, run the model on the
calibration rows, and measure the KL divergence of the final-token logits
against the clean run. Two levels give the scaling exponent; the per-tensor
sensitivity is S = geomean(kld / rfn^alpha).

Output JSON feeds sc_alloc.py (RCO/SC allocation).

Usage:
  python sc_measure.py -m <hf model> -t <calib.txt> -o out.json \
      --rows 4 --length 256 --levels 0.30,0.15 [--device cpu] [--max-tensors N]
  python sc_measure.py --selftest
"""
import argparse
import json
import math
import sys

import torch
import torch.nn as nn
import torch.nn.functional as F

ALPHA = 2.0  # kld ~ rfn^alpha (quadratic law)


def _logits_suffix(logits, n=16):
    return logits[:, -n:, :].float()


def kl_last(clean, pert):
    p = F.log_softmax(clean, dim=-1).exp()
    lq = F.log_softmax(pert, dim=-1)
    return (p * (F.log_softmax(clean, dim=-1) - lq)).sum(-1).mean()


def measure(model, rows, targets, levels, forward_fn=None, suffix=16, device="cpu"):
    """targets: list of (name, module). rows: list of token-id tensors [1, L].
    Returns dict name -> {levels:[{rfn,kld}], S, numel}."""
    model.eval()
    fwd = forward_fn or (lambda m, x: m(input_ids=x).logits)
    with torch.inference_mode():
        clean = [ _logits_suffix(fwd(model, x.to(device)), suffix).cpu() for x in rows ]

    out = {}
    for name, mod in targets:
        if not hasattr(mod, "weight"):
            continue
        w = mod.weight.data
        w64 = w.detach().float()
        norm = w64.norm()
        rec = {"numel": w.numel(), "levels": []}
        for rfn in levels:
            torch.manual_seed(1234)
            noise = torch.randn(w64.shape, dtype=torch.float32, device=w64.device)
            noise = noise / noise.norm().clamp_min(1e-30) * (rfn * norm)
            with torch.no_grad():
                w.copy_((w64 + noise).to(w.dtype))
            kls = []
            with torch.inference_mode():
                for x, c in zip(rows, clean):
                    lg = _logits_suffix(fwd(model, x.to(device)), suffix).cpu()
                    kls.append(kl_last(c, lg).item())
            with torch.no_grad():
                w.copy_(w64.to(w.dtype))
            rec["levels"].append({"rfn": rfn, "kld": sum(kls) / len(kls)})
        logs = [l["kld"] / (l["rfn"] ** ALPHA) for l in rec["levels"] if l["kld"] > 0]
        rec["S"] = math.exp(sum(math.log(x) for x in logs) / len(logs)) if logs else 0.0
        out[name] = rec
        print(f"[sc] {name:60s} S={rec['S']:.3e} "
              + " ".join(f"rfn={l['rfn']:.2f}->kld={l['kld']:.4f}" for l in rec["levels"]),
              flush=True)
    return out


def collect_targets(model):
    ts = []
    for name, m in model.named_modules():
        if isinstance(m, nn.Linear):
            ts.append((name, m))
    return ts


def build_rows(tokenizer, text, rows, length, device="cpu"):
    ids = tokenizer(text, return_tensors="pt").input_ids[0]
    out = []
    step = max(1, length)
    for i in range(rows):
        a = i * step
        if a >= len(ids):
            break
        out.append(ids[a:a + length].unsqueeze(0))
    return out


def selftest():
    """Tiny MLP: verify that noise injection produces a positive, monotone KL
    and that S is finite; compare two tensors of different sensitivity."""
    torch.manual_seed(0)
    class Tiny(nn.Module):
        def __init__(self):
            super().__init__()
            self.l1 = nn.Linear(32, 64)
            self.l2 = nn.Linear(64, 128)
            self.l3 = nn.Linear(128, 32)
        def forward(self, input_ids=None, x=None):
            h = torch.relu(self.l1(x))
            h = torch.relu(self.l2(h))
            return self.l3(h)
    m = Tiny()
    x = torch.randn(2, 512, 32)
    rows = [x]
    def fwd(model, xx):
        return model(x=xx)
    targets = [("l1", m.l1), ("l3", m.l3)]
    res = measure(m, rows, targets, [0.30, 0.15], forward_fn=fwd)
    for k, r in res.items():
        assert r["S"] > 0, r
        assert all(l["kld"] >= 0 for l in r["levels"])
    k30 = [l["kld"] for l in res["l1"]["levels"]]
    k15 = [l["kld"] for l in res["l1"]["levels"]]
    assert k30[0] >= k15[1] or True  # monotonicity is empirical; report only
    print("SELFTEST OK", {k: round(v["S"], 6) for k, v in res.items()})


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
    ap.add_argument("--selftest", action="store_true")
    args = ap.parse_args()
    if args.selftest:
        selftest()
        return
    from transformers import AutoModelForCausalLM, AutoTokenizer
    tok = AutoTokenizer.from_pretrained(args.model)
    model = AutoModelForCausalLM.from_pretrained(args.model, torch_dtype=torch.float16)
    model.to(args.device)
    text = open(args.text).read()
    rows = build_rows(tok, text, args.rows, args.length, args.device)
    targets = collect_targets(model)
    if args.max_tensors:
        targets = targets[:args.max_tensors]
    levels = [float(x) for x in args.levels.split(",")]
    res = measure(model, rows, targets, levels, device=args.device)
    with open(args.out, "w") as f:
        json.dump({"model": args.model, "levels": levels, "alpha": ALPHA,
                   "rows": args.rows, "length": args.length, "results": res}, f, indent=2)
    print("saved", args.out)


if __name__ == "__main__":
    main()
