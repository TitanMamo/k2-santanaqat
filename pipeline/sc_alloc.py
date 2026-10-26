#!/usr/bin/env python
"""SC + RCO allocation: per-tensor bitrate under an exact bpw budget.

Cost model (EXL3 SC):
    kld(t, K) = S_t * rfn_t(K)^alpha
    rfn_t(K)  = rfn_ref(t) * ratio^(K_ref(t) - K)      (amplitude-halving law)
S_t comes from sc_measure.py; rfn anchors may come from a probe JSON
(per-tensor) or use a global anchor. The allocator is the exact greedy for a
separable convex objective (marginal KLD reduction per bit, + exchange repair),
as in exllamav3's sc_optimize; tie groups keep fused GEMMs at one K.

Usage:
  python sc_alloc.py -m measure.json -o grid_map.json --bpw 3.0 \
      [--menu 2,3,4,5] [--ratio 1.96] [--alpha 2.0] [--anchor 2:0.292] \
      [--rfn-ref probe.json] [--tie "q_proj+k_proj+v_proj,gate_proj+up_proj"]
  python sc_alloc.py --selftest
"""
import argparse
import heapq
import json
import math
import sys
from collections import defaultdict


def rfn_at(anchors, name, k, ratio):
    ak, ar = anchors.get(name, anchors.get("__default__", (2, 0.292)))
    return ar * ratio ** (ak - k)


def kld_at(S, anchors, name, k, alpha, ratio, alpha_low=None):
    ak, _ = anchors.get(name, anchors.get("__default__", (2, 0.292)))
    base = S * rfn_at(anchors, name, max(k, ak), ratio) ** alpha
    if k < ak:
        base *= (rfn_at(anchors, name, k, ratio) / rfn_at(anchors, name, ak, ratio)) ** (alpha_low or alpha)
    return base


def tie_groups(names, tie_spec):
    """tie_spec like 'q_proj+k_proj+v_proj,gate_proj+up_proj'; group by leaf
    within the same parent module. Returns {name: group_key}."""
    sets = [set(p.split("+")) for p in tie_spec.split(",")] if tie_spec.strip() else []
    out = {}
    for n in names:
        parent, _, leaf = n.rpartition(".")
        key = n
        for gi, s in enumerate(sets):
            if leaf in s:
                key = (parent, gi)
                break
        out[n] = key
    return out


def allocate(S, numel, menu, budget_bpw, anchors, alpha=2.0, ratio=1.96,
             alpha_low=None, tie_spec=""):
    names = list(S.keys())
    tmap = tie_groups(names, tie_spec)
    groups = defaultdict(list)
    for n in names:
        groups[tmap[n]].append(n)
    gkeys = list(groups.keys())
    sum_numel = sum(numel.values())
    budget = int(round(budget_bpw * sum_numel))
    gnumel = [sum(numel[n] for n in groups[g]) for g in gkeys]
    kmin, kmax = min(menu), max(menu)
    k = {g: kmin for g in gkeys}
    spent = kmin * sum_numel
    assert spent <= budget, f"budget {budget_bpw} below menu min {kmin}"

    def gkld(g, kk):
        return sum(kld_at(S[n], anchors, n, kk, alpha, ratio, alpha_low) for n in groups[g])

    heap = []
    for i, g in enumerate(gkeys):
        if kmax > kmin:
            gain = gkld(g, kmin) - gkld(g, kmin + 1)
            heapq.heappush(heap, (-gain / gnumel[i], i, kmin + 1))
    while heap:
        neg, i, nk = heapq.heappop(heap)
        if spent + gnumel[i] > budget:
            continue
        k[gkeys[i]] = nk
        spent += gnumel[i]
        if nk < kmax:
            gain = gkld(gkeys[i], nk) - gkld(gkeys[i], nk + 1)
            heapq.heappush(heap, (-gain / gnumel[i], i, nk + 1))

    # exchange repair (same idea as EXL3: promote a stranded group by demoting
    # cheaper-loss groups)
    for _ in range(100):
        leftover = budget - spent
        cands = []
        for j, g in enumerate(gkeys):
            if k[g] > kmin:
                loss = gkld(g, k[g] - 1) - gkld(g, k[g])
                cands.append((loss / gnumel[j], loss, gnumel[j], j))
        cands.sort()
        best = None
        for i, g in enumerate(gkeys):
            if k[g] >= kmax:
                continue
            gain = gkld(g, k[g]) - gkld(g, k[g] + 1)
            need = gnumel[i] - leftover
            dem, loss_sum = [], 0.0
            for _, loss, nj, j in cands:
                if need <= 0:
                    break
                if j == i:
                    continue
                dem.append(j); loss_sum += loss; need -= nj
            if need > 0:
                continue
            delta = gain - loss_sum
            if delta > 0 and (best is None or delta > best[0]):
                best = (delta, i, dem)
        if best is None:
            break
        _, i, dem = best
        k[gkeys[i]] += 1; spent += gnumel[i]
        for j in dem:
            k[gkeys[j]] -= 1; spent -= gnumel[j]

    assign = {n: k[tmap[n]] for n in names}
    achieved = spent / sum_numel
    pred = sum(kld_at(S[n], anchors, n, assign[n], alpha, ratio, alpha_low) for n in names)
    hist = defaultdict(lambda: [0, 0])
    for n in names:
        hist[assign[n]][0] += 1
        hist[assign[n]][1] += numel[n]
    return dict(target_bpw=budget_bpw, achieved_bpw=round(achieved, 4),
                predicted_kld=round(pred, 6), alpha=alpha, ratio=ratio,
                tensors=assign,
                histogram={str(kk): v for kk, v in sorted(hist.items())})


def _load_measure(path):
    d = json.load(open(path))
    S, numel = {}, {}
    for name, r in d["results"].items():
        S[name] = r["S"]; numel[name] = r["numel"]
    return d, S, numel


def _load_anchors(path, names):
    a = {}
    if path:
        d = json.load(open(path))
        for r in d.get("results", []):
            a[r["key"]] = (r["K"], r["rfn"])
    for n in names:
        if n not in a:
            a["__default__"] = a.get("__default__", (2, 0.292))
    return a


def selftest():
    import random
    random.seed(0)
    names = [f"l{i}.w" for i in range(8)]
    S = {n: 10 ** random.uniform(-4, -2) for n in names}
    numel = {n: random.choice([896 * 896, 4864 * 896]) for n in names}
    anchors = {"__default__": (2, 0.292)}
    res = allocate(S, numel, [2, 3, 4, 5], 3.0, anchors, tie_spec="")
    assert abs(res["achieved_bpw"] - 3.0) < 0.1, res
    assert all(2 <= v <= 5 for v in res["tensors"].values())
    # monotone in budget
    lo = allocate(S, numel, [2, 3, 4, 5], 2.5, anchors)["predicted_kld"]
    hi = allocate(S, numel, [2, 3, 4, 5], 4.0, anchors)["predicted_kld"]
    assert hi <= lo, (lo, hi)
    # exact-budget tie groups
    res2 = allocate(S, numel, [2, 3, 4, 5], 3.0, anchors,
                    tie_spec="w")
    assert len(res2["tensors"]) == len(names)
    print("SELFTEST OK", res["histogram"], "kld", res["predicted_kld"])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("-m", "--measure")
    ap.add_argument("-o", "--out", required=False)
    ap.add_argument("--bpw", type=float, required=False)
    ap.add_argument("--menu", default="2,3,4,5")
    ap.add_argument("--ratio", type=float, default=1.96)
    ap.add_argument("--alpha", type=float, default=2.0)
    ap.add_argument("--alpha-low", type=float, default=None)
    ap.add_argument("--anchor", default="2:0.292")
    ap.add_argument("--rfn-ref", default=None)
    ap.add_argument("--tie", default="q_proj+k_proj+v_proj,gate_proj+up_proj")
    ap.add_argument("--selftest", action="store_true")
    args = ap.parse_args()
    if args.selftest:
        selftest(); return
    d, S, numel = _load_measure(args.measure)
    anchors = _load_anchors(args.rfn_ref, list(S))
    if not args.rfn_ref:
        ak, ar = args.anchor.split(":")
        anchors["__default__"] = (int(ak), float(ar))
    menu = [int(x) for x in args.menu.split(",")]
    res = allocate(S, numel, menu, args.bpw, anchors, alpha=args.alpha,
                   ratio=args.ratio, alpha_low=args.alpha_low, tie_spec=args.tie)
    res["model"] = d.get("model")
    with open(args.out, "w") as f:
        json.dump(res, f, indent=2)
    print(json.dumps({k: res[k] for k in ("target_bpw", "achieved_bpw",
                                          "predicted_kld", "histogram")}, indent=2))
    print("saved", args.out)


if __name__ == "__main__":
    main()
