#!/usr/bin/env python
"""parity.py — compare sensitivity sets from sc_measure.py / sc_measure_stream.py.

Usage: parity.py <ref.json> <new.json> [<ref.json> <new.json> ...]
Prints, per pair: tensor count, Spearman rho, max/median relative S deviation, worst tensor,
and (if an allocation grid sits next to the json) whether the K assignment is identical.

The point of the allocation column: identical S is only interesting because it produces the
same recipe. If the K assignment matches, the pack is the same pack.
"""
import json
import math
import os
import sys


def load(path):
    d = json.load(open(path))
    return {k: v["S"] for k, v in d.get("results", {}).items()
            if isinstance(v, dict) and "S" in v and math.isfinite(v["S"])}


def ranks(xs):
    order = sorted(range(len(xs)), key=lambda i: xs[i])
    r = [0.0] * len(xs)
    i = 0
    while i < len(order):                      # average ties
        j = i
        while j + 1 < len(order) and xs[order[j + 1]] == xs[order[i]]:
            j += 1
        avg = (i + j) / 2.0
        for k in range(i, j + 1):
            r[order[k]] = avg
        i = j + 1
    return r


def spearman(a, b):
    ra, rb = ranks(a), ranks(b)
    n = len(a)
    ma, mb = sum(ra) / n, sum(rb) / n
    num = sum((x - ma) * (y - mb) for x, y in zip(ra, rb))
    den = math.sqrt(sum((x - ma) ** 2 for x in ra) * sum((y - mb) ** 2 for y in rb))
    return num / den if den else float("nan")


def alloc_for(path):
    """A grid written next to the measurement by sc_alloc.py, if any."""
    cand = path.replace(".json", ".alloc.json")
    if os.path.exists(cand):
        return json.load(open(cand)).get("tensors")
    return None


def compare(ref, new):
    R, N = load(ref), load(new)
    keys = sorted(set(R) & set(N))
    if not keys:
        print(f"  {os.path.basename(new):26s} no common tensors")
        return
    a, b = [R[k] for k in keys], [N[k] for k in keys]
    dev = sorted(((abs(x - y) / max(abs(x), 1e-12), k) for x, y, k in zip(a, b, keys)), reverse=True)
    med = dev[len(dev) // 2][0]
    print(f"  {os.path.basename(new):26s} n={len(keys):4d}  rho={spearman(a, b):.6f}  "
          f"max={dev[0][0] * 100:8.4f}%  median={med * 100:8.5f}%  worst={dev[0][1].split('.')[-2]}")
    ar, an = alloc_for(ref), alloc_for(new)
    if ar and an:
        same = sum(1 for k in ar if an.get(k) == ar[k])
        print(f"    allocation: {same}/{len(ar)} identical K" if same == len(ar)
              else f"    allocation: {same}/{len(ar)} identical K  <-- DIFFERS")


def main(argv):
    if len(argv) < 3 or len(argv) % 2 != 1:
        print(__doc__)
        return 2
    for i in range(1, len(argv), 2):
        compare(argv[i], argv[i + 1])
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv))
