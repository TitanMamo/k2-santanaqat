#!/usr/bin/env python
"""EXL3 quant -> plain GGUF with the folded effective weights (comparison baseline).

The EXL3/K2 runtime computes, for a linear with trellis weights `w_rot` (stored layout
[in, out]) and side scales su [in], sv [out]:

    y = sv * H( w_rot^T ( H(x * su) ) )          H = Sylvester-128 / sqrt(128)

This tool folds that whole expression into one matrix

    W_eff = diag(su) * H_in * w_rot * H_out * diag(sv)        (torch layout [in, out])

and writes W_eff.T as a plain F32 tensor into a copy of the stock F16 GGUF, keeping the
name, the metadata, the biases and every non-quantized tensor byte-for-byte. The result
is an unmodified llama.cpp model that computes exactly the EXL3-quantized function (up to
f32 storage), so its top-1 / KLD on a fixed corpus is directly comparable to K2's.

The token embedding is kept from the F16 base (vendors ship it unquantized); an EXL3
`-eb` embed trellis has no suh/svh and is skipped. A quantized head (trellis + suh/svh)
is folded like any other tensor.

Usage:
  python k2_exl3_dequant.py --exl3 DIR --f16 BASE.gguf --out OUT.gguf [--device cuda]
"""
import argparse
import glob
import math
import os
import sys

import numpy as np
import torch

sys.path.insert(0, "/home/titan/qat_tools")
sys.path.insert(0, "/home/titan/exllamav3")

from k2_gguf_rewrite import read_gguf, write_gguf, hf_to_gguf  # noqa: E402

TYPE_F32 = 0
TYPE_F16 = 1


def sylvester(n: int) -> torch.Tensor:
    h = torch.ones(1, 1)
    while h.shape[0] < n:
        h = torch.cat([torch.cat([h, h], 1), torch.cat([h, -h], 1)], 0)
    return h / math.sqrt(n)


def fold(w_rot: torch.Tensor, su: torch.Tensor, sv: torch.Tensor, h: torch.Tensor) -> torch.Tensor:
    """w_rot [in, out] f16 -> W_eff [in, out] f32 for y = sv*H(w_rot^T H(x*su))."""
    in_f, out_f = w_rot.shape
    bs = h.shape[0]
    assert in_f % bs == 0 and out_f % bs == 0, (w_rot.shape, bs)
    w = w_rot.float()
    # left: H applied to every 128-row block of the input dimension
    w = torch.einsum("ij,njk->nik", h, w.view(in_f // bs, bs, out_f)).reshape(in_f, out_f)
    # right: H applied to every 128-column block of the output dimension
    w = torch.einsum("nij,jk->nik", w.view(in_f, out_f // bs, bs), h).reshape(in_f, out_f)
    return w * su[:, None] * sv[None, :]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--exl3", required=True)
    ap.add_argument("--f16", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--dtype", choices=["f32", "f16"], default="f32",
                    help="storage of the folded weights (f32 = exact)")
    args = ap.parse_args()

    from exllamav3.ext import exllamav3_ext as ext
    from safetensors import safe_open

    dev = args.device
    h = sylvester(128).to(dev).float()
    g = read_gguf(args.f16)
    print("f16 gguf:", len(g["infos"]), "tensors, align", g["align"])

    shards = sorted(glob.glob(os.path.join(args.exl3, "*.safetensors")))
    if not shards:
        raise SystemExit("no safetensors in " + args.exl3)
    print("shards:", len(shards))

    ttype = TYPE_F32 if args.dtype == "f32" else TYPE_F16
    tsz = 4 if args.dtype == "f32" else 2

    new = {}
    folded = skipped = 0
    for sp in shards:
        with safe_open(sp, framework="pt") as sf:
            keys = set(sf.keys())
            for k in sorted(keys):
                if not k.endswith(".trellis"):
                    continue
                base = k[: -len(".trellis")]
                if base + ".suh" not in keys or base + ".svh" not in keys:
                    print("  skip (no suh/svh):", base)
                    skipped += 1
                    continue
                gname = hf_to_gguf(base + ".weight")
                if gname is None:
                    print("  skip (no gguf mapping):", base)
                    skipped += 1
                    continue
                tr = sf.get_tensor(k).to(dev)
                K = int(tr.shape[-1] // 16)
                in_f, out_f = int(tr.shape[0]) * 16, int(tr.shape[1]) * 16
                w = torch.empty((in_f, out_f), dtype=torch.float16, device=dev)
                ext.reconstruct(w, tr, K, False, True)
                torch.cuda.synchronize()
                su = sf.get_tensor(base + ".suh").to(dev).float()
                sv = sf.get_tensor(base + ".svh").to(dev).float()
                w_eff = fold(w, su, sv, h)
                if args.dtype == "f16":
                    blob = w_eff.half().contiguous().T.cpu().numpy().tobytes()
                else:
                    blob = w_eff.contiguous().T.cpu().numpy().tobytes()
                new[gname] = ([in_f, out_f], ttype, blob)
                folded += 1
                print(f"  {gname:28s} K={K} [{in_f}x{out_f}] "
                      f"rms={w_eff.pow(2).mean().sqrt().item():.4f}")
                del tr, w, su, sv, w_eff, blob
                torch.cuda.empty_cache()

    if not new:
        raise SystemExit("nothing folded")
    print(f"folded {folded} tensors ({args.dtype}), skipped {skipped}")

    extra = [
        ("k2.baseline.kind", 8, _s("exl3_folded")),
        ("k2.baseline.source", 8, _s(os.path.abspath(args.exl3))),
    ]
    write_gguf(args.out, g, new, extra)
    print("wrote", args.out, round(os.path.getsize(args.out) / 1e9, 2), "GB",
          f"({folded} x {tsz} bytes/elt + kept tensors)")


def _s(v: str) -> bytes:
    import struct
    b = v.encode()
    return struct.pack("<Q", len(b)) + b


if __name__ == "__main__":
    main()
