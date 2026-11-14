#!/usr/bin/env python
"""Per-tensor audit of a K2 (exl3) pack against its HF source weights.

Reconstructs each trellis exactly as the runtime does and folds the side scales into one
matrix, then compares with the HF weight:

    W_eff = diag(su) H_in w_rot H_out diag(sv)          (torch [in, out])
    rfn   = ||W_eff^T - W_hf||_F / ||W_hf||_F           (W_hf is [out, in])

For fused-QKV models the HF tensor is one matrix [Q;K;V] and the pack holds split
q/k/v tensors, so the HF side is sliced at [0:q, q:q+kv, q+kv:].

Usage: k2_tensor_audit.py --pack DIR --hf DIR --layer 10 [--tag NAME]
"""
import argparse, glob, json, math, os, sys
import torch
from safetensors import safe_open

sys.path.insert(0, "/home/titan/qat_tools")
sys.path.insert(0, "/home/titan/exllamav3")


def sylvester(n: int) -> torch.Tensor:
    h = torch.ones(1, 1)
    while h.shape[0] < n:
        h = torch.cat([torch.cat([h, h], 1), torch.cat([h, -h], 1)], 0)
    return h / math.sqrt(n)


def fold(w_rot, su, sv, h):
    in_f, out_f = w_rot.shape
    bs = h.shape[0]
    assert in_f % bs == 0 and out_f % bs == 0, (w_rot.shape, bs)
    w = w_rot.float()
    w = torch.einsum("ij,njk->nik", h, w.view(in_f // bs, bs, out_f)).reshape(in_f, out_f)
    w = torch.einsum("nij,jk->nik", w.view(in_f, out_f // bs, bs), h).reshape(in_f, out_f)
    return w * su[:, None] * sv[None, :]


def pack_tensors(pack):
    """map base name -> (trellis, suh, svh) shard handles for a directory of safetensors"""
    out = {}
    for sp in sorted(glob.glob(os.path.join(pack, "*.safetensors"))):
        f = safe_open(sp, framework="pt")
        for k in f.keys():
            if k.endswith(".trellis"):
                base = k[: -len(".trellis")]
                out[base] = sp
    return out


def hf_index(hf):
    idx = os.path.join(hf, "model.safetensors.index.json")
    if os.path.exists(idx):
        return json.load(open(idx))["weight_map"]
    sp = os.path.join(hf, "model.safetensors")
    with safe_open(sp, framework="pt") as f:
        return {k: "model.safetensors" for k in f.keys()}


def get_hf(hf, wmap, key):
    sp = os.path.join(hf, wmap[key])
    with safe_open(sp, framework="pt") as f:
        return f.get_tensor(key).float()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--pack", required=True)
    ap.add_argument("--hf", required=True)
    ap.add_argument("--layer", type=int, default=10)
    ap.add_argument("--tag", default="")
    ap.add_argument("--fused-qkv", action="store_true")
    args = ap.parse_args()
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    from exllamav3.ext import exllamav3_ext as ext
    H = sylvester(128).to(dev).float()
    L = f"model.layers.{args.layer}"
    bases = {n: f"{L}.{n}" for n in
             ("self_attn.q_proj", "self_attn.k_proj", "self_attn.v_proj", "self_attn.out_proj",
              "mlp.gate_proj", "mlp.up_proj", "mlp.down_proj")}
    have = pack_tensors(args.pack)
    wmap = hf_index(args.hf)
    fused = None
    if args.fused_qkv:
        fused = get_hf(args.hf, wmap, f"{L}.self_attn.q_k_v_proj.weight")   # [6144, 2560]
        qd = fused.shape[0] - 2 * 1024
    print(f"=== {args.tag or args.pack}  layer {args.layer} ===")
    print(f"{'tensor':22s} {'K':>2s} {'rfn':>8s} {'rmse':>9s} {'W_hf rows':>10s}")
    for nice, base in bases.items():
        if base not in have:
            print(f"  {nice:20s} (not in pack)"); continue
        sp = have[base]
        with safe_open(sp, framework="pt") as f:
            tr = f.get_tensor(base + ".trellis").to(dev)
            su = f.get_tensor(base + ".suh").to(dev).float()
            sv = f.get_tensor(base + ".svh").to(dev).float()
        K = int(tr.shape[-1] // 16)
        in_f, out_f = int(tr.shape[0]) * 16, int(tr.shape[1]) * 16
        w = torch.empty((in_f, out_f), dtype=torch.float16, device=dev)
        ext.reconstruct(w, tr, K, False, True)
        torch.cuda.synchronize()
        weff = fold(w, su, sv, H).T.contiguous()                     # [out, in]
        if args.fused_qkv and nice in ("self_attn.q_proj", "self_attn.k_proj", "self_attn.v_proj"):
            kv = 1024
            sl = {"self_attn.q_proj": slice(0, qd),
                  "self_attn.k_proj": slice(qd, qd + kv),
                  "self_attn.v_proj": slice(qd + kv, qd + 2 * kv)}[nice]
            whf = fused[sl].contiguous()
        else:
            gname = base + ".weight"          # spark2_5 keeps out_proj in HF as well
            whf = get_hf(args.hf, wmap, gname)[:out_f, :in_f].contiguous()
        whf = whf.to(dev).float()
        d = weff - whf
        rfn = (d.norm() / whf.norm()).item()
        rmse = (d.pow(2).mean().sqrt()).item()
        print(f"  {nice:20s} {K:2d} {rfn:8.4f} {rmse:9.5f} {tuple(whf.shape)!s:>10s}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
