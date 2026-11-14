"""modal_sc_7b.py - a NATIVE (f16) 7B SC sensitivity measurement on a cloud GPU.

Why: sc_body_7b.json on this box is synthetic -- the 1.5B's S values with 7B shapes
(identical S on 196/196 tensors), because the 15.2 GB f16 model does not fit the 3060.
The streamed SC needs the whole model on one device and has no layer offload, so the
only honest way to get real 7B sensitivities is a >=24 GB card.

Version pinning matters: the 1.5B reference sc.json was measured with
transformers 5.5.0 / torch 2.8.0 on the box, so the image pins the same pair --
S values are only comparable within one stack.

GPU: L4 has no capacity on this account (2026-11-14: "waiting to be scheduled on a
GPU_L4 worker"); A10G/A100/T4 provision immediately. A10G is enough here
(23.7 GB VRAM, 7B f16 = 15.2 GB). Override with K2_GPU.

    modal run qat_tools/modal_sc_7b.py
    modal run qat_tools/modal_sc_7b.py --rows 4 --length 256
"""
import os

import modal

app = modal.App("k2-sc-7b")
vol = modal.Volume.from_name("distill-vol", create_if_missing=False)

GPU = os.environ.get("K2_GPU", "A10G")

image = (
    modal.Image.debian_slim(python_version="3.11")
    .pip_install("torch==2.8.0", "transformers==5.5.0", "numpy", "safetensors", "accelerate")
    .add_local_file("/home/titan/qat_tools/sc_measure_stream.py", "/root/sc_measure_stream.py")
    # the streamed tool imports the reference implementation (collect_targets, build_rows,
    # kl_last, _logits_suffix, ALPHA, the seeded noise) - it must sit next to it.
    .add_local_file("/home/titan/qat_tools/sc_measure.py", "/root/sc_measure.py")
    .add_local_file("/mnt/Data/qat_k2/wikitext-2-raw/wiki.test.raw", "/root/wiki.test.raw")
)


@app.function(image=image, gpu=GPU, timeout=7200, volumes={"/vol": vol})
def measure(model: str = "Qwen/Qwen2.5-7B",
            rows: int = 4,
            length: int = 256,
            levels: str = "0.30,0.15",
            out: str = "/vol/k2/sc/sc_body_7b_native.json") -> str:
    import subprocess

    import torch
    from huggingface_hub import snapshot_download

    p = torch.cuda.get_device_properties(0)
    print(f"gpu {p.name} | {p.total_memory/1e9:.1f} GB", flush=True)
    os.environ["HF_HOME"] = "/vol/k2/hf"
    os.environ["HF_HUB_CACHE"] = "/vol/k2/hf/hub"   # hub 1.x ignores HF_HOME for cache root

    # Pull the 15 GB once and commit it, so a later failure does not pay for it again.
    t = snapshot_download(model, cache_dir="/vol/k2/hf/hub")
    vol.commit()
    print(f"model cached at {t}", flush=True)

    os.makedirs(os.path.dirname(out), exist_ok=True)
    env = dict(os.environ, HF_HOME="/vol/k2/hf", HF_HUB_CACHE="/vol/k2/hf/hub", TOKENIZERS_PARALLELISM="false")
    cmd = ["python", "/root/sc_measure_stream.py",
           "-m", model, "-t", "/root/wiki.test.raw", "-o", out,
           "--rows", str(rows), "--length", str(length), "--levels", levels,
           "--device", "cuda"]
    print(" ".join(cmd), flush=True)
    subprocess.run(cmd, check=True, env=env)
    vol.commit()
    print(f"saved {out}  {os.path.getsize(out)/1e6:.1f} MB", flush=True)
    return out


@app.local_entrypoint()
def main(rows: int = 4, length: int = 256):
    print(measure.remote(rows=rows, length=length))
