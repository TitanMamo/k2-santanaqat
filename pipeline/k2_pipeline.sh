#!/bin/bash
# K2 full pipeline (dense models): SC -> RCO alloc -> recipe -> EXL3 convert (wiki trace)
#   -> [embed dequant] -> scale-QAT (CE + 10*KL, head frozen) -> export -> K2 GGUF -> validate -> battery.
#
# Usage: ./k2_pipeline.sh <hf_dir> <name> [body_bpw=3.34] [embed_bits=0] [steps=300] [chunks=20]
#   embed_bits: 0 = keep the f16 embed; 4 = exllamav3 -eb 4 (row trellis codec) + dequant gate.
# Env: F16_BASE (f16 GGUF with the same tokenizer; built from the HF if unset),
#      GPU_BIN/CPU_BIN (default the k2 worktree builds), SKIP_SC=1 to reuse sc.json.
set -euo pipefail

HF=${1:?usage: k2_pipeline.sh <hf_dir> <name> [bpw] [embed_bits] [steps] [chunks]}
NAME=${2:?}
BPW=${3:-3.34}
EMB=${4:-0}
STEPS=${5:-300}
CHUNKS=${6:-20}

ROOT=/mnt/Data/qat_k2
P=$ROOT/pipeline_$NAME
mkdir -p "$P/logs"
WIKI=$ROOT/wikitext-2-raw/wiki.train.raw   # CALIBRATION corpus: must be DISJOINT from the eval
TRACE=$ROOT/wiki_train_trace.safetensors   # built from the calibration corpus, not the eval
GPU_BIN=${GPU_BIN:-/home/titan/ik_llama-k2/build-cuda/bin}
CPU_BIN=${CPU_BIN:-/home/titan/ik_llama-k2/build/bin}
F16_BASE=${F16_BASE:-}
export LD_LIBRARY_PATH=/home/titan/cuda-libs${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}
source /home/titan/miniconda3/etc/profile.d/conda.sh
conda activate qat-box

echo "=== K2 pipeline: $NAME (body $BPW bpw, embed_bits $EMB) -> $P"

if [ -z "$F16_BASE" ] || [ ! -f "$F16_BASE" ]; then
    F16_BASE=$P/base-f16.gguf
    if [ ! -f "$F16_BASE" ]; then
        echo "[0/8] building the f16 base GGUF from the HF model"
        python /home/titan/ik_llama-k2/convert_hf_to_gguf.py "$HF" --outtype f16 --outfile "$F16_BASE" \
            2>&1 | tee "$P/logs/base.log" | tail -3
    fi
fi

echo "[1/8] SC measurement (wiki)"
if [ "${SKIP_SC:-0}" != "1" ] || [ ! -f "$P/sc.json" ]; then
    python /home/titan/qat_tools/sc_measure.py -m "$HF" -t "$WIKI" -o "$P/sc.json" \
        --rows 4 --length 256 --levels 0.30,0.15 --device cuda 2>&1 | tee "$P/logs/sc.log" | tail -4
fi

echo "[2/8] RCO allocation (body $BPW, head 6)"
python - "$P" <<'PY'
import json, sys
p = sys.argv[1]
d = json.load(open(p + "/sc.json"))
d["results"].pop("lm_head", None)
json.dump(d, open(p + "/sc_body.json", "w"))
PY
MENU=$([ "$BPW" = "3.0" ] && echo "2,3,4,5" || echo "3,4,5")
python /home/titan/qat_tools/sc_alloc.py -m "$P/sc_body.json" -o "$P/grid.json" --bpw "$BPW" \
    --menu "$MENU" --ratio 1.96 --alpha 2.0 --anchor 3:0.155 \
    --tie "q_proj+k_proj+v_proj,gate_proj+up_proj" 2>&1 | tee "$P/logs/alloc.log" | tail -12

echo "[3/8] recipe"
python - "$P" "$HF" <<'PY'
import json, sys
p, hf = sys.argv[1], sys.argv[2]
g = json.load(open(p + "/grid.json"))
lines = [f"model: {hf}", f"measurement: {p}/sc.json",
         f"target_bpw: {g['target_bpw']}", f"achieved_bpw: {g['achieved_bpw']}",
         f"predicted_kld: {g['predicted_kld']}", "head_bits: 6",
         "tie: q_proj+k_proj+v_proj,gate_proj+up_proj", "tensors:"]
lines += [f"  {n}: {k}" for n, k in sorted(g["tensors"].items())]
open(p + "/recipe.yaml", "w").write("\n".join(lines) + "\n")
print(f"recipe: {len(g['tensors'])} tensors")
PY

echo "[4/8] EXL3 conversion (wiki trace)"
CONV=(python /home/titan/exllamav3/convert.py -i "$HF" -o "$P/exl3" -w "$P/work" \
      -rcp "$P/recipe.yaml" -cd "$TRACE" -hb 6 -d 0)
[ "$EMB" != "0" ] && CONV+=(-eb "$EMB")
"${CONV[@]}" 2>&1 | tee "$P/logs/convert.log" | tail -4

EMBARGS=()
if [ "$EMB" != "0" ]; then
    echo "[5/8] embed dequant (-eb $EMB quality gate)"
    python /home/titan/qat_tools/k2_embed_dequant.py --exl3 "$P/exl3" --out "$P/embed.safetensors" \
        2>&1 | tee "$P/logs/embed.log" | tail -2
    EMBARGS=(--embed-override "$P/embed.safetensors")
else
    echo "[5/8] embed: f16 (no override)"
fi

EXL3_QAT="$P/exl3"
if [ "$STEPS" -gt 0 ]; then
    echo "[6/8] scale-QAT (CE + 10*KL, head frozen, $STEPS steps)"
    python /home/titan/qat_tools/k2_qat_pilot.py --hf "$HF" --exl3 "$P/exl3" --steps "$STEPS" \
        --lr 2e-5 --head-lr 0 --loss kl+ce --kl-weight 10 --eval-chunks 4 --eval-every 100 \
        --out "$P/scales.pt" "${EMBARGS[@]}" 2>&1 | tee "$P/logs/qat.log" | tail -8
    python /home/titan/qat_tools/k2_apply_scales.py --exl3 "$P/exl3" --scales "$P/scales.pt" \
        --out "$P/exl3_qat" 2>&1 | tail -1
    EXL3_QAT="$P/exl3_qat"
else
    echo "[6/8] QAT skipped (STEPS=0): using the PTQ scales"
fi

echo "[7/8] transcode + validate"
# the quantized K2E embed (I16 trellis + signs + k2.embed.*); the runtime decodes it
EMBOPT=()
[ "$EMB" != "0" ] && EMBOPT=(--embed-k2e "$P/exl3")
python /home/titan/qat_tools/k2_gguf_rewrite.py --f16 "$F16_BASE" --exl3 "$EXL3_QAT" \
    --out "$P/k2.gguf" "${EMBOPT[@]}" 2>&1 | tail -2
python /home/titan/qat_tools/k2_validate.py "$P/k2.gguf" "$EXL3_QAT" 2>&1 | tail -1

echo "[8/8] battery (PPL $CHUNKS chunks, generation, domain)"
"$GPU_BIN/llama-perplexity" -m "$P/k2.gguf" -f "$WIKI" -c 512 --chunks "$CHUNKS" -ngl 99 -t 6 \
    2>&1 | grep "Final estimate" | tee "$P/logs/ppl.log"
"$GPU_BIN/llama-cli" -m "$P/k2.gguf" -ngl 99 -n 16 -p "The capital of France is" -t 6 \
    --no-warmup --temp 0 2>&1 | grep "The capital of France" | head -1 | tee "$P/logs/gen.log"
if [ -f "$ROOT/domain_test.txt" ]; then
    "$GPU_BIN/llama-perplexity" -m "$P/k2.gguf" -f "$ROOT/domain_test.txt" -c 512 --chunks 4 \
        -ngl 99 -t 6 2>&1 | grep "Final estimate" | tee "$P/logs/domain.log"
fi
echo "=== done: $P/k2.gguf"
