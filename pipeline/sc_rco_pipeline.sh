#!/bin/bash
# SC -> RCO -> (grid_map) -> CAT-Q pipeline skeleton (2026-10-20).
# Nothing here runs the GPU by itself except the CAT-Q/fit step; the model-level
# stages are meant for a quiet window. Paths are examples.
set -e
cd /home/titan/qat_tools

# 0. calibration text (wiki or an in-domain trace; MUST match the eval domain
#    you care about - the chat-calibrated SC recipe lost on wikitext, 2026-10-18)
CAL=${CAL:-/mnt/Data/ik_llama.cpp/quant_test/ppl_wikitext2.txt}

# 1. SC: per-tensor sensitivity (noise injection + KLD)
python sc_measure.py -m "${MODEL:-unsloth/Qwen2.5-0.5B}" -t "$CAL" \
    -o measure.json --rows "${ROWS:-8}" --length "${LEN:-512}" \
    --levels 0.30,0.15 --device "${DEV:-cpu}"

# 2. SC/RCO allocation: exact budget, tie groups for fused GEMMs
python sc_alloc.py -m measure.json -o grid_map.json \
    --bpw "${BPW:-3.0}" --menu 2,3,4,5 --tie "q_proj+k_proj+v_proj,gate_proj+up_proj"

# 3. K2 writer consumes grid_map.json (per-tensor K -> IQ{2,3,4,5}_T16)
#    (writer: qat_tools/k2_write.py - not built yet; S0/S4 in K2_GRID_DESIGN.md)

# 4. CAT-Q polish (PTQ LM/ST per tensor, weights frozen) or QAT LM/ST in the loop
#    library: catq.CATQLinear(linear, gs=64) + catq.fit_catq(...)
#    (per-tensor driver: to be wired with the trainer's GRID_MAP dispatch)

echo "pipeline stages done (see grid_map.json)"
