#!/bin/bash
set -e
echo "=== Re-mech-interp Pythia 160M + 410M in FP32 (fix NaN baseline) at $(date) ==="

echo "=== Pythia 160M fp32 ==="
python3 -u pythia_mechinterp.py \
    --model EleutherAI/pythia-160m --revision step143000 \
    --features-json pythia_160m_features.json \
    --n-examples 2000 --batch-size 8 --top-k 30 \
    --dtype fp32 \
    --out pythia_160m_mechinterp_fp32.json

echo "=== Pythia 410M fp32 ==="
python3 -u pythia_mechinterp.py \
    --model EleutherAI/pythia-410m --revision step143000 \
    --features-json pythia_410m_features.json \
    --n-examples 2000 --batch-size 4 --top-k 80 \
    --dtype fp32 \
    --out pythia_410m_mechinterp_fp32.json

echo "=== DONE at $(date) ==="
