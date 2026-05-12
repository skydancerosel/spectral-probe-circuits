#!/bin/bash
set -e
echo "=== Tier A: natural-text mech-interp on Pythia 160M + 410M + layer BOS analysis ==="
echo "Started at $(date)"

echo ""
echo "=== Pythia 160M natural-text (fp32) ==="
python3 -u pythia_mechinterp_naturaltext.py \
    --model EleutherAI/pythia-160m --revision step143000 \
    --batch-file natural_induction_batch.pt \
    --features-json pythia_160m_features.json \
    --batch-size 8 --top-k 30 \
    --dtype fp32 \
    --out pythia_160m_mechinterp_naturaltext.json

echo ""
echo "=== Pythia 410M natural-text (fp32) ==="
python3 -u pythia_mechinterp_naturaltext.py \
    --model EleutherAI/pythia-410m --revision step143000 \
    --batch-file natural_induction_batch.pt \
    --features-json pythia_410m_features.json \
    --batch-size 4 --top-k 80 \
    --dtype fp32 \
    --out pythia_410m_mechinterp_naturaltext.json

echo ""
echo "=== Tier A #2: Layer-wise BOS distribution analysis ==="
python3 -u tier_a_layer_bos_analysis.py

echo ""
echo "=== DONE at $(date) ==="
