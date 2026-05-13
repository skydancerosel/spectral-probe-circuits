#!/usr/bin/env bash
# Tier A: natural-text mech-interp on Pythia 160M + 410M + layer-BOS analysis.
set -e
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$SCRIPT_DIR"
RESULTS="$(cd "$SCRIPT_DIR/../results" && pwd)"

echo "=== Tier A: natural-text mech-interp on Pythia 160M + 410M + layer BOS analysis ==="
echo "Started at $(date)"

echo ""
echo "=== Pythia 160M natural-text (fp32) ==="
python3 -u pythia/pythia_mechinterp_naturaltext.py \
    --model EleutherAI/pythia-160m --revision step143000 \
    --batch-file "$RESULTS/natural_induction_batch.pt" \
    --features-json "$RESULTS/pythia_160m_features.json" \
    --batch-size 8 --top-k 30 \
    --dtype fp32 \
    --out "$RESULTS/pythia_160m_mechinterp_naturaltext.json"

echo ""
echo "=== Pythia 410M natural-text (fp32) ==="
python3 -u pythia/pythia_mechinterp_naturaltext.py \
    --model EleutherAI/pythia-410m --revision step143000 \
    --batch-file "$RESULTS/natural_induction_batch.pt" \
    --features-json "$RESULTS/pythia_410m_features.json" \
    --batch-size 4 --top-k 80 \
    --dtype fp32 \
    --out "$RESULTS/pythia_410m_mechinterp_naturaltext.json"

echo ""
echo "=== Tier A #2: Layer-wise BOS distribution analysis ==="
python3 -u tier_a/tier_a_layer_bos_analysis.py

echo ""
echo "=== DONE at $(date) ==="
