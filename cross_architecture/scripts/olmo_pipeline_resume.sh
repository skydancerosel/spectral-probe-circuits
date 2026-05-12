#!/bin/bash
set -e
echo "=== OLMo pipeline RESUME (Phase 3 onward) at $(date) ==="

echo "=== [5/7] Phase 3: ablation K=45 on synthetic ==="
python3 -u olmo_ablation.py \
    --features-json olmo_phase1_features.json \
    --mechinterp-json olmo_mechinterp.json \
    --n-examples 2000 --top-k 45 --induction-threshold 50.0 \
    --out olmo_ablation_k45.json

echo "=== [6/7] Phase 3: ablation K=6 on synthetic ==="
python3 -u olmo_ablation.py \
    --features-json olmo_phase1_features.json \
    --mechinterp-json olmo_mechinterp.json \
    --n-examples 2000 --top-k 6 --induction-threshold 50.0 \
    --out olmo_ablation_k6.json

echo "=== [7/7] Phase 3b: natural-text causal ablation ==="
python3 -u olmo_ablation_naturaltext.py \
    --mechinterp-json olmo_mechinterp.json \
    --induction-threshold 50.0 \
    --batch-file natural_induction_batch.pt \
    --batch-size 4 \
    --out olmo_ablation_naturaltext.json

echo "=== ALL DONE at $(date) ==="
