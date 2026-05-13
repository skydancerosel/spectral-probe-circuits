#!/bin/bash
set -e  # exit on any error
echo "=== OLMo-1B-0724-hf full pipeline starting at $(date) ==="

# Phase 1: per-head PR trajectory
echo "=== [1/7] Phase 1: per-head PR trajectory at 10 revisions ==="
python3 -u olmo_per_head.py \
    --n-examples 500 --batch-size 4 --dtype fp16 \
    --out olmo_phase1_trajectory.json

# Compute features (integral, etc.) inline
echo "=== [2/7] Computing trajectory features ==="
python3 -c "
import json, numpy as np
d = json.load(open('olmo_phase1_trajectory.json'))
L, H = d['n_layer'], d['num_heads']
toks = np.array(d['ckpt_tokens_B'], dtype=float)
log_t = np.log(toks)
features = {'trajectory_steps': d['ckpt_step'], 'trajectory_tokens_B': d['ckpt_tokens_B'], 'features': {}}
for l in range(L):
    for h in range(H):
        pr = np.array(d['pr'][f'L{l}_H{h}'])
        features['features'][f'L{l}_H{h}'] = {
            'integral': float(np.trapz(np.maximum(pr - 1.0, 0), log_t)),
            'final_pr': float(pr[-1]),
            'max_pr': float(pr.max()),
            'min_pr': float(pr.min()),
            'spread': float(pr.max() - pr.min()),
        }
json.dump(features, open('olmo_phase1_features.json', 'w'), indent=2)
print('wrote olmo_phase1_features.json')
"

# Phase 2: mech-interp on synthetic
echo "=== [3/7] Phase 2: mech-interp on synthetic batch ==="
python3 -u olmo_mechinterp.py \
    --features-json olmo_phase1_features.json \
    --n-examples 2000 --batch-size 4 --top-k 45 \
    --out olmo_mechinterp.json

# Phase 2b: mech-interp on natural-text
echo "=== [4/7] Phase 2b: mech-interp on natural-text batch ==="
python3 -u olmo_mechinterp_naturaltext.py \
    --features-json olmo_phase1_features.json \
    --batch-file natural_induction_batch.pt \
    --batch-size 4 --top-k 45 \
    --out olmo_mechinterp_naturaltext.json

# Phase 3 K=45 ablation on synthetic
echo "=== [5/7] Phase 3: ablation K=45 on synthetic ==="
python3 -u olmo_ablation.py \
    --features-json olmo_phase1_features.json \
    --mechinterp-json olmo_mechinterp.json \
    --n-examples 2000 --top-k 45 --induction-threshold 50.0 \
    --out olmo_ablation_k45.json

# Phase 3 K=6 ablation on synthetic
echo "=== [6/7] Phase 3: ablation K=6 on synthetic ==="
python3 -u olmo_ablation.py \
    --features-json olmo_phase1_features.json \
    --mechinterp-json olmo_mechinterp.json \
    --n-examples 2000 --top-k 6 --induction-threshold 50.0 \
    --out olmo_ablation_k6.json

# Phase 3b: natural-text ablation on auto-discovered induction circuit
echo "=== [7/7] Phase 3b: natural-text causal ablation ==="
python3 -u olmo_ablation_naturaltext.py \
    --mechinterp-json olmo_mechinterp.json \
    --induction-threshold 50.0 \
    --batch-file natural_induction_batch.pt \
    --batch-size 4 \
    --out olmo_ablation_naturaltext.json

echo "=== ALL DONE at $(date) ==="
