#!/bin/bash
set -e
echo "=== Pythia-1B full pipeline starting at $(date) ==="

echo "=== [1/7] Phase 1: per-head PR trajectory at 10 revisions ==="
python3 -u pythia_per_head.py \
    --model EleutherAI/pythia-1b \
    --n-examples 500 --batch-size 4 --dtype fp16 \
    --out pythia_phase1_trajectory.json

echo "=== [2/7] Computing trajectory features ==="
python3 -c "
import json, numpy as np
d = json.load(open('pythia_phase1_trajectory.json'))
L, H = d['n_layer'], d['num_heads']
# Pythia tokens_B may be small/zero for early revisions; use steps if all-zero
toks = np.array(d['ckpt_tokens_B'], dtype=float)
if toks.min() <= 0:
    print('  using step count as x-axis (some revisions have 0 tokens)')
    x = np.array(d['ckpt_step'], dtype=float)
else:
    x = toks
log_x = np.log(np.maximum(x, 1.0))
features = {'trajectory_steps': d['ckpt_step'], 'trajectory_tokens_B': d['ckpt_tokens_B'], 'features': {}}
for l in range(L):
    for h in range(H):
        pr = np.array(d['pr'][f'L{l}_H{h}'])
        features['features'][f'L{l}_H{h}'] = {
            'integral': float(np.trapz(np.maximum(pr - 1.0, 0), log_x)),
            'final_pr': float(pr[-1]),
            'max_pr': float(pr.max()),
            'min_pr': float(pr.min()),
            'spread': float(pr.max() - pr.min()),
        }
json.dump(features, open('pythia_phase1_features.json', 'w'), indent=2)
print('wrote pythia_phase1_features.json')
"

echo "=== [3/7] Phase 2: mech-interp on synthetic batch ==="
python3 -u pythia_mechinterp.py \
    --model EleutherAI/pythia-1b --revision step143000 \
    --features-json pythia_phase1_features.json \
    --n-examples 2000 --batch-size 4 --top-k 45 \
    --out pythia_mechinterp.json

echo "=== [4/7] Phase 2b: mech-interp on natural-text batch ==="
python3 -u pythia_mechinterp_naturaltext.py \
    --model EleutherAI/pythia-1b --revision step143000 \
    --features-json pythia_phase1_features.json \
    --batch-file natural_induction_batch.pt \
    --batch-size 4 --top-k 45 \
    --out pythia_mechinterp_naturaltext.json

echo "=== [5/7] Phase 3: ablation K=45 on synthetic ==="
python3 -u pythia_ablation.py \
    --model EleutherAI/pythia-1b --revision step143000 \
    --features-json pythia_phase1_features.json \
    --mechinterp-json pythia_mechinterp.json \
    --n-examples 2000 --top-k 45 --induction-threshold 50.0 \
    --out pythia_ablation_k45.json

echo "=== [6/7] Phase 3: ablation K=6 on synthetic ==="
python3 -u pythia_ablation.py \
    --model EleutherAI/pythia-1b --revision step143000 \
    --features-json pythia_phase1_features.json \
    --mechinterp-json pythia_mechinterp.json \
    --n-examples 2000 --top-k 6 --induction-threshold 50.0 \
    --out pythia_ablation_k6.json

echo "=== [7/7] Phase 3b: natural-text causal ablation ==="
python3 -u pythia_ablation_naturaltext.py \
    --model EleutherAI/pythia-1b --revision step143000 \
    --mechinterp-json pythia_mechinterp.json \
    --induction-threshold 50.0 \
    --batch-file natural_induction_batch.pt \
    --batch-size 4 \
    --out pythia_ablation_naturaltext.json

echo "=== ALL DONE at $(date) ==="
