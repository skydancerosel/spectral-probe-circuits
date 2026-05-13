#!/usr/bin/env bash
# Prev-token circuit ablation across the three 1B models.
# Uses the final-checkpoint mech-interp JSONs from the per-revision sweep.
set -euo pipefail
cd "$(dirname "$0")"
mkdir -p prev_token_circuit

# Pythia 1B
if [ ! -f prev_token_circuit/pythia1b_ablation.json ]; then
  echo "=== Pythia 1B prev-token ablation ==="
  python3 -u prev_token_circuit_ablation.py \
    --arch pythia --model EleutherAI/pythia-1b --revision main \
    --mechinterp-json per_revision_mechinterp/pythia1b_mechinterp_step143000.json \
    --out prev_token_circuit/pythia1b_ablation.json
fi

# OLMo 1B
if [ ! -f prev_token_circuit/olmo_ablation.json ]; then
  echo ""
  echo "=== OLMo 1B prev-token ablation ==="
  python3 -u prev_token_circuit_ablation.py \
    --arch olmo --model allenai/OLMo-1B-0724-hf --revision main \
    --mechinterp-json per_revision_mechinterp/olmo_mechinterp_step1454000-tokens3048B.json \
    --out prev_token_circuit/olmo_ablation.json
fi

# OLMoE 1B-7B
if [ ! -f prev_token_circuit/olmoe_ablation.json ]; then
  echo ""
  echo "=== OLMoE 1B-7B prev-token ablation ==="
  python3 -u prev_token_circuit_ablation.py \
    --arch olmoe --model allenai/OLMoE-1B-7B-0924 --revision main \
    --mechinterp-json per_revision_mechinterp/olmoe_mechinterp_step1220000-tokens5117B.json \
    --out prev_token_circuit/olmoe_ablation.json
fi

echo ""; echo "=== Prev-token circuit ablation complete ==="
ls -la prev_token_circuit/
