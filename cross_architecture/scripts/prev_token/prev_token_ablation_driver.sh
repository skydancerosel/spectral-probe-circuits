#!/usr/bin/env bash
# Prev-token circuit ablation across the three 1B models.
# Uses the final-checkpoint mech-interp JSONs from the per-revision sweep.
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$SCRIPT_DIR"
RESULTS="$(cd "$SCRIPT_DIR/../results" && pwd)"
OUT="$RESULTS/prev_token_circuit"
MI="$RESULTS/per_revision_mechinterp"
mkdir -p "$OUT"

# Pythia 1B
if [ ! -f "$OUT/pythia1b_ablation.json" ]; then
  echo "=== Pythia 1B prev-token ablation ==="
  python3 -u prev_token_circuit_ablation.py \
    --arch pythia --model EleutherAI/pythia-1b --revision main \
    --mechinterp-json "$MI/pythia1b_mechinterp_step143000.json" \
    --out "$OUT/pythia1b_ablation.json"
fi

# OLMo 1B
if [ ! -f "$OUT/olmo_ablation.json" ]; then
  echo ""
  echo "=== OLMo 1B prev-token ablation ==="
  python3 -u prev_token_circuit_ablation.py \
    --arch olmo --model allenai/OLMo-1B-0724-hf --revision main \
    --mechinterp-json "$MI/olmo_mechinterp_step1454000-tokens3048B.json" \
    --out "$OUT/olmo_ablation.json"
fi

# OLMoE 1B-7B
if [ ! -f "$OUT/olmoe_ablation.json" ]; then
  echo ""
  echo "=== OLMoE 1B-7B prev-token ablation ==="
  python3 -u prev_token_circuit_ablation.py \
    --arch olmoe --model allenai/OLMoE-1B-7B-0924 --revision main \
    --mechinterp-json "$MI/olmoe_mechinterp_step1220000-tokens5117B.json" \
    --out "$OUT/olmoe_ablation.json"
fi

echo ""; echo "=== Prev-token circuit ablation complete ==="
ls -la "$OUT/"
