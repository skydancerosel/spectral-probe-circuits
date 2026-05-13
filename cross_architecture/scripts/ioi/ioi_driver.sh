#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")"
RESULTS="$(cd ../../results && pwd)"
OUT="$RESULTS/ioi"
MI="$RESULTS/per_revision_mechinterp"
mkdir -p "$OUT"

if [ ! -f "$OUT/pythia1b_ioi.json" ]; then
  echo "=== Pythia 1B IOI ==="
  python3 -u ioi_eval.py --arch pythia --model EleutherAI/pythia-1b \
    --mechinterp-json "$MI/pythia1b_mechinterp_step143000.json" \
    --out "$OUT/pythia1b_ioi.json"
fi

if [ ! -f "$OUT/olmo_ioi.json" ]; then
  echo ""; echo "=== OLMo 1B IOI ==="
  python3 -u ioi_eval.py --arch olmo --model allenai/OLMo-1B-0724-hf \
    --mechinterp-json "$MI/olmo_mechinterp_step1454000-tokens3048B.json" \
    --out "$OUT/olmo_ioi.json"
fi

if [ ! -f "$OUT/olmoe_ioi.json" ]; then
  echo ""; echo "=== OLMoE 1B-7B IOI ==="
  python3 -u ioi_eval.py --arch olmoe --model allenai/OLMoE-1B-7B-0924 \
    --mechinterp-json "$MI/olmoe_mechinterp_step1220000-tokens5117B.json" \
    --out "$OUT/olmoe_ioi.json"
fi

echo ""; echo "=== IOI eval complete ==="
ls -la "$OUT/"
