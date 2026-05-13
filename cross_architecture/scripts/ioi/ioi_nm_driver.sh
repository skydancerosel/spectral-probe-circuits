#!/usr/bin/env bash
# Name-mover ablation across the three 1B models. Reads IOI mech-interp output, picks
# top-5 name-mover candidates, ablates + matched-random + full-layer upper bound.
set -euo pipefail
cd "$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"  # cd to scripts/ root
OUT="$(cd ../results && pwd)/ioi"
mkdir -p "$OUT"

for arch_model in "pythia EleutherAI/pythia-1b pythia1b" "olmo allenai/OLMo-1B-0724-hf olmo" "olmoe allenai/OLMoE-1B-7B-0924 olmoe"; do
  read -r arch model short <<< "$arch_model"
  out="$OUT/${short}_ioi_nm_ablation.json"
  mi="$OUT/${short}_ioi_mechinterp.json"
  if [ -f "$out" ]; then echo "SKIP $out"; continue; fi
  echo "=== $arch name-mover ablation ==="
  python3 -u ioi_name_mover_ablation.py --arch "$arch" --model "$model" \
    --ioi-mechinterp-json "$mi" --top-k 5 --out "$out"
done
echo ""; echo "=== done ==="; ls -la "$OUT"/*nm_ablation*
