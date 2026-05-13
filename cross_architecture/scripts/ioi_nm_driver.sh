#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")"

for arch_model in "pythia EleutherAI/pythia-1b pythia1b" "olmo allenai/OLMo-1B-0724-hf olmo" "olmoe allenai/OLMoE-1B-7B-0924 olmoe"; do
  read -r arch model short <<< "$arch_model"
  out="ioi/${short}_ioi_nm_ablation.json"
  mi="ioi/${short}_ioi_mechinterp.json"
  if [ -f "$out" ]; then echo "SKIP $out (exists)"; continue; fi
  echo "=== $arch name-mover ablation ==="
  python3 -u ioi_name_mover_ablation.py --arch "$arch" --model "$model" \
    --ioi-mechinterp-json "$mi" --top-k 5 --out "$out"
done
echo ""; echo "=== done ==="; ls -la ioi/*nm_ablation*
