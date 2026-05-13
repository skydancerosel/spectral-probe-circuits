#!/usr/bin/env bash
# IOI mech-interp + circuit ablation across the three 1B models.
set -euo pipefail
cd "$(dirname "$0")"
mkdir -p ioi

# Step A: IOI mech-interp (name-mover selectivity per head)
for arch_model in "pythia EleutherAI/pythia-1b" "olmo allenai/OLMo-1B-0724-hf" "olmoe allenai/OLMoE-1B-7B-0924"; do
  read -r arch model <<< "$arch_model"
  short=$(echo "$arch" | sed 's/pythia/pythia1b/')
  out="ioi/${short}_ioi_mechinterp.json"
  if [ -f "$out" ]; then
    echo "SKIP $out (exists)"
    continue
  fi
  echo "=== $arch IOI mech-interp ==="
  python3 -u ioi_mechinterp.py --arch "$arch" --model "$model" --out "$out"
done

echo ""; echo "=== IOI mech-interp complete ==="; ls -la ioi/
