#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")"

# Run S-Inhibition ablation across all three 1B models. Even though Pythia's IOI is
# upstream-driven, running the screen on all 3 lets us check whether S-Inhibition has
# differential causal role (cross-model).
for arch_model in "pythia EleutherAI/pythia-1b pythia1b" "olmo allenai/OLMo-1B-0724-hf olmo" "olmoe allenai/OLMoE-1B-7B-0924 olmoe"; do
  read -r arch model short <<< "$arch_model"
  out="ioi/${short}_ioi_si_ablation.json"
  mi="ioi/${short}_ioi_mechinterp.json"
  if [ -f "$out" ]; then echo "SKIP $out (exists)"; continue; fi
  echo "=== $arch S-Inhibition ablation ==="
  python3 -u ioi_s_inhibition_ablation.py --arch "$arch" --model "$model" \
    --ioi-mechinterp-json "$mi" --top-k 5 --out "$out"
done
echo ""; echo "=== done ==="; ls -la ioi/*si_ablation*
