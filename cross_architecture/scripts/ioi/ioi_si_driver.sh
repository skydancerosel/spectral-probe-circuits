#!/usr/bin/env bash
# S-Inhibition ablation across all three 1B models. Reads IOI mech-interp output, picks
# top-5 subject-attending heads (subj/io ≥ 2, subj_max ≥ 0.1), ablates + matched-random
# + name-mover-union condition.
set -euo pipefail
cd "$(dirname "$0")"
OUT="$(cd ../../results && pwd)/ioi"

for arch_model in "pythia EleutherAI/pythia-1b pythia1b" "olmo allenai/OLMo-1B-0724-hf olmo" "olmoe allenai/OLMoE-1B-7B-0924 olmoe"; do
  read -r arch model short <<< "$arch_model"
  out="$OUT/${short}_ioi_si_ablation.json"
  mi="$OUT/${short}_ioi_mechinterp.json"
  if [ -f "$out" ]; then echo "SKIP $out"; continue; fi
  echo "=== $arch S-Inhibition ablation ==="
  python3 -u ioi_s_inhibition_ablation.py --arch "$arch" --model "$model" \
    --ioi-mechinterp-json "$mi" --top-k 5 --out "$out"
done
echo ""; echo "=== done ==="; ls -la "$OUT"/*si_ablation*
