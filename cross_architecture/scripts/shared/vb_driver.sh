#!/usr/bin/env bash
# Variable-binding full pipeline across the three 1B models.
set -euo pipefail
cd "$(dirname "$0")"

SHORTS=(pythia1b olmo olmoe)
ARCHES=(pythia olmo olmoe)
MODELS=("EleutherAI/pythia-1b" "allenai/OLMo-1B-0724-hf" "allenai/OLMoE-1B-7B-0924")

declare -a FINAL=(
  "per_revision_mechinterp/pythia1b_mechinterp_step143000.json"
  "per_revision_mechinterp/olmo_mechinterp_step1454000-tokens3048B.json"
  "per_revision_mechinterp/olmoe_mechinterp_step1220000-tokens5117B.json"
)

mkdir -p vb

for i in 0 1 2; do
  short=${SHORTS[$i]}
  arch=${ARCHES[$i]}
  model=${MODELS[$i]}
  final=${FINAL[$i]}
  out_mi="vb/${short}_vb_mechinterp.json"
  out_ab="vb/${short}_vb_ablation.json"
  if [ -f "$out_ab" ]; then echo "SKIP $out_ab"; continue; fi
  echo "=== $arch variable-binding full pipeline ==="
  python3 -u vb_full.py --arch "$arch" --model "$model" \
    --final-mechinterp-json "$final" --top-k 5 \
    --out-mechinterp "$out_mi" --out-ablation "$out_ab"
done
echo "DONE"
ls vb/
