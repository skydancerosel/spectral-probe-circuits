#!/usr/bin/env bash
# Matched-random sweep across all 4 task screens × 3 models × 10 seeds.
# Uses plain bash arrays (works on macOS bash 3.2).
set -euo pipefail
cd "$(dirname "$0")"

SHORTS=(pythia1b olmo olmoe)
ARCHES=(pythia olmo olmoe)
MODELS=("EleutherAI/pythia-1b" "allenai/OLMo-1B-0724-hf" "allenai/OLMoE-1B-7B-0924")
TASKS=(ioi_nm ioi_si gt succ)

mkdir -p matched_random_sweep

for i in 0 1 2; do
  short=${SHORTS[$i]}
  arch=${ARCHES[$i]}
  model=${MODELS[$i]}
  for task in "${TASKS[@]}"; do
    case "$task" in
      ioi_nm) abl="ioi/${short}_ioi_nm_ablation.json" ;;
      ioi_si) abl="ioi/${short}_ioi_si_ablation.json" ;;
      gt)     abl="gt/${short}_gt_ablation.json" ;;
      succ)   abl="succ/${short}_succ_ablation.json" ;;
    esac
    out="matched_random_sweep/${short}_${task}_mr_sweep.json"
    if [ -f "$out" ]; then echo "SKIP $out"; continue; fi
    if [ ! -f "$abl" ]; then echo "MISSING $abl"; continue; fi
    echo "=== $short × $task ==="
    python3 -u matched_random_sweep.py --arch "$arch" --model "$model" \
      --task "$task" --ablation-json "$abl" --n-seeds 10 --out "$out"
  done
done
echo "DONE"
ls matched_random_sweep/
