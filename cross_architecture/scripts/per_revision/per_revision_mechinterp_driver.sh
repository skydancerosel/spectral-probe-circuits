#!/usr/bin/env bash
# Per-revision mech-interp across the three 1B models.
# For each model, run mech-interp classification at all 10 trajectory revisions
# (same revisions as Phase 1 PR trajectory). Goal: track emergence of BOS-class
# heads and induction-selective heads over training, not just at the final ckpt.
#
# Output: ../../results/per_revision_mechinterp/{pythia1b,olmo,olmoe}_mechinterp_{rev}.json
#
# Runs sequentially (one GPU job at a time per project rule).
set -euo pipefail

# cd to the scripts/ root so subdir paths resolve uniformly
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$SCRIPT_DIR"
RESULTS="$(cd "$SCRIPT_DIR/../results" && pwd)"
OUT="$RESULTS/per_revision_mechinterp"
mkdir -p "$OUT"

PYTHIA_REVS=(step1 step4 step16 step64 step256 step512 step3000 step10000 step38000 step143000)
OLMO_REVS=(step1000-tokens2B step2000-tokens4B step5000-tokens10B step11000-tokens23B step25000-tokens52B \
           step56000-tokens117B step126000-tokens264B step285000-tokens597B step644000-tokens1350B step1454000-tokens3048B)
OLMOE_REVS=(step5000-tokens20B step10000-tokens41B step25000-tokens104B step50000-tokens209B \
            step100000-tokens419B step200000-tokens838B step400000-tokens1677B \
            step600000-tokens2516B step800000-tokens3355B step1220000-tokens5117B)

echo "=== Pythia 1B per-revision mech-interp (fp32; 10 revisions) ==="
for rev in "${PYTHIA_REVS[@]}"; do
    out="$OUT/pythia1b_mechinterp_${rev}.json"
    if [ -f "$out" ]; then echo "SKIP $rev"; continue; fi
    echo "--- Pythia 1B @ $rev ---"
    python3 -u pythia/pythia_mechinterp.py --model EleutherAI/pythia-1b --revision "$rev" \
        --dtype fp32 --out "$out" \
        --features-json "$RESULTS/pythia_1b_phase1_features.json"
done

echo ""
echo "=== OLMo 1B per-revision mech-interp (fp16; 10 revisions) ==="
for rev in "${OLMO_REVS[@]}"; do
    out="$OUT/olmo_mechinterp_${rev}.json"
    if [ -f "$out" ]; then echo "SKIP $rev"; continue; fi
    echo "--- OLMo 1B @ $rev ---"
    python3 -u olmo/olmo_mechinterp.py --model allenai/OLMo-1B-0724-hf --revision "$rev" \
        --out "$out" \
        --features-json "$RESULTS/olmo_phase1_features.json"
done

echo ""
echo "=== OLMoE 1B-7B per-revision mech-interp (fp16; 10 revisions) ==="
for rev in "${OLMOE_REVS[@]}"; do
    out="$OUT/olmoe_mechinterp_${rev}.json"
    if [ -f "$out" ]; then echo "SKIP $rev"; continue; fi
    echo "--- OLMoE 1B-7B @ $rev ---"
    python3 -u olmoe/olmoe_mechinterp.py --model allenai/OLMoE-1B-7B-0924 --revision "$rev" \
        --out "$out" \
        --features-json "$RESULTS/olmoe_phase1_features.json"
done

echo ""; echo "=== Per-revision mech-interp sweep complete ==="
ls -la "$OUT/"
