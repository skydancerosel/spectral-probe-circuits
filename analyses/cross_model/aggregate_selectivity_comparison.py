"""
aggregate_selectivity_comparison.py

Test the user's question (item 1): is the class-mix shift with scale
about distribution (sum conserved) or dilution (sum decreases)?

For each model + class, compute:
  - max selectivity
  - mean selectivity over heads with sel >= 30x
  - sum of selectivities over heads with sel >= 30x
  - count of heads with sel >= 30x

If "distribution" hypothesis is correct: sum is roughly conserved across
scales (capability spread across more heads but total signal preserved).
If "dilution" hypothesis is correct: sum decreases with scale.
"""

import json
from pathlib import Path
import numpy as np

REPO = Path(__file__).resolve().parent

# Files
karpathy_mech = json.load(open(REPO / "results/induction_heads_mechinterp_124m.json"))
prev_mech = json.load(open(REPO / "results/prev_token_mechinterp_124m.json"))
pythia160_mech = json.load(open(REPO / "results/pythia_mechinterp.json"))
pythia410_mech = json.load(open(REPO / "results/pythia_410m_mechinterp.json"))


def extract_per_head_selectivity_124m():
    """Karpathy 124M: induction + prev-token from separate mechinterp files."""
    out = {}
    for k, v in karpathy_mech["per_head"].items():
        parts = k.replace(" ", "").replace("L", "").split("H")
        L, H = int(parts[0]), int(parts[1])
        out.setdefault((L, H), {})["induction"] = v["selectivity"]
    for k, v in prev_mech["selectivity_per_head"].items():
        parts = k.replace("L", "").split("H")
        L, H = int(parts[0]), int(parts[1])
        out.setdefault((L, H), {})["previous-token"] = v["selectivity"]
    return out


def extract_per_head_selectivity_pythia(mech_data, classes=("induction", "previous-token", "self", "first-token")):
    """Pythia: only top-30 picks have full selectivity in mech-interp output."""
    # We only have classifications for top-30 picks
    out = {}
    for c in mech_data["classifications"]:
        L, H = c["layer"], c["head"]
        out[(L, H)] = c.get("all_selectivities", {})
    return out


print("=" * 100)
print("Aggregate selectivity by class across models")
print("=" * 100)

models = [
    ("Karpathy 124M", extract_per_head_selectivity_124m(), {"induction", "previous-token"}),
    ("Pythia 160M (top-30 only)", extract_per_head_selectivity_pythia(pythia160_mech),
        {"induction", "previous-token", "self", "first-token"}),
    ("Pythia 410M (top-30 only)", extract_per_head_selectivity_pythia(pythia410_mech),
        {"induction", "previous-token", "self", "first-token"}),
]

for cls in ["induction", "previous-token", "self", "first-token"]:
    print(f"\n--- Class: {cls} ---")
    print(f"  {'model':<30} {'count >=30x':>12} {'max':>8} {'mean (>=30)':>12} {'sum (>=30)':>12}")
    for name, sel_data, available_classes in models:
        if cls not in available_classes:
            print(f"  {name:<30}  (not measured)")
            continue
        all_sels = []
        for (L, H), d in sel_data.items():
            v = d.get(cls)
            if v is not None:
                all_sels.append(v)
        all_sels = np.array(all_sels)
        ge30 = all_sels[all_sels >= 30]
        print(f"  {name:<30} {len(ge30):>12} {all_sels.max() if len(all_sels) else 0:>8.0f} "
              f"{ge30.mean() if len(ge30) else 0:>12.1f} {ge30.sum() if len(ge30) else 0:>12.0f}")

# Caveat
print("\nNOTE: Karpathy 124M's induction/prev-token data covers ALL 144 heads.")
print("      Pythia 160M and 410M only have top-30 picks classified — sums underestimate.")
print("      Comparison is approximate; relative trends are what matter.")
