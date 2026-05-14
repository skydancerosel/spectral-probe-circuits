"""Figure 3: greater-than ablation effects across three 1B configurations.

Two panels: left = Δtop-1 (percentage points), right = Δlogit_diff.
For each of three models, four ablation conditions grouped:
  - top-5 GT screen (task-specific)
  - matched-random in same layers (null)
  - induction circuit (≥50× induction sel — different screen)
  - prev-token circuit (best-class prev-token, ≥100× sel — different screen)

The visual story: Pythia is GT-screen-dominated on top-1; OLMo is GT-screen-affected
only on logit_diff (margin-not-argmax); OLMoE is more sensitive to prev-token
circuit ablation than to its own GT-specific heads (compositional substrate).
"""
from __future__ import annotations
import json
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

HERE = Path(__file__).parent
WORKTREE = HERE.parent

MODELS = [
    ("Pythia 1B\n(Pile · dense)",       "pythia1b"),
    ("OLMo 1B\n(DCLM · dense)",         "olmo"),
    ("OLMoE 1B-7B\n(DCLM · MoE)",       "olmoe"),
]

CONDITIONS = [
    # (label, key_substring_in_condition_name, color)
    ("top-5 GT screen",        "ablate_top",                    "#8c2d04"),   # dark amber — task-specific
    ("matched-random",         "ablate_matched_random",         "#bbbbbb"),   # grey — null
    ("induction circuit",      "ablate_induction_circuit",      "#d62728"),   # red — different screen
    ("prev-token circuit",     "ablate_prev_token_circuit",     "#1f77b4"),   # blue — different screen
]


def load_gt_ablation(short):
    with open(WORKTREE / f"gt/{short}_gt_ablation.json") as f:
        return json.load(f)


def find_condition(conds, key_substr):
    for c in conds:
        if key_substr in c["condition"]:
            return c
    return None


def get_deltas():
    """Returns {model_short: {condition_label: (delta_top1_pp, delta_logit_diff)}}"""
    out = {}
    for label, short in MODELS:
        d = load_gt_ablation(short)
        base = d["conditions"][0]
        b1 = base["top1_above"]
        bl = base["logit_diff_above_below"]
        out[short] = {}
        for cond_label, key, _ in CONDITIONS:
            c = find_condition(d["conditions"], key)
            if c is None:
                out[short][cond_label] = (None, None)
                continue
            d_top1 = (c["top1_above"] - b1) * 100
            d_logit = c["logit_diff_above_below"] - bl
            out[short][cond_label] = (d_top1, d_logit)
    return out


# -------- Plot --------

deltas = get_deltas()

plt.rcParams.update({"font.size": 10.5})
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13.5, 4.8),
                                gridspec_kw={"width_ratios": [1.0, 1.0]})

N_MODELS = len(MODELS)
N_CONDS = len(CONDITIONS)
BAR_WIDTH = 0.18
group_centers = np.arange(N_MODELS)

# Left panel: Δtop-1
for ci, (cond_label, _, color) in enumerate(CONDITIONS):
    offset = (ci - (N_CONDS - 1) / 2) * BAR_WIDTH
    xs = group_centers + offset
    ys = [deltas[short][cond_label][0] for _, short in MODELS]
    bars = ax1.bar(xs, ys, BAR_WIDTH * 0.95, color=color, edgecolor="black",
                   linewidth=0.7, label=cond_label, zorder=3)
    # Value labels
    for x, y in zip(xs, ys):
        if y is None: continue
        # Place label above the bar if y >= 0, below if y < 0
        va = "bottom" if y >= -2 else "top"
        offset_y = 0.6 if y >= -2 else -0.8
        ax1.text(x, y + offset_y, f"{y:+.1f}", ha="center", va=va,
                 fontsize=7.5, color="0.15", zorder=4)

ax1.axhline(0, color="black", lw=0.6, zorder=2)
ax1.set_xticks(group_centers)
ax1.set_xticklabels([m[0] for m in MODELS], fontsize=9.5)
ax1.set_ylabel("Δ top-1  (percentage points)")
ax1.set_title("A   Δ top-1 (P[argmax > start])  on ablation", loc="left",
              fontsize=11.0, fontweight="bold", pad=8)
ax1.set_ylim(-78, 8)
ax1.grid(True, axis="y", alpha=0.3, ls="--", lw=0.6)

# Right panel: Δlogit_diff
for ci, (cond_label, _, color) in enumerate(CONDITIONS):
    offset = (ci - (N_CONDS - 1) / 2) * BAR_WIDTH
    xs = group_centers + offset
    ys = [deltas[short][cond_label][1] for _, short in MODELS]
    bars = ax2.bar(xs, ys, BAR_WIDTH * 0.95, color=color, edgecolor="black",
                   linewidth=0.7, label=cond_label, zorder=3)
    for x, y in zip(xs, ys):
        if y is None: continue
        va = "bottom" if y >= -0.15 else "top"
        offset_y = 0.05 if y >= -0.15 else -0.06
        ax2.text(x, y + offset_y, f"{y:+.2f}", ha="center", va=va,
                 fontsize=7.5, color="0.15", zorder=4)

ax2.axhline(0, color="black", lw=0.6, zorder=2)
ax2.set_xticks(group_centers)
ax2.set_xticklabels([m[0] for m in MODELS], fontsize=9.5)
ax2.set_ylabel("Δ logit_diff  (above − below)")
ax2.set_title("B   Δ logit_diff (mean[logit > start] − mean[logit ≤ start])  on ablation",
              loc="left", fontsize=11.0, fontweight="bold", pad=8)
ax2.set_ylim(-5.0, 0.5)
ax2.grid(True, axis="y", alpha=0.3, ls="--", lw=0.6)

# Single shared legend at the top
handles, labels = ax1.get_legend_handles_labels()
fig.legend(handles, labels, loc="upper center", ncol=4, bbox_to_anchor=(0.5, 1.04),
           fontsize=10, frameon=True, framealpha=0.94, edgecolor="0.6")

fig.suptitle("Greater-than: same task, three different ablation profiles across configurations",
             fontsize=12.5, fontweight="bold", y=1.10)
plt.tight_layout()

out_png = HERE / "gt_ablation_figure.png"
out_pdf = HERE / "gt_ablation_figure.pdf"
plt.savefig(out_png, dpi=220, bbox_inches="tight")
plt.savefig(out_pdf, bbox_inches="tight")
print(f"wrote {out_png}")
print(f"wrote {out_pdf}")

print("\nΔ values used:")
for label, short in MODELS:
    print(f"\n  {label.replace(chr(10), ' '):>30}:")
    for cond_label, _, _ in CONDITIONS:
        dt, dl = deltas[short][cond_label]
        print(f"    {cond_label:<24}  Δtop1={dt:+6.2f}pp   Δlogit_diff={dl:+6.3f}")
