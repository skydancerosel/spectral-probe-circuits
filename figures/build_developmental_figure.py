"""Figure 2: three-panel per-head spectral trajectories across the three 1B-class
configurations. Each panel shows max(PR_t − 1, 0) over fraction of training for
three identified heads — induction (red), prev-token (blue), BOS (grey) — with
X markers at the revision where each head's capability selectivity crosses its
threshold.

Output: figures/developmental_three_panel.{png,pdf}
"""
from __future__ import annotations
import json, math
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

HERE = Path(__file__).parent
WORKTREE = HERE.parent


def load_per_revision(prefix, revs):
    out = {}
    for rev in revs:
        with open(WORKTREE / f"per_revision_mechinterp/{prefix}_mechinterp_{rev}.json") as f:
            out[rev] = json.load(f)["all_head_selectivity"]
    return out


def load_pr_traj(model_prefix):
    with open(WORKTREE / f"{model_prefix}_phase1_trajectory.json") as f:
        d = json.load(f)
    return d["pr"]


# Per-model configuration
PANELS = [
    {
        "name": "Pythia 1B  ·  Pile · dense",
        "pr_prefix": "pythia", "mi_prefix": "pythia1b",
        "revs": ['step1', 'step4', 'step16', 'step64', 'step256', 'step512',
                 'step3000', 'step10000', 'step38000', 'step143000'],
        "toks_B": [0.002, 0.008, 0.033, 0.134, 0.536, 1.073, 6.291, 20.971, 79.691, 299.892],
        "total_B": 299.892,
        "heads": [
            ("L4_H4",  "induction",      "induction  L4·H4",      50.0, "#d62728"),
            ("L3_H5",  "previous-token", "prev-token  L3·H5",    100.0, "#1f77b4"),
            ("L4_H1",  "first-token",    "BOS / sink  L4·H1",     30.0, "#7f7f7f"),
        ],
    },
    {
        "name": "OLMo 1B  ·  DCLM · dense",
        "pr_prefix": "olmo", "mi_prefix": "olmo",
        "revs": ['step1000-tokens2B','step2000-tokens4B','step5000-tokens10B','step11000-tokens23B',
                 'step25000-tokens52B','step56000-tokens117B','step126000-tokens264B',
                 'step285000-tokens597B','step644000-tokens1350B','step1454000-tokens3048B'],
        "toks_B": [2, 4, 10, 23, 52, 117, 264, 597, 1350, 3048],
        "total_B": 3048,
        "heads": [
            ("L2_H11",  "induction",      "induction  L2·H11",      50.0, "#d62728"),
            ("L11_H10", "previous-token", "prev-token  L11·H10",   100.0, "#1f77b4"),
            ("L6_H12",  "first-token",    "BOS / sink  L6·H12",     30.0, "#7f7f7f"),
        ],
    },
    {
        "name": "OLMoE 1B-7B  ·  DCLM · MoE",
        "pr_prefix": "olmoe", "mi_prefix": "olmoe",
        "revs": ['step5000-tokens20B','step10000-tokens41B','step25000-tokens104B','step50000-tokens209B',
                 'step100000-tokens419B','step200000-tokens838B','step400000-tokens1677B',
                 'step600000-tokens2516B','step800000-tokens3355B','step1220000-tokens5117B'],
        "toks_B": [20, 41, 104, 209, 419, 838, 1677, 2516, 3355, 5117],
        "total_B": 5117,
        "heads": [
            ("L7_H0",   "induction",      "induction  L7·H0",       50.0, "#d62728"),
            ("L8_H10",  "previous-token", "prev-token  L8·H10",    100.0, "#1f77b4"),
            ("L11_H11", "first-token",    "BOS / sink  L11·H11",    30.0, "#7f7f7f"),
        ],
    },
]


def formation_idx(sel_data, head, cap, thr, revs):
    for i, rev in enumerate(revs):
        v = sel_data[rev].get(head, {}).get(cap)
        if v is not None and not (isinstance(v, float) and math.isnan(v)) and v >= thr:
            return i
    return None


# -------- Plot --------

plt.rcParams.update({"font.size": 10.5, "axes.labelsize": 10.5, "axes.titlesize": 11.0})

fig, axes = plt.subplots(1, 3, figsize=(15.5, 4.6), sharey=True)

for ax, cfg in zip(axes, PANELS):
    PR = load_pr_traj(cfg["pr_prefix"])
    SEL = load_per_revision(cfg["mi_prefix"], cfg["revs"])
    toks_frac = np.array(cfg["toks_B"]) / cfg["total_B"]

    # Plot curves
    for head, cap, label, thr, color in cfg["heads"]:
        pr = np.array(PR[head])
        pr_sig = np.maximum(pr - 1.0, 1e-3)
        ax.plot(toks_frac, pr_sig, "-o", color=color, lw=2.0, ms=4.5,
                markeredgecolor=color, markerfacecolor=color, zorder=3)
        # formation marker
        fi = formation_idx(SEL, head, cap, thr, cfg["revs"])
        if fi is not None:
            ax.plot(toks_frac[fi], pr_sig[fi], marker="X", ms=16,
                    mfc=color, mec="black", mew=1.4, zorder=6)

    # Direct labels for each curve — placed at chosen positions per panel
    # Strategy: at the right edge of the plot near each curve's final y-value
    label_positions = {
        # (head, model_pr_prefix) -> (x_data, y_data, ha)
        # Pythia 1B
        ("L3_H5", "pythia"):  (0.005, 75, "left"),
        ("L4_H4", "pythia"):  (0.0025, 1.3, "left"),
        ("L4_H1", "pythia"):  (0.55, 8.5, "center"),
        # OLMo 1B
        ("L2_H11", "olmo"):   (0.001, 45, "left"),
        ("L11_H10","olmo"):   (0.6, 25, "center"),
        ("L6_H12", "olmo"):   (0.0008, 3.0, "left"),
        # OLMoE 1B-7B
        ("L7_H0",   "olmoe"): (0.0042, 55, "left"),
        ("L8_H10",  "olmoe"): (0.6, 38, "center"),
        ("L11_H11", "olmoe"): (0.0042, 12, "left"),
    }
    for head, cap, label, thr, color in cfg["heads"]:
        key = (head, cfg["pr_prefix"])
        if key in label_positions:
            x, y, ha = label_positions[key]
            ax.text(x, y, label, color=color, fontsize=10.0, fontweight="bold",
                    ha=ha, va="center", zorder=7,
                    bbox=dict(boxstyle="round,pad=0.16", fc="white", ec="none", alpha=0.85))

    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlim(5e-6, 1.4)
    ax.set_ylim(0.05, 200)
    ax.set_xlabel("fraction of training")
    ax.set_title(cfg["name"], loc="left", fontsize=10.5, fontweight="bold", pad=6)
    ax.grid(True, which="major", alpha=0.3, ls="--", lw=0.6)
    ax.grid(True, which="minor", alpha=0.12, ls=":", lw=0.4)

axes[0].set_ylabel("per-head spectral signal\nmax(PR − 1, 0)")

# Shared annotation for X markers — placed in the leftmost panel
axes[0].text(0.98, 0.04,
             "X  formation event:\n     selectivity ≥ threshold",
             transform=axes[0].transAxes, ha="right", va="bottom", fontsize=8.6,
             bbox=dict(boxstyle="round,pad=0.3", fc="white", ec="0.55", alpha=0.95))

fig.suptitle("PR rises at or before capability-selectivity formation, across three 1B configurations",
             fontsize=12, fontweight="bold", y=1.02)
plt.tight_layout()

out_png = HERE / "developmental_three_panel.png"
out_pdf = HERE / "developmental_three_panel.pdf"
plt.savefig(out_png, dpi=220, bbox_inches="tight")
plt.savefig(out_pdf, bbox_inches="tight")
print(f"wrote {out_png}")
print(f"wrote {out_pdf}")
