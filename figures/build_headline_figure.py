"""Build the two-panel headline figure for the methodology paper.

Panel A: Pythia 1B per-revision PR trajectories for three named heads
         (induction L4_H4, prev-token L3_H5, BOS L4_H1) with formation markers
         at the revision where each capability's selectivity crosses threshold.

Panel B: Cross-configuration formation timing — three 1B-class configurations
         (Pythia Pile dense / OLMo DCLM dense / OLMoE DCLM MoE) with the
         induction, prev-token, and BOS-attractor formation fractions per
         configuration. Configuration on x; training fraction (log) on y.

Output: figures/headline_two_panel.{png,pdf}
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


def load_per_revision(prefix: str, revisions: list[str]) -> dict:
    out = {}
    for rev in revisions:
        p = WORKTREE / f"per_revision_mechinterp/{prefix}_mechinterp_{rev}.json"
        with open(p) as f:
            out[rev] = json.load(f)["all_head_selectivity"]
    return out


def load_pr_trajectory(model_prefix: str):
    with open(WORKTREE / f"{model_prefix}_phase1_trajectory.json") as f:
        d = json.load(f)
    return d["pr"], d["revisions"], d["ckpt_tokens_B"]


# -------- Panel A setup --------

PYTHIA_REVS = ['step1', 'step4', 'step16', 'step64', 'step256', 'step512',
               'step3000', 'step10000', 'step38000', 'step143000']
PYTHIA_TOKS = [0.002, 0.008, 0.033, 0.134, 0.536, 1.073, 6.291, 20.971, 79.691, 299.892]
PYTHIA_TOTAL_B = 299.892

PR, _, _ = load_pr_trajectory("pythia")
SEL = load_per_revision("pythia1b", PYTHIA_REVS)

# (head, capability, label, threshold, color, label_offset_y_in_data_units)
PANEL_A_HEADS = [
    ("L3_H5", "previous-token", "prev-token  L3·H5",  100.0, "#1f77b4", 1.6),
    ("L4_H4", "induction",      "induction  L4·H4",    50.0, "#d62728", 1.5),
    ("L4_H1", "first-token",    "BOS / sink  L4·H1",   30.0, "#7f7f7f", 1.55),
]


def formation_token_idx(head, capability, threshold):
    for i, rev in enumerate(PYTHIA_REVS):
        v = SEL[rev].get(head, {}).get(capability)
        if v is not None and not (isinstance(v, float) and math.isnan(v)) and v >= threshold:
            return PYTHIA_TOKS[i], i
    return None, None


# -------- Panel B setup --------

CONFIGS = [
    ("Pythia 1B\n(Pile · dense)", "pythia1b", 299.892, PYTHIA_REVS, PYTHIA_TOKS),
    ("OLMo 1B\n(DCLM · dense)", "olmo", 3048.0,
     ['step1000-tokens2B', 'step2000-tokens4B', 'step5000-tokens10B', 'step11000-tokens23B', 'step25000-tokens52B',
      'step56000-tokens117B', 'step126000-tokens264B', 'step285000-tokens597B', 'step644000-tokens1350B', 'step1454000-tokens3048B'],
     [2, 4, 10, 23, 52, 117, 264, 597, 1350, 3048]),
    ("OLMoE 1B-7B\n(DCLM · MoE)", "olmoe", 5117.0,
     ['step5000-tokens20B', 'step10000-tokens41B', 'step25000-tokens104B', 'step50000-tokens209B', 'step100000-tokens419B',
      'step200000-tokens838B', 'step400000-tokens1677B', 'step600000-tokens2516B', 'step800000-tokens3355B', 'step1220000-tokens5117B'],
     [20, 41, 104, 209, 419, 838, 1677, 2516, 3355, 5117]),
]


def count_heads(per_rev, capability, threshold):
    return sum(1 for v in per_rev.values()
               if (s := v.get(capability)) is not None and not (isinstance(s, float) and math.isnan(s)) and s >= threshold)


def panel_b_form_tokens(prefix, revs, toks, capability, head_thr, count_thr, total_B=None,
                         frac_threshold=None):
    """Return tokens at first revision where #heads-with-capability >= count_thr
    (or BOS fraction >= frac_threshold if specified)."""
    per_rev_all = load_per_revision(prefix, revs)
    for rev, t in zip(revs, toks):
        if frac_threshold is not None:
            sel = per_rev_all[rev]
            n = count_heads(sel, capability, head_thr)
            frac = 100.0 * n / len(sel)
            if frac >= frac_threshold:
                return t
        else:
            n = count_heads(per_rev_all[rev], capability, head_thr)
            if n >= count_thr:
                return t
    return None


CIRCUITS_B = [
    ("induction circuit\n(≥3 heads, ≥50× sel)",       "induction",       50,  3,  None,  "*",   "#d62728"),
    ("prev-token circuit\n(≥10 heads, ≥100× sel)",    "previous-token", 100, 10,  None,  "^",   "#1f77b4"),
    ("BOS attractor\n(≥10% of heads, ≥30× sel)",      "first-token",     30,  0,  10.0,  "v",   "#7f7f7f"),
]


# -------- Plot --------

plt.rcParams.update({"font.size": 10.5, "axes.labelsize": 11, "axes.titlesize": 12})
fig, (axA, axB) = plt.subplots(1, 2, figsize=(12.5, 5.0),
                                gridspec_kw={"width_ratios": [1.15, 1.0]})

# === Panel A ===
toks_arr = np.array(PYTHIA_TOKS) / PYTHIA_TOTAL_B  # x = fraction

# Plot curves first
formation_pts = []
for head, cap, label, thr, color, _ in PANEL_A_HEADS:
    pr = np.array(PR[head])
    pr_sig = np.maximum(pr - 1.0, 1e-3)
    axA.plot(toks_arr, pr_sig, "-o", color=color, lw=2.0, ms=4.5, zorder=3,
             markeredgecolor=color, markerfacecolor=color)
    ft, fi = formation_token_idx(head, cap, thr)
    if ft is not None:
        formation_pts.append((head, label, color, ft / PYTHIA_TOTAL_B, pr_sig[fi]))

# Then formation markers (so they sit on top)
for head, label, color, fx, fy in formation_pts:
    axA.plot(fx, fy, marker="X", ms=18, mfc=color, mec="black", mew=1.6, zorder=6)

# Direct labels — placed near the curves with small offsets
# Computed positions to avoid overlap:
label_positions = {
    "L3_H5": (0.005, 70, "left"),      # prev-token: high curve, label on upper left
    "L4_H4": (0.0025, 1.3, "left"),     # induction: middle curve, label below the pre-formation dip
    "L4_H1": (0.55, 8.5,  "center"),    # BOS: low-mid curve, label on top right
}
for head, cap, label, thr, color, _ in PANEL_A_HEADS:
    x, y, ha = label_positions[head]
    axA.text(x, y, label, color=color, fontsize=10.5, fontweight="bold",
             ha=ha, va="center", zorder=7,
             bbox=dict(boxstyle="round,pad=0.18", fc="white", ec="none", alpha=0.85))

axA.set_xscale("log")
axA.set_yscale("log")
axA.set_xlabel("fraction of training (Pythia 1B; total 300B tokens)")
axA.set_ylabel("per-head spectral signal  max(PR − 1, 0)")
axA.set_title("A   Spectral signal rises before capability selectivity crosses threshold",
              loc="left", fontsize=11.0, pad=8, fontweight="bold")
axA.set_xlim(5e-6, 1.4)
axA.set_ylim(0.05, 200)
axA.grid(True, which="major", alpha=0.3, ls="--", lw=0.6)
axA.grid(True, which="minor", alpha=0.12, ls=":", lw=0.4)

# Annotation box for the X-marker meaning
axA.text(0.98, 0.04,
         "X  formation event:\n     capability selectivity\n     crosses threshold",
         transform=axA.transAxes, ha="right", va="bottom", fontsize=8.8,
         bbox=dict(boxstyle="round,pad=0.32", fc="white", ec="0.55", alpha=0.95))

# === Panel B ===
b_data = {}
for label, prefix, total_B, revs, toks in CONFIGS:
    b_data[label] = {}
    for circ_label, cap, head_thr, count_thr, frac_thr, _, _ in CIRCUITS_B:
        t = panel_b_form_tokens(prefix, revs, toks, cap, head_thr, count_thr,
                                 total_B=total_B, frac_threshold=frac_thr)
        b_data[label][cap] = (t / total_B) if t is not None else None

xpos = np.arange(len(CONFIGS))
N_CIRCUITS = len(CIRCUITS_B)
DODGE = 0.18   # horizontal offset between circuit-markers within a config

for ci, (circ_label, cap, _, _, _, marker, color) in enumerate(CIRCUITS_B):
    dx = (ci - (N_CIRCUITS - 1) / 2) * DODGE
    for xi, (label, prefix, total_B, revs, toks) in enumerate(CONFIGS):
        y = b_data[label][cap]
        if y is None:
            continue
        axB.scatter(xi + dx, y, s=210, marker=marker, color=color,
                    edgecolors="black", linewidths=1.3, zorder=5)
        # Optional small text near the point
        axB.text(xi + dx, y * 0.78, f"{y*100:.2f}%", ha="center", va="top",
                 fontsize=7.8, color="0.25")

# Reference horizontal lines at 1% and 0.1%
for ref, ref_label, ls in [(0.01, "1% of training", (0, (4, 3))),
                            (0.001, "0.1%", (0, (1, 3)))]:
    axB.axhline(ref, color="0.7", ls=ls, lw=0.9, zorder=1)

axB.set_xticks(xpos)
axB.set_xticklabels([c[0] for c in CONFIGS], fontsize=9.5)
axB.set_yscale("log")
axB.set_xlim(-0.65, len(CONFIGS) - 0.35)
axB.set_ylim(2e-3, 0.3)
axB.set_yticks([0.003, 0.01, 0.03, 0.1])
axB.set_yticklabels(["0.3%", "1%", "3%", "10%"])
axB.set_ylabel("training fraction at circuit formation")
axB.set_title("B   Task circuits form within ~2% of training; attention sinks form later",
              loc="left", fontsize=11.0, pad=8, fontweight="bold")
axB.grid(True, which="major", axis="y", alpha=0.3, ls="--", lw=0.6)

# Direct labels for circuits (inside the panel, top area)
legend_handles = []
for circ_label, cap, _, _, _, marker, color in CIRCUITS_B:
    short = {
        "induction": "★  induction circuit",
        "previous-token": "▲  prev-token circuit",
        "first-token": "▼  BOS attractor",
    }[cap]
    legend_handles.append(plt.Line2D([0], [0], marker=marker, color="w",
                                      markerfacecolor=color, markeredgecolor="black",
                                      markersize=11, label={
                                          "induction": "induction circuit  (≥3 heads, ≥50× sel)",
                                          "previous-token": "prev-token circuit  (≥10 heads, ≥100× sel)",
                                          "first-token": "BOS attractor  (≥10% of heads, ≥30× sel)",
                                      }[cap]))
axB.legend(handles=legend_handles, loc="upper right", fontsize=8.5, frameon=True,
           framealpha=0.94, edgecolor="0.6", handlelength=1.5)

plt.tight_layout(pad=1.2)

out_png = HERE / "headline_two_panel.png"
out_pdf = HERE / "headline_two_panel.pdf"
plt.savefig(out_png, dpi=220, bbox_inches="tight")
plt.savefig(out_pdf, bbox_inches="tight")
print(f"wrote {out_png}")
print(f"wrote {out_pdf}")

print("\nPanel B formation fractions:")
for label, prefix, total_B, *_ in CONFIGS:
    print(f"  {label!r}:")
    for _, cap, *_ in CIRCUITS_B:
        f = b_data[label][cap]
        print(f"    {cap}: {f*100:.2f}%" if f is not None else f"    {cap}: never")
