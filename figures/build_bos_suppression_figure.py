"""Build Pythia 1B BOS-suppression-of-induction bar chart for Paper 1 §5.3.

Shows the non-monotone matched-random-ablation curve in Pythia 1B's induction
layers: random ablation of 3 heads is baseline-like, but random ablation of
6 or 11 heads RAISES top-1 above baseline (16.7% / 36.1% vs 4.05% baseline),
then ablation of 32 heads dismantles the circuit (drops to 0.40%). The screen-
ablation at >=50x (3 heads) is shown for contrast (drops to 0.25%).
"""
from __future__ import annotations

import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

REPO = Path(__file__).resolve().parents[1]
RESULTS = REPO / "cross_architecture" / "results"
OUT_DIR = REPO / "figures"


def main():
    out_pdf = OUT_DIR / "bos_suppression_pythia_1b.pdf"
    out_png = OUT_DIR / "bos_suppression_pythia_1b.png"

    d = json.load(open(RESULTS / "ablation_threshold_sweep_pythia_1b.json"))
    results = {r["name"]: r for r in d["results"]}
    base_top1 = results["baseline"]["acc_top1"]

    # Conditions to display
    rows = []
    for tag, label, color in [
        ("baseline",                                 "baseline\n(no ablation)",        "gray"),
        ("ablate_induction_screen_>=50x (3h)",       "ablate screen $\\geq 50\\times$\n(3 heads, real circuit)", "C3"),
        ("matched_random_>=50x (3h)",                "matched-random\n$n{=}3$",         "C0"),
        ("matched_random_>=30x (6h)",                "matched-random\n$n{=}6$",         "C0"),
        ("matched_random_>=10x (11h)",               "matched-random\n$n{=}11$",        "C0"),
        ("matched_random_>=2x (32h)",                "matched-random\n$n{=}32$",        "C0"),
    ]:
        r = results.get(tag)
        if r is None:
            continue
        rows.append((label, r["acc_top1"], r.get("n_ablated", 0), color))

    fig, ax = plt.subplots(figsize=(10, 4.5))
    x = list(range(len(rows)))
    labels = [r[0] for r in rows]
    vals = [r[1] * 100 for r in rows]
    colors = [r[3] for r in rows]
    bars = ax.bar(x, vals, color=colors, edgecolor="black", lw=0.7)

    # baseline reference line
    ax.axhline(base_top1 * 100, color="gray", linestyle="--", lw=1.0,
               label=f"baseline ({base_top1*100:.2f}%)")

    # value labels above bars
    for i, (b, v) in enumerate(zip(bars, vals)):
        ax.text(b.get_x() + b.get_width()/2, v + 0.7, f"{v:.2f}%",
                ha="center", va="bottom", fontsize=9,
                fontweight="bold" if v > base_top1 * 100 * 1.5 else "normal")

    ax.set_xticks(x)
    ax.set_xticklabels(labels, fontsize=9)
    ax.set_ylabel("synthetic-induction top-1 acc (%)")
    ax.set_title("Pythia 1B: matched-random ablation of 6-11 heads in induction-circuit layers raises\n"
                 "top-1 above baseline (BOS-attractor suppression released)", fontsize=11)
    ax.legend(loc="upper left", fontsize=9)
    ax.grid(True, axis="y", alpha=0.3)
    ax.set_ylim(0, max(vals) * 1.15)
    fig.tight_layout()
    fig.savefig(out_pdf, bbox_inches="tight")
    fig.savefig(out_png, bbox_inches="tight", dpi=160)
    print(f"wrote {out_pdf}")
    print(f"wrote {out_png}")


if __name__ == "__main__":
    main()
