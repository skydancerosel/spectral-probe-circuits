"""Build per-model ablation-floor curves figure for Paper 1 §5.2.

Plots induction top-1 vs ablation threshold T for each of the three 1B-class
models, on one shared x-axis. Shows visually that per-model T* differs:
OLMo 1B saturates at T=100× (2 heads), OLMoE plateaus at T=30-50× (4 heads),
Pythia 1B keeps descending past T=50× into the multi-role band (T=10-30×).
"""
from __future__ import annotations

import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

REPO = Path(__file__).resolve().parents[1]
RESULTS = REPO / "cross_architecture" / "results"
OUT_DIR = REPO / "figures"


def load_curve(path):
    d = json.load(open(path))
    base = next(r for r in d["results"] if r["name"] == "baseline")
    screens = [r for r in d["results"] if r["name"].startswith("ablate_induction_screen")]
    randoms = [r for r in d["results"] if r["name"].startswith("matched_random")]
    # parse threshold T from name like "ablate_induction_screen_>=50x (3h)"
    def T_of(name):
        return float(name.split(">=")[1].split("x")[0])
    pts = []
    for s in screens:
        T = T_of(s["name"])
        pts.append((T, s["acc_top1"], s["n_ablated"]))
    pts.sort()
    # matched-random — pick by threshold
    rnd_by_T = {}
    for r in randoms:
        T = float(r["name"].split(">=")[1].split("x")[0])
        rnd_by_T[T] = r["acc_top1"]
    return base["acc_top1"], pts, rnd_by_T


def main():
    out_pdf = OUT_DIR / "ablation_floor_curves.pdf"
    out_png = OUT_DIR / "ablation_floor_curves.png"

    models = [
        ("Pythia 1B",   "ablation_threshold_sweep_pythia_1b.json", "C0", "o"),
        ("OLMo 1B",     "ablation_threshold_sweep_olmo.json",      "C1", "s"),
        ("OLMoE 1B-7B", "ablation_threshold_sweep_olmoe.json",     "C2", "^"),
    ]

    fig, ax = plt.subplots(figsize=(8.5, 5))

    for name, fname, color, marker in models:
        baseline, pts, _ = load_curve(RESULTS / fname)
        Ts = [p[0] for p in pts]
        top1s = [p[1] for p in pts]
        n_heads = [p[2] for p in pts]
        # Fraction of baseline performance RETAINED after ablation
        retained = [t1 / baseline * 100 for t1 in top1s]

        ax.plot(Ts, retained, color=color, marker=marker, lw=2.2, markersize=9,
                label=f"{name} (baseline {baseline*100:.2f}%)")
        for T, r, nh in zip(Ts, retained, n_heads):
            ax.annotate(f"$n{{=}}{nh}$",
                        xy=(T, r), xytext=(7, 5),
                        textcoords="offset points", fontsize=8, color=color)

    ax.set_xscale("log")
    ax.set_xticks([2, 10, 30, 50, 100])
    ax.set_xticklabels(["2", "10", "30", "50", "100"])
    ax.invert_xaxis()  # large T (fewer heads ablated) on the left
    ax.set_xlabel("induction-selectivity threshold $T$  (larger $T$ $\\to$ fewer heads ablated)",
                  fontsize=10)
    ax.set_ylabel("fraction of baseline induction top-1 retained (%)", fontsize=10)
    ax.set_title("Per-model ablation-floor curves:\nat what threshold does ablation hit the floor?",
                 fontsize=11)
    ax.axhline(0, color="gray", linestyle=":", lw=0.8)
    ax.axhline(100, color="gray", linestyle=":", lw=0.8, alpha=0.5)
    ax.legend(loc="upper left", fontsize=10, framealpha=0.95)
    ax.grid(True, alpha=0.3)
    ax.set_ylim(-2, 15)

    # annotation: shaded region indicating "ablation floor reached"
    ax.axhspan(-2, 2, color="green", alpha=0.07)
    ax.text(2.5, 1, "ablation floor", fontsize=8, color="darkgreen", style="italic")

    fig.tight_layout()
    fig.savefig(out_pdf, bbox_inches="tight")
    fig.savefig(out_png, bbox_inches="tight", dpi=160)
    print(f"wrote {out_pdf}")
    print(f"wrote {out_png}")


if __name__ == "__main__":
    main()
