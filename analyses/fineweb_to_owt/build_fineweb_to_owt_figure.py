"""
build_fineweb_to_owt_figure.py

Produces the four-panel comparison figure from the JSON outputs of
fineweb_to_owt_analysis.py.

Panels:
  (A) Per-head PR trajectory across the FineWeb -> OWT boundary, for the
      top-N (by FineWeb-endpoint induction selectivity) heads.
  (B) Whole-model BOS-class fraction across the FineWeb -> OWT boundary.
  (C) Endpoint comparison: induction-selectivity values at the FineWeb
      endpoint vs the OWT endpoint for each (L, H) head that crosses 50x
      in either phase. Diagonal = same selectivity in both phases.
  (D) Capability-selectivity trajectory through the OWT phase for the
      FineWeb-endpoint induction heads (induction sel, prev-token sel,
      first-token sel each as a stacked subpanel).

Run:
  python build_fineweb_to_owt_figure.py \
      --input-dir results/fineweb_to_owt \
      --output figures/fineweb_to_owt_panel.pdf
"""

import argparse
import json
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


def load_jsons(input_dir):
    return {
        "pr": json.loads((input_dir / "per_head_pr_trajectory.json").read_text()),
        "bos": json.loads((input_dir / "bos_fraction_trajectory.json").read_text()),
        "endpoints": json.loads((input_dir / "mech_interp_endpoints.json").read_text()),
        "sel": json.loads((input_dir / "circuit_selectivity_trajectory.json").read_text())
        if (input_dir / "circuit_selectivity_trajectory.json").exists() else None,
    }


def cumulative_step(phase_list, step_list, fineweb_end_step):
    """Translate phase-relative step counts into a continuous x axis so the
    two phases plot side by side. OWT steps start where FineWeb steps end."""
    xs = []
    for phase, step in zip(phase_list, step_list):
        if phase == "fineweb":
            xs.append(step)
        else:
            xs.append(fineweb_end_step + step)
    return np.array(xs)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--input-dir", type=Path, default=Path("results/fineweb_to_owt"))
    ap.add_argument("--output", type=Path, default=Path("figures/fineweb_to_owt_panel.pdf"))
    ap.add_argument("--top-n", type=int, default=12,
                    help="number of heads to show in panel A (by FineWeb-endpoint induction sel)")
    args = ap.parse_args()

    d = load_jsons(args.input_dir)
    args.output.parent.mkdir(parents=True, exist_ok=True)

    # Determine FineWeb-end step from pr trajectory
    fw_steps_pr = [s for (p, s) in zip(d["pr"]["phase"], d["pr"]["step"]) if p == "fineweb"]
    if not fw_steps_pr:
        raise SystemExit("no FineWeb steps in PR trajectory")
    fw_end = max(fw_steps_pr)

    fig, axes = plt.subplots(2, 2, figsize=(13, 9))

    # ---------- Panel A: per-head PR across boundary ----------
    ax = axes[0, 0]
    # Pick top-N FineWeb-endpoint induction heads
    fw_circuit = d["endpoints"]["fineweb"]["induction_circuit_50x"]
    fw_circuit.sort(key=lambda lhs: lhs[2], reverse=True)
    top_heads = [(L, H) for (L, H, _) in fw_circuit[: args.top_n]]
    if not top_heads:
        # Fall back: use top by FineWeb-endpoint PR
        last_pr = {k: d["pr"]["pr"][k][len(fw_steps_pr) - 1] for k in d["pr"]["pr"]}
        top = sorted(last_pr.items(), key=lambda kv: kv[1], reverse=True)[: args.top_n]
        top_heads = [tuple(int(x[1:]) for x in k.split("_")) for (k, _) in top]

    xs = cumulative_step(d["pr"]["phase"], d["pr"]["step"], fw_end)
    cmap = plt.cm.viridis(np.linspace(0, 0.85, len(top_heads)))
    for color, (L, H) in zip(cmap, top_heads):
        key = f"L{L}_H{H}"
        ax.plot(xs, d["pr"]["pr"][key], color=color, alpha=0.85, lw=1.2,
                label=f"L{L}H{H}")
    ax.axvline(fw_end, color="black", lw=1, linestyle="--", alpha=0.7)
    ax.text(fw_end, ax.get_ylim()[1] * 0.95, "  FineWeb $\\rightarrow$ OWT",
            ha="left", va="top", fontsize=9, color="black")
    ax.set_xscale("symlog", linthresh=200)
    ax.set_xlabel("training step (cumulative across both phases)")
    ax.set_ylabel("PR (participation ratio)")
    ax.set_title("(A) Per-head PR across the FineWeb $\\rightarrow$ OWT boundary")
    ax.legend(fontsize=7, loc="upper left", ncol=2, frameon=False)
    ax.grid(alpha=0.3)

    # ---------- Panel B: BOS-class fraction across boundary ----------
    ax = axes[0, 1]
    xs_b = cumulative_step(d["bos"]["phase"], d["bos"]["step"], fw_end)
    ax.plot(xs_b, d["bos"]["bos_fraction"], color="#c44e52", lw=1.6, marker="o", ms=3)
    ax.axvline(fw_end, color="black", lw=1, linestyle="--", alpha=0.7)
    ax.text(fw_end, ax.get_ylim()[1] * 0.92, "  FineWeb $\\rightarrow$ OWT",
            ha="left", va="top", fontsize=9, color="black")
    ax.set_xscale("symlog", linthresh=200)
    ax.set_xlabel("training step")
    ax.set_ylabel("BOS-class fraction (whole-model, $\\geq 30\\times$)")
    ax.set_title("(B) BOS-attractor fraction across the boundary")
    ax.grid(alpha=0.3)
    ax.set_ylim(0, max(0.4, max(d["bos"]["bos_fraction"]) * 1.15))

    # ---------- Panel C: induction-selectivity FineWeb endpoint vs OWT endpoint ----------
    ax = axes[1, 0]
    fw_sel = np.array(d["endpoints"]["fineweb"]["selectivities"]["induction"])
    owt_sel = np.array(d["endpoints"]["owt"]["selectivities"]["induction"])
    # Plot all heads, but mark heads that cross 50x in either phase
    n_layer, n_head = fw_sel.shape
    fw_flat = fw_sel.flatten()
    owt_flat = owt_sel.flatten()
    crosses = (fw_flat >= 50) | (owt_flat >= 50)
    ax.scatter(fw_flat[~crosses], owt_flat[~crosses], s=8, c="#bbbbbb", alpha=0.55,
               label="all other heads")
    ax.scatter(fw_flat[crosses], owt_flat[crosses], s=36, c="#c44e52",
               edgecolors="black", linewidths=0.4,
               label=f"heads with induction sel $\\geq 50\\times$ in either phase")
    # Diagonal line
    lim_lo = min(fw_flat.min(), owt_flat.min(), 1.0) * 0.9
    lim_hi = max(fw_flat.max(), owt_flat.max()) * 1.05
    ax.plot([lim_lo, lim_hi], [lim_lo, lim_hi], "k--", lw=0.8, alpha=0.5)
    ax.set_xscale("symlog", linthresh=1)
    ax.set_yscale("symlog", linthresh=1)
    ax.set_xlabel(f"induction selectivity at FineWeb endpoint "
                  f"(step {d['endpoints']['fineweb']['step']})")
    ax.set_ylabel(f"induction selectivity at OWT endpoint "
                  f"(step {d['endpoints']['owt']['step']})")
    jac = d["endpoints"]["set_comparison"]["jaccard_50x"]
    n_shared = len(d["endpoints"]["set_comparison"]["shared_50x"])
    n_fw_only = len(d["endpoints"]["set_comparison"]["fineweb_only_50x"])
    n_owt_only = len(d["endpoints"]["set_comparison"]["owt_only_50x"])
    ax.set_title(f"(C) Induction-circuit identity stability: Jaccard $= {jac:.2f}$  "
                 f"(shared $= {n_shared}$, FW-only $= {n_fw_only}$, OWT-only $= {n_owt_only}$)")
    ax.legend(fontsize=8, loc="lower right", frameon=False)
    ax.grid(alpha=0.3)

    # ---------- Panel D: selectivity trajectory for FineWeb-endpoint induction heads ----------
    ax = axes[1, 1]
    if d["sel"] is None or not d["sel"].get("induction"):
        ax.text(0.5, 0.5, "no FineWeb endpoint induction circuit\n(skipped)",
                ha="center", va="center", transform=ax.transAxes, fontsize=12)
        ax.set_xticks([])
        ax.set_yticks([])
    else:
        xs_s = cumulative_step(d["sel"]["phase"], d["sel"]["step"], fw_end)
        ind_keys = list(d["sel"]["induction"].keys())
        for k in ind_keys:
            ax.plot(xs_s, d["sel"]["induction"][k], "-",
                    lw=1.3, alpha=0.85, label=k)
        ax.axhline(50, color="black", lw=0.5, linestyle=":", alpha=0.4)
        ax.axvline(fw_end, color="black", lw=1, linestyle="--", alpha=0.7)
        ax.set_xscale("symlog", linthresh=200)
        ax.set_yscale("symlog", linthresh=1)
        ax.set_xlabel("training step (cumulative)")
        ax.set_ylabel("induction selectivity")
        ax.set_title("(D) Selectivity through OWT for FineWeb-endpoint induction heads")
        ax.legend(fontsize=8, ncol=2, frameon=False)
        ax.grid(alpha=0.3)

    fig.suptitle("Karpathy GPT-2 124M: data-distribution shift FineWeb $\\rightarrow$ OWT",
                 fontsize=13, fontweight="bold", y=1.00)
    fig.tight_layout()
    fig.savefig(args.output, bbox_inches="tight")
    png_out = args.output.with_suffix(".png")
    fig.savefig(png_out, bbox_inches="tight", dpi=150)
    print(f"wrote {args.output}")
    print(f"wrote {png_out}")


if __name__ == "__main__":
    main()
