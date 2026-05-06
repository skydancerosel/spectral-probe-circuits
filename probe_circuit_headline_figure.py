"""
Single composite headline figure for the probe-retrieval circuit result.

Combines, in one canvas:
  (A) Per-head PR heatmap on s42 across training, with probe_in_acc curve above
      → shows spectral identification: 4 L0 heads stand out
  (B) Per-head attention to KEY position (s42, ckpt=4000)
      → mechanistic confirmation: same 4 heads attend to KEY
  (C) Causal ablation (s42, ckpt=4000 + 800)
      → ablating those 4 heads tanks probe_in; controls do nothing
  (D) Cross-seed asymmetry: s42 ablated with s271 candidates (no effect)
      vs s271 ablated with s42 candidates (substantial effect)
      → s42 = L0-localized, s271 = distributed

Output: analyses/probe_circuit_headline.png (300 DPI, blog-ready)
"""

import json
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec

REPO = Path(__file__).resolve().parent.parent
ANALYSES = REPO / "analyses"

CIRCUIT_HEADS_S42 = [3, 6, 14, 15]
CONTROL_HEADS_S42 = [0, 1, 5, 7]


def main():
    # ── Load data ─────────────────────────────────────────────────────
    # The original s42 run was saved without an _s42 suffix
    spectral = json.load(open(ANALYSES / "probe_circuit_per_head.json"))
    mechinterp = json.load(open(ANALYSES / "probe_circuit_mechinterp.json"))
    ablation_s42 = json.load(open(ANALYSES / "probe_circuit_ablation.json"))
    ablation_s42_with_s271 = json.load(open(ANALYSES /
        "probe_circuit_ablation_s42_with_s271_candidates.json"))
    ablation_s271 = json.load(open(ANALYSES / "probe_circuit_ablation_s271_multi.json"))

    n_layer = spectral["n_layer"]
    n_head = spectral["n_head"]
    steps_pretrain = np.array(spectral["ckpt_step"])
    head_labels = [f"L{l}H{h}" for l in range(n_layer) for h in range(n_head)]

    # PR matrix [128, n_steps]
    PR = np.zeros((n_layer*n_head, len(steps_pretrain)))
    for l in range(n_layer):
        for h in range(n_head):
            PR[l*n_head + h] = spectral["pr"][f"L{l}_H{h}"]

    pc = spectral["probe_curve"]
    # tuples may be 3 or 4 elements depending on which version saved the JSON
    pc_steps = np.array([row[0] for row in pc])
    pc_pin = np.array([row[1] for row in pc], dtype=float)

    # ── Set up canvas ─────────────────────────────────────────────────
    fig = plt.figure(figsize=(16, 11))
    gs = gridspec.GridSpec(
        3, 2,
        height_ratios=[0.6, 4, 3.5],
        width_ratios=[1.6, 1.0],
        hspace=0.25, wspace=0.22,
    )

    # ── Panel A (top): probe_in_acc curve ─────────────────────────────
    ax_pin = fig.add_subplot(gs[0, 0])
    ax_pin.plot(pc_steps, pc_pin, color="tab:red", lw=2)
    ax_pin.set_ylabel("probe_in_acc", fontsize=10)
    ax_pin.set_xlim(steps_pretrain[0], steps_pretrain[-1])
    ax_pin.set_ylim(-0.05, 1.1)
    ax_pin.set_xticklabels([])
    ax_pin.grid(True, alpha=0.3)
    ax_pin.set_title("(A) Spectral identification of circuit heads (s42)",
                      fontsize=11, loc="left", weight="bold")

    # Heatmap of per-head PR over training
    ax_hm = fig.add_subplot(gs[1, 0], sharex=ax_pin)
    im = ax_hm.imshow(PR, aspect="auto", cmap="viridis",
                      extent=[steps_pretrain[0], steps_pretrain[-1],
                              n_layer*n_head - 0.5, -0.5])
    ax_hm.set_xlabel("pretraining step", fontsize=10)
    ax_hm.set_ylabel("(layer, head) — 128 total", fontsize=10)
    # Annotate the L0 row range with red bracket on the left
    for h in CIRCUIT_HEADS_S42:
        row = 0 * n_head + h   # L0
        ax_hm.axhline(row, color="tab:red", lw=0.5, alpha=0.4, xmax=0.04)
        ax_hm.text(steps_pretrain[0] - 200, row, f"L0H{h}", color="tab:red",
                    fontsize=8, va="center", ha="right", weight="bold")
    cbar = plt.colorbar(im, ax=ax_hm, shrink=0.8, pad=0.02)
    cbar.set_label("PR (effective rank, head_dim=32)", fontsize=9)

    # ── Panel B (top-right): attention to KEY ─────────────────────────
    ax_attn = fig.add_subplot(gs[0:2, 1])
    n = n_head
    attn_to_key = np.array(mechinterp["attn_to_key"])
    attn_baseline = np.array(mechinterp["attn_to_random_position"])
    colors = ["tab:red" if h in CIRCUIT_HEADS_S42 else "tab:gray" if h in CONTROL_HEADS_S42 else "tab:blue"
              for h in range(n)]
    width = 0.4
    x = np.arange(n)
    ax_attn.bar(x - width/2, attn_to_key, width, color=colors, edgecolor="k",
                linewidth=0.5, label="attention → KEY position")
    ax_attn.bar(x + width/2, attn_baseline, width, color="lightgray",
                edgecolor="k", linewidth=0.5,
                label="attention → other positions (baseline)")
    ax_attn.set_xticks(x)
    ax_attn.set_xticklabels([f"H{h}" for h in range(n)], fontsize=8)
    ax_attn.set_xlabel("Layer-0 head index", fontsize=10)
    ax_attn.set_ylabel("attention from query-read position", fontsize=10)
    ax_attn.set_title("(B) Mechanistic confirmation:\n"
                       "circuit heads attend back to KEY position",
                       fontsize=11, loc="left", weight="bold")
    ax_attn.legend(fontsize=8, loc="upper left")
    ax_attn.grid(True, alpha=0.3, axis="y")

    # ── Panel C (bottom-left): causal ablation on s42 ─────────────────
    ax_abl = fig.add_subplot(gs[2, 0])
    # Extract relevant conditions from s42 ablation
    s42_data = {}  # {ckpt: {condition: pin}}
    for r in ablation_s42["conditions"]:
        s42_data.setdefault(r["ckpt_step"], {})[r["name"]] = r["probe_in_acc"]

    conditions_show = [
        ("baseline", "baseline (no ablation)", "tab:green"),
        ("ablate_circuit", "ablate L0H{3,6,14,15}\n(spectral picks)", "tab:red"),
        ("ablate_matched_control_L0H0157", "ablate L0H{0,1,5,7}\n(matched control)", "tab:gray"),
        ("ablate_random_L0", "ablate random L0 set\n(control)", "lightgray"),
    ]
    ckpt_show = [800, 4000]
    bar_w = 0.18
    x_base = np.arange(len(conditions_show))
    for i, ck in enumerate(ckpt_show):
        offset = (i - 0.5) * bar_w * 1.1
        vals = [s42_data[ck].get(name, 0) for name, _, _ in conditions_show]
        cs = [c for _, _, c in conditions_show]
        ax_abl.bar(x_base + offset, vals, bar_w, color=cs,
                   edgecolor="k", linewidth=0.5, label=f"ckpt step={ck}")
        for j, v in enumerate(vals):
            ax_abl.text(x_base[j] + offset, v + 0.01, f"{v:.3f}", ha="center",
                         fontsize=7)
    ax_abl.set_xticks(x_base)
    ax_abl.set_xticklabels([label for _, label, _ in conditions_show], fontsize=9)
    ax_abl.set_ylabel("probe_in_acc after ablation", fontsize=10)
    ax_abl.set_title("(C) Causal ablation on s42:\n"
                      "circuit-head ablation tanks probe_in; controls don't",
                      fontsize=11, loc="left", weight="bold")
    ax_abl.set_ylim(0, 1.1)
    ax_abl.legend(fontsize=8, loc="lower right")
    ax_abl.grid(True, alpha=0.3, axis="y")

    # ── Panel D (bottom-right): cross-seed asymmetry ──────────────────
    ax_xs = fig.add_subplot(gs[2, 1])
    # s42 conditions (with s271 candidates):
    s42_xs_data = {r["name"]: r["probe_in_acc"]
                    for r in ablation_s42_with_s271["conditions"]
                    if r["ckpt_step"] == 4000}
    # s271 conditions:
    s271_data = {r["name"]: r["probe_in_acc"]
                  for r in ablation_s271["conditions"]
                  if r["ckpt_step"] == 2000}

    groups = [
        ("baseline", "baseline"),
        ("ablate L0H{3,6,14,15}\n(s42 spectral)", "ablate_s42_circuit_on_s271"),
        ("ablate L6H{1,10}+L7H{9,15}\n(s271 spectral)", "ablate_s271_circuit_L6L7"),
        ("ablate matched random\nL6+L7 (control)", "ablate_matched_random_L6L7"),
    ]
    s42_vals = [s42_xs_data.get(g[1], 0) if g[1] != "baseline" else s42_xs_data.get("baseline", 0)
                for g in groups]
    s271_vals = [s271_data.get(g[1], 0) if g[1] != "baseline" else s271_data.get("baseline", 0)
                  for g in groups]
    x = np.arange(len(groups))
    ax_xs.bar(x - 0.2, s42_vals, 0.4, color="tab:blue",
              edgecolor="k", linewidth=0.5, label="s42 (ckpt 4000)")
    ax_xs.bar(x + 0.2, s271_vals, 0.4, color="tab:orange",
              edgecolor="k", linewidth=0.5, label="s271 (ckpt 2000)")
    ax_xs.set_xticks(x)
    ax_xs.set_xticklabels([g[0] for g in groups], fontsize=8)
    ax_xs.set_ylabel("probe_in_acc", fontsize=10)
    ax_xs.set_title("(D) Cross-seed asymmetry:\n"
                     "s42 = L0-localized; s271 = distributed (uses both)",
                     fontsize=11, loc="left", weight="bold")
    ax_xs.set_ylim(0, 1.1)
    ax_xs.legend(fontsize=9, loc="lower left")
    ax_xs.grid(True, alpha=0.3, axis="y")
    for xi, (a, b) in enumerate(zip(s42_vals, s271_vals)):
        ax_xs.text(xi - 0.2, a + 0.02, f"{a:.2f}", ha="center", fontsize=7)
        ax_xs.text(xi + 0.2, b + 0.02, f"{b:.2f}", ha="center", fontsize=7)

    fig.suptitle(
        "Probe-retrieval circuit in TS-51M:\n"
        "spectral identification → causal ablation → mechanistic confirmation → cross-seed asymmetry",
        fontsize=13, weight="bold", y=0.995,
    )

    out_png = ANALYSES / "probe_circuit_headline.png"
    fig.savefig(out_png, dpi=150, bbox_inches="tight")
    print(f"saved {out_png}")


if __name__ == "__main__":
    main()
