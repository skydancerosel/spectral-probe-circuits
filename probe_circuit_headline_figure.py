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

REPO = Path(__file__).resolve().parent
ANALYSES = REPO / "results"

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
    fig = plt.figure(figsize=(17, 14))
    # Outer gridspec: top half (rows 0-1: A's curve + heatmap, B) vs bottom half (C, D)
    # Use a nested layout so we get tight spacing between A's curve and heatmap
    # but generous spacing between the top row (A heatmap / B) and the bottom row (C / D)
    # so panel C/D titles don't collide with A/B xlabels.
    gs_outer = gridspec.GridSpec(
        2, 1,
        height_ratios=[4.6, 3.5],
        hspace=0.20,
    )
    gs_top = gridspec.GridSpecFromSubplotSpec(
        2, 2,
        subplot_spec=gs_outer[0],
        height_ratios=[0.6, 4],
        width_ratios=[1.6, 1.0],
        hspace=0.05, wspace=0.30,
    )
    gs_bot = gridspec.GridSpecFromSubplotSpec(
        1, 2,
        subplot_spec=gs_outer[1],
        width_ratios=[1.6, 1.0],
        wspace=0.30,
    )

    # ── Panel A (top): probe_in_acc curve ─────────────────────────────
    ax_pin = fig.add_subplot(gs_top[0, 0])
    ax_pin.plot(pc_steps, pc_pin, color="tab:red", lw=2)
    ax_pin.set_ylabel("probe_in_acc", fontsize=10)
    ax_pin.set_xlim(steps_pretrain[0], steps_pretrain[-1])
    ax_pin.set_ylim(-0.05, 1.1)
    ax_pin.set_xticklabels([])
    ax_pin.grid(True, alpha=0.3)
    ax_pin.set_title("(A) Spectral identification of circuit heads (s42)",
                      fontsize=11, loc="left", weight="bold")

    # Heatmap of per-head PR over training
    ax_hm = fig.add_subplot(gs_top[1, 0], sharex=ax_pin)
    im = ax_hm.imshow(PR, aspect="auto", cmap="viridis",
                      extent=[steps_pretrain[0], steps_pretrain[-1],
                              n_layer*n_head - 0.5, -0.5])
    ax_hm.set_xlabel("pretraining step", fontsize=10)
    ax_hm.set_ylabel("(layer, head) — 128 total", fontsize=10)
    # Annotate the L0 circuit heads with a red bracket + a single combined label
    # (individual labels overlap because L0H3..L0H15 are within the top 12 of 128 rows)
    bracket_top = min(CIRCUIT_HEADS_S42)
    bracket_bot = max(CIRCUIT_HEADS_S42)
    ax_hm.plot([steps_pretrain[0] - 100, steps_pretrain[0] - 100],
                [bracket_top - 0.5, bracket_bot + 0.5],
                color="tab:red", lw=2, clip_on=False)
    ax_hm.text(steps_pretrain[0] - 250, (bracket_top + bracket_bot) / 2,
                "L0H{3,6,14,15}", color="tab:red",
                fontsize=9, va="center", ha="right", weight="bold")
    for h in CIRCUIT_HEADS_S42:
        row = 0 * n_head + h
        ax_hm.axhline(row, color="tab:red", lw=0.5, alpha=0.4, xmax=0.03)
    cbar = plt.colorbar(im, ax=ax_hm, shrink=0.8, pad=0.02)
    cbar.set_label("PR (effective rank, head_dim=32)", fontsize=9)

    # ── Panel B (top-right): attention to KEY ─────────────────────────
    ax_attn = fig.add_subplot(gs_top[:, 1])
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
    ax_abl = fig.add_subplot(gs_bot[0, 0])
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

    # ── Panel D (bottom-right): 6-seed cross-seed asymmetry as heatmap ──
    ax_xs = fig.add_subplot(gs_bot[0, 1])

    ablation_s42_v2 = json.load(open(ANALYSES / "probe_circuit_ablation_s42.json"))
    ablation_s271_v2 = json.load(open(ANALYSES / "probe_circuit_ablation_s271.json"))
    ablation_s149 = json.load(open(ANALYSES / "probe_circuit_ablation_s149.json"))
    ablation_s256 = json.load(open(ANALYSES / "probe_circuit_ablation_s256.json"))
    ablation_s123 = json.load(open(ANALYSES / "probe_circuit_ablation_s123.json"))
    ablation_s314 = json.load(open(ANALYSES / "probe_circuit_ablation_s314.json"))

    s42_data = {r["name"]: r["probe_in_acc"]
                 for r in ablation_s42_v2["conditions"]
                 if r["ckpt_step"] == 4000}
    s271_data = {r["name"]: r["probe_in_acc"]
                  for r in ablation_s271_v2["conditions"]
                  if r["ckpt_step"] == 2000}
    s149_data = {r["name"]: r["probe_in_acc"]
                  for r in ablation_s149["conditions"]
                  if r["ckpt_step"] == 4000}
    s256_data = {r["name"]: r["probe_in_acc"]
                  for r in ablation_s256["conditions"]
                  if r["ckpt_step"] == 4000}
    s123_data = {r["name"]: r["probe_in_acc"]
                  for r in ablation_s123["conditions"]
                  if r["ckpt_step"] == 4000}
    s314_data = {r["name"]: r["probe_in_acc"]
                  for r in ablation_s314["conditions"]
                  if r["ckpt_step"] == 4000}

    NA = np.nan
    # Columns: baseline, ablate_s42, ablate_s271, ablate_s149, ablate_s256, ablate_s123, ablate_s314
    # Rows: s42, s271, s149, s256, s123, s314
    grid = np.array([
        # s42 row
        [s42_data.get("baseline", NA),
         s42_data.get("ablate_s42_circuit_L0", NA),
         s42_data.get("ablate_s271_circuit_on_s42", NA),
         s42_data.get("ablate_s149_circuit_on_s42", NA),
         s42_data.get("ablate_s256_circuit_on_s42", NA),
         NA, NA],  # s123 picks on s42 — not measured (predicted ~baseline since no head overlap)
        # s271 row
        [s271_data.get("baseline", NA),
         s271_data.get("ablate_s42_circuit_on_s271", NA),
         s271_data.get("ablate_s271_circuit_L6L7", NA),
         s271_data.get("ablate_s149_circuit_on_s271", NA),
         s271_data.get("ablate_s256_circuit_on_s271", NA),
         NA, NA],
        # s149 row
        [s149_data.get("baseline", NA),
         s149_data.get("ablate_s42_circuit_on_s149", NA),
         s149_data.get("ablate_s271_circuit_on_s149", NA),
         s149_data.get("ablate_s149_circuit_L6L7", NA),
         s149_data.get("ablate_s256_circuit_on_s149", NA),
         s149_data.get("ablate_s123_circuit_on_s149", NA),
         s149_data.get("ablate_s314_circuit_on_s149", NA)],
        # s256 row
        [s256_data.get("baseline", NA),
         s256_data.get("ablate_s42_circuit_on_s256", NA),
         s256_data.get("ablate_s271_circuit_on_s256", NA),
         s256_data.get("ablate_s149_circuit_on_s256", NA),
         s256_data.get("ablate_s256_circuit_L5L6L7", NA),
         s256_data.get("ablate_s123_circuit_on_s256", NA),
         s256_data.get("ablate_s314_circuit_on_s256", NA)],
        # s123 row
        [s123_data.get("baseline", NA),
         s123_data.get("ablate_s42_circuit_on_s123", NA),
         s123_data.get("ablate_s271_circuit_on_s123", NA),
         s123_data.get("ablate_s149_circuit_on_s123", NA),
         s123_data.get("ablate_s256_circuit_on_s123", NA),
         s123_data.get("ablate_s123_circuit_L5L6L7", NA),
         NA],  # s314 picks on s123 — not measured
        # s314 row
        [s314_data.get("baseline", NA),
         s314_data.get("ablate_s42_circuit_on_s314", NA),
         s314_data.get("ablate_s271_circuit_on_s314", NA),
         s314_data.get("ablate_s149_circuit_on_s314", NA),
         s314_data.get("ablate_s256_circuit_on_s314", NA),
         s314_data.get("ablate_s123_circuit_on_s314", NA),
         s314_data.get("ablate_s314_circuit_L5L7", NA)],
    ])

    masked = np.ma.array(grid, mask=np.isnan(grid))
    cmap = plt.cm.RdYlGn.copy()
    cmap.set_bad("lightgray")
    im = ax_xs.imshow(masked, cmap=cmap, vmin=0, vmax=1.0, aspect="auto")
    for i in range(grid.shape[0]):
        for j in range(grid.shape[1]):
            v = grid[i, j]
            if np.isnan(v):
                ax_xs.text(j, i, "—", ha="center", va="center",
                            fontsize=8, color="dimgray")
            else:
                color = "white" if v < 0.45 else "black"
                weight = "bold" if i + 1 == j else "normal"  # diagonal: row i, col i+1 (skip baseline col)
                ax_xs.text(j, i, f"{v:.2f}", ha="center", va="center",
                            fontsize=9, color=color, weight=weight)
    ax_xs.set_xticks(range(7))
    ax_xs.set_xticklabels(["baseline",
                             "ablate\ns42 picks",
                             "ablate\ns271 picks",
                             "ablate\ns149 picks",
                             "ablate\ns256 picks",
                             "ablate\ns123 picks",
                             "ablate\ns314 picks"],
                            fontsize=7)
    ax_xs.set_yticks(range(6))
    ax_xs.set_yticklabels(["s42\n@4000",
                             "s271\n@2000",
                             "s149\n@4000",
                             "s256\n@4000",
                             "s123\n@4000",
                             "s314\n@4000"], fontsize=8)
    ax_xs.set_title("(D) Cross-seed asymmetry on six seeds:\n"
                     "bold = own picks tank own circuit; gray = not measured",
                     fontsize=11, loc="left", weight="bold")
    cbar = plt.colorbar(im, ax=ax_xs, shrink=0.7, pad=0.02)
    cbar.set_label("probe_in_acc after ablation", fontsize=8)

    fig.suptitle(
        "Probe-retrieval circuit in TS-51M (n=6 seeds):\n"
        "spectral identification → causal ablation → mechanistic confirmation → cross-seed asymmetry",
        fontsize=13, weight="bold", y=0.995,
    )

    out_png = ANALYSES / "probe_circuit_headline.png"
    fig.savefig(out_png, dpi=150, bbox_inches="tight")
    print(f"saved {out_png}")


if __name__ == "__main__":
    main()
