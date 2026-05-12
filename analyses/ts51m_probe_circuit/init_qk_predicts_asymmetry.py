"""
init_qk_predicts_asymmetry.py

Test #7 from the "what's next" list (speculative but quick):
  Can we predict the cross-seed asymmetry (s42 → L0-only;
  s271/s149/s256 → distributed across L5/L6/L7) from features of the
  initial Q/K matrices?

If YES, the cross-seed asymmetry isn't random — it's determined by
init. That's a strong "lottery ticket"-flavored claim.

Approach:
  For each of the 4 seeds, load the step-1 checkpoint (essentially
  random init except for the first optimizer step). For each (layer,
  head) compute features of the per-head Q/K matrices:

    1. Frobenius norm of W_QK = W_Q @ W_K.T (head-internal)
    2. Spectral entropy / participation ratio of W_QK eigenvalues
    3. Top-singular-value of W_QK
    4. Effective rank

Then ask: do these features at init differ between s42 (which becomes
L0-only) and s271/s149/s256 (which become distributed) at the layers
where the asymmetry shows up?

Output: analyses/init_qk_predicts_asymmetry.{json,png}
"""

import json
import sys
from pathlib import Path
import numpy as np
import torch
import matplotlib.pyplot as plt

REPO = Path(__file__).resolve().parent

SEEDS = ["s42", "s271", "s149", "s256"]
SEED_OUTCOME = {
    "s42": "L0-only",
    "s271": "distributed (L6+L7)",
    "s149": "distributed (L6+L7)",
    "s256": "distributed (L5+L6+L7)",
}
SEED_PICKS = {
    "s42": {0: [3, 6, 14, 15]},
    "s271": {6: [1, 10], 7: [9, 15]},
    "s149": {6: [2, 5, 6, 7], 7: [13]},
    "s256": {5: [10], 6: [2, 4], 7: [6, 13]},
}

D_MODEL = 512
HEAD_DIM = 32


def load_seed_step1(seed):
    path = REPO / f"runs/beta2_ablation/pilot_wd0.5_lr0.001_lp2.0_b20.95_{seed}/ckpt_000001.pt"
    return torch.load(path, map_location="cpu", weights_only=True)["model_state_dict"]


def per_head_qk_features(state_dict, layer_idx):
    """For each head in `layer_idx`, compute features of the QK matrix."""
    qkv = state_dict[f"blocks.{layer_idx}.attn.qkv.weight"]  # [3*d_model, d_model]
    n_head = D_MODEL // HEAD_DIM
    out = []
    for h in range(n_head):
        W_Q_h = qkv[h*HEAD_DIM:(h+1)*HEAD_DIM, :]                              # [32, 512]
        W_K_h = qkv[D_MODEL + h*HEAD_DIM:D_MODEL + (h+1)*HEAD_DIM, :]          # [32, 512]
        # QK product (in residual-stream input space): W_QK = W_Q^T @ W_K  [512, 512]
        # But for head-internal structure: W_q W_k^T is [32, 32]
        # Use SVD of W_QK = W_Q^T @ W_K [d_model, d_model] — this captures the
        # full-rank structure of the head's attention pattern at init
        W_QK = W_Q_h.T @ W_K_h  # [d_model, d_model] but rank ≤ head_dim
        # Singular values: only the top head_dim are nonzero
        U, S, Vh = torch.linalg.svd(W_QK, full_matrices=False)
        S = S.numpy()
        S_topk = S[:HEAD_DIM]
        # Features
        frob = float(np.linalg.norm(S_topk))
        top_sv = float(S_topk[0])
        # Spectral entropy → effective rank
        s2 = S_topk ** 2
        s2_sum = s2.sum() + 1e-12
        p = s2 / s2_sum
        H_ent = -float((p * np.log(p + 1e-12)).sum())
        eff_rank = float(np.exp(H_ent))
        # Spectral gap (top-1 vs top-2)
        gap_12 = float(S_topk[0] - S_topk[1]) if len(S_topk) >= 2 else float("nan")
        # Median over top
        med = float(np.median(S_topk))

        out.append({"head": h,
                     "frob_norm": frob,
                     "top_sv": top_sv,
                     "effective_rank": eff_rank,
                     "gap_top1_top2": gap_12,
                     "median_sv": med})
    return out


def main():
    print(f"Loading step-1 checkpoints for {len(SEEDS)} seeds...")
    seed_data = {}
    for seed in SEEDS:
        sd = load_seed_step1(seed)
        seed_data[seed] = sd
    print("  loaded")

    # For each seed, compute QK features for layers 0, 5, 6, 7 (the layers
    # where any seed has spectral picks)
    target_layers = [0, 5, 6, 7]
    results = {seed: {} for seed in SEEDS}
    for seed in SEEDS:
        for L in target_layers:
            results[seed][L] = per_head_qk_features(seed_data[seed], L)

    # Print compact summary
    feature_names = ["frob_norm", "top_sv", "effective_rank", "gap_top1_top2"]

    for L in target_layers:
        print(f"\n{'='*84}")
        print(f"LAYER {L} — per-head features at init (step 1)")
        print(f"{'='*84}")
        for fname in feature_names:
            print(f"\n  {fname} (mean ± std across all 16 heads in L{L}):")
            print(f"  {'seed':<8} {'mean':>10} {'std':>10} {'picks_mean':>12} {'picks_min':>12} {'picks_max':>12}")
            for seed in SEEDS:
                vals = np.array([h[fname] for h in results[seed][L]])
                pick_indices = SEED_PICKS[seed].get(L, [])
                if pick_indices:
                    pick_vals = np.array([results[seed][L][h][fname] for h in pick_indices])
                    pm, pl, ph = pick_vals.mean(), pick_vals.min(), pick_vals.max()
                    pm_s, pl_s, ph_s = f"{pm:>12.4f}", f"{pl:>12.4f}", f"{ph:>12.4f}"
                else:
                    pm_s, pl_s, ph_s = f"{'(no picks)':>12}", f"{'':>12}", f"{'':>12}"
                print(f"  {seed:<8} {vals.mean():>10.4f} {vals.std():>10.4f} {pm_s} {pl_s} {ph_s}")

    # The big test: do PICKS show systematic feature differences from
    # NON-PICKS in the same layer for the same seed?
    print(f"\n{'='*84}")
    print("SYSTEMATIC TEST: pick heads vs non-pick heads, within each seed/layer")
    print(f"{'='*84}")
    summary = {}
    for seed in SEEDS:
        for L, picks in SEED_PICKS[seed].items():
            non_picks = [h for h in range(D_MODEL // HEAD_DIM) if h not in picks]
            for fname in feature_names:
                pick_vals = np.array([results[seed][L][h][fname] for h in picks])
                non_pick_vals = np.array([results[seed][L][h][fname] for h in non_picks])
                diff_z = (pick_vals.mean() - non_pick_vals.mean()) / (non_pick_vals.std() + 1e-9)
                key = f"{seed}_L{L}_{fname}"
                summary[key] = {
                    "pick_mean": float(pick_vals.mean()),
                    "non_pick_mean": float(non_pick_vals.mean()),
                    "non_pick_std": float(non_pick_vals.std()),
                    "z_score_of_pick_mean": float(diff_z),
                    "n_picks": len(picks),
                }
                if abs(diff_z) > 0.5:  # report any non-trivial difference
                    direction = "HIGHER" if diff_z > 0 else "LOWER"
                    print(f"  {seed} L{L} {fname:<20}: picks {direction} than non-picks "
                          f"(pick_mean={pick_vals.mean():.4f}, "
                          f"non_pick_mean={non_pick_vals.mean():.4f}, "
                          f"z={diff_z:+.2f})")

    # Save
    out = {"per_seed_per_layer": {seed: {str(L): results[seed][L] for L in target_layers}
                                    for seed in SEEDS},
           "pick_vs_nonpick_summary": summary,
           "seed_outcomes": SEED_OUTCOME,
           "seed_picks": {seed: {str(L): hs for L, hs in SEED_PICKS[seed].items()}
                           for seed in SEEDS}}
    out_json = REPO / "results/init_qk_predicts_asymmetry.json"
    with open(out_json, "w") as f:
        json.dump(out, f, indent=2)
    print(f"\nwrote {out_json}")

    # Plot: bar chart of pick-vs-nonpick z-scores across all (seed, layer)
    fig, axes = plt.subplots(1, len(feature_names), figsize=(5*len(feature_names), 5))
    for i, fname in enumerate(feature_names):
        ax = axes[i]
        labels = []
        zs = []
        colors = []
        for seed in SEEDS:
            for L in sorted(SEED_PICKS[seed].keys()):
                key = f"{seed}_L{L}_{fname}"
                if key in summary:
                    labels.append(f"{seed}\nL{L}")
                    zs.append(summary[key]["z_score_of_pick_mean"])
                    colors.append("tab:red" if seed == "s42" else "tab:blue")
        x = np.arange(len(labels))
        ax.bar(x, zs, color=colors, edgecolor="k", linewidth=0.5)
        ax.set_xticks(x)
        ax.set_xticklabels(labels, fontsize=8)
        ax.set_ylabel("z-score of pick-mean vs non-pick distribution")
        ax.set_title(f"{fname}")
        ax.axhline(0, color="k", lw=0.5)
        ax.axhline(1, color="gray", lw=0.5, ls="--", alpha=0.5)
        ax.axhline(-1, color="gray", lw=0.5, ls="--", alpha=0.5)
        ax.grid(True, alpha=0.3, axis="y")

    fig.suptitle("Do the spectral picks differ from non-picks in their initial QK features?\n"
                  "z-score: how many σ above/below the non-pick distribution the pick-mean sits.",
                  fontsize=11, weight="bold", y=1.02)
    fig.tight_layout()
    out_png = REPO / "results/init_qk_predicts_asymmetry.png"
    fig.savefig(out_png, dpi=130, bbox_inches="tight")
    print(f"saved {out_png}")


if __name__ == "__main__":
    main()
