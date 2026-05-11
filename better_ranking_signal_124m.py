"""
better_ranking_signal_124m.py

Question: PR-spread (max - min over training) is the ranking signal we use
to pick the top-k spectral picks. The capability survey on GPT-2 124M
showed PR-spread doesn't perfectly rank by capability strength:
  - L6H9 has 8,105x prev-token selectivity but ranks only 14 by PR-spread
  - L7H4 has 184x induction selectivity but ranks 23

Hypothesis: an alternative feature of the PR trajectory might rank
high-selectivity heads more accurately.

Candidates tested:
  1. spread          (current baseline) — max - min
  2. final_pr        — value at last ckpt
  3. max_pr          — peak value
  4. peak_step       — when peak occurred (later = ?)
  5. max_rate        — max derivative (sharpest jump)
  6. rise_time       — steps from PR=2 to PR=15
  7. mean_post_grok  — mean PR after step where it first exceeds 15
  8. slope_at_inflection — slope at the steepest derivative point
  9. integral        — area under the (PR - 1) curve (total content-dependence over training)
  10. composite_1    — spread * (1 / rise_time) (penalize slow transitions)

Ground truth: a head's "capability strength" = max selectivity across
{induction, previous-token, self} from the existing mechinterp data.
A head is a "true capability head" if max selectivity > 30x.

For each ranking signal, compute precision-at-k = fraction of top-k by
that signal that are also true capability heads.

Output: analyses/better_ranking_signal_124m.{json,png}
"""

import json
import sys
from pathlib import Path
import numpy as np
import torch
import torch.nn.functional as F
import matplotlib.pyplot as plt

REPO = Path(__file__).resolve().parent
sys.path.insert(0, str(REPO))
from induction_heads_per_head_124m import (
    GPT, GPTConfig, load_karpathy_ckpt, build_induction_batch
)

KARPATHY_DIR = REPO / "karpathy_llmc/runs/gpt2_fineweb10B"
SELECTIVITY_THRESHOLD = 30.0


def compute_features(pr_traj, steps):
    """Given a PR trajectory (list of PR values) and corresponding steps,
    return a dict of trajectory features."""
    pr = np.array(pr_traj, dtype=np.float64)
    s = np.array(steps, dtype=np.float64)
    feats = {}
    feats["spread"] = float(pr.max() - pr.min())
    feats["final_pr"] = float(pr[-1])
    feats["max_pr"] = float(pr.max())
    feats["min_pr"] = float(pr.min())
    feats["peak_step"] = float(s[pr.argmax()])

    # Derivative features
    if len(pr) >= 2:
        dpr = np.gradient(pr, s)
        feats["max_rate"] = float(np.abs(dpr).max())
        feats["mean_abs_rate"] = float(np.abs(dpr).mean())
    else:
        feats["max_rate"] = 0.0
        feats["mean_abs_rate"] = 0.0

    # Rise time: steps to go from PR=2 to PR=15 (typical content-dep onset to mid-saturation)
    rise_low_idx = np.argmax(pr > 2)
    rise_high_idx = np.argmax(pr > 15)
    if pr.max() < 15:
        feats["rise_time"] = float("inf")
    else:
        feats["rise_time"] = float(s[rise_high_idx] - s[rise_low_idx])

    # Mean PR post-saturation (after first crossing PR=15)
    if pr.max() < 15:
        feats["mean_post_grok"] = float("nan")
    else:
        feats["mean_post_grok"] = float(pr[rise_high_idx:].mean())

    # Integral of (PR - baseline_PR) over training — total content-dep magnitude
    feats["integral"] = float(np.trapz(np.maximum(pr - 1.0, 0), s))

    # Composite: spread per unit rise-time (rewards sharp transitions)
    if np.isfinite(feats["rise_time"]) and feats["rise_time"] > 0:
        feats["composite_sharpness"] = feats["spread"] / feats["rise_time"]
    else:
        feats["composite_sharpness"] = 0.0

    # Slope at inflection point (steepest derivative location)
    if len(pr) >= 2:
        feats["slope_at_inflection"] = float(np.abs(dpr).max())
    else:
        feats["slope_at_inflection"] = 0.0

    return feats


def get_self_attention_selectivity(model, tokens, last_pos, n_layer, n_head,
                                     head_dim, device, batch_size=32):
    """Measure self-attention selectivity (attn at last_pos to last_pos itself)
    per (layer, head). Returns selectivity matrix [n_layer, n_head]."""
    n, T = tokens.shape
    attn_to_self = np.zeros((n_layer, n_head))
    attn_to_random = np.zeros((n_layer, n_head))
    captured = {}
    handles = []
    for L in range(n_layer):
        attn = model.transformer.h[L].attn
        def make_hook(L=L):
            def hook(module, ainputs, output):
                B, T, _ = output.shape
                C = output.shape[-1] // 3
                q, k, v = output.split(C, dim=2)
                q = q.view(B, T, n_head, head_dim).transpose(1, 2)
                k = k.view(B, T, n_head, head_dim).transpose(1, 2)
                q_last = q[:, :, -1:, :]
                scores = (q_last @ k.transpose(-2, -1)) / (head_dim ** 0.5)
                w = F.softmax(scores, dim=-1)
                captured[L] = w[:, :, 0, :].detach()
            return hook
        handles.append(attn.c_attn.register_forward_hook(make_hook()))
    rng = np.random.RandomState(0)
    avoid = {0, last_pos, last_pos - 1}
    sample = []
    for _ in range(50):
        rp = rng.randint(0, last_pos)
        if rp not in avoid:
            sample.append(rp)
    try:
        with torch.no_grad():
            attn_self_sum = torch.zeros(n_layer, n_head)
            attn_rand_sum = torch.zeros(n_layer, n_head)
            n_seen = 0
            for start in range(0, n, batch_size):
                end = min(start + batch_size, n)
                tok = tokens[start:end].to(device)
                _ = model(tok)
                B = end - start
                for L in range(n_layer):
                    w = captured[L]
                    attn_self_sum[L] += w[:, :, last_pos].sum(dim=0).cpu()
                    attn_rand_sum[L] += w[:, :, sample].mean(dim=2).sum(dim=0).cpu()
                n_seen += B
            attn_to_self = (attn_self_sum / n_seen).numpy()
            attn_to_random = (attn_rand_sum / n_seen).numpy()
    finally:
        for h in handles:
            h.remove()
    selectivity = attn_to_self / np.maximum(attn_to_random, 1e-8)
    return selectivity


def main():
    device = "mps" if torch.backends.mps.is_available() else "cpu"
    print(f"device = {device}")

    # Load PR trajectories
    spec = json.load(open(REPO / "results/induction_heads_per_head_124m.json"))
    n_layer = spec["n_layer"]
    n_head = spec["n_head"]
    steps = spec["ckpt_step"]

    print(f"Loaded PR trajectories for {n_layer * n_head} heads, {len(steps)} ckpts")

    # Compute features per head
    print("Computing trajectory features...")
    features = {}
    for L in range(n_layer):
        for H in range(n_head):
            feats = compute_features(spec["pr"][f"L{L}_H{H}"], steps)
            features[(L, H)] = feats

    # Load existing selectivity data for ground truth
    print("Loading mechinterp selectivities...")
    induction_mech = json.load(open(REPO / "results/induction_heads_mechinterp_124m.json"))
    prev_token_mech = json.load(open(REPO / "results/prev_token_mechinterp_124m.json"))

    # induction_mech only has selectivity for all 144 heads (per_head dict)
    induction_sel = {}
    for k, v in induction_mech["per_head"].items():
        # parse 'L 8H8' -> (8, 8)
        parts = k.replace(" ", "").replace("L", "").split("H")
        L, H = int(parts[0]), int(parts[1])
        induction_sel[(L, H)] = v["selectivity"]

    prev_token_sel = {}
    for k, v in prev_token_mech["selectivity_per_head"].items():
        parts = k.replace("L", "").split("H")
        L, H = int(parts[0]), int(parts[1])
        prev_token_sel[(L, H)] = v["selectivity"]

    # Need to compute self-attention selectivity for all 144 heads (we only had top-30)
    print("Computing self-attention selectivity for all 144 heads...")
    cfg = GPTConfig()
    model = GPT(cfg).to(device)
    final_ckpt = sorted(KARPATHY_DIR.glob("ckpt_*.pt"))[-1]
    load_karpathy_ckpt(model, final_ckpt, device)
    model.eval()
    rng = np.random.RandomState(42)
    tokens, _, _ = build_induction_batch(n_examples=2000, seq_len=256, rng=rng)
    last_pos = tokens.shape[1] - 1
    self_sel_matrix = get_self_attention_selectivity(model, tokens, last_pos,
                                                       cfg.n_layer, cfg.n_head,
                                                       cfg.n_embd // cfg.n_head, device)
    self_sel = {(L, H): float(self_sel_matrix[L, H])
                 for L in range(cfg.n_layer) for H in range(cfg.n_head)}

    # Ground truth: max selectivity across all classes per head
    capability_strength = {}
    is_capability_head = {}
    for L in range(n_layer):
        for H in range(n_head):
            s = max(induction_sel.get((L, H), 0),
                     prev_token_sel.get((L, H), 0),
                     self_sel.get((L, H), 0))
            capability_strength[(L, H)] = s
            is_capability_head[(L, H)] = s >= SELECTIVITY_THRESHOLD

    n_capability = sum(is_capability_head.values())
    print(f"  {n_capability} of {n_layer * n_head} heads are capability heads (max sel >= {SELECTIVITY_THRESHOLD}x)")

    # For each ranking signal, compute precision-at-k
    feature_names = ["spread", "final_pr", "max_pr", "max_rate", "mean_abs_rate",
                      "mean_post_grok", "integral", "composite_sharpness",
                      "slope_at_inflection"]
    # Higher is better for all of these (we want big PR / fast change / etc.)
    # peak_step we don't directly use as ranking
    # rise_time: lower is better — invert via 1/rise_time, but we have composite_sharpness already
    # min_pr: lower is better — we want heads that started concentrated; invert below

    # negative-ranking features (lower is better, so we flip sign)
    flip_features = ["min_pr"]

    ks = [5, 10, 15, 20, 30, 50]

    ranking_results = {}
    for fname in feature_names:
        scores = [(L, H, features[(L, H)].get(fname, float("nan")))
                   for L in range(n_layer) for H in range(n_head)]
        valid = [r for r in scores if np.isfinite(r[2])]
        valid.sort(key=lambda r: -r[2])  # higher first
        ranked_heads = [(L, H) for L, H, _ in valid]
        precision = {}
        for k in ks:
            top_k = ranked_heads[:k]
            n_real = sum(1 for h in top_k if is_capability_head.get(h, False))
            precision[k] = n_real / k if k > 0 else 0
        # Also compute mean ground-truth selectivity in top-k (a continuous quality score)
        mean_sel_at_k = {}
        for k in ks:
            top_k = ranked_heads[:k]
            sels = [capability_strength[h] for h in top_k]
            mean_sel_at_k[k] = float(np.mean(sels)) if sels else 0
        ranking_results[fname] = {
            "precision_at_k": precision,
            "mean_capability_strength_at_k": mean_sel_at_k,
            "top_5": [{"head": f"L{L}H{H}", "score": s, "selectivity": capability_strength[(L,H)]}
                      for L, H, s in valid[:5]],
        }

    # Print summary
    print(f"\n{'='*100}")
    print("Precision-at-k for each ranking signal (fraction of top-k that are real capability heads):")
    print(f"{'='*100}")
    print(f"  {'feature':<22}" + "  ".join(f"k={k:>3}" for k in ks))
    for fname in feature_names:
        row = ranking_results[fname]["precision_at_k"]
        print(f"  {fname:<22}" + "  ".join(f"{row[k]:>5.2f}" for k in ks))

    print(f"\n{'='*100}")
    print("Mean capability-strength selectivity in top-k (continuous quality, higher = better):")
    print(f"{'='*100}")
    print(f"  {'feature':<22}" + "  ".join(f"k={k:>3}" for k in ks))
    for fname in feature_names:
        row = ranking_results[fname]["mean_capability_strength_at_k"]
        print(f"  {fname:<22}" + "  ".join(f"{row[k]:>5.0f}" for k in ks))

    # Best feature per k
    print(f"\n{'='*100}")
    print("Best ranking signal at each k (by precision):")
    print(f"{'='*100}")
    for k in ks:
        best = max(feature_names,
                    key=lambda f: ranking_results[f]["precision_at_k"][k])
        print(f"  k={k}: {best} (precision={ranking_results[best]['precision_at_k'][k]:.2f}, "
              f"vs spread={ranking_results['spread']['precision_at_k'][k]:.2f})")

    # Save
    out = {
        "selectivity_threshold": SELECTIVITY_THRESHOLD,
        "n_capability_heads": n_capability,
        "n_total_heads": n_layer * n_head,
        "ranking_features": ranking_results,
        "ks": ks,
        "feature_names": feature_names,
    }
    out_json = REPO / "results/better_ranking_signal_124m.json"
    with open(out_json, "w") as f:
        json.dump(out, f, indent=2)
    print(f"\nwrote {out_json}")

    # Plot precision-at-k curves for each feature
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))

    # Precision-at-k
    ax = axes[0]
    for fname in feature_names:
        row = ranking_results[fname]["precision_at_k"]
        vals = [row[k] for k in ks]
        ls = "-" if fname != "spread" else "--"
        lw = 1.5 if fname != "spread" else 3.0
        ax.plot(ks, vals, "o-", label=fname, linestyle=ls, linewidth=lw)
    ax.set_xlabel("k (top-k by ranking signal)", fontsize=11)
    ax.set_ylabel("precision-at-k\n(fraction of top-k that are real capability heads)", fontsize=11)
    ax.set_title("Precision-at-k for alternative ranking signals\n"
                  "(spread is current baseline, dashed line)", fontsize=11)
    ax.legend(fontsize=9, loc="lower left")
    ax.grid(True, alpha=0.3)
    ax.set_ylim(0, 1.05)

    # Mean selectivity in top-k
    ax = axes[1]
    for fname in feature_names:
        row = ranking_results[fname]["mean_capability_strength_at_k"]
        vals = [row[k] for k in ks]
        ls = "-" if fname != "spread" else "--"
        lw = 1.5 if fname != "spread" else 3.0
        ax.plot(ks, vals, "o-", label=fname, linestyle=ls, linewidth=lw)
    ax.set_xlabel("k (top-k by ranking signal)", fontsize=11)
    ax.set_ylabel("mean capability strength in top-k\n(continuous score, higher = better)", fontsize=11)
    ax.set_title("Mean ground-truth selectivity in top-k", fontsize=11)
    ax.legend(fontsize=9, loc="upper right")
    ax.grid(True, alpha=0.3)
    ax.set_yscale("log")

    fig.tight_layout()
    out_png = REPO / "results/better_ranking_signal_124m.png"
    fig.savefig(out_png, dpi=130, bbox_inches="tight")
    print(f"saved {out_png}")


if __name__ == "__main__":
    main()
