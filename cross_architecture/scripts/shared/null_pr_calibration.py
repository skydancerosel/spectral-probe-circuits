"""Null-PR calibration for the K-cutoff hyperparameter.

For each model, compute per-head PR at the final checkpoint on a random-token
batch (no structure). Compare to the real PR-final from features.json. Heads
where real_PR > null_PR are doing something on structured input that they
don't do on random input — the principled definition of "specialized."

This gives a per-model, model-agnostic K-cutoff that does NOT require any
ablation or ground-truth labels. The "18% conserved fraction" rule becomes
a *prediction* of this method (or a counter-prediction) rather than an input.

Usage:
  python null_pr_calibration.py \
      --family {pythia|olmo|olmoe} \
      --model <HF model id> \
      --revision <revision> \
      --features-json <existing PR features JSON> \
      --out <output JSON>
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time

import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))  # noqa: E402
from induction_utils import build_induction_batch, compute_pr  # noqa: E402


def get_o_proj(model, family, layer_idx):
    if family == "pythia":
        return model.gpt_neox.layers[layer_idx].attention.dense
    return model.model.layers[layer_idx].self_attn.o_proj


def per_head_pr_at_last(model, tokens, n_layer, n_head, head_dim,
                        device, family, batch_size=4, log_prefix=""):
    """Gather per-head o_proj input at the last position across the batch,
    compute PR(L, H) on the [n_examples, head_dim] matrix."""
    n = tokens.shape[0]
    last = tokens.shape[1] - 1
    captured = {}
    out = torch.zeros(n, n_layer, n_head, head_dim, dtype=torch.float32)

    def make_hook(L):
        def hook(_module, args):
            captured[L] = args[0].detach()
        return hook

    handles = [get_o_proj(model, family, L).register_forward_pre_hook(make_hook(L))
               for L in range(n_layer)]
    try:
        with torch.no_grad():
            for start in range(0, n, batch_size):
                end = min(start + batch_size, n)
                tok = tokens[start:end].to(device)
                _ = model(tok)
                B = end - start
                for L in range(n_layer):
                    x = captured[L]
                    xr = x.view(B, x.shape[1], n_head, head_dim)
                    out[start:end, L] = xr[:, last, :, :].cpu().float()
                if start % (batch_size * 25) == 0:
                    print(f"    {log_prefix}{start + B}/{n}", flush=True)
    finally:
        for h in handles:
            h.remove()

    pr = np.zeros((n_layer, n_head))
    for L in range(n_layer):
        for H in range(n_head):
            pr[L, H] = compute_pr(out[:, L, H, :])
    return pr


def random_token_batch(n_examples, seq_len, vocab_size, seed=2026):
    rng = np.random.RandomState(seed)
    tokens = rng.randint(0, vocab_size, size=(n_examples, seq_len), dtype=np.int64)
    return torch.from_numpy(tokens)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--family", choices=["pythia", "olmo", "olmoe"], required=True)
    ap.add_argument("--model", required=True)
    ap.add_argument("--revision", default="main")
    ap.add_argument("--features-json", required=True,
                    help="existing features JSON with per-head final_pr / integral")
    ap.add_argument("--out", required=True)
    ap.add_argument("--n-examples", type=int, default=2000)
    ap.add_argument("--seq-len", type=int, default=256)
    ap.add_argument("--batch-size", type=int, default=4)
    ap.add_argument("--null-seed", type=int, default=2026)
    ap.add_argument("--also-induction", action="store_true",
                    help="Also compute PR on induction batch for sanity check vs features.json")
    args = ap.parse_args()

    device = "mps" if torch.backends.mps.is_available() else "cpu"
    print(f"device = {device}")

    print(f"Loading {args.model}@{args.revision}...")
    t0 = time.time()
    if args.family == "pythia":
        from transformers import GPTNeoXForCausalLM as ModelClass
    elif args.family == "olmo":
        from transformers import OlmoForCausalLM as ModelClass
    elif args.family == "olmoe":
        from transformers import OlmoeForCausalLM as ModelClass
    model = ModelClass.from_pretrained(args.model, revision=args.revision,
                                       dtype=torch.float16).to(device).eval()
    cfg = model.config
    n_layer = cfg.num_hidden_layers
    n_head = cfg.num_attention_heads
    head_dim = cfg.hidden_size // n_head
    vocab_size = cfg.vocab_size
    print(f"  loaded in {time.time() - t0:.0f}s  L={n_layer} H={n_head} hd={head_dim} V={vocab_size}")

    print(f"\nBuilding NULL batch: {args.n_examples} × {args.seq_len} random ints from [0, {vocab_size})")
    null_tokens = random_token_batch(args.n_examples, args.seq_len, vocab_size, args.null_seed)
    print(f"  shape: {tuple(null_tokens.shape)}")

    print("\n[NULL] Per-head PR at last position on random-token batch...")
    null_pr = per_head_pr_at_last(model, null_tokens, n_layer, n_head, head_dim,
                                   device, args.family, args.batch_size,
                                   log_prefix="null  ")
    print(f"  null_PR: min={null_pr.min():.2f} max={null_pr.max():.2f} "
          f"mean={null_pr.mean():.2f} median={np.median(null_pr):.2f}")

    # Real PR from features.json
    feat = json.load(open(args.features_json))["features"]
    real_pr = np.zeros((n_layer, n_head))
    real_integral = np.zeros((n_layer, n_head))
    for L in range(n_layer):
        for H in range(n_head):
            k = f"L{L}_H{H}"
            real_pr[L, H] = feat[k]["final_pr"]
            real_integral[L, H] = feat[k]["integral"]
    print(f"  real_PR (final, induction batch): min={real_pr.min():.2f} max={real_pr.max():.2f} "
          f"mean={real_pr.mean():.2f} median={np.median(real_pr):.2f}")

    # Optionally also re-measure real PR ourselves to confirm
    if args.also_induction:
        print("\n[REAL] Per-head PR on induction batch (sanity check vs features.json)...")
        rng = np.random.RandomState(42)
        ind_tokens, _, _ = build_induction_batch(args.n_examples, args.seq_len, rng=rng)
        real_pr_recomp = per_head_pr_at_last(model, ind_tokens, n_layer, n_head, head_dim,
                                              device, args.family, args.batch_size,
                                              log_prefix="real  ")
        print(f"  recomputed-real_PR: min={real_pr_recomp.min():.2f} "
              f"max={real_pr_recomp.max():.2f} mean={real_pr_recomp.mean():.2f}")
    else:
        real_pr_recomp = None

    # Cutoff candidates
    null_max = float(null_pr.max())
    null_99 = float(np.quantile(null_pr, 0.99))
    null_95 = float(np.quantile(null_pr, 0.95))
    null_median = float(np.median(null_pr))

    # Per-head signal: real - null (positional, same (L,H))
    delta = real_pr - null_pr

    # K candidates
    total = n_layer * n_head
    K_above_null_max = int((real_pr > null_max).sum())
    K_above_null_99 = int((real_pr > null_99).sum())
    K_above_null_95 = int((real_pr > null_95).sum())
    K_perhead_delta_pos = int((delta > 0).sum())
    K_18pct = round(0.18 * total)

    print(f"\n=== K-cutoff candidates (total heads: {total}) ===")
    print(f"  real_PR > null_max     ({null_max:.2f}):   K = {K_above_null_max}   ({K_above_null_max/total:.1%})")
    print(f"  real_PR > null_99      ({null_99:.2f}):   K = {K_above_null_99}   ({K_above_null_99/total:.1%})")
    print(f"  real_PR > null_95      ({null_95:.2f}):   K = {K_above_null_95}   ({K_above_null_95/total:.1%})")
    print(f"  real_PR(L,H) > null_PR(L,H) (per-head):   K = {K_perhead_delta_pos}   ({K_perhead_delta_pos/total:.1%})")
    print(f"  18% conserved-fraction band:               K = {K_18pct}   (18.0%)")

    out = {
        "model": args.model,
        "revision": args.revision,
        "family": args.family,
        "n_examples": args.n_examples,
        "seq_len": args.seq_len,
        "n_layer": n_layer,
        "n_head_per_layer": n_head,
        "total_heads": total,
        "null_seed": args.null_seed,
        "null_pr": null_pr.tolist(),
        "real_pr_final_from_features": real_pr.tolist(),
        "real_integral_from_features": real_integral.tolist(),
        "real_pr_recomputed": real_pr_recomp.tolist() if real_pr_recomp is not None else None,
        "null_stats": {
            "min": float(null_pr.min()),
            "max": null_max,
            "p95": null_95,
            "p99": null_99,
            "median": null_median,
            "mean": float(null_pr.mean()),
        },
        "K_candidates": {
            "above_null_max": K_above_null_max,
            "above_null_99": K_above_null_99,
            "above_null_95": K_above_null_95,
            "per_head_delta_positive": K_perhead_delta_pos,
            "conserved_fraction_18pct": K_18pct,
        },
    }
    with open(args.out, "w") as f:
        json.dump(out, f, indent=2)
    print(f"\nwrote {args.out}")


if __name__ == "__main__":
    main()
