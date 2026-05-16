"""Null-selectivity calibration for the induction-circuit-membership threshold.

For each head, compute induction-selectivity twice:
  REAL: target = ab+1 (true induction-target position)
  NULL: target = random non-special position (induction structure destroyed)

The null is drawn N_NULLS times to give a distribution. A head's real
selectivity must exceed null_99 (or null_max) to be "doing induction
above noise." The point of this calibration: the per-model threshold falls
out of the data instead of being inherited from one model's ablation sweep.

Computationally: re-uses the existing attention-extraction pipeline, then
picks different target positions to compute real vs null selectivity.
Same forward pass, different selectors.
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
from induction_utils import build_induction_batch  # noqa: E402


def reconstruct_ab_indices(tokens, targets):
    n, T = tokens.shape
    ab_indices = torch.zeros(n, dtype=torch.long)
    for i in range(n):
        match = (tokens[i, :-1] == targets[i]).nonzero(as_tuple=True)[0]
        if len(match) == 1:
            ab_indices[i] = match[0].item() - 1
        else:
            ab_indices[i] = -1
    return ab_indices


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--family", choices=["pythia", "olmo", "olmoe"], required=True)
    ap.add_argument("--model", required=True)
    ap.add_argument("--revision", default="main")
    ap.add_argument("--out", required=True)
    ap.add_argument("--n-examples", type=int, default=2000)
    ap.add_argument("--seq-len", type=int, default=256)
    ap.add_argument("--batch-size", type=int, default=4)
    ap.add_argument("--n-nulls", type=int, default=20,
                    help="number of random-null-position draws per example")
    ap.add_argument("--null-seed", type=int, default=2026)
    ap.add_argument("--dtype", default="fp16", choices=["fp16", "fp32", "bf16"],
                    help="inference dtype; use fp32 for smaller Pythia models "
                         "(160M / 410M) that NaN in fp16 mid-layers")
    args = ap.parse_args()
    dtype_map = {"fp16": torch.float16, "fp32": torch.float32, "bf16": torch.bfloat16}
    model_dtype = dtype_map[args.dtype]

    device = "mps" if torch.backends.mps.is_available() else "cpu"
    print(f"device = {device}")

    rng = np.random.RandomState(42)
    tokens, _, targets = build_induction_batch(args.n_examples, args.seq_len, rng=rng)
    T = tokens.shape[1]
    last_pos = T - 1
    prev_pos = T - 2
    ab_indices = reconstruct_ab_indices(tokens, targets)
    valid = ab_indices >= 0
    print(f"  valid examples: {int(valid.sum().item())}/{tokens.shape[0]}")

    print(f"\nLoading {args.model}@{args.revision} (eager attention)...")
    t0 = time.time()
    if args.family == "pythia":
        from transformers import GPTNeoXForCausalLM as ModelClass
    elif args.family == "olmo":
        from transformers import OlmoForCausalLM as ModelClass
    elif args.family == "olmoe":
        from transformers import OlmoeForCausalLM as ModelClass
    model = ModelClass.from_pretrained(args.model, revision=args.revision,
                                       dtype=model_dtype,
                                       attn_implementation="eager").to(device).eval()
    cfg = model.config
    n_layer = cfg.num_hidden_layers
    n_head = cfg.num_attention_heads
    print(f"  loaded in {time.time() - t0:.0f}s  L={n_layer} H={n_head}")

    print(f"\nExtracting attentions at last query position...")
    n = tokens.shape[0]
    attn_at_last = torch.zeros(n, n_layer, n_head, T, dtype=torch.float32)
    t0 = time.time()
    with torch.no_grad():
        for start in range(0, n, args.batch_size):
            end = min(start + args.batch_size, n)
            tok = tokens[start:end].to(device)
            out = model(tok, output_attentions=True)
            for L in range(n_layer):
                attn_at_last[start:end, L] = out.attentions[L][:, :, last_pos, :].float().cpu()
            del out
            if device == "mps":
                torch.mps.empty_cache()
            if start % (args.batch_size * 25) == 0 and start > 0:
                rate = (start + args.batch_size) / (time.time() - t0)
                eta = (n - start - args.batch_size) / rate
                print(f"    {end}/{n}  ({rate:.1f} ex/s, ETA {eta:.0f}s)", flush=True)
    print(f"  attention extraction done in {time.time() - t0:.0f}s")

    # Baseline (random non-special positions, 50 of them) — same as mechinterp scripts
    rng_b = np.random.RandomState(0)
    avoid = {0, last_pos, prev_pos}
    avoid.update(range(prev_pos - 4, prev_pos))
    sample = []
    while len(sample) < 50:
        rp = rng_b.randint(1, last_pos)
        if rp not in avoid and rp not in sample:
            sample.append(rp)
    baseline = attn_at_last[:, :, :, sample].mean(dim=(0, 3)).numpy()

    # REAL induction selectivity: target = ab+1
    induction_sum = torch.zeros(n_layer, n_head, dtype=torch.float64)
    valid_examples = []
    for i in range(n):
        if not valid[i]:
            continue
        ab = ab_indices[i].item()
        if ab < 0 or ab + 1 >= T:
            continue
        induction_sum += attn_at_last[i, :, :, ab + 1].double()
        valid_examples.append((i, ab))
    n_valid = len(valid_examples)
    real_attn = (induction_sum / max(n_valid, 1)).numpy()
    real_sel = real_attn / np.maximum(baseline, 1e-8)
    print(f"\n[REAL] per-head induction-selectivity (target=ab+1):")
    print(f"  min={real_sel.min():.2f}  max={real_sel.max():.2f}  "
          f"mean={real_sel.mean():.2f}  median={np.median(real_sel):.2f}")

    # NULL distribution: random target position (not in avoid, not ab, not ab+1)
    # repeated N_NULLS times to get a distribution per head
    print(f"\n[NULL] running {args.n_nulls} random-target draws...")
    rng_n = np.random.RandomState(args.null_seed)
    null_sels = []
    for null_iter in range(args.n_nulls):
        null_sum = torch.zeros(n_layer, n_head, dtype=torch.float64)
        for i, ab in valid_examples:
            # Pick a random non-special position that isn't the induction target
            while True:
                rp = rng_n.randint(1, last_pos)
                if rp not in avoid and rp != ab + 1 and rp != ab:
                    break
            null_sum += attn_at_last[i, :, :, rp].double()
        null_attn = (null_sum / n_valid).numpy()
        ns = null_attn / np.maximum(baseline, 1e-8)
        null_sels.append(ns)
        if null_iter % 5 == 0:
            print(f"    null iter {null_iter + 1}/{args.n_nulls}: "
                  f"max={ns.max():.2f}  median={np.median(ns):.2f}", flush=True)
    null_sels = np.array(null_sels)  # [n_nulls, n_layer, n_head]
    print(f"  null distribution: max-across-all={null_sels.max():.2f}  "
          f"p99={np.quantile(null_sels, 0.99):.2f}  "
          f"p95={np.quantile(null_sels, 0.95):.2f}  "
          f"median={np.median(null_sels):.2f}")

    # Per-head null statistics
    null_per_head_max = null_sels.max(axis=0)        # max null across the n_nulls iterations per head
    null_per_head_p99 = np.quantile(null_sels, 0.99, axis=0)
    null_per_head_p95 = np.quantile(null_sels, 0.95, axis=0)
    null_per_head_median = np.median(null_sels, axis=0)

    # Global-null statistics (pool over all heads × iters)
    null_global_p99 = float(np.quantile(null_sels, 0.99))
    null_global_p95 = float(np.quantile(null_sels, 0.95))
    null_global_max = float(null_sels.max())
    null_global_median = float(np.median(null_sels))

    total = n_layer * n_head
    print(f"\n=== Threshold candidates (total heads: {total}) ===")
    for thresh_name, thresh_val in [
        ("real_sel > 1.5",        1.5),
        ("real_sel > 2",          2.0),
        ("real_sel > 3",          3.0),
        ("real_sel > 5",          5.0),
        ("real_sel > null_global_p95", null_global_p95),
        ("real_sel > null_global_p99", null_global_p99),
        ("real_sel > null_global_max", null_global_max),
        ("real_sel > 10  (current 'descriptive')", 10.0),
        ("real_sel > 30  (current class-assignment)", 30.0),
        ("real_sel > 50  (current circuit-membership)", 50.0),
        ("real_sel > 100", 100.0),
    ]:
        K = int((real_sel > thresh_val).sum())
        print(f"  {thresh_name:<50} (T={thresh_val:>7.2f}):  K = {K:>3}   "
              f"({K/total:.1%})")

    # Per-head signal vs noise (real > null_per_head_p99)
    K_above_perhead_p99 = int((real_sel > null_per_head_p99).sum())
    K_above_perhead_max = int((real_sel > null_per_head_max).sum())
    print(f"\n  Per-head:")
    print(f"  real_sel(L,H) > null(L,H) at p99: K = {K_above_perhead_p99} ({K_above_perhead_p99/total:.1%})")
    print(f"  real_sel(L,H) > null(L,H) at max: K = {K_above_perhead_max} ({K_above_perhead_max/total:.1%})")

    # Per-head ratio
    ratio = real_sel / np.maximum(null_per_head_max, 1e-8)
    print(f"\n  Per-head ratio (real / null_max):")
    print(f"    median={np.median(ratio):.2f}  p95={np.quantile(ratio,0.95):.2f}  "
          f"max={ratio.max():.2f}")
    for r in [1.0, 2.0, 3.0, 5.0, 10.0]:
        K = int((ratio > r).sum())
        print(f"    ratio > {r:>5.1f}:  K = {K:>3} ({K/total:.1%})")

    out = {
        "model": args.model,
        "revision": args.revision,
        "family": args.family,
        "n_examples": args.n_examples,
        "n_valid": n_valid,
        "n_nulls": args.n_nulls,
        "null_seed": args.null_seed,
        "n_layer": n_layer,
        "n_head_per_layer": n_head,
        "total_heads": total,
        "real_sel": real_sel.tolist(),
        "null_sels_all": null_sels.tolist(),       # [n_nulls, L, H]
        "null_per_head_max": null_per_head_max.tolist(),
        "null_per_head_p99": null_per_head_p99.tolist(),
        "null_per_head_p95": null_per_head_p95.tolist(),
        "null_global_stats": {
            "max": null_global_max,
            "p99": null_global_p99,
            "p95": null_global_p95,
            "median": null_global_median,
        },
        "real_sel_stats": {
            "max": float(real_sel.max()),
            "min": float(real_sel.min()),
            "median": float(np.median(real_sel)),
            "mean": float(real_sel.mean()),
        },
    }
    with open(args.out, "w") as f:
        json.dump(out, f, indent=2)
    print(f"\nwrote {args.out}")


if __name__ == "__main__":
    main()
