"""
induction_heads_mechinterp_124m.py

Mechanistic confirmation: at the second-A position in the induction-eval
batch, do the spectrally-identified heads attend back to position
(first-A + 1), which is where B sits and where a working induction head
should attend?

Run AFTER induction_heads_per_head_124m.py. Reads top picks from there.

Output: analyses/induction_heads_mechinterp_124m.{json,png}
"""

import json
import sys
import re
from pathlib import Path
import numpy as np
import torch
import torch.nn as nn
from torch.nn import functional as F

REPO = Path(__file__).resolve().parent
sys.path.insert(0, str(REPO))
from induction_heads_per_head_124m import (
    GPT, GPTConfig, load_karpathy_ckpt, build_induction_batch
)

KARPATHY_CKPT_DIR = REPO / "karpathy_llmc/runs/gpt2_fineweb10B"


def attention_weights_per_head(model, tokens, n_layer, n_head, head_dim, device,
                                batch_size=32):
    """Return attention weights at the LAST query position, per (layer, head).

    Shape: [B, n_layer, n_head, T] — attention from position T-1 to all T positions.
    """
    n, T = tokens.shape
    out = torch.zeros(n, n_layer, n_head, T, dtype=torch.float32)

    captured = {}

    handles = []
    for L in range(n_layer):
        attn = model.transformer.h[L].attn
        ln = model.transformer.h[L].ln_1
        # We need q, k from c_attn. Hook c_attn to capture qkv.
        def make_hook(L=L):
            def hook(module, ainputs, output):
                # output shape: [B, T, 3*C]
                B, T, _ = output.shape
                C = output.shape[-1] // 3
                q, k, v = output.split(C, dim=2)
                q = q.view(B, T, n_head, head_dim).transpose(1, 2)
                k = k.view(B, T, n_head, head_dim).transpose(1, 2)
                # Compute attention scores for last query position only
                # q_last: [B, H, 1, D]; k: [B, H, T, D]
                q_last = q[:, :, -1:, :]
                scores = (q_last @ k.transpose(-2, -1)) / (head_dim ** 0.5)
                # Causal mask: last position can attend to all positions
                attn_w = F.softmax(scores, dim=-1)
                captured[L] = attn_w[:, :, 0, :].detach()  # [B, H, T]
            return hook
        handles.append(attn.c_attn.register_forward_hook(make_hook()))

    try:
        with torch.no_grad():
            for start in range(0, n, batch_size):
                end = min(start + batch_size, n)
                tok = tokens[start:end].to(device)
                _ = model(tok)
                B = end - start
                for L in range(n_layer):
                    out[start:end, L] = captured[L].cpu()
    finally:
        for h in handles:
            h.remove()

    return out


def main():
    device = "mps" if torch.backends.mps.is_available() else "cpu"
    print(f"device = {device}")

    # Load top picks from per_head output
    per_head_json = REPO / "results/induction_heads_per_head_124m.json"
    if not per_head_json.exists():
        print(f"ERROR: missing {per_head_json}")
        print("Run induction_heads_per_head_124m.py first.")
        sys.exit(1)

    spectral = json.load(open(per_head_json))
    n_layer = spectral["n_layer"]
    n_head = spectral["n_head"]
    head_dim = spectral["head_dim"]
    steps = np.array(spectral["ckpt_step"])

    # Compute spread per head, sort, take top picks
    flat = []
    for L in range(n_layer):
        for H in range(n_head):
            arr = np.array(spectral["pr"][f"L{L}_H{H}"])
            flat.append((L, H, float(arr.min()), float(arr.max()),
                          float(arr.max() - arr.min())))
    flat.sort(key=lambda x: -x[4])
    top_picks = [(L, H) for L, H, _, _, _ in flat[:8]]
    print(f"Top 8 spectral picks by PR spread:")
    for L, H, lo, hi, sp in flat[:8]:
        print(f"  L{L}H{H}  spread={sp:.2f}  ({lo:.2f} -> {hi:.2f})")

    # Load model at final checkpoint
    cfg = GPTConfig()
    model = GPT(cfg).to(device)
    final_ckpt = sorted(KARPATHY_CKPT_DIR.glob("ckpt_*.pt"))[-1]
    step = load_karpathy_ckpt(model, final_ckpt, device)
    model.eval()
    print(f"Loaded final ckpt step={step}")

    # Build induction batch (same RNG as per_head for parity)
    rng = np.random.RandomState(42)
    tokens, positions, targets = build_induction_batch(n_examples=2000, seq_len=256,
                                                        rng=rng)

    # Find position of first A in each example: it's at position ab_idx
    # which is the position just before B (at ab_idx+1 = position with target token).
    # Reconstruct: scan tokens to find where the target token (B) appears.
    # The "induction target position" we want attention TO is the position of B,
    # i.e. the position right after first A.
    # Simpler: scan each example for the position of `targets[i]` in tokens[i, :seq_len-1].
    target_positions = torch.zeros(tokens.shape[0], dtype=torch.long)
    for i in range(tokens.shape[0]):
        # B (= targets[i]) was placed at ab_idx+1; it should appear exactly once
        # in tokens[i, :-1]
        match = (tokens[i, :-1] == targets[i]).nonzero(as_tuple=True)[0]
        if len(match) > 0:
            target_positions[i] = match[0].item()
        else:
            target_positions[i] = -1  # shouldn't happen given our construction

    valid = (target_positions >= 0)
    print(f"Valid examples (B found exactly): {valid.sum().item()} / {tokens.shape[0]}")

    # Compute attention weights
    print("Computing attention weights at last query position over batch...")
    attn_weights = attention_weights_per_head(model, tokens, n_layer, n_head,
                                                head_dim, device)
    # attn_weights: [n_examples, n_layer, n_head, seq_len]

    # For each (L, H), measure mean attention to target_positions vs random other positions
    n_examples = tokens.shape[0]
    seq_len = tokens.shape[1]
    rng_a = np.random.RandomState(0)

    results = {
        "step": int(step),
        "top_picks": [{"layer": L, "head": H} for L, H in top_picks],
        "n_examples": int(valid.sum().item()),
        "per_head": {},
    }

    for L in range(n_layer):
        for H in range(n_head):
            attn_to_target = []
            attn_to_random = []
            for i in range(n_examples):
                if not valid[i]:
                    continue
                tp = target_positions[i].item()
                # Random "other" position not equal to tp
                rp = rng_a.randint(1, seq_len - 1)
                while rp == tp:
                    rp = rng_a.randint(1, seq_len - 1)
                attn_to_target.append(attn_weights[i, L, H, tp].item())
                attn_to_random.append(attn_weights[i, L, H, rp].item())
            mean_target = float(np.mean(attn_to_target))
            mean_random = float(np.mean(attn_to_random))
            selectivity = mean_target / max(mean_random, 1e-6)
            results["per_head"][f"L{L}H{H}"] = {
                "attn_to_target_pos": mean_target,
                "attn_to_random_pos": mean_random,
                "selectivity": selectivity,
            }

    # Print top picks' mech-interp scores
    print("\nMech-interp scores for top spectral picks:")
    print(f"  {'head':<8} {'attn->B':>10} {'attn->random':>14} {'selectivity':>12}")
    for L, H in top_picks:
        r = results["per_head"][f"L{L}H{H}"]
        print(f"  L{L:>2}H{H:<3} {r['attn_to_target_pos']:>10.4f} "
              f"{r['attn_to_random_pos']:>14.4f} {r['selectivity']:>12.1f}x")

    print("\nAll heads sorted by selectivity (top 15):")
    sel_sorted = sorted(results["per_head"].items(),
                        key=lambda x: -x[1]["selectivity"])
    print(f"  {'head':<8} {'selectivity':>12}")
    for k, v in sel_sorted[:15]:
        print(f"  {k:<8} {v['selectivity']:>12.1f}x")

    out_json = REPO / "results/induction_heads_mechinterp_124m.json"
    with open(out_json, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nwrote {out_json}")


if __name__ == "__main__":
    main()
