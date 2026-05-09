"""
prev_token_mechinterp_124m.py

Pivot from IOI (which the karpathy 124M doesn't reliably implement) to
previous-token heads — a simpler, well-characterized natural capability.

A 'previous-token head' attends from position t back to position t-1.
Found in essentially every transformer LM; emerges early.

Pipeline:
  Reuse the synthetic induction batch (random tokens with a specific
  A/B/A structure). At the last position (= 'second A' in induction
  terms, but for prev-token analysis we just care about the position
  itself), measure attention from position 255 to position 254 for
  each (layer, head). High attn(255→254) = previous-token head.

Then ask:
  - Do the top spectral picks (from induction_heads_per_head_124m.json)
    include the previous-token heads?
  - If so, the picks that were 'false positives' for induction may
    actually be true positives for prev-token.

Output: analyses/prev_token_mechinterp_124m.{json,png}
"""

import json
import sys
from pathlib import Path
import numpy as np
import torch
import torch.nn.functional as F
import math

REPO = Path(__file__).resolve().parent
sys.path.insert(0, str(REPO))
from induction_heads_per_head_124m import (
    GPT, GPTConfig, load_karpathy_ckpt, build_induction_batch
)

KARPATHY_DIR = REPO / "karpathy_llmc/runs/gpt2_fineweb10B"


def attention_at_last_position(model, tokens, n_layer, n_head, head_dim, device,
                                batch_size=32):
    """Return attention from the LAST position (T-1) to all positions, per
    (layer, head). Shape [B, n_layer, n_head, T]."""
    n, T = tokens.shape
    out = torch.zeros(n, n_layer, n_head, T, dtype=torch.float32)
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

    cfg = GPTConfig()
    model = GPT(cfg).to(device)
    final_ckpt = sorted(KARPATHY_DIR.glob("ckpt_*.pt"))[-1]
    step = load_karpathy_ckpt(model, final_ckpt, device)
    model.eval()
    print(f"Loaded step={step}")

    # Same batch as induction analysis
    rng = np.random.RandomState(42)
    tokens, positions, targets = build_induction_batch(n_examples=2000, seq_len=256,
                                                        rng=rng)

    print("Computing attention at last position over 2000-example batch...")
    attn = attention_at_last_position(model, tokens, cfg.n_layer, cfg.n_head,
                                       cfg.n_embd // cfg.n_head, device)
    # attn: [n_examples, n_layer, n_head, T]

    T = tokens.shape[1]
    last_pos = T - 1
    prev_pos = T - 2

    # Mean attention to prev-pos and to other positions (uniform baseline)
    n = tokens.shape[0]
    rng_b = np.random.RandomState(0)
    attn_to_prev = attn[:, :, :, prev_pos].mean(dim=0).numpy()  # [L, H]
    attn_to_random = np.zeros_like(attn_to_prev)
    # for each (L, H), avg over 50 random positions != prev_pos and != last_pos
    sample_positions = []
    for _ in range(50):
        rp = rng_b.randint(0, last_pos)
        if rp != prev_pos:
            sample_positions.append(rp)
    attn_to_random = attn[:, :, :, sample_positions].mean(dim=(0, 3)).numpy()
    # selectivity = attn(t → t-1) / attn(t → random other)
    selectivity = attn_to_prev / np.maximum(attn_to_random, 1e-8)

    # Print top heads by previous-token attention selectivity
    flat = []
    for L in range(cfg.n_layer):
        for H in range(cfg.n_head):
            flat.append((L, H, float(attn_to_prev[L, H]),
                          float(attn_to_random[L, H]),
                          float(selectivity[L, H])))
    flat.sort(key=lambda x: -x[4])  # sort by selectivity desc
    print(f"\nTop 15 heads by previous-token attention selectivity:")
    print(f"  {'head':<8} {'attn(t→t-1)':>13} {'attn(t→rand)':>13} {'selectivity':>12}")
    for L, H, ap, ar, sel in flat[:15]:
        print(f"  L{L:>2}H{H:<3} {ap:>13.4f} {ar:>13.4f} {sel:>12.1f}x")

    # Cross-reference with spectral picks from induction_heads_per_head
    print("\nLoading spectral picks from induction_heads_per_head_124m.json...")
    spec = json.load(open(REPO / "results/induction_heads_per_head_124m.json"))
    pr_dict = spec["pr"]
    spread = []
    for k, v in pr_dict.items():
        arr = np.array(v)
        L, H = int(k.split("_")[0][1:]), int(k.split("_")[1][1:])
        spread.append((L, H, float(arr.max() - arr.min())))
    spread.sort(key=lambda x: -x[2])
    top_spectral = spread[:8]
    top_spectral_set = {(L, H) for L, H, _ in top_spectral}

    # Also load the induction-attention selectivity from prior mechinterp
    mech = json.load(open(REPO / "results/induction_heads_mechinterp_124m.json"))

    print(f"\n{'='*78}")
    print("Cross-classification of top 8 spectral picks:")
    print(f"{'='*78}")
    print(f"  {'head':<8} {'spread':>8} {'induction_sel':>14} {'prev_token_sel':>16}  {'classification':<30}")
    for L, H, sp in top_spectral:
        ind_sel = mech["per_head"].get(f"L{L}H{H}", {}).get("selectivity", 0.0)
        pt_sel = float(selectivity[L, H])
        if pt_sel > 30:
            cls = "PREV-TOKEN HEAD"
        elif ind_sel > 30:
            cls = "INDUCTION HEAD"
        else:
            cls = "neither (other content-dep)"
        print(f"  L{L:>2}H{H:<3} {sp:>8.2f} {ind_sel:>14.1f}x {pt_sel:>16.1f}x  {cls:<30}")

    # All heads with prev-token selectivity > 30, regardless of spectral rank
    print(f"\n{'='*78}")
    print("All heads with prev-token selectivity > 30x — and where they rank by spread:")
    print(f"{'='*78}")
    print(f"  {'head':<8} {'prev_token_sel':>16} {'spread':>8} {'spread_rank':>12}")
    high_pt = [r for r in flat if r[4] > 30]
    spread_rank_lookup = {(L, H): rank+1 for rank, (L, H, _) in enumerate(spread)}
    for L, H, ap, ar, sel in high_pt:
        rank = spread_rank_lookup.get((L, H), -1)
        sp_val = next((s for ll, hh, s in spread if ll == L and hh == H), -1)
        in_top8 = " ✓ in top 8 spectral" if (L, H) in top_spectral_set else ""
        print(f"  L{L:>2}H{H:<3} {sel:>16.1f}x {sp_val:>8.2f} {rank:>12}{in_top8}")

    # Save
    out = {
        "step": int(step),
        "n_examples": n,
        "selectivity_per_head": {
            f"L{L}H{H}": {"attn_to_prev": float(attn_to_prev[L, H]),
                           "attn_to_random": float(attn_to_random[L, H]),
                           "selectivity": float(selectivity[L, H])}
            for L in range(cfg.n_layer) for H in range(cfg.n_head)
        },
        "top_15_by_prev_token_sel": [
            {"head": f"L{L}H{H}", "selectivity": float(sel)}
            for L, H, _, _, sel in flat[:15]
        ],
    }
    out_json = REPO / "results/prev_token_mechinterp_124m.json"
    with open(out_json, "w") as f:
        json.dump(out, f, indent=2)
    print(f"\nwrote {out_json}")


if __name__ == "__main__":
    main()
