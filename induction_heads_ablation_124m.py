"""
induction_heads_ablation_124m.py

Causal verification: does ablating the spectrally-identified heads in
karpathy_llmc GPT-2 124M tank the induction-prediction loss more than
matched controls?

Run AFTER induction_heads_per_head_124m.py.

Output: analyses/induction_heads_ablation_124m.json
"""

import json
import sys
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


def make_pre_hook(ablated_heads, n_head, head_dim):
    """Hook that zeroes specified heads in c_proj's input — same as TS-51M
    ablation: per-head attention output before c_proj projection."""
    def pre_hook(_module, ainputs):
        x = ainputs[0]
        B, T, C = x.shape
        xr = x.view(B, T, n_head, head_dim).clone()
        for h in ablated_heads:
            xr[:, :, h, :] = 0.0
        return (xr.view(B, T, C),)
    return pre_hook


def evaluate_induction_loss(model, tokens, positions, targets, device, batch_size=64):
    """For each example, compute the loss of predicting `targets[i]` at position
    `positions[i]+1`-th-prediction (i.e., logits at positions[i] should predict targets[i+1] but
    here we want logits at the second-A position to predict B = targets[i]).

    Wait: in our induction batch, second A is at position seq_len-1 (= positions[i]).
    The logits at position seq_len-1 predict the NEXT token, which doesn't exist
    in a length-seq_len input. So we need positions[i] to be seq_len-1 and we
    measure logit at that position predicting targets[i].

    That is, we want: at the second-A position, the model should predict B.
    In autoregressive terms, this is: given the prefix ending in A, predict B.
    So the logit-at-position-(seq_len-1) is the one we care about.
    """
    n = tokens.shape[0]
    losses = []
    accs_top1 = []
    accs_top5 = []
    with torch.no_grad():
        for start in range(0, n, batch_size):
            end = min(start + batch_size, n)
            tok = tokens[start:end].to(device)
            pos = positions[start:end].to(device)
            tgt = targets[start:end].to(device)
            logits = model(tok)  # [B, T, V]
            B = end - start
            logit_at_pos = logits[torch.arange(B, device=device), pos]  # [B, V]
            loss = F.cross_entropy(logit_at_pos, tgt, reduction="none")  # [B]
            losses.append(loss.cpu().numpy())
            top1 = logit_at_pos.argmax(dim=-1)
            accs_top1.append((top1 == tgt).float().cpu().numpy())
            top5 = logit_at_pos.topk(5, dim=-1).indices  # [B, 5]
            accs_top5.append((top5 == tgt.unsqueeze(-1)).any(dim=-1).float().cpu().numpy())
    return {
        "loss": float(np.concatenate(losses).mean()),
        "acc_top1": float(np.concatenate(accs_top1).mean()),
        "acc_top5": float(np.concatenate(accs_top5).mean()),
    }


def run_condition(model, ablation_spec, tokens, positions, targets, device,
                   n_head, head_dim):
    """ablation_spec: dict {layer_idx: [head_indices]}.
       Empty dict = baseline."""
    handles = []
    for layer_idx, heads in ablation_spec.items():
        if not heads:
            continue
        h = model.transformer.h[layer_idx].attn.c_proj.register_forward_pre_hook(
            make_pre_hook(heads, n_head, head_dim)
        )
        handles.append(h)
    try:
        r = evaluate_induction_loss(model, tokens, positions, targets, device)
    finally:
        for h in handles:
            h.remove()
    return r


def main():
    device = "mps" if torch.backends.mps.is_available() else "cpu"
    print(f"device = {device}")

    # Load top picks from per_head output
    per_head_json = REPO / "results/induction_heads_per_head_124m.json"
    if not per_head_json.exists():
        print(f"ERROR: missing {per_head_json}")
        sys.exit(1)
    spectral = json.load(open(per_head_json))
    n_layer = spectral["n_layer"]
    n_head = spectral["n_head"]
    head_dim = spectral["head_dim"]

    # Get top picks by spread
    flat = []
    for L in range(n_layer):
        for H in range(n_head):
            arr = np.array(spectral["pr"][f"L{L}_H{H}"])
            flat.append((L, H, float(arr.max() - arr.min())))
    flat.sort(key=lambda x: -x[2])
    top_picks = flat[:6]
    spectral_picks_by_layer = {}
    for L, H, _ in top_picks:
        spectral_picks_by_layer.setdefault(L, []).append(H)

    print(f"Top 6 spectral picks (by PR spread): "
          f"{[(L, H) for L, H, _ in top_picks]}")

    # Load model at final checkpoint
    cfg = GPTConfig()
    model = GPT(cfg).to(device)
    final_ckpt = sorted(KARPATHY_CKPT_DIR.glob("ckpt_*.pt"))[-1]
    step = load_karpathy_ckpt(model, final_ckpt, device)
    model.eval()
    print(f"Loaded final ckpt step={step}")

    # Build induction batch
    rng = np.random.RandomState(42)
    tokens, positions, targets = build_induction_batch(n_examples=2000, seq_len=256,
                                                        rng=rng)

    # Conditions: spectral picks, individual heads, controls
    rng_c = np.random.RandomState(123)
    # Build matched-random control: same number of heads, same layers, but different head indices
    matched_random = {}
    for L, picks in spectral_picks_by_layer.items():
        eligible = [h for h in range(n_head) if h not in picks]
        matched_random[L] = sorted(rng_c.choice(eligible, size=len(picks), replace=False).tolist())

    conditions = [("baseline", {}),
                  ("ablate_spectral_picks", spectral_picks_by_layer),
                  ("ablate_matched_random", matched_random)]
    # Add individual head ablations
    for L, H, _ in top_picks:
        conditions.append((f"ablate_L{L}H{H}", {L: [H]}))
    # Upper bound: ablate all heads in spectral-pick layers
    upper = {L: list(range(n_head)) for L in spectral_picks_by_layer}
    conditions.append(("ablate_full_spectral_layers", upper))

    print(f"\nRunning {len(conditions)} conditions on induction batch (n=2000):")
    print(f"  {'condition':<35} {'loss':>8} {'top1':>8} {'top5':>8}")
    results = []
    for name, spec in conditions:
        r = run_condition(model, spec, tokens, positions, targets, device,
                           n_head, head_dim)
        print(f"  {name:<35} {r['loss']:>8.4f} {r['acc_top1']:>8.4f} {r['acc_top5']:>8.4f}")
        results.append({"name": name, "spec": {str(k): v for k, v in spec.items()}, **r})

    out_json = REPO / "results/induction_heads_ablation_124m.json"
    with open(out_json, "w") as f:
        json.dump({"step": int(step),
                    "spectral_picks": [(L, H) for L, H, _ in top_picks],
                    "matched_random": {str(k): v for k, v in matched_random.items()},
                    "conditions": results}, f, indent=2)
    print(f"\nwrote {out_json}")


if __name__ == "__main__":
    main()
