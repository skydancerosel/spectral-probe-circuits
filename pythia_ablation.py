"""
pythia_ablation.py

Causal verification on Pythia 160M: does ablating the spectrally-identified
heads (by integral ranking) tank the induction-prediction loss more than
matched-random controls?

Conditions:
  - baseline
  - ablate top-6 spectral picks (by integral)
  - ablate matched-random 6 heads (same layers, different head indices)
  - ablate the 2 specifically-classified-as-induction heads (L8H2, L5H0)
  - ablate top-1 individual heads in spectral picks
  - upper bound: ablate all heads in the spectral-pick layers

Same induction batch as per-head + mech-interp.

Output: analyses/pythia_ablation.json
"""

import json
import sys
from pathlib import Path
import numpy as np
import torch
import torch.nn.functional as F

REPO = Path(__file__).resolve().parent
sys.path.insert(0, str(REPO))
from induction_heads_per_head_124m import build_induction_batch
from transformers import GPTNeoXForCausalLM

MODEL_NAME = "EleutherAI/pythia-160m"


def make_pre_hook(ablated_heads, n_head, head_dim):
    """Hook that zeroes specified heads in the input to attention.dense
    (which is the per-head V output before output projection)."""
    def pre_hook(_module, ainputs):
        x = ainputs[0]
        B, T, C = x.shape
        xr = x.view(B, T, n_head, head_dim).clone()
        for h in ablated_heads:
            xr[:, :, h, :] = 0.0
        return (xr.view(B, T, C),)
    return pre_hook


def evaluate_induction(model, tokens, positions, targets, device, batch_size=32):
    """For each example, compute the loss/accuracy of predicting targets[i]
    at positions[i] (which is seq_len-1, the second-A position)."""
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
            out = model(tok)
            logits = out.logits  # [B, T, V]
            B = end - start
            logit_at_pos = logits[torch.arange(B, device=device), pos]
            loss = F.cross_entropy(logit_at_pos, tgt, reduction="none")
            losses.append(loss.cpu().numpy())
            top1 = logit_at_pos.argmax(dim=-1)
            accs_top1.append((top1 == tgt).float().cpu().numpy())
            top5 = logit_at_pos.topk(5, dim=-1).indices
            accs_top5.append((top5 == tgt.unsqueeze(-1)).any(dim=-1).float().cpu().numpy())
    return {
        "loss": float(np.concatenate(losses).mean()),
        "acc_top1": float(np.concatenate(accs_top1).mean()),
        "acc_top5": float(np.concatenate(accs_top5).mean()),
    }


def run_condition(model, ablation_spec, tokens, positions, targets, device,
                   n_head, head_dim):
    handles = []
    for layer_idx, heads in ablation_spec.items():
        if not heads:
            continue
        h = model.gpt_neox.layers[layer_idx].attention.dense.register_forward_pre_hook(
            make_pre_hook(heads, n_head, head_dim)
        )
        handles.append(h)
    try:
        r = evaluate_induction(model, tokens, positions, targets, device)
    finally:
        for h in handles:
            h.remove()
    return r


def main():
    device = "mps" if torch.backends.mps.is_available() else "cpu"
    print(f"device = {device}")

    # Load Pythia 160M (final ckpt, eager attn for consistency with mechinterp)
    print("Loading Pythia 160M @ step143000...")
    model = GPTNeoXForCausalLM.from_pretrained(MODEL_NAME, revision="step143000",
                                                  attn_implementation="eager")
    model = model.to(device).eval()
    cfg = model.config
    n_layer = cfg.num_hidden_layers
    n_head = cfg.num_attention_heads
    head_dim = cfg.hidden_size // n_head

    # Same batch
    rng = np.random.RandomState(42)
    tokens, positions, targets = build_induction_batch(n_examples=2000, seq_len=256,
                                                        rng=rng)

    # Top-6 picks by INTEGRAL (from pythia_per_head)
    spec = json.load(open(REPO / "results/pythia_per_head.json"))
    steps = np.array(spec["ckpt_step"])
    integrals = []
    for L in range(n_layer):
        for H in range(n_head):
            arr = np.array(spec["pr"][f"L{L}_H{H}"])
            integ = float(np.trapz(np.maximum(arr - 1.0, 0), steps))
            integrals.append((L, H, integ))
    integrals.sort(key=lambda x: -x[2])
    top_6 = integrals[:6]
    print(f"Top-6 by integral: {[(L, H) for L, H, _ in top_6]}")

    # Build conditions
    spectral_picks_by_layer = {}
    for L, H, _ in top_6:
        spectral_picks_by_layer.setdefault(L, []).append(H)

    rng_c = np.random.RandomState(123)
    matched_random = {}
    for L, picks in spectral_picks_by_layer.items():
        eligible = [h for h in range(n_head) if h not in picks]
        matched_random[L] = sorted(rng_c.choice(eligible, size=len(picks),
                                                  replace=False).tolist())

    # Specifically-classified-as-induction heads (from mechinterp)
    mech = json.load(open(REPO / "results/pythia_mechinterp.json"))
    induction_heads = {}
    for c in mech["classifications"]:
        if c["classification"] == "induction":
            induction_heads.setdefault(c["layer"], []).append(c["head"])

    upper_layers = list(spectral_picks_by_layer.keys())
    upper_bound = {L: list(range(n_head)) for L in upper_layers}

    conditions = [("baseline", {}),
                  ("ablate_top6_spectral_by_integral", spectral_picks_by_layer),
                  ("ablate_matched_random", matched_random),
                  ("ablate_induction_only_L8H2_L5H0", induction_heads),
                  ("ablate_full_spectral_pick_layers", upper_bound)]
    # Add individual top picks
    for L, H, _ in top_6:
        conditions.append((f"ablate_L{L}H{H}_only", {L: [H]}))

    print(f"\nRunning {len(conditions)} conditions on induction batch (n=2000):")
    print(f"  {'condition':<45} {'loss':>8} {'top1':>8} {'top5':>8}")
    results = []
    for name, spec in conditions:
        r = run_condition(model, spec, tokens, positions, targets, device,
                           n_head, head_dim)
        print(f"  {name:<45} {r['loss']:>8.4f} {r['acc_top1']:>8.4f} {r['acc_top5']:>8.4f}")
        results.append({"name": name,
                         "spec": {str(k): v for k, v in spec.items()},
                         **r})

    out_json = REPO / "results/pythia_ablation.json"
    with open(out_json, "w") as f:
        json.dump({"model": MODEL_NAME, "step": 143000,
                    "spectral_picks_top6": [(L, H) for L, H, _ in top_6],
                    "matched_random": {str(k): v for k, v in matched_random.items()},
                    "induction_heads": {str(k): v for k, v in induction_heads.items()},
                    "conditions": results}, f, indent=2)
    print(f"\nwrote {out_json}")


if __name__ == "__main__":
    main()
