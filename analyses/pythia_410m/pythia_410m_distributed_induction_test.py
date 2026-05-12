"""
pythia_410m_distributed_induction_test.py

Tests whether the 77% induction drop on 410M (vs 95-99% on 124M / 160M)
reflects a methodologically-incomplete targeting OR a genuinely-distributed
circuit at scale.

Key new condition: ablate ALL heads with induction selectivity > 50x,
regardless of primary class. If this reaches 95%+ drop, the previous
77% drop was just incomplete targeting (extended-class condition missed
some 2nd-class contributors). If it stays at ~77-90%, induction is
genuinely distributed at 410M scale and the methodology can't fully
account for it via discrete-head ablation.

Compute induction selectivity for ALL 384 heads of Pythia 410M, find
those exceeding 50x, and ablate the entire set.
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

MODEL_NAME = "EleutherAI/pythia-410m"


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


def make_pre_hook(ablated_heads, n_head, head_dim):
    def pre_hook(_module, ainputs):
        x = ainputs[0]
        B, T, C = x.shape
        xr = x.view(B, T, n_head, head_dim).clone()
        for h in ablated_heads:
            xr[:, :, h, :] = 0.0
        return (xr.view(B, T, C),)
    return pre_hook


def evaluate_induction(model, tokens, positions, targets, device, batch_size=16):
    n = tokens.shape[0]
    losses, accs1, accs5 = [], [], []
    with torch.no_grad():
        for start in range(0, n, batch_size):
            end = min(start + batch_size, n)
            tok = tokens[start:end].to(device)
            pos = positions[start:end].to(device)
            tgt = targets[start:end].to(device)
            out = model(tok)
            B = end - start
            logit_at_pos = out.logits[torch.arange(B, device=device), pos]
            loss = F.cross_entropy(logit_at_pos, tgt, reduction="none")
            losses.append(loss.cpu().numpy())
            top1 = logit_at_pos.argmax(dim=-1)
            accs1.append((top1 == tgt).float().cpu().numpy())
            top5 = logit_at_pos.topk(5, dim=-1).indices
            accs5.append((top5 == tgt.unsqueeze(-1)).any(dim=-1).float().cpu().numpy())
            del out
    return {"loss": float(np.concatenate(losses).mean()),
            "acc_top1": float(np.concatenate(accs1).mean()),
            "acc_top5": float(np.concatenate(accs5).mean())}


def run_condition(model, ablation_spec, tokens, positions, targets, device,
                   n_head, head_dim):
    handles = []
    for layer_idx, heads in ablation_spec.items():
        if not heads:
            continue
        h = model.gpt_neox.layers[layer_idx].attention.dense.register_forward_pre_hook(
            make_pre_hook(heads, n_head, head_dim))
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

    print("Loading Pythia 410M @ step143000 (eager attention)...")
    model = GPTNeoXForCausalLM.from_pretrained(MODEL_NAME, revision="step143000",
                                                  attn_implementation="eager")
    model = model.to(device).eval()
    cfg = model.config
    n_layer = cfg.num_hidden_layers
    n_head = cfg.num_attention_heads
    head_dim = cfg.hidden_size // n_head

    rng = np.random.RandomState(42)
    tokens, positions, targets = build_induction_batch(n_examples=2000, seq_len=256,
                                                        rng=rng)
    T = tokens.shape[1]
    last_pos = T - 1
    ab_indices = reconstruct_ab_indices(tokens, targets)
    valid = ab_indices >= 0

    # Compute attention at last position for all heads (all 384)
    print("Computing attention weights for ALL 384 heads (batch_size=4)...")
    attn_at_last = torch.zeros(tokens.shape[0], n_layer, n_head, T)
    batch_size = 4
    n = tokens.shape[0]
    with torch.no_grad():
        for start in range(0, n, batch_size):
            end = min(start + batch_size, n)
            tok = tokens[start:end].to(device)
            out = model(tok, output_attentions=True)
            for L in range(n_layer):
                attn_at_last[start:end, L] = out.attentions[L][:, :, last_pos, :].cpu()
            del out
            if device == "mps":
                torch.mps.empty_cache()
            if start % 200 == 0:
                print(f"  progress: {end}/{n}")

    # Induction selectivity for all 384 heads
    rng_b = np.random.RandomState(0)
    avoid = {0, last_pos, last_pos - 1}
    sample_pos = []
    for _ in range(50):
        rp = rng_b.randint(1, last_pos)
        if rp not in avoid:
            sample_pos.append(rp)
    baseline = attn_at_last[:, :, :, sample_pos].mean(dim=(0, 3)).numpy()  # [L, H]

    induction_sum = torch.zeros(n_layer, n_head, dtype=torch.float64)
    n_valid = 0
    for i in range(n):
        if not valid[i]:
            continue
        ab = ab_indices[i].item()
        if ab < 0 or ab + 1 >= T:
            continue
        induction_sum += attn_at_last[i, :, :, ab + 1].double()
        n_valid += 1
    induction_attn = (induction_sum / max(n_valid, 1)).numpy()
    induction_sel = induction_attn / np.maximum(baseline, 1e-8)

    # Find ALL heads with induction selectivity > 50x
    threshold = 50.0
    extended_set = {}
    selectivity_records = []
    for L in range(n_layer):
        for H in range(n_head):
            sel = float(induction_sel[L, H])
            if sel >= threshold:
                extended_set.setdefault(L, []).append(H)
                selectivity_records.append({"layer": L, "head": H, "selectivity": sel})

    selectivity_records.sort(key=lambda r: -r["selectivity"])
    print(f"\nHeads with induction selectivity >= {threshold}x: {len(selectivity_records)}")
    print(f"  {'head':<8} {'selectivity':>12}")
    for r in selectivity_records[:20]:
        print(f"  L{r['layer']:>2}H{r['head']:<3} {r['selectivity']:>12.1f}x")
    if len(selectivity_records) > 20:
        print(f"  ... and {len(selectivity_records) - 20} more")

    # For comparison, also compute the previously-used "extended" set
    # (primary induction OR 2nd-class >100x from mechinterp output)
    mech = json.load(open(REPO / "results/pythia_410m_mechinterp.json"))
    prev_extended = {}
    for c in mech["classifications"]:
        if c["classification"] == "induction":
            prev_extended.setdefault(c["layer"], []).append(c["head"])
        elif c["second_class"] == "induction" and c["second_selectivity"] >= 100:
            prev_extended.setdefault(c["layer"], []).append(c["head"])

    print(f"\nPrevious 'extended' set (primary induction OR 2nd-class >100x):")
    for L, heads in prev_extended.items():
        print(f"  L{L}: {heads}")

    # Run conditions
    conditions = [
        ("baseline", {}),
        ("ablate_extended_prev (primary OR 2nd >100x)", prev_extended),
        ("ablate_ALL_induction_selective_>=50x", extended_set),
    ]
    # And same set with stricter thresholds for comparison
    for thr in [30, 100, 200]:
        s = {}
        for L in range(n_layer):
            for H in range(n_head):
                if induction_sel[L, H] >= thr:
                    s.setdefault(L, []).append(H)
        n_total = sum(len(v) for v in s.values())
        conditions.append((f"ablate_ALL_induction_selective_>={thr}x ({n_total} heads)", s))

    print(f"\n{'='*100}")
    print(f"Pythia 410M ablation conditions:")
    print(f"{'='*100}")
    print(f"  {'condition':<60} {'loss':>8} {'top1':>8} {'top5':>8}")
    results = []
    for name, sp in conditions:
        n_heads = sum(len(v) for v in sp.values()) if sp else 0
        r = run_condition(model, sp, tokens, positions, targets, device,
                           n_head, head_dim)
        print(f"  {name:<60} {r['loss']:>8.4f} {r['acc_top1']:>8.4f} {r['acc_top5']:>8.4f}")
        results.append({"name": name, "n_heads_ablated": n_heads,
                         "spec": {str(k): v for k, v in sp.items()}, **r})

    out_json = REPO / "results/pythia_410m_distributed_induction_test.json"
    with open(out_json, "w") as f:
        json.dump({"model": MODEL_NAME, "step": 143000,
                    "induction_selective_heads_ge_50x": selectivity_records,
                    "n_heads_ge_50x": len(selectivity_records),
                    "conditions": results}, f, indent=2)
    print(f"\nwrote {out_json}")


if __name__ == "__main__":
    main()
