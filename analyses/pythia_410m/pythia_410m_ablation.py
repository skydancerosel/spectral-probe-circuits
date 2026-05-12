"""pythia_410m_ablation.py — same as pythia_ablation.py but for 410M."""
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

    print("Loading Pythia 410M @ step143000...")
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

    spec = json.load(open(REPO / "results/pythia_410m_per_head.json"))
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

    spectral_picks = {}
    for L, H, _ in top_6:
        spectral_picks.setdefault(L, []).append(H)

    rng_c = np.random.RandomState(123)
    matched_random = {}
    for L, picks in spectral_picks.items():
        eligible = [h for h in range(n_head) if h not in picks]
        matched_random[L] = sorted(rng_c.choice(eligible, size=len(picks),
                                                  replace=False).tolist())

    # Induction-classified heads from mechinterp (primary class = induction)
    mech = json.load(open(REPO / "results/pythia_410m_mechinterp.json"))
    induction_heads = {}
    for c in mech["classifications"]:
        if c["classification"] == "induction":
            induction_heads.setdefault(c["layer"], []).append(c["head"])
    # Also include heads where induction is 2nd class with selectivity > 100x
    induction_heads_extended = dict(induction_heads)
    for c in mech["classifications"]:
        if c["second_class"] == "induction" and c["second_selectivity"] >= 100:
            induction_heads_extended.setdefault(c["layer"], []).append(c["head"])

    upper_bound = {L: list(range(n_head)) for L in spectral_picks.keys()}

    conditions = [
        ("baseline", {}),
        ("ablate_top6_spectral_by_integral", spectral_picks),
        ("ablate_matched_random", matched_random),
        ("ablate_induction_only (primary class)", induction_heads),
        ("ablate_induction_extended (incl 2nd-class >100x)", induction_heads_extended),
        ("ablate_full_spectral_pick_layers", upper_bound),
    ]
    for L, H, _ in top_6:
        conditions.append((f"ablate_L{L}H{H}_only", {L: [H]}))

    print(f"\nRunning {len(conditions)} conditions on induction batch (n=2000):")
    print(f"  {'condition':<54} {'loss':>8} {'top1':>8} {'top5':>8}")
    results = []
    for name, sp in conditions:
        r = run_condition(model, sp, tokens, positions, targets, device,
                           n_head, head_dim)
        print(f"  {name:<54} {r['loss']:>8.4f} {r['acc_top1']:>8.4f} {r['acc_top5']:>8.4f}")
        results.append({"name": name, "spec": {str(k): v for k, v in sp.items()}, **r})

    out_json = REPO / "results/pythia_410m_ablation.json"
    with open(out_json, "w") as f:
        json.dump({"model": MODEL_NAME, "step": 143000,
                    "spectral_picks_top6": [(L, H) for L, H, _ in top_6],
                    "matched_random": {str(k): v for k, v in matched_random.items()},
                    "induction_heads": {str(k): v for k, v in induction_heads.items()},
                    "induction_heads_extended": {str(k): v for k, v in induction_heads_extended.items()},
                    "conditions": results}, f, indent=2)
    print(f"\nwrote {out_json}")


if __name__ == "__main__":
    main()
