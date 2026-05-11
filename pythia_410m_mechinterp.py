"""pythia_410m_mechinterp.py — same as pythia_mechinterp.py but for 410M."""
import json
import sys
from pathlib import Path
import numpy as np
import torch

REPO = Path(__file__).resolve().parent
sys.path.insert(0, str(REPO))
from induction_heads_per_head_124m import build_induction_batch
from transformers import GPTNeoXForCausalLM

MODEL_NAME = "EleutherAI/pythia-410m"
TOP_N = 30


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
    print(f"  arch: {n_layer}L x {cfg.hidden_size}d x {n_head}h (head_dim={head_dim})")

    rng = np.random.RandomState(42)
    tokens, _, targets = build_induction_batch(n_examples=2000, seq_len=256, rng=rng)
    T = tokens.shape[1]
    last_pos = T - 1
    prev_pos = T - 2
    ab_indices = reconstruct_ab_indices(tokens, targets)
    valid = ab_indices >= 0
    print(f"  valid examples: {int(valid.sum().item())}/{tokens.shape[0]}")

    # 410M with eager attention is bigger — smaller batch
    print("\nExtracting attention weights at last query position (batch_size=4 for 410M)...")
    n = tokens.shape[0]
    attn_at_last = torch.zeros(n, n_layer, n_head, T)
    batch_size = 4
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
                print(f"    progress: {end}/{n}")

    print("Computing per-class selectivity...")
    rng_b = np.random.RandomState(0)
    sample = []
    avoid = {0, last_pos, prev_pos}
    avoid.update(range(prev_pos - 4, prev_pos))
    for _ in range(50):
        rp = rng_b.randint(1, last_pos)
        if rp not in avoid:
            sample.append(rp)
    baseline = attn_at_last[:, :, :, sample].mean(dim=(0, 3)).numpy()

    cls_attn = {}
    cls_attn["previous-token"] = attn_at_last[:, :, :, prev_pos].mean(dim=0).numpy()
    cls_attn["self"] = attn_at_last[:, :, :, last_pos].mean(dim=0).numpy()
    cls_attn["first-token"] = attn_at_last[:, :, :, 0].mean(dim=0).numpy()
    local_pos = [prev_pos - k for k in range(1, 5)]
    cls_attn["local"] = attn_at_last[:, :, :, local_pos].mean(dim=(0, 3)).numpy()

    induction_sum = torch.zeros(n_layer, n_head, dtype=torch.float64)
    duplicate_sum = torch.zeros(n_layer, n_head, dtype=torch.float64)
    n_valid = 0
    for i in range(n):
        if not valid[i]:
            continue
        ab = ab_indices[i].item()
        if ab < 0 or ab + 1 >= T:
            continue
        induction_sum += attn_at_last[i, :, :, ab + 1].double()
        duplicate_sum += attn_at_last[i, :, :, ab].double()
        n_valid += 1
    cls_attn["induction"] = (induction_sum / max(n_valid, 1)).numpy()
    cls_attn["duplicate-token"] = (duplicate_sum / max(n_valid, 1)).numpy()

    classes = ["induction", "previous-token", "duplicate-token",
                "first-token", "self", "local"]
    selectivity = {cls: cls_attn[cls] / np.maximum(baseline, 1e-8) for cls in classes}

    spec = json.load(open(REPO / "results/pythia_410m_per_head.json"))
    steps = np.array(spec["ckpt_step"])
    integrals = []
    for L in range(n_layer):
        for H in range(n_head):
            arr = np.array(spec["pr"][f"L{L}_H{H}"])
            integ = float(np.trapz(np.maximum(arr - 1.0, 0), steps))
            sp = float(arr.max() - arr.min())
            integrals.append((L, H, integ, sp))
    integrals.sort(key=lambda x: -x[2])

    print(f"\n{'='*100}")
    print(f"Top-{TOP_N} Pythia 410M picks by INTEGRAL, classified:")
    print(f"{'='*100}")
    print(f"  {'rank':>4} {'head':<10} {'integral':>10}  {'best class':<18} {'sel':>10}  {'2nd class':<18} {'sel':>10}")

    classifications = []
    for rank, (L, H, integ, sp) in enumerate(integrals[:TOP_N], 1):
        per_cls = [(cls, float(selectivity[cls][L, H])) for cls in classes]
        per_cls.sort(key=lambda x: -x[1])
        best_cls, best_sel = per_cls[0]
        sec_cls, sec_sel = per_cls[1]
        classification = best_cls if best_sel >= 30.0 else "unclassified"
        classifications.append({
            "rank": rank, "layer": L, "head": H,
            "integral": integ, "spread": sp,
            "classification": classification,
            "best_class": best_cls, "best_selectivity": best_sel,
            "second_class": sec_cls, "second_selectivity": sec_sel,
            "all_selectivities": {cls: float(selectivity[cls][L, H]) for cls in classes},
        })
        marker = "" if best_sel >= 30.0 else "  ← UNCLASSIFIED"
        print(f"  {rank:>4} L{L:>2}H{H:<3}  {integ:>10.0f}  {best_cls:<18} {best_sel:>10.1f}x  {sec_cls:<18} {sec_sel:>10.1f}x{marker}")

    print(f"\nPrecision-at-k:")
    for k in [5, 10, 15, 20, 30]:
        cls = sum(1 for c in classifications[:k] if c["classification"] != "unclassified")
        print(f"  k={k}: {cls/k:.2f}  ({cls}/{k} classified)")

    cls_counts = {}
    for c in classifications:
        cls_counts[c["classification"]] = cls_counts.get(c["classification"], 0) + 1
    print(f"\nClass breakdown across top-{TOP_N}:")
    for k, v in sorted(cls_counts.items(), key=lambda x: -x[1]):
        print(f"  {k:<18} {v}")

    out = {
        "model": MODEL_NAME, "step": 143000,
        "selectivity_threshold": 30.0,
        "ranking": "integral",
        "n_examples_valid": int(valid.sum().item()),
        "classifications": classifications,
        "class_breakdown_top30": cls_counts,
    }
    out_json = REPO / "results/pythia_410m_mechinterp.json"
    with open(out_json, "w") as f:
        json.dump(out, f, indent=2)
    print(f"\nwrote {out_json}")


if __name__ == "__main__":
    main()
