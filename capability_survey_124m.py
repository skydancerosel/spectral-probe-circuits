"""
capability_survey_124m.py

Cross-classify the top-30 spectral picks (by PR spread) on karpathy_llmc
GPT-2 124M into known head-capability classes. Goal: how far down the
spectral-pick ranking is every pick still a recognizable capability head?

Classes tested (each = a per-head attention-pattern measurement on the
existing 2000-example synthetic batch, taken from the LAST query position):

  1. previous-token       attn(t → t-1)
  2. induction            attn(t → ab_idx + 1)   (where B = induction-target was placed)
  3. duplicate-token      attn(t → ab_idx)        (earlier occurrence of token at t)
  4. BOS / first-token    attn(t → 0)
  5. self                 attn(t → t)
  6. local (≤5 back)      mean(attn over t-5..t-1, excluding t-1 which is prev-token)

A head is classified into the class for which its selectivity (attn-to-
target / attn-to-uniform-other) is highest, IF that selectivity > 30×.
Otherwise classified as "unclassified".

Output: analyses/capability_survey_124m.{json,png}
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

CLASSES = ["previous-token", "induction", "duplicate-token", "first-token",
            "self", "local"]
SELECTIVITY_THRESHOLD = 30.0
TOP_K_TO_SURVEY = 30


def attention_at_last_position(model, tokens, n_layer, n_head, head_dim, device,
                                batch_size=32):
    """Return attention weights at LAST query position. Shape [B, L, H, T]."""
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


def reconstruct_ab_indices(tokens, targets):
    """For each example in the induction batch, find ab_idx (the position of
    the FIRST A, == position right before B). We can recover this by finding
    where the target token (= B) appears in tokens[:, :-1]; ab_idx = that
    position - 1.
    """
    n, T = tokens.shape
    ab_indices = torch.zeros(n, dtype=torch.long)
    for i in range(n):
        match = (tokens[i, :-1] == targets[i]).nonzero(as_tuple=True)[0]
        if len(match) == 1:
            ab_indices[i] = match[0].item() - 1   # B is at ab_idx+1; A at ab_idx
        else:
            ab_indices[i] = -1
    return ab_indices


def measure_selectivity(attn, n_examples, n_layer, n_head, ab_indices,
                         valid, T, rng):
    """For each (L, H), measure mean attn to each target position class
    averaged over valid examples; also a uniform-other baseline."""
    last_pos = T - 1
    prev_pos = T - 2
    rng_b = np.random.RandomState(0)
    # Sample a few random positions for the baseline; avoid ALL the
    # specific positions we're testing
    avoid = {0, last_pos, prev_pos}
    sample = []
    for _ in range(50):
        rp = rng_b.randint(1, last_pos)
        if rp not in avoid:
            sample.append(rp)
    baseline_attn = attn[:, :, :, sample].mean(dim=(0, 3)).numpy()  # [L, H]

    out = {}
    for cls in CLASSES + ["baseline"]:
        out[cls] = np.zeros((n_layer, n_head), dtype=np.float64)

    valid_count = int(valid.sum().item())

    # previous-token: attn(t → t-1)
    out["previous-token"] = attn[:, :, :, prev_pos].mean(dim=0).numpy()

    # first-token: attn(t → 0)
    out["first-token"] = attn[:, :, :, 0].mean(dim=0).numpy()

    # self: attn(t → t)
    out["self"] = attn[:, :, :, last_pos].mean(dim=0).numpy()

    # local (mean over t-5..t-2, EXCLUDING t-1 which is prev-token)
    local_positions = [last_pos - k for k in range(2, 6)]
    out["local"] = attn[:, :, :, local_positions].mean(dim=(0, 3)).numpy()

    # induction: attn(t → ab_idx + 1) for each valid example
    # duplicate-token: attn(t → ab_idx)
    induction_sum = torch.zeros(n_layer, n_head, dtype=torch.float64)
    duplicate_sum = torch.zeros(n_layer, n_head, dtype=torch.float64)
    n_valid = 0
    for i in range(n_examples):
        if not valid[i]:
            continue
        ab = ab_indices[i].item()
        if ab < 0 or ab + 1 >= T:
            continue
        induction_sum += attn[i, :, :, ab + 1].double()
        duplicate_sum += attn[i, :, :, ab].double()
        n_valid += 1
    out["induction"] = (induction_sum / max(n_valid, 1)).numpy()
    out["duplicate-token"] = (duplicate_sum / max(n_valid, 1)).numpy()

    out["baseline"] = baseline_attn

    # Selectivity per class
    selectivity = {cls: out[cls] / np.maximum(out["baseline"], 1e-8)
                    for cls in CLASSES}
    return out, selectivity


def main():
    device = "mps" if torch.backends.mps.is_available() else "cpu"
    print(f"device = {device}")

    cfg = GPTConfig()
    model = GPT(cfg).to(device)
    final_ckpt = sorted(KARPATHY_DIR.glob("ckpt_*.pt"))[-1]
    step = load_karpathy_ckpt(model, final_ckpt, device)
    model.eval()
    print(f"loaded step={step}")

    rng = np.random.RandomState(42)
    tokens, positions, targets = build_induction_batch(n_examples=2000, seq_len=256,
                                                        rng=rng)
    T = tokens.shape[1]

    print("Reconstructing ab_indices...")
    ab_indices = reconstruct_ab_indices(tokens, targets)
    valid = ab_indices >= 0
    print(f"  valid examples: {int(valid.sum().item())}/{tokens.shape[0]}")

    print("Computing per-head attention at last position...")
    attn = attention_at_last_position(model, tokens, cfg.n_layer, cfg.n_head,
                                       cfg.n_embd // cfg.n_head, device)

    print("Computing per-class selectivity...")
    raw, selectivity = measure_selectivity(attn, tokens.shape[0],
                                             cfg.n_layer, cfg.n_head,
                                             ab_indices, valid, T, rng)

    # Load spectral picks (by PR spread)
    spec = json.load(open(REPO / "results/induction_heads_per_head_124m.json"))
    spread = []
    for k, v in spec["pr"].items():
        arr = np.array(v)
        L, H = int(k.split("_")[0][1:]), int(k.split("_")[1][1:])
        spread.append((L, H, float(arr.max() - arr.min())))
    spread.sort(key=lambda x: -x[2])

    # Classify each top-K pick
    print(f"\n{'='*88}")
    print(f"Top-{TOP_K_TO_SURVEY} spectral picks, classified by dominant capability:")
    print(f"{'='*88}")
    print(f"  {'rank':>4} {'head':<8} {'spread':>7}  {'best class':<18} {'sel':>9}  {'2nd class':<18} {'sel':>9}")

    classifications = []
    for rank, (L, H, sp) in enumerate(spread[:TOP_K_TO_SURVEY], 1):
        per_cls = [(cls, float(selectivity[cls][L, H])) for cls in CLASSES]
        per_cls.sort(key=lambda x: -x[1])
        best_cls, best_sel = per_cls[0]
        sec_cls, sec_sel = per_cls[1]
        classification = best_cls if best_sel >= SELECTIVITY_THRESHOLD else "unclassified"
        classifications.append({
            "rank": rank, "layer": L, "head": H, "spread": sp,
            "classification": classification,
            "best_class": best_cls, "best_selectivity": best_sel,
            "second_class": sec_cls, "second_selectivity": sec_sel,
            "all_selectivities": {cls: float(selectivity[cls][L, H]) for cls in CLASSES},
        })
        marker = "" if best_sel >= SELECTIVITY_THRESHOLD else "  ← UNCLASSIFIED"
        print(f"  {rank:>4} L{L:>2}H{H:<3} {sp:>7.2f}  {best_cls:<18} {best_sel:>9.1f}x  "
              f"{sec_cls:<18} {sec_sel:>9.1f}x{marker}")

    # Precision-at-k
    print(f"\n{'='*88}")
    print(f"Precision-at-k: fraction of top-k picks classified as a known capability")
    print(f"{'='*88}")
    print(f"  {'k':>4} {'precision':>10} {'classified':>11} {'unclassified':>13}")
    precision_at_k = {}
    for k in [1, 5, 10, 15, 20, 25, 30]:
        classified = sum(1 for c in classifications[:k] if c["classification"] != "unclassified")
        prec = classified / k
        precision_at_k[k] = prec
        print(f"  {k:>4} {prec:>10.2f} {classified:>11} {(k - classified):>13}")

    # Class breakdown
    print(f"\n{'='*88}")
    print(f"Class breakdown across top-{TOP_K_TO_SURVEY}:")
    print(f"{'='*88}")
    cls_counts = {}
    for c in classifications:
        cls_counts[c["classification"]] = cls_counts.get(c["classification"], 0) + 1
    for cls, cnt in sorted(cls_counts.items(), key=lambda x: -x[1]):
        print(f"  {cls:<18} {cnt}")

    # Save
    out = {
        "step": int(step),
        "n_examples_valid": int(valid.sum().item()),
        "selectivity_threshold": SELECTIVITY_THRESHOLD,
        "classifications": classifications,
        "precision_at_k": precision_at_k,
        "class_breakdown_top30": cls_counts,
    }
    out_json = REPO / "results/capability_survey_124m.json"
    with open(out_json, "w") as f:
        json.dump(out, f, indent=2)
    print(f"\nwrote {out_json}")

    # Plot: precision-at-k + class breakdown
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    ks = sorted(precision_at_k.keys())
    pks = [precision_at_k[k] for k in ks]
    axes[0].plot(ks, pks, "o-", color="tab:blue", markersize=8, lw=2)
    axes[0].set_ylim(0, 1.05)
    axes[0].set_xlabel("k (number of top spectral picks)")
    axes[0].set_ylabel("precision: fraction classified as a known capability")
    axes[0].set_title(f"Precision-at-k for spectral picks on GPT-2 124M / FineWeb-10B\n"
                       f"selectivity threshold = {SELECTIVITY_THRESHOLD}× over uniform-other baseline")
    axes[0].axhline(1.0, color="gray", lw=0.5, ls="--", alpha=0.5)
    axes[0].grid(True, alpha=0.3)
    for k, p in zip(ks, pks):
        axes[0].annotate(f"{p:.2f}", (k, p), textcoords="offset points",
                          xytext=(0, 10), ha="center", fontsize=10)

    # Class breakdown bar chart
    items = sorted(cls_counts.items(), key=lambda x: -x[1])
    labels = [it[0] for it in items]
    counts = [it[1] for it in items]
    colors = ["tab:gray" if l == "unclassified" else "tab:green" for l in labels]
    x = np.arange(len(items))
    axes[1].bar(x, counts, color=colors, edgecolor="k", linewidth=0.5)
    axes[1].set_xticks(x)
    axes[1].set_xticklabels(labels, rotation=30, ha="right", fontsize=10)
    axes[1].set_ylabel("count in top-30")
    axes[1].set_title(f"Capability-class breakdown across top-{TOP_K_TO_SURVEY} spectral picks")
    axes[1].grid(True, alpha=0.3, axis="y")
    for xi, c in enumerate(counts):
        axes[1].text(xi, c + 0.2, str(c), ha="center", fontsize=10, weight="bold")

    fig.tight_layout()
    out_png = REPO / "results/capability_survey_124m.png"
    fig.savefig(out_png, dpi=130, bbox_inches="tight")
    print(f"saved {out_png}")


if __name__ == "__main__":
    main()
