"""
pythia_410m_extended_survey.py

Extends the capability survey from top-30 to top-80 on Pythia 410M
(matching the head-count-normalized ratio: 30/144 ≈ 80/384).
Also computes the PR-integral elbow for each model and the
fraction-of-identifiable-heads-at-matched-k metric.

Steps:
  1. Re-extract attention at last query position for all 384 Pythia 410M
     heads (eager attention)
  2. Compute selectivity for all 6 classes (induction, prev-token,
     duplicate-token, first-token, self, local) for all 384 heads
  3. Classify top-80 by integral
  4. Compute integral distribution + elbow per model (using cached
     per-head data where available)
  5. Report fraction-of-identifiable-heads at matched-k across models
"""

import json
import sys
from pathlib import Path
import numpy as np
import torch
import matplotlib.pyplot as plt

REPO = Path(__file__).resolve().parent
sys.path.insert(0, str(REPO))
from induction_heads_per_head_124m import build_induction_batch
from transformers import GPTNeoXForCausalLM

MODEL_NAME = "EleutherAI/pythia-410m"
TOP_N = 80
SEL_THR = 30.0


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
    print(f"  arch: {n_layer}L x {cfg.hidden_size}d x {n_head}h = {n_layer*n_head} total heads")

    rng = np.random.RandomState(42)
    tokens, _, targets = build_induction_batch(n_examples=2000, seq_len=256, rng=rng)
    T = tokens.shape[1]
    last_pos = T - 1
    prev_pos = T - 2
    ab_indices = reconstruct_ab_indices(tokens, targets)
    valid = ab_indices >= 0

    print(f"\nExtracting attention at last position for ALL {n_layer*n_head} heads...")
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
            if start % 400 == 0:
                print(f"  progress: {end}/{n}")

    print("\nComputing selectivity for all classes, all heads...")
    rng_b = np.random.RandomState(0)
    avoid = {0, last_pos, prev_pos}
    avoid.update(range(prev_pos - 4, prev_pos))
    sample = []
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

    # Compute per-head integrals
    spec = json.load(open(REPO / "results/pythia_410m_per_head.json"))
    steps = np.array(spec["ckpt_step"])
    integrals = []
    for L in range(n_layer):
        for H in range(n_head):
            arr = np.array(spec["pr"][f"L{L}_H{H}"])
            integ = float(np.trapz(np.maximum(arr - 1.0, 0), steps))
            integrals.append((L, H, integ))
    integrals.sort(key=lambda x: -x[2])

    # Classify top-N by integral
    print(f"\n{'='*100}")
    print(f"Top-{TOP_N} Pythia 410M picks by INTEGRAL, classified:")
    print(f"{'='*100}")
    classifications = []
    for rank, (L, H, integ) in enumerate(integrals[:TOP_N], 1):
        per_cls = [(cls, float(selectivity[cls][L, H])) for cls in classes]
        per_cls.sort(key=lambda x: -x[1])
        best_cls, best_sel = per_cls[0]
        sec_cls, sec_sel = per_cls[1]
        classification = best_cls if best_sel >= SEL_THR else "unclassified"
        classifications.append({
            "rank": rank, "layer": L, "head": H, "integral": integ,
            "classification": classification,
            "best_class": best_cls, "best_selectivity": best_sel,
            "second_class": sec_cls, "second_selectivity": sec_sel,
            "all_selectivities": {cls: float(selectivity[cls][L, H]) for cls in classes},
        })

    # Print summary
    print(f"  {'rank':>4} {'head':<8} {'best':<18} {'sel':>9}  {'2nd':<18} {'sel':>9}")
    for c in classifications:
        marker = "" if c["classification"] != "unclassified" else "  ←UN"
        print(f"  {c['rank']:>4} L{c['layer']:>2}H{c['head']:<3} "
              f"{c['best_class']:<18} {c['best_selectivity']:>9.1f}x  "
              f"{c['second_class']:<18} {c['second_selectivity']:>9.1f}x{marker}")

    # Precision-at-k
    print(f"\nPrecision-at-k for Pythia 410M (k extends to {TOP_N}):")
    for k in [5, 10, 15, 20, 30, 50, 80]:
        if k > len(classifications):
            continue
        cls = sum(1 for c in classifications[:k] if c["classification"] != "unclassified")
        print(f"  k={k}: {cls/k:.2f}  ({cls}/{k} classified)  "
              f"[fraction of total head pool: {cls/(n_layer*n_head)*100:.1f}%]")

    # Class breakdown
    cls_counts = {}
    for c in classifications:
        cls_counts[c["classification"]] = cls_counts.get(c["classification"], 0) + 1
    print(f"\nClass breakdown across top-{TOP_N}:")
    for k, v in sorted(cls_counts.items(), key=lambda x: -x[1]):
        print(f"  {k:<18} {v}")

    # ── ELBOW analysis ────────────────────────────────────────────────
    print(f"\n{'='*100}")
    print("PR-integral elbow analysis across models")
    print(f"{'='*100}")

    # For each model, get sorted integrals + plot the distribution
    model_data = {}
    # karpathy 124M
    karp = json.load(open(REPO / "results/induction_heads_per_head_124m.json"))
    karp_steps = np.array(karp["ckpt_step"])
    karp_integrals = []
    for L in range(karp["n_layer"]):
        for H in range(karp["n_head"]):
            arr = np.array(karp["pr"][f"L{L}_H{H}"])
            karp_integrals.append(float(np.trapz(np.maximum(arr - 1.0, 0), karp_steps)))
    karp_integrals = sorted(karp_integrals, reverse=True)
    model_data["karpathy 124M (144 heads)"] = karp_integrals

    # Pythia 160M
    pyth160 = json.load(open(REPO / "results/pythia_per_head.json"))
    pyth160_steps = np.array(pyth160["ckpt_step"])
    pyth160_integrals = []
    for L in range(pyth160["n_layer"]):
        for H in range(pyth160["n_head"]):
            arr = np.array(pyth160["pr"][f"L{L}_H{H}"])
            pyth160_integrals.append(float(np.trapz(np.maximum(arr - 1.0, 0), pyth160_steps)))
    pyth160_integrals = sorted(pyth160_integrals, reverse=True)
    model_data["Pythia 160M (144 heads)"] = pyth160_integrals

    # Pythia 410M
    pyth410_integrals = sorted([integ for _, _, integ in integrals], reverse=True)
    model_data["Pythia 410M (384 heads)"] = pyth410_integrals

    # Find elbow via knee point: maximize distance to line from first to last
    def find_elbow(values):
        v = np.array(values)
        n = len(v)
        x = np.arange(n)
        # Line from (0, v[0]) to (n-1, v[-1])
        x1, y1 = 0.0, v[0]
        x2, y2 = n - 1, v[-1]
        # Distance from each point to the line
        # |a*x_i + b*y_i + c| / sqrt(a^2+b^2) where line is a*x + b*y + c = 0
        a = y2 - y1
        b = x1 - x2
        c = x2*y1 - x1*y2
        dist = np.abs(a * x + b * v + c) / np.sqrt(a*a + b*b)
        return int(np.argmax(dist))

    print(f"\n  {'model':<28} {'elbow k':>10} {'elbow integral':>16} {'k=30 / total':>14}")
    elbow_results = {}
    for name, vals in model_data.items():
        elbow_k = find_elbow(vals)
        elbow_integral = vals[elbow_k]
        total = len(vals)
        print(f"  {name:<28} {elbow_k:>10} {elbow_integral:>16.0f} {30/total*100:>13.1f}%")
        elbow_results[name] = {"elbow_k": elbow_k,
                                "elbow_integral": elbow_integral,
                                "total_heads": total}

    # Plot integral distributions
    fig, axes = plt.subplots(1, 2, figsize=(14, 5.5))
    ax = axes[0]
    for name, vals in model_data.items():
        n = len(vals)
        ax.plot(range(n), vals, label=f"{name} (elbow @ k={elbow_results[name]['elbow_k']})")
        ek = elbow_results[name]["elbow_k"]
        ax.axvline(ek, color="gray", lw=0.5, ls="--", alpha=0.4)
        ax.scatter([ek], [vals[ek]], s=80, edgecolor="k", zorder=5)
    ax.set_xlabel("rank by PR-integral (high to low)")
    ax.set_ylabel("PR-integral")
    ax.set_yscale("log")
    ax.set_title("PR-integral distribution per model\n"
                  "(elbow point = knee of the curve, marker)", fontsize=11)
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.3)

    # Right panel: fraction of heads classified at matched k
    ax = axes[1]
    matched_ks = {"karpathy 124M (144 heads)": 30,
                   "Pythia 160M (144 heads)": 30,
                   "Pythia 410M (384 heads)": TOP_N}

    # Loaded existing data
    karp_classified = 28  # from existing capability_survey_124m
    pyth160_classified = 27  # from pythia_mechinterp top-30
    pyth410_classified = sum(1 for c in classifications if c["classification"] != "unclassified")

    fractions = {
        "karpathy 124M": (karp_classified, 144, 30),
        "Pythia 160M": (pyth160_classified, 144, 30),
        "Pythia 410M": (pyth410_classified, 384, TOP_N),
    }
    print(f"\n  {'model':<18} {'k':>4} {'classified':>11} {'total heads':>12} {'frac classified':>17}")
    for name, (cls, total, k) in fractions.items():
        frac = cls / total * 100
        print(f"  {name:<18} {k:>4} {cls:>11} {total:>12} {frac:>16.1f}%")

    names = list(fractions.keys())
    fracs = [fractions[n][0] / fractions[n][1] * 100 for n in names]
    ks = [fractions[n][2] for n in names]
    bars = ax.bar(names, fracs, color=["tab:blue", "tab:orange", "tab:green"],
                    edgecolor="k")
    ax.set_ylabel("% of total heads classified at matched-k")
    ax.set_title("Fraction of heads doing identifiable computation\n"
                  "(at head-count-matched k)", fontsize=11)
    for b, f, k in zip(bars, fracs, ks):
        ax.text(b.get_x() + b.get_width()/2, f + 0.3,
                 f"{f:.1f}%\n(k={k})", ha="center", fontsize=10)
    ax.grid(True, alpha=0.3, axis="y")
    ax.set_ylim(0, max(fracs) * 1.3)

    fig.tight_layout()
    out_png = REPO / "results/pythia_410m_extended_survey.png"
    fig.savefig(out_png, dpi=130, bbox_inches="tight")
    print(f"\nsaved {out_png}")

    # Save
    out = {
        "model": MODEL_NAME, "step": 143000,
        "top_n": TOP_N,
        "selectivity_threshold": SEL_THR,
        "classifications": classifications,
        "class_breakdown_top_n": cls_counts,
        "elbow_results": elbow_results,
        "fraction_at_matched_k": fractions,
    }
    out_json = REPO / "results/pythia_410m_extended_survey.json"
    with open(out_json, "w") as f:
        json.dump(out, f, indent=2)
    print(f"wrote {out_json}")


if __name__ == "__main__":
    main()
