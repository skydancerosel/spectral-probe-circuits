"""
capability_survey_multi_pos_124m.py

Artifact check on the top-30 capability survey: are the 14 heads classified
as 'self-attention' at query position 255 (the second-A position in the
synthetic induction batch) genuinely self-attending heads, or is "self"
just an artifact of measuring at the LAST position in a causal LM?

Test:
  For each top-30 spectral pick, measure attention from query positions
  {50, 100, 150, 200, 255} to all positions. Then for each query position
  classify the head into a position-agnostic capability class:
    - previous-token:   attn(p → p-1)
    - self:             attn(p → p)
    - first-token:      attn(p → 0)
    - local:            mean(attn(p → p-5..p-2))
  (Induction and duplicate-token are skipped — those are batch-structure
  specific to position 255.)

If a head classifies as 'self' at position 255 but as something else at
positions 50/100/150/200, the position-255 classification was an
artifact. If it consistently classifies as self across positions, the
class is real.

Output: analyses/capability_survey_multi_pos_124m.{json,png}
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

POSITION_AGNOSTIC_CLASSES = ["previous-token", "self", "first-token", "local"]
SELECTIVITY_THRESHOLD = 30.0
TOP_K_TO_CHECK = 30
QUERY_POSITIONS = [50, 100, 150, 200, 255]


def attention_at_query_position(model, tokens, query_pos, n_layer, n_head, head_dim,
                                 device, batch_size=32):
    """Return attention at the given query position to all 256 positions, per
    (layer, head). Shape [B, L, H, T]."""
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
                # Take query at query_pos
                q_qp = q[:, :, query_pos:query_pos+1, :]
                scores = (q_qp @ k.transpose(-2, -1)) / (head_dim ** 0.5)
                # Causal mask: query at query_pos can only attend to positions ≤ query_pos
                # (Already implicit in the SDPA causal computation; here we apply manually)
                mask = torch.zeros_like(scores)
                mask[:, :, :, query_pos+1:] = float("-inf")
                scores = scores + mask
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


def classify_pos_agnostic(attn, query_pos, n_layer, n_head):
    """For each (L, H), measure each position-agnostic class's mean attn
    and a uniform-other baseline. Return per-class selectivity matrix."""
    T = attn.shape[-1]
    rng = np.random.RandomState(0)
    # Baseline: random valid attention positions (≤ query_pos), excluding
    # the specific class targets
    avoid = {0, query_pos, query_pos - 1}
    avoid.update(range(query_pos - 5, query_pos - 1))  # 'local' targets
    sample = []
    for _ in range(50):
        rp = rng.randint(0, query_pos)
        if rp not in avoid:
            sample.append(rp)
    baseline = attn[:, :, :, sample].mean(dim=(0, 3)).numpy()  # [L, H]

    out_attn = {}
    out_attn["previous-token"] = attn[:, :, :, query_pos - 1].mean(dim=0).numpy()
    out_attn["self"] = attn[:, :, :, query_pos].mean(dim=0).numpy()
    out_attn["first-token"] = attn[:, :, :, 0].mean(dim=0).numpy()
    local_positions = [query_pos - k for k in range(2, 6)]
    out_attn["local"] = attn[:, :, :, local_positions].mean(dim=(0, 3)).numpy()

    selectivity = {cls: out_attn[cls] / np.maximum(baseline, 1e-8)
                    for cls in POSITION_AGNOSTIC_CLASSES}
    return selectivity, out_attn, baseline


def main():
    device = "mps" if torch.backends.mps.is_available() else "cpu"
    print(f"device = {device}")

    cfg = GPTConfig()
    model = GPT(cfg).to(device)
    final_ckpt = sorted(KARPATHY_DIR.glob("ckpt_*.pt"))[-1]
    step = load_karpathy_ckpt(model, final_ckpt, device)
    model.eval()
    print(f"loaded step={step}")

    # Use the same induction batch but ignore the planted structure;
    # for non-255 query positions the structure doesn't match anyway
    rng = np.random.RandomState(42)
    tokens, _, _ = build_induction_batch(n_examples=2000, seq_len=256,
                                          rng=rng)

    # Compute attention at each query position
    print(f"\nMeasuring attention at query positions {QUERY_POSITIONS}...")
    results_per_pos = {}
    for qp in QUERY_POSITIONS:
        print(f"  query position {qp}...")
        attn = attention_at_query_position(model, tokens, qp,
                                            cfg.n_layer, cfg.n_head,
                                            cfg.n_embd // cfg.n_head, device)
        selectivity, attn_means, baseline = classify_pos_agnostic(
            attn, qp, cfg.n_layer, cfg.n_head)
        results_per_pos[qp] = {
            "selectivity": {cls: selectivity[cls] for cls in POSITION_AGNOSTIC_CLASSES},
            "attn_means": attn_means,
            "baseline": baseline,
        }

    # Load top-30 spectral picks
    spec = json.load(open(REPO / "results/induction_heads_per_head_124m.json"))
    spread = []
    for k, v in spec["pr"].items():
        arr = np.array(v)
        L, H = int(k.split("_")[0][1:]), int(k.split("_")[1][1:])
        spread.append((L, H, float(arr.max() - arr.min())))
    spread.sort(key=lambda x: -x[2])
    top_picks = spread[:TOP_K_TO_CHECK]

    # Load original capability_survey to know each pick's classification at p=255
    orig = json.load(open(REPO / "results/capability_survey_124m.json"))
    orig_class = {(c["layer"], c["head"]): c["classification"]
                   for c in orig["classifications"]}

    # For each top pick, classify at each query position (using only the
    # position-agnostic classes)
    classifications_per_pos = []
    for rank, (L, H, sp) in enumerate(top_picks, 1):
        per_pos_class = {}
        for qp in QUERY_POSITIONS:
            sels = {cls: float(results_per_pos[qp]["selectivity"][cls][L, H])
                     for cls in POSITION_AGNOSTIC_CLASSES}
            best = max(sels.items(), key=lambda x: x[1])
            classification = best[0] if best[1] >= SELECTIVITY_THRESHOLD else "unclassified"
            per_pos_class[qp] = {"classification": classification,
                                  "best_selectivity": best[1],
                                  "all_selectivities": sels}
        classifications_per_pos.append({
            "rank": rank, "layer": L, "head": H, "spread": sp,
            "orig_class_at_p255": orig_class.get((L, H), "?"),
            "per_position": per_pos_class,
        })

    # Print summary, focusing on heads originally classified as 'self'
    print(f"\n{'='*100}")
    print(f"CONSISTENCY TABLE: top-30 spectral picks classified at each query position")
    print(f"{'='*100}")
    headers = ["rank", "head", "orig@255"] + [f"@{qp}" for qp in QUERY_POSITIONS]
    print(f"  {'rank':>4} {'head':<8} {'orig@255':<18} " + " ".join(f"{h:<14}" for h in [f'@{qp}' for qp in QUERY_POSITIONS]))
    for c in classifications_per_pos:
        row = [str(c["rank"]), f"L{c['layer']}H{c['head']}", c["orig_class_at_p255"]]
        for qp in QUERY_POSITIONS:
            row.append(c["per_position"][qp]["classification"])
        print(f"  {row[0]:>4} {row[1]:<8} {row[2]:<18} " + " ".join(f"{r:<14}" for r in row[3:]))

    # Focus: heads originally classified as 'self' at p=255 — what are they at other positions?
    print(f"\n{'='*100}")
    print("ARTIFACT CHECK: heads originally classified 'self' at p=255")
    print(f"{'='*100}")
    self_picks = [c for c in classifications_per_pos if c["orig_class_at_p255"] == "self"]
    self_at_other = 0
    for c in self_picks:
        # Count how often it classifies as self at non-255 positions
        non_255_classes = [c["per_position"][qp]["classification"] for qp in QUERY_POSITIONS if qp != 255]
        self_count = sum(1 for cls in non_255_classes if cls == "self")
        prev_count = sum(1 for cls in non_255_classes if cls == "previous-token")
        first_count = sum(1 for cls in non_255_classes if cls == "first-token")
        local_count = sum(1 for cls in non_255_classes if cls == "local")
        unclass_count = sum(1 for cls in non_255_classes if cls == "unclassified")
        is_consistent_self = (self_count >= 3)  # ≥3 of 4 non-255 positions
        if is_consistent_self:
            self_at_other += 1
        print(f"  L{c['layer']}H{c['head']:<3} (rank {c['rank']:>2}): "
              f"self={self_count}/4, prev={prev_count}/4, first={first_count}/4, "
              f"local={local_count}/4, unclass={unclass_count}/4 "
              f"{'← CONSISTENT SELF' if is_consistent_self else '← NOT CONSISTENT'}")

    print(f"\n  Of {len(self_picks)} heads originally 'self' at p=255: "
          f"{self_at_other} are CONSISTENTLY self across positions ({self_at_other/len(self_picks)*100:.0f}%)")

    # Same check for previous-token picks (sanity check)
    print(f"\n{'='*100}")
    print("SANITY CHECK: heads originally classified 'previous-token' at p=255")
    print(f"{'='*100}")
    prev_picks = [c for c in classifications_per_pos if c["orig_class_at_p255"] == "previous-token"]
    prev_at_other = 0
    for c in prev_picks:
        non_255_classes = [c["per_position"][qp]["classification"] for qp in QUERY_POSITIONS if qp != 255]
        prev_count = sum(1 for cls in non_255_classes if cls == "previous-token")
        is_consistent_prev = (prev_count >= 3)
        if is_consistent_prev:
            prev_at_other += 1
        print(f"  L{c['layer']}H{c['head']:<3} (rank {c['rank']:>2}): prev={prev_count}/4 "
              f"{'← CONSISTENT PREV' if is_consistent_prev else '← NOT CONSISTENT'}")
    print(f"\n  Of {len(prev_picks)} heads originally 'prev-token' at p=255: "
          f"{prev_at_other} are CONSISTENTLY prev across positions ({prev_at_other/len(prev_picks)*100:.0f}%)")

    # Save
    out = {
        "step": int(step),
        "query_positions": QUERY_POSITIONS,
        "selectivity_threshold": SELECTIVITY_THRESHOLD,
        "classifications_per_pos": classifications_per_pos,
        "self_artifact_summary": {
            "n_orig_self": len(self_picks),
            "n_consistent_self": self_at_other,
            "consistency_rate": self_at_other / max(len(self_picks), 1),
        },
        "prev_token_sanity_summary": {
            "n_orig_prev": len(prev_picks),
            "n_consistent_prev": prev_at_other,
            "consistency_rate": prev_at_other / max(len(prev_picks), 1),
        },
    }
    out_json = REPO / "results/capability_survey_multi_pos_124m.json"
    with open(out_json, "w") as f:
        json.dump(out, f, indent=2)
    print(f"\nwrote {out_json}")


if __name__ == "__main__":
    main()
