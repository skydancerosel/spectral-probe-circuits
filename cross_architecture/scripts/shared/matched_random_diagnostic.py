"""Phase 2 diagnostic for matched-random N=11 release in Pythia 1B.

Tests three hypotheses for why matched-random produced 36% top-1 while
sink-only N=11 (same target count, smaller layer set) produced only 5%:

  H1 (seed-luck): matched-random's 36% was a lucky composition. Different
      random seeds should produce different magnitudes, possibly much
      smaller. Run with 8 seeds.

  H2 (L6/L11 inclusion): matched-random hit L6H4 and L11H1, both outside
      our "overlap layers" L4/5/7/10. Maybe those two heads are the
      cause. Conditions: ablate {L6H4, L11H1} alone; ablate matched-
      random minus {L6H4, L11H1}.

  H3 (L7H4 multi-class head): one of the 11 heads was a "self" head
      with strong prev-token + local + self selectivities. Conditions:
      ablate just L7H4; ablate matched-random minus L7H4.

Plus baseline + original matched-random replication.
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time

import numpy as np
import torch
import torch.nn.functional as F

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__))))
from induction_utils import build_induction_batch


def make_pre_hook(heads_in_layer, head_dim):
    lo_his = [(h * head_dim, (h + 1) * head_dim) for h in heads_in_layer]
    def pre_hook(_module, ainputs):
        x = ainputs[0].clone()
        for lo, hi in lo_his:
            x[..., lo:hi] = 0
        return (x,) + ainputs[1:]
    return pre_hook


def evaluate(model, tokens, targets, device, head_dim, spec, batch_size=4):
    handles = []
    for L, heads in spec.items():
        if heads:
            h = model.gpt_neox.layers[L].attention.dense.register_forward_pre_hook(
                make_pre_hook(heads, head_dim))
            handles.append(h)
    try:
        n = tokens.shape[0]
        last = tokens.shape[1] - 1
        correct = 0
        with torch.no_grad():
            for start in range(0, n, batch_size):
                end = min(start + batch_size, n)
                tok = tokens[start:end].to(device)
                tgt = targets[start:end].to(device)
                logits = model(tok).logits[:, last, :]
                correct += (logits.argmax(dim=-1) == tgt).sum().item()
        return {"n": n, "acc_top1": correct / n}
    finally:
        for h in handles:
            h.remove()


def sample_matched_random(seed, layers_with_count, n_layer, n_head, exclude_heads):
    """Replicate the matched-random sampling logic: for each layer, pick
    'count' random heads not in exclude_heads[layer]."""
    rng = np.random.RandomState(seed)
    spec = {}
    for L, count in layers_with_count.items():
        excluded = set(exclude_heads.get(L, []))
        eligible = [h for h in range(n_head) if h not in excluded]
        n_sample = min(count, len(eligible))
        if n_sample > 0:
            spec[L] = sorted(rng.choice(eligible, size=n_sample, replace=False).tolist())
    return spec


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="EleutherAI/pythia-1b")
    ap.add_argument("--revision", default="step143000")
    ap.add_argument("--n-examples", type=int, default=2000)
    ap.add_argument("--seq-len", type=int, default=256)
    ap.add_argument("--batch-size", type=int, default=4)
    ap.add_argument("--out", default="cross_architecture/results/matched_random_diagnostic.json")
    args = ap.parse_args()

    device = "mps" if torch.backends.mps.is_available() else "cpu"
    print(f"device = {device}")

    rng_batch = np.random.RandomState(42)
    tokens, _, targets = build_induction_batch(args.n_examples, args.seq_len, rng=rng_batch)

    print(f"Loading {args.model}@{args.revision}...")
    t0 = time.time()
    from transformers import GPTNeoXForCausalLM
    model = GPTNeoXForCausalLM.from_pretrained(args.model, revision=args.revision,
                                                dtype=torch.float16).to(device).eval()
    cfg = model.config
    n_layer, n_head = cfg.num_hidden_layers, cfg.num_attention_heads
    head_dim = cfg.hidden_size // n_head
    print(f"  L={n_layer} H={n_head} hd={head_dim}")

    # Reconstruct: original matched-random spec layers + per-layer counts
    # matches the >=10x induction screen's layer distribution from the
    # ablation_threshold_sweep_pythia_1b.json result
    # Original screen heads (induction >= 10x): need to look up
    sweep = json.load(open("cross_architecture/results/ablation_threshold_sweep_pythia_1b.json"))
    screen = next(r for r in sweep["results"]
                  if r["name"].startswith("ablate_induction_screen_>=10"))
    mr_orig = next(r for r in sweep["results"]
                   if r["name"].startswith("matched_random_>=10"))
    screen_spec = {int(L): heads for L, heads in screen["spec"].items()}
    mr_orig_spec = {int(L): heads for L, heads in mr_orig["spec"].items()}
    print(f"\n  Original ≥10x induction-screen spec: {screen_spec}")
    print(f"  Original matched-random spec (seed=123): {mr_orig_spec}")
    print(f"  Original matched-random top-1: {mr_orig['acc_top1']:.4f}")

    # Per-layer counts (mirror screen's per-layer count, for re-sampling)
    layers_with_count = {L: len(heads) for L, heads in screen_spec.items()}
    print(f"  Per-layer screen counts (mirrored by matched-random): {layers_with_count}")

    conditions = []

    # 1. baseline
    conditions.append(("baseline", {}))

    # 2. Replicate original matched-random (seed=123)
    conditions.append(("matched_random_original_seed=123", mr_orig_spec))

    # 3. H1: seed sweep — 8 fresh seeds
    for seed in [1, 7, 42, 99, 256, 314, 700, 999]:
        spec = sample_matched_random(seed, layers_with_count, n_layer, n_head, screen_spec)
        conditions.append((f"matched_random_seed={seed}", spec))

    # 4. H2: L6/L11 isolation tests (the two layers outside our overlap set)
    nonoverlap_picks = {6: [4], 11: [1]}  # L6H4 + L11H1 from the original matched-random
    conditions.append(("only_L6H4_and_L11H1", nonoverlap_picks))
    # matched-random minus L6 minus L11
    mr_minus_67_11 = {L: hs for L, hs in mr_orig_spec.items() if L not in [6, 11]}
    conditions.append(("matched_random_minus_L6_L11", mr_minus_67_11))

    # 5. H3: L7H4 isolation (the multi-class self head)
    conditions.append(("only_L7H4", {7: [4]}))
    mr_minus_l7h4 = {L: [h for h in hs if not (L == 7 and h == 4)] for L, hs in mr_orig_spec.items()}
    mr_minus_l7h4 = {L: hs for L, hs in mr_minus_l7h4.items() if hs}
    conditions.append(("matched_random_minus_L7H4", mr_minus_l7h4))

    print(f"\n=== Running {len(conditions)} conditions ===")
    print(f"  {'condition':<48} {'n':>3} {'top1':>8}")
    print("  " + "-" * 65)
    results = []
    for name, spec in conditions:
        t0 = time.time()
        r = evaluate(model, tokens, targets, device, head_dim, spec, args.batch_size)
        elapsed = time.time() - t0
        n = sum(len(v) for v in spec.values())
        print(f"  {name:<48} {n:>3} {r['acc_top1']:>8.4f}  [{elapsed:.0f}s]", flush=True)
        results.append({"name": name, "spec": {str(k): v for k, v in spec.items()},
                        "n_ablated": n, **r})
        with open(args.out, "w") as f:
            json.dump({"model": args.model, "revision": args.revision,
                       "original_screen_spec": {str(L): h for L, h in screen_spec.items()},
                       "original_mr_spec":     {str(L): h for L, h in mr_orig_spec.items()},
                       "layers_with_count":    {str(L): c for L, c in layers_with_count.items()},
                       "results": results}, f, indent=2)

    # Summary stats
    seed_results = [r for r in results if r["name"].startswith("matched_random_seed=")]
    if seed_results:
        s = [r["acc_top1"] for r in seed_results]
        print(f"\n  Seed sweep summary: n={len(s)}, mean={np.mean(s):.4f}, "
              f"std={np.std(s):.4f}, min={min(s):.4f}, max={max(s):.4f}")
    print(f"\nwrote {args.out}")


if __name__ == "__main__":
    main()
