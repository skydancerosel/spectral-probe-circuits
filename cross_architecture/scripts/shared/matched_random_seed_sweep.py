"""Seed-sweep matched-random ablation for OLMo / OLMoE (generalizing the
Pythia 1B diagnostic). Tests whether the single-seed matched-random results
reported in Paper 1 §6.x are seed-stable or seed-luck.

For each model, mirror the original matched-random sampling logic:
  - Use per-layer counts equal to the ≥10× induction-screen distribution
  - Sample matched-random with 8 fresh seeds, plus the original seed=123
  - Report distribution; the original number's quantile within the
    distribution tells us whether to trust it as a point estimate.
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


def get_o_proj(model, family, L):
    if family == "pythia":
        return model.gpt_neox.layers[L].attention.dense
    return model.model.layers[L].self_attn.o_proj


def make_pre_hook(heads, head_dim):
    lo_his = [(h * head_dim, (h + 1) * head_dim) for h in heads]
    def pre_hook(_module, ainputs):
        x = ainputs[0].clone()
        for lo, hi in lo_his:
            x[..., lo:hi] = 0
        return (x,) + ainputs[1:]
    return pre_hook


def evaluate(model, tokens, targets, device, family, head_dim, spec, batch_size=4):
    handles = []
    for L, hs in spec.items():
        if hs:
            h = get_o_proj(model, family, L).register_forward_pre_hook(
                make_pre_hook(hs, head_dim))
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


def sample_mr(seed, layers_with_count, n_layer, n_head, exclude_heads):
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
    ap.add_argument("--family", choices=["pythia", "olmo", "olmoe"], required=True)
    ap.add_argument("--model", required=True)
    ap.add_argument("--revision", default="main")
    ap.add_argument("--ablation-sweep-json", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--n-examples", type=int, default=2000)
    ap.add_argument("--seq-len", type=int, default=256)
    ap.add_argument("--batch-size", type=int, default=4)
    ap.add_argument("--screen-threshold", type=int, default=10,
                    help="Use the >=Nx induction screen layer composition")
    args = ap.parse_args()

    device = "mps" if torch.backends.mps.is_available() else "cpu"
    print(f"device = {device}")

    rng = np.random.RandomState(42)
    tokens, _, targets = build_induction_batch(args.n_examples, args.seq_len, rng=rng)

    print(f"Loading {args.model}@{args.revision}...")
    t0 = time.time()
    if args.family == "pythia":
        from transformers import GPTNeoXForCausalLM as M
    elif args.family == "olmo":
        from transformers import OlmoForCausalLM as M
    else:
        from transformers import OlmoeForCausalLM as M
    model = M.from_pretrained(args.model, revision=args.revision,
                              dtype=torch.float16).to(device).eval()
    cfg = model.config
    n_layer, n_head = cfg.num_hidden_layers, cfg.num_attention_heads
    head_dim = cfg.hidden_size // n_head
    print(f"  L={n_layer} H={n_head} hd={head_dim} (loaded in {time.time()-t0:.0f}s)")

    sweep = json.load(open(args.ablation_sweep_json))
    screen = next(r for r in sweep["results"]
                  if r["name"].startswith(f"ablate_induction_screen_>={args.screen_threshold}"))
    mr_orig = next(r for r in sweep["results"]
                   if r["name"].startswith(f"matched_random_>={args.screen_threshold}"))
    screen_spec = {int(L): hs for L, hs in screen["spec"].items()}
    mr_orig_spec = {int(L): hs for L, hs in mr_orig["spec"].items()}
    layers_with_count = {L: len(hs) for L, hs in screen_spec.items()}

    print(f"  screen >=10x spec: {screen_spec}")
    print(f"  original MR spec (seed=123): {mr_orig_spec}; reported top-1={mr_orig['acc_top1']:.4f}")
    print(f"  per-layer screen counts: {layers_with_count}")

    conditions = [
        ("baseline", {}),
        ("matched_random_original_seed=123", mr_orig_spec),
    ]
    for seed in [1, 7, 42, 99, 256, 314, 700, 999]:
        spec = sample_mr(seed, layers_with_count, n_layer, n_head, screen_spec)
        conditions.append((f"matched_random_seed={seed}", spec))

    print(f"\n=== Running {len(conditions)} conditions ===")
    print(f"  {'condition':<44} {'n':>3} {'top1':>8}")
    print("  " + "-" * 60)
    results = []
    for name, spec in conditions:
        t0 = time.time()
        r = evaluate(model, tokens, targets, device, args.family, head_dim, spec, args.batch_size)
        elapsed = time.time() - t0
        n = sum(len(v) for v in spec.values())
        print(f"  {name:<44} {n:>3} {r['acc_top1']:>8.4f}  [{elapsed:.0f}s]", flush=True)
        results.append({"name": name, "spec": {str(k): v for k, v in spec.items()},
                        "n_ablated": n, **r})
        with open(args.out, "w") as f:
            json.dump({"model": args.model, "revision": args.revision,
                       "family": args.family,
                       "screen_spec": {str(L): h for L, h in screen_spec.items()},
                       "original_mr_spec": {str(L): h for L, h in mr_orig_spec.items()},
                       "layers_with_count": {str(L): c for L, c in layers_with_count.items()},
                       "results": results}, f, indent=2)

    seed_results = [r for r in results if r["name"].startswith("matched_random_seed=")]
    s = [r["acc_top1"] for r in seed_results]
    base = next(r["acc_top1"] for r in results if r["name"] == "baseline")
    orig = next(r["acc_top1"] for r in results if "original" in r["name"])
    print(f"\n  Summary:")
    print(f"    baseline:               {base:.4f}")
    print(f"    original seed=123:      {orig:.4f}")
    print(f"    8-seed sweep: mean={np.mean(s):.4f}  std={np.std(s):.4f}  "
          f"min={min(s):.4f}  max={max(s):.4f}")
    orig_quantile = float((np.array(s) <= orig).mean())
    print(f"    original quantile in seed dist: {orig_quantile:.2f}")
    print(f"\nwrote {args.out}")


if __name__ == "__main__":
    main()
