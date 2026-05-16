"""Per-model ablation-floor sweep over induction-membership threshold.

For each model in {Pythia 1B, OLMo 1B, OLMoE 1B-7B}, ablate the all-head
induction-selectivity screen at multiple thresholds T and measure synthetic-
induction acc_top1. The ablation-floor T is the smallest threshold at which
adding lower-selectivity heads no longer drops performance further. Compare
across models to test whether the uniform 50x default (Pythia 410M ablation
floor) is still right at 1B scale.

Reuses build_induction_batch from cross_architecture/scripts/shared/induction_utils.py
and the hook/eval pattern from {pythia,olmo,olmoe}_ablation.py.

Conditions per model:
  - baseline (no ablation)
  - ablate_induction_screen_>=Tx for each T in --thresholds (head set from
    all_head_selectivity['induction'], no top-K gating)
  - matched_random_>=Tx for each T (layer-matched non-overlap, fixed seed)
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

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))  # noqa: E402
from induction_utils import build_induction_batch  # noqa: E402


def make_pre_hook(heads_in_layer, head_dim):
    lo_his = [(h * head_dim, (h + 1) * head_dim) for h in heads_in_layer]
    def pre_hook(_module, ainputs):
        x = ainputs[0].clone()
        for lo, hi in lo_his:
            x[..., lo:hi] = 0
        return (x,) + ainputs[1:]
    return pre_hook


def get_attn_o_proj(model, family, layer_idx):
    if family == "pythia":
        return model.gpt_neox.layers[layer_idx].attention.dense
    return model.model.layers[layer_idx].self_attn.o_proj


def evaluate(model, tokens, targets, device, batch_size):
    n = tokens.shape[0]
    last = tokens.shape[1] - 1
    losses, accs1, accs5 = [], [], []
    sum_logit_B = 0.0
    with torch.no_grad():
        for start in range(0, n, batch_size):
            end = min(start + batch_size, n)
            tok = tokens[start:end].to(device)
            tgt = targets[start:end].to(device)
            logits = model(tok).logits[:, last, :]
            loss = F.cross_entropy(logits, tgt, reduction="none")
            losses.append(loss.cpu().numpy())
            accs1.append((logits.argmax(dim=-1) == tgt).float().cpu().numpy())
            top5 = logits.topk(5, dim=-1).indices
            accs5.append((top5 == tgt.unsqueeze(-1)).any(dim=-1).float().cpu().numpy())
            sum_logit_B += logits.gather(-1, tgt.unsqueeze(-1)).squeeze(-1).sum().item()
    return {
        "loss": float(np.concatenate(losses).mean()),
        "acc_top1": float(np.concatenate(accs1).mean()),
        "acc_top5": float(np.concatenate(accs5).mean()),
        "mean_logit_B": float(sum_logit_B / n),
    }


def run_condition(model, spec, tokens, targets, device, head_dim, batch_size, family):
    handles = []
    for layer_idx, heads in spec.items():
        if not heads:
            continue
        h = get_attn_o_proj(model, family, layer_idx).register_forward_pre_hook(
            make_pre_hook(heads, head_dim))
        handles.append(h)
    try:
        return evaluate(model, tokens, targets, device, batch_size)
    finally:
        for h in handles:
            h.remove()


def parse_head_key(k):
    L = int(k.split("_")[0][1:])
    H = int(k.split("_")[1][1:])
    return L, H


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--family", choices=["pythia", "olmo", "olmoe"], required=True)
    ap.add_argument("--model", required=True)
    ap.add_argument("--revision", default="main")
    ap.add_argument("--mechinterp-json", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--thresholds", type=float, nargs="+",
                    default=[10, 20, 30, 50, 100, 200, 500])
    ap.add_argument("--n-examples", type=int, default=2000)
    ap.add_argument("--seq-len", type=int, default=256)
    ap.add_argument("--batch-size", type=int, default=4)
    ap.add_argument("--random-seed", type=int, default=123)
    args = ap.parse_args()

    device = "mps" if torch.backends.mps.is_available() else "cpu"
    print(f"device = {device}")

    rng = np.random.RandomState(42)
    tokens, _, targets = build_induction_batch(args.n_examples, args.seq_len, rng=rng)
    print(f"induction batch: {tuple(tokens.shape)}; query position = {tokens.shape[1] - 1}")

    print(f"Loading {args.model}@{args.revision}...")
    t0 = time.time()
    if args.family == "pythia":
        from transformers import GPTNeoXForCausalLM as ModelClass
    elif args.family == "olmo":
        from transformers import OlmoForCausalLM as ModelClass
    elif args.family == "olmoe":
        from transformers import OlmoeForCausalLM as ModelClass
    model = ModelClass.from_pretrained(args.model, revision=args.revision,
                                       dtype=torch.float16)
    model = model.to(device).eval()
    cfg = model.config
    n_layer = cfg.num_hidden_layers
    n_head = cfg.num_attention_heads
    head_dim = cfg.hidden_size // n_head
    print(f"  loaded in {time.time() - t0:.0f}s  L={n_layer} H={n_head} hd={head_dim}")

    mech = json.load(open(args.mechinterp_json))
    all_sel = mech["all_head_selectivity"]

    screen_sets = {}
    for T in args.thresholds:
        screen = {}
        for k, sels in all_sel.items():
            if sels.get("induction", 0.0) >= T:
                L, H = parse_head_key(k)
                screen.setdefault(L, []).append(H)
        n_h = sum(len(v) for v in screen.values())
        screen_sets[T] = (screen, n_h)
        print(f"  T={T:>5}x : {n_h:>3} heads at induction >= {T}x")

    rng_c = np.random.RandomState(args.random_seed)
    matched_random_sets = {}
    for T, (screen, n_h) in screen_sets.items():
        mr = {}
        for L, picks in screen.items():
            eligible = [h for h in range(n_head) if h not in picks]
            n_sample = min(len(picks), len(eligible))
            mr[L] = sorted(rng_c.choice(eligible, size=n_sample, replace=False).tolist())
        matched_random_sets[T] = mr

    conditions = [("baseline", {})]
    seen_sizes = set()
    for T in args.thresholds:
        screen, n_h = screen_sets[T]
        if n_h == 0:
            continue
        # Skip if a smaller threshold already produced the same head count
        # (would just duplicate the row); but always include the highest
        # threshold for visibility.
        sig = (n_h, tuple(sorted((L, tuple(sorted(Hs))) for L, Hs in screen.items())))
        if sig in seen_sizes:
            print(f"  [skip] T={T:>5}x: same head set as previous threshold")
            continue
        seen_sizes.add(sig)
        conditions.append((f"ablate_induction_screen_>={T:.0f}x ({n_h}h)", screen))
        conditions.append((f"matched_random_>={T:.0f}x ({n_h}h)", matched_random_sets[T]))

    print(f"\nRunning {len(conditions)} conditions (n={args.n_examples}, batch={args.batch_size}):")
    print(f"  {'condition':<58} {'n_h':>4} {'loss':>8} {'top1':>8} {'top5':>8} {'logitB':>9}")
    print("  " + "-" * 95)

    results = []
    for name, spec in conditions:
        t0 = time.time()
        r = run_condition(model, spec, tokens, targets, device, head_dim,
                          args.batch_size, args.family)
        elapsed = time.time() - t0
        n_h = sum(len(v) for v in spec.values())
        print(f"  {name:<58} {n_h:>4} {r['loss']:>8.4f} {r['acc_top1']:>8.4f} "
              f"{r['acc_top5']:>8.4f} {r['mean_logit_B']:>+9.3f}  [{elapsed:.0f}s]",
              flush=True)
        results.append({"name": name,
                        "spec": {str(k): v for k, v in spec.items()},
                        "n_ablated": n_h,
                        **r})
        with open(args.out, "w") as f:
            json.dump({"model": args.model, "revision": args.revision,
                       "family": args.family,
                       "n_examples": args.n_examples,
                       "thresholds": args.thresholds,
                       "screen_set_sizes": {f"{T:.0f}": n for T, (_, n) in screen_sets.items()},
                       "results": results}, f, indent=2)

    print(f"\nwrote {args.out}")


if __name__ == "__main__":
    main()
