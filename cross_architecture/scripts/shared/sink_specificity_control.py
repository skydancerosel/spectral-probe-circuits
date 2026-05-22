"""Sink-specificity control for the Paper 1 §6.3 BOS-suppression finding.

The matched-random release (4.05% → 36.05% at N=11 in overlap layers L4,L5,L7,L10)
could be either:
  (a) sink-specific: pure-sink heads suppress induction; removing them releases it.
  (b) layer-perturbation artifact: any heads removed from those layers release induction.

This script runs 4 additional ablation conditions on Pythia 1B to distinguish (a) vs (b):
  (3) sink-only N=11 in overlap layers L4,L5,L7,L10
  (4) sink-only N=4 in overlap layers (small-N sink baseline)
  (5) other-only N=4 in overlap layers (small-N non-sink baseline)
  (6) sink-only N=11 in non-overlap sink-rich layers L11..L14

Comparisons:
  (4) vs (5) at matched N=4: does sink-only specifically release?
  (3) vs matched-random N=11: do sinks ALONE explain the matched-random release?
  (3) vs (6): is the release layer-localized?

Output: cross_architecture/results/sink_specificity_control_pythia_1b.json
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


SINK_THRESHOLD = 30.0
INDUCTION_THRESHOLD = 30.0
OVERLAP_LAYERS = [4, 5, 7, 10]
NON_OVERLAP_LAYERS = [11, 12, 13, 14]


def make_pre_hook(heads_in_layer, head_dim):
    lo_his = [(h * head_dim, (h + 1) * head_dim) for h in heads_in_layer]
    def pre_hook(_module, ainputs):
        x = ainputs[0].clone()
        for lo, hi in lo_his:
            x[..., lo:hi] = 0
        return (x,) + ainputs[1:]
    return pre_hook


def evaluate(model, tokens, targets, device, head_dim, ablation_spec, batch_size=4):
    handles = []
    for layer_idx, heads in ablation_spec.items():
        if heads:
            h = model.gpt_neox.layers[layer_idx].attention.dense.register_forward_pre_hook(
                make_pre_hook(heads, head_dim))
            handles.append(h)
    try:
        n = tokens.shape[0]
        last = tokens.shape[1] - 1
        correct = 0
        top5_correct = 0
        logit_B_sum = 0.0
        loss_sum = 0.0
        with torch.no_grad():
            for start in range(0, n, batch_size):
                end = min(start + batch_size, n)
                tok = tokens[start:end].to(device)
                tgt = targets[start:end].to(device)
                logits = model(tok).logits[:, last, :]
                top1 = logits.argmax(dim=-1)
                top5 = logits.topk(5, dim=-1).indices
                correct += (top1 == tgt).sum().item()
                top5_correct += (top5 == tgt.unsqueeze(-1)).any(dim=-1).sum().item()
                logit_B_sum += logits.gather(-1, tgt.unsqueeze(-1)).squeeze(-1).sum().item()
                loss_sum += F.cross_entropy(logits, tgt, reduction="sum").item()
        return {"n": n, "acc_top1": correct / n, "acc_top5": top5_correct / n,
                "mean_logit_B": logit_B_sum / n, "loss": loss_sum / n}
    finally:
        for h in handles:
            h.remove()


def classify_heads(mech, n_layer, n_head):
    ahs = mech["all_head_selectivity"]
    cls = {}
    for L in range(n_layer):
        for H in range(n_head):
            v = ahs[f"L{L}_H{H}"]
            is_sink = v.get("first-token", 0) >= SINK_THRESHOLD
            is_ind  = v.get("induction", 0)   >= INDUCTION_THRESHOLD
            if is_sink and is_ind:   c = "multirole"
            elif is_sink:            c = "sink"
            elif is_ind:             c = "induction"
            else:                    c = "other"
            cls[(L, H)] = (c, v.get("first-token", 0), v.get("induction", 0))
    return cls


def pick_heads(cls_filter, layers, classification, n_target, sort_by="first-token"):
    pool = []
    for (L, H), (cls, ft, ind) in classification.items():
        if L not in layers: continue
        if cls != cls_filter: continue
        score = ft if sort_by == "first-token" else ind
        pool.append((L, H, cls, score))
    pool.sort(key=lambda x: -x[3])
    selected = pool[:n_target]
    spec = {}
    for L, H, _, _ in selected:
        spec.setdefault(L, []).append(H)
    return spec, selected


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="EleutherAI/pythia-1b")
    ap.add_argument("--revision", default="step143000")
    ap.add_argument("--mechinterp-json", default="cross_architecture/results/pythia_1b_mechinterp.json")
    ap.add_argument("--n-examples", type=int, default=2000)
    ap.add_argument("--seq-len", type=int, default=256)
    ap.add_argument("--batch-size", type=int, default=4)
    ap.add_argument("--out", default="cross_architecture/results/sink_specificity_control_pythia_1b.json")
    args = ap.parse_args()

    device = "mps" if torch.backends.mps.is_available() else "cpu"
    print(f"device = {device}")

    rng = np.random.RandomState(42)
    tokens, _, targets = build_induction_batch(args.n_examples, args.seq_len, rng=rng)
    print(f"  batch: {tuple(tokens.shape)}")

    print(f"Loading {args.model}@{args.revision}...")
    t0 = time.time()
    from transformers import GPTNeoXForCausalLM
    model = GPTNeoXForCausalLM.from_pretrained(args.model, revision=args.revision,
                                                dtype=torch.float16).to(device).eval()
    cfg = model.config
    n_layer = cfg.num_hidden_layers
    n_head = cfg.num_attention_heads
    head_dim = cfg.hidden_size // n_head
    print(f"  loaded in {time.time() - t0:.0f}s; L={n_layer} H={n_head} hd={head_dim}")

    mech = json.load(open(args.mechinterp_json))
    classification = classify_heads(mech, n_layer, n_head)

    sink_11_overlap, sel_sink_11 = pick_heads("sink", OVERLAP_LAYERS, classification, 11)
    sink_4_overlap,  sel_sink_4  = pick_heads("sink", OVERLAP_LAYERS, classification, 4)
    other_4_overlap, sel_other_4 = pick_heads("other", OVERLAP_LAYERS, classification, 4)
    sink_11_no,      sel_sink_no = pick_heads("sink", NON_OVERLAP_LAYERS, classification, 11)

    conditions = [
        ("baseline", {}),
        (f"sink_only_N11_overlap_L4-5-7-10", sink_11_overlap),
        (f"sink_only_N4_overlap",  sink_4_overlap),
        (f"other_only_N4_overlap", other_4_overlap),
        (f"sink_only_N11_nonoverlap_L11-14", sink_11_no),
    ]

    print(f"\n=== Selected heads (top-by-first-token-selectivity) ===")
    for name, spec in conditions:
        n = sum(len(v) for v in spec.values())
        print(f"  {name}: n={n}, spec={spec}")

    print(f"\n=== Running {len(conditions)} conditions ===")
    print(f"  {'condition':<60} {'n':>3} {'top1':>8} {'top5':>8} {'logitB':>9} {'loss':>7}")
    print("  " + "-" * 100)
    results = []
    for name, spec in conditions:
        t0 = time.time()
        r = evaluate(model, tokens, targets, device, head_dim, spec, args.batch_size)
        elapsed = time.time() - t0
        n = sum(len(v) for v in spec.values())
        print(f"  {name:<60} {n:>3} {r['acc_top1']:>8.4f} {r['acc_top5']:>8.4f} "
              f"{r['mean_logit_B']:>+9.3f} {r['loss']:>7.3f}  [{elapsed:.0f}s]", flush=True)
        results.append({"name": name, "spec": {str(k): v for k, v in spec.items()},
                        "n_ablated": n, **r})
        with open(args.out, "w") as f:
            json.dump({"model": args.model, "revision": args.revision,
                       "n_examples": args.n_examples,
                       "overlap_layers": OVERLAP_LAYERS,
                       "non_overlap_layers": NON_OVERLAP_LAYERS,
                       "sink_threshold": SINK_THRESHOLD,
                       "induction_threshold": INDUCTION_THRESHOLD,
                       "results": results}, f, indent=2)

    print(f"\nwrote {args.out}")


if __name__ == "__main__":
    main()
