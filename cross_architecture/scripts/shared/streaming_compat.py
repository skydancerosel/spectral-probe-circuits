"""Experiment B: StreamingLLM cache-compatibility test for induction.

Setup: replicate StreamingLLM's KV-cache structure -- the model "sees" only
the first 4 tokens (the attention-sink slot) plus the last W tokens
(sliding window). Implement this by truncating the input sequence to
[tokens[0:4], tokens[T-W:T]] and feeding it forward. Measure induction
top-1 acc on the query position (last token).

Two regimes per example based on whether the induction-target position
ab+1 falls in the visible set:

  (a) ab+1 in {0,1,2,3} OR ab+1 in [T-W, T-1] : visible; circuit can fire
  (b) ab+1 in (3, T-W)                       : invisible; circuit cannot fire

Hypothesis: visible-set top-1 approximates the baseline; invisible-set
top-1 drops to chance. If true: the induction circuit identified by our
methodology operates within the StreamingLLM cache footprint, so the
methodology remains useful in streaming-deployed regimes for queries
whose context is recent enough.

Note on positional encoding: this is a truncation test, not a full
StreamingLLM emulation. Real StreamingLLM rolls position indices on the
cached tokens so the model sees them as adjacent. Truncation here uses
the natural shorter-sequence positions (0..3+W-1) which differ from
StreamingLLM's rolling assignment. For our purpose -- testing whether
the induction circuit can still operate when only sinks + recent
context is kept -- truncation is the cleanest first test; full RoPE
position-rolling is a follow-up.
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__))))
from induction_utils import build_induction_batch


def reconstruct_ab_indices(tokens, targets):
    n, T = tokens.shape
    ab = torch.zeros(n, dtype=torch.long)
    for i in range(n):
        match = (tokens[i, :-1] == targets[i]).nonzero(as_tuple=True)[0]
        ab[i] = match[0].item() - 1 if len(match) == 1 else -1
    return ab


def make_streamed_tokens(tokens, n_sink=4, window=64):
    """Return a [n, n_sink + window] tensor: first n_sink + last window tokens."""
    n, T = tokens.shape
    if window > T - n_sink:
        window = T - n_sink
    sinks = tokens[:, :n_sink]
    recent = tokens[:, T - window:T]
    return torch.cat([sinks, recent], dim=1)


def visible_in_stream(ab_plus_1, T, n_sink=4, window=64):
    """Bool tensor: True if the induction-target position is in the visible set."""
    return (ab_plus_1 < n_sink) | (ab_plus_1 >= T - window)


def evaluate(model, tokens, targets, device, batch_size=4):
    n = tokens.shape[0]
    last = tokens.shape[1] - 1
    correct = 0
    top5_correct = 0
    logit_B_sum = 0.0
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
    return {
        "n": n,
        "acc_top1": correct / n,
        "acc_top5": top5_correct / n,
        "mean_logit_B": logit_B_sum / n,
    }


def evaluate_per_example(model, tokens, targets, device, batch_size=4):
    n = tokens.shape[0]
    last = tokens.shape[1] - 1
    flags = torch.zeros(n, dtype=torch.bool)
    with torch.no_grad():
        for start in range(0, n, batch_size):
            end = min(start + batch_size, n)
            tok = tokens[start:end].to(device)
            tgt = targets[start:end].to(device)
            logits = model(tok).logits[:, last, :]
            flags[start:end] = (logits.argmax(dim=-1).cpu() == tgt.cpu())
    return flags


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--family", choices=["pythia", "olmo", "olmoe"], required=True)
    ap.add_argument("--model", required=True)
    ap.add_argument("--revision", default="main")
    ap.add_argument("--out", required=True)
    ap.add_argument("--n-examples", type=int, default=2000)
    ap.add_argument("--seq-len", type=int, default=256)
    ap.add_argument("--batch-size", type=int, default=4)
    ap.add_argument("--n-sink", type=int, default=4)
    ap.add_argument("--windows", type=int, nargs="+",
                    default=[32, 64, 128, 200, 252])
    ap.add_argument("--dtype", default="fp16", choices=["fp16", "fp32", "bf16"])
    args = ap.parse_args()
    dtype_map = {"fp16": torch.float16, "fp32": torch.float32, "bf16": torch.bfloat16}

    device = "mps" if torch.backends.mps.is_available() else "cpu"
    print(f"device = {device}")

    rng = np.random.RandomState(42)
    tokens, _, targets = build_induction_batch(args.n_examples, args.seq_len, rng=rng)
    T = tokens.shape[1]
    ab = reconstruct_ab_indices(tokens, targets)
    ab_plus_1 = ab + 1
    valid_mask = (ab >= 0) & (ab + 1 < T)
    print(f"  batch: {tuple(tokens.shape)}; valid={int(valid_mask.sum())}/{tokens.shape[0]}")

    print(f"Loading {args.model}@{args.revision}...")
    t0 = time.time()
    if args.family == "pythia":
        from transformers import GPTNeoXForCausalLM as M
    elif args.family == "olmo":
        from transformers import OlmoForCausalLM as M
    elif args.family == "olmoe":
        from transformers import OlmoeForCausalLM as M
    model = M.from_pretrained(args.model, revision=args.revision,
                              dtype=dtype_map[args.dtype]).to(device).eval()
    print(f"  loaded in {time.time() - t0:.0f}s")

    # Baseline: full seq_len
    print(f"\n=== Baseline (full seq_len={T}) ===")
    base = evaluate(model, tokens, targets, device, args.batch_size)
    base_flags = evaluate_per_example(model, tokens, targets, device, args.batch_size)
    print(f"  top1={base['acc_top1']:.4f}  top5={base['acc_top5']:.4f}  logit_B={base['mean_logit_B']:+.3f}")

    out = {
        "model": args.model, "revision": args.revision,
        "family": args.family, "seq_len": T, "n_sink": args.n_sink,
        "n_examples": args.n_examples,
        "baseline": base,
        "windows": [],
    }

    for W in args.windows:
        print(f"\n=== Streaming window W={W} (visible: tokens[0:{args.n_sink}] + tokens[T-{W}:T]) ===")
        streamed = make_streamed_tokens(tokens, n_sink=args.n_sink, window=W)
        visible = visible_in_stream(ab_plus_1, T, n_sink=args.n_sink, window=W)
        n_visible = int(visible.sum())
        n_invisible = (args.n_examples - n_visible)
        print(f"  visible_examples={n_visible}/{args.n_examples} ({n_visible/args.n_examples:.1%})")

        result = evaluate(model, streamed, targets, device, args.batch_size)
        stream_flags = evaluate_per_example(model, streamed, targets, device, args.batch_size)

        # Decompose by visibility
        visible_idx = visible.nonzero(as_tuple=True)[0]
        invisible_idx = (~visible).nonzero(as_tuple=True)[0]
        if len(visible_idx) > 0:
            vis_acc = stream_flags[visible_idx].float().mean().item()
        else:
            vis_acc = float("nan")
        if len(invisible_idx) > 0:
            invis_acc = stream_flags[invisible_idx].float().mean().item()
        else:
            invis_acc = float("nan")

        print(f"  overall top1={result['acc_top1']:.4f}  top5={result['acc_top5']:.4f}")
        print(f"  visible-set   top1={vis_acc:.4f}  (ab+1 in cache)")
        print(f"  invisible-set top1={invis_acc:.4f}  (ab+1 evicted)")

        out["windows"].append({
            "window": W,
            "n_visible": n_visible,
            "fraction_visible": n_visible / args.n_examples,
            "overall_top1": result["acc_top1"],
            "overall_top5": result["acc_top5"],
            "overall_logit_B": result["mean_logit_B"],
            "visible_set_top1": vis_acc,
            "invisible_set_top1": invis_acc,
        })

        with open(args.out, "w") as f:
            json.dump(out, f, indent=2)

    print(f"\nwrote {args.out}")


if __name__ == "__main__":
    main()
