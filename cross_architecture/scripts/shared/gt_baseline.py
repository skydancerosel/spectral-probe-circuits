"""Quick baseline greater-than evaluation across the three 1B models. No ablation."""
from __future__ import annotations
import argparse, time, json
from pathlib import Path
import torch
from transformers import AutoTokenizer
from gt_batch import build_gt_batch, build_two_digit_token_table, evaluate_gt


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--arch", required=True, choices=["pythia", "olmo", "olmoe"])
    ap.add_argument("--model", required=True)
    ap.add_argument("--revision", default="main")
    ap.add_argument("--n-examples", type=int, default=500)
    ap.add_argument("--batch-size", type=int, default=4)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    device = "mps" if torch.backends.mps.is_available() else "cpu"
    print(f"device: {device}")
    tk = AutoTokenizer.from_pretrained(args.model)
    tokens, decades, records = build_gt_batch(tk, n_examples=args.n_examples)
    two_digit_ids = build_two_digit_token_table(tk)
    print(f"batch: {tuple(tokens.shape)}; 2-digit token table: {len(two_digit_ids)}")

    print(f"loading {args.model}@{args.revision} (arch={args.arch})...")
    t0 = time.time()
    if args.arch == "pythia":
        from transformers import GPTNeoXForCausalLM
        model = GPTNeoXForCausalLM.from_pretrained(args.model, revision=args.revision, dtype=torch.float16)
    elif args.arch == "olmo":
        from transformers import OlmoForCausalLM
        model = OlmoForCausalLM.from_pretrained(args.model, revision=args.revision, dtype=torch.float16)
    else:
        from transformers import OlmoeForCausalLM
        model = OlmoeForCausalLM.from_pretrained(args.model, revision=args.revision, dtype=torch.float16)
    model = model.to(device).eval()
    print(f"  loaded in {time.time()-t0:.0f}s")

    t0 = time.time()
    r = evaluate_gt(model, tokens, decades, two_digit_ids, device, batch_size=args.batch_size)
    r["wall_s"] = time.time() - t0
    print(f"\n=== {args.arch} greater-than baseline ===")
    print(f"  top1_above (P[argmax > start]):           {r['top1_above']*100:6.2f}%")
    print(f"  prob_above (Σ prob > start):              {r['prob_above']*100:6.2f}%")
    print(f"  prob_below (Σ prob ≤ start):              {r['prob_below']*100:6.2f}%")
    print(f"  mean logit_diff (above - below):          {r['logit_diff_above_below']:+6.3f}")
    print(f"  wall: {r['wall_s']:.0f}s")
    with open(args.out, "w") as f:
        json.dump({"model": args.model, "arch": args.arch, "n_examples": args.n_examples, **r}, f, indent=2)
    print(f"\nwrote {args.out}")


if __name__ == "__main__":
    main()
