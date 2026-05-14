"""Greater-than mech-interp: for each (L, H), compute mean attention from the
final query position (pos 11) to the start-decade position (pos 7) over the
batch, and to all other positions (baseline). The greater-than selectivity is
attn_to_decade / mean_attn_to_other.
"""
from __future__ import annotations
import argparse, json, time
from pathlib import Path
import torch
from transformers import AutoTokenizer
from gt_batch import build_gt_batch


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
    tk = AutoTokenizer.from_pretrained(args.model)
    tokens, decades, records = build_gt_batch(tk, n_examples=args.n_examples)
    n, T = tokens.shape
    print(f"batch: {n}x{T}")

    print(f"loading {args.model}@{args.revision} (arch={args.arch}) eager attention...")
    t0 = time.time()
    if args.arch == "pythia":
        from transformers import GPTNeoXForCausalLM
        model = GPTNeoXForCausalLM.from_pretrained(args.model, revision=args.revision,
                                                    dtype=torch.float16, attn_implementation="eager")
    elif args.arch == "olmo":
        from transformers import OlmoForCausalLM
        model = OlmoForCausalLM.from_pretrained(args.model, revision=args.revision,
                                                  dtype=torch.float16, attn_implementation="eager")
    else:
        from transformers import OlmoeForCausalLM
        model = OlmoeForCausalLM.from_pretrained(args.model, revision=args.revision,
                                                   dtype=torch.float16, attn_implementation="eager")
    model = model.to(device).eval()
    cfg = model.config
    n_layer, n_head = cfg.num_hidden_layers, cfg.num_attention_heads
    print(f"  loaded in {time.time()-t0:.0f}s; L={n_layer} H={n_head}")

    DECADE_POS = 7
    QUERY_POS = T - 1  # 11
    # Per-(L, H) totals
    decade_attn = torch.zeros(n_layer, n_head, dtype=torch.float64)
    other_attn  = torch.zeros(n_layer, n_head, dtype=torch.float64)
    century_attn = torch.zeros(n_layer, n_head, dtype=torch.float64)  # for completeness; pos 6 = " 17"
    n_used = 0

    with torch.no_grad():
        for s in range(0, n, args.batch_size):
            e = min(s + args.batch_size, n)
            tok = tokens[s:e].to(device)
            out = model(tok, output_attentions=True)
            # attentions: tuple of n_layer tensors [B, n_head, T, T]
            for L, attn in enumerate(out.attentions):
                a_last = attn[:, :, QUERY_POS, :].cpu().double()  # [B, n_head, T]
                # For each example
                for bi in range(e - s):
                    decade_attn[L]  += a_last[bi, :, DECADE_POS]
                    century_attn[L] += a_last[bi, :, DECADE_POS - 1]  # pos 6 = " CC"
                    total = a_last[bi].sum(-1)
                    other_attn[L] += total - a_last[bi, :, DECADE_POS] - a_last[bi, :, DECADE_POS - 1]
            n_used = e
            if s % (args.batch_size * 25) == 0:
                print(f"  {n_used}/{n}")
            del out

    decade_mean = decade_attn / n
    century_mean = century_attn / n
    other_mean = other_attn / n  # over T - 2 positions

    eps = 1e-8
    # baseline = mean attention per other position
    other_per_pos = other_mean / (T - 2)
    gt_sel = decade_mean / other_per_pos.clamp_min(eps)

    # Build all-head dict and a top-25 list
    all_sel = {}
    for L in range(n_layer):
        for H in range(n_head):
            all_sel[f"L{L}_H{H}"] = {
                "decade_attn": float(decade_mean[L, H]),
                "century_attn": float(century_mean[L, H]),
                "gt_selectivity": float(gt_sel[L, H]),
            }
    flat = [(L, H, float(gt_sel[L, H]), float(decade_mean[L, H]), float(century_mean[L, H]))
            for L in range(n_layer) for H in range(n_head)]
    flat.sort(key=lambda x: -x[2])

    print(f"\nTop 20 greater-than candidates (attn from pos {QUERY_POS} to pos {DECADE_POS}):")
    print(f"  {'head':>10}  {'gt_sel':>9}  {'attn_decade':>11}  {'attn_century':>12}")
    for L, H, sel, da, ca in flat[:20]:
        print(f"  L{L}_H{H:<3}  {sel:>9.2f}  {da:>11.4f}  {ca:>12.4f}")

    out_d = {
        "model": args.model, "arch": args.arch, "revision": args.revision,
        "n_examples": n, "n_layer": n_layer, "n_head": n_head,
        "decade_pos": DECADE_POS, "query_pos": QUERY_POS,
        "all_head_gt_selectivity": all_sel,
        "top25_gt_candidates": [
            {"L": L, "H": H, "gt_sel": s, "decade_attn": da, "century_attn": ca}
            for L, H, s, da, ca in flat[:25]
        ],
    }
    with open(args.out, "w") as f:
        json.dump(out_d, f, indent=2)
    print(f"\nwrote {args.out}")


if __name__ == "__main__":
    main()
