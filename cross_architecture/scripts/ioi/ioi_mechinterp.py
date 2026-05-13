"""IOI-specific mech-interp: per-(L,H) attention at the final "to" position to:
  - IO position (target name's earlier occurrence)
  - subject first occurrence
  - subject second occurrence

Name-mover selectivity = mean_attn_to_IO / mean_attn_to_subject_avg.
Heads with high name-mover selectivity at the final position are candidates for
direct name-mover heads (output the IO name into the residual stream).

Output: all_head_ioi_selectivity dict (L_H -> {io_attn, subj_first, subj_second, name_mover_selectivity}).
"""
from __future__ import annotations

import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))  # noqa: E402
import argparse, json, time
import numpy as np
import torch
from transformers import AutoTokenizer
from ioi_batch import build_ioi_batch


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
    tokens, targets, distractors, io_p, sf_p, ss_p, records = build_ioi_batch(tk, n_examples=args.n_examples)
    n, T = tokens.shape
    print(f"IOI batch: {n}x{T}")

    print(f"Loading {args.model}@{args.revision} (arch={args.arch}) eager attention...")
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
    print(f"  loaded in {time.time()-t0:.0f}s  L={n_layer} H={n_head}")

    # Sum attention from pos T-1 to: IO, subj_first, subj_second positions
    # Shape per layer: [n, n_head, T] (attention from pos last)
    io_attn_sum   = torch.zeros(n_layer, n_head, dtype=torch.float64)
    sf_attn_sum   = torch.zeros(n_layer, n_head, dtype=torch.float64)
    ss_attn_sum   = torch.zeros(n_layer, n_head, dtype=torch.float64)
    other_attn_sum = torch.zeros(n_layer, n_head, dtype=torch.float64)
    n_examples_used = 0

    last_pos = T - 1
    with torch.no_grad():
        for s in range(0, n, args.batch_size):
            e = min(s + args.batch_size, n)
            tok = tokens[s:e].to(device)
            B = e - s
            out = model(tok, output_attentions=True)
            # attentions: tuple of n_layer tensors, each [B, n_head, T, T]
            for L, attn in enumerate(out.attentions):
                a_last = attn[:, :, last_pos, :]  # [B, n_head, T]
                # For each example in batch, gather attention at IO/sf/ss positions
                for bi in range(B):
                    idx = s + bi
                    io_attn_sum[L]  += a_last[bi, :, io_p[idx]].cpu().double()
                    sf_attn_sum[L]  += a_last[bi, :, sf_p[idx]].cpu().double()
                    ss_attn_sum[L]  += a_last[bi, :, ss_p[idx]].cpu().double()
                    # "other" = everything that isn't IO, subj_first, subj_second
                    total = a_last[bi].cpu().double().sum(-1)
                    other_attn_sum[L] += total - a_last[bi, :, io_p[idx]].cpu().double() \
                                                - a_last[bi, :, sf_p[idx]].cpu().double() \
                                                - a_last[bi, :, ss_p[idx]].cpu().double()
            n_examples_used = e
            if s % (args.batch_size * 25) == 0:
                print(f"  {n_examples_used}/{n}")
            del out

    io_attn = io_attn_sum / n
    sf_attn = sf_attn_sum / n
    ss_attn = ss_attn_sum / n
    other_attn = other_attn_sum / n

    # Per-head name-mover selectivity:
    #   nm_sel = mean_attn_IO / max(mean_attn_subj_first, mean_attn_subj_second, small_eps)
    eps = 1e-6
    subj_max = torch.max(sf_attn, ss_attn).clamp_min(eps)
    nm_sel = io_attn / subj_max

    # Save all-head matrix
    all_sel = {}
    for L in range(n_layer):
        for H in range(n_head):
            all_sel[f"L{L}_H{H}"] = {
                "io_attn": float(io_attn[L, H]),
                "subj_first_attn": float(sf_attn[L, H]),
                "subj_second_attn": float(ss_attn[L, H]),
                "other_attn": float(other_attn[L, H]),
                "name_mover_selectivity": float(nm_sel[L, H]),
            }

    # Sort by name-mover selectivity, print top 25
    flat = [(L, H, float(nm_sel[L, H]), float(io_attn[L, H]), float(sf_attn[L, H]), float(ss_attn[L, H]))
            for L in range(n_layer) for H in range(n_head)]
    flat.sort(key=lambda x: -x[2])
    print(f"\nTop 25 name-mover heads (highest IO/subj attention ratio at pos T-1):")
    print(f"  {'head':>8}  {'nm_sel':>8}  {'attn_IO':>8}  {'attn_sf':>8}  {'attn_ss':>8}")
    for L, H, sel, io_a, sf_a, ss_a in flat[:25]:
        print(f"  L{L}_H{H:<3}  {sel:>8.2f}  {io_a:>8.4f}  {sf_a:>8.4f}  {ss_a:>8.4f}")

    out_d = {
        "model": args.model, "revision": args.revision, "arch": args.arch,
        "n_examples": n, "n_layer": n_layer, "n_head": n_head,
        "last_pos": last_pos,
        "all_head_ioi_selectivity": all_sel,
        "top25_name_mover": [{"L": L, "H": H, "nm_sel": s, "io_attn": ia, "sf_attn": sa, "ss_attn": ssa}
                              for L, H, s, ia, sa, ssa in flat[:25]],
    }
    with open(args.out, "w") as f: json.dump(out_d, f, indent=2)
    print(f"\nwrote {args.out}")


if __name__ == "__main__":
    main()
