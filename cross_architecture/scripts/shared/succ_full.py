"""Successor task — full pipeline per model: baseline + mech-interp + ablations,
one model load, three artifacts saved.

Mech-interp screen: at the query position (pos 4, the last shown item), measure
attention from pos 4 to pos 4 (self-attention at the query). Per Gould et al.
(2023), successor heads attend strongly to the current token and apply an
OV-circuit transformation that increments to the next-in-sequence.

Ablation conditions:
  - baseline
  - top-K successor heads (self-attention at query position, top-K)
  - matched-random in the same layers
  - induction circuit (from per_revision_mechinterp final ckpt, ≥50× ind sel)
  - prev-token circuit (best-class prev-token, ≥100× prev sel)
"""
from __future__ import annotations
import argparse, json, math, time
import numpy as np
import torch
from transformers import AutoTokenizer
from succ_batch import build_succ_batch, evaluate_succ
from prev_token_circuit_ablation import make_pre_hook, get_layer_module, pick_circuit


def run_condition(model, arch, spec, tokens, target_ids, device, head_dim, batch_size):
    handles = []
    for L, hs in spec.items():
        if not hs: continue
        h = get_layer_module(model, arch, L).register_forward_pre_hook(make_pre_hook(hs, head_dim))
        handles.append(h)
    try:
        return evaluate_succ(model, tokens, target_ids, device, batch_size)
    finally:
        for h in handles: h.remove()


def to_spec(heads):
    spec = {}
    for tup in heads:
        L, H = (tup[0], tup[1]) if isinstance(tup, tuple) else (tup["L"], tup["H"])
        spec.setdefault(L, []).append(H)
    return spec


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--arch", required=True, choices=["pythia", "olmo", "olmoe"])
    ap.add_argument("--model", required=True)
    ap.add_argument("--revision", default="main")
    ap.add_argument("--final-mechinterp-json", required=True)
    ap.add_argument("--top-k", type=int, default=5)
    ap.add_argument("--batch-size", type=int, default=4)
    ap.add_argument("--random-seed", type=int, default=123)
    ap.add_argument("--out-mechinterp", required=True)
    ap.add_argument("--out-ablation", required=True)
    args = ap.parse_args()

    device = "mps" if torch.backends.mps.is_available() else "cpu"
    tk = AutoTokenizer.from_pretrained(args.model)
    tokens, target_ids, records = build_succ_batch(tk)
    n, T = tokens.shape
    print(f"successor batch: {n}x{T} ({len(records)} unique sequences)")

    # Mech-interp pass: need eager attention to get output_attentions
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
    head_dim = cfg.hidden_size // n_head
    print(f"  loaded in {time.time()-t0:.0f}s; L={n_layer} H={n_head} hd={head_dim}")

    # Per-(L, H) self-attention at query position
    QUERY_POS = T - 1
    self_attn_sum = torch.zeros(n_layer, n_head, dtype=torch.float64)
    prev_attn_sum = torch.zeros(n_layer, n_head, dtype=torch.float64)
    other_attn_sum = torch.zeros(n_layer, n_head, dtype=torch.float64)
    n_used = 0

    with torch.no_grad():
        for s in range(0, n, args.batch_size):
            e = min(s + args.batch_size, n)
            tok = tokens[s:e].to(device)
            out = model(tok, output_attentions=True)
            for L, attn in enumerate(out.attentions):
                a_last = attn[:, :, QUERY_POS, :].cpu().double()  # [B, H, T]
                for bi in range(e - s):
                    self_attn_sum[L]  += a_last[bi, :, QUERY_POS]
                    prev_attn_sum[L]  += a_last[bi, :, QUERY_POS - 1]
                    total = a_last[bi].sum(-1)
                    other_attn_sum[L] += total - a_last[bi, :, QUERY_POS] - a_last[bi, :, QUERY_POS - 1]
            n_used = e
            del out

    self_mean = self_attn_sum / n
    prev_mean = prev_attn_sum / n
    other_mean = other_attn_sum / n
    other_per_pos = other_mean / (T - 2)
    eps = 1e-8
    succ_sel = self_mean / other_per_pos.clamp_min(eps)

    all_sel = {}
    for L in range(n_layer):
        for H in range(n_head):
            all_sel[f"L{L}_H{H}"] = {
                "self_attn": float(self_mean[L, H]),
                "prev_attn": float(prev_mean[L, H]),
                "succ_selectivity": float(succ_sel[L, H]),
            }
    flat = [(L, H, float(succ_sel[L, H]), float(self_mean[L, H]), float(prev_mean[L, H]))
            for L in range(n_layer) for H in range(n_head)]
    flat.sort(key=lambda x: -x[2])

    print(f"\nTop 15 successor candidates (self-attention at query pos {QUERY_POS}):")
    print(f"  {'head':>10}  {'succ_sel':>9}  {'attn_self':>10}  {'attn_prev':>10}")
    for L, H, sel, sa, pa in flat[:15]:
        print(f"  L{L}_H{H:<3}  {sel:>9.2f}  {sa:>10.4f}  {pa:>10.4f}")

    mi_out = {
        "model": args.model, "arch": args.arch, "revision": args.revision,
        "n_examples": n, "n_layer": n_layer, "n_head": n_head,
        "query_pos": QUERY_POS,
        "all_head_succ_selectivity": all_sel,
        "top25_succ_candidates": [
            {"L": L, "H": H, "succ_sel": s, "self_attn": sa, "prev_attn": pa}
            for L, H, s, sa, pa in flat[:25]
        ],
    }
    with open(args.out_mechinterp, "w") as f:
        json.dump(mi_out, f, indent=2)
    print(f"\nwrote {args.out_mechinterp}")

    # Now reload without eager attention for faster forward passes in ablation
    print(f"\nreloading {args.model} without eager attention for ablation...")
    del model
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

    # Top-K successor heads
    top_succ = flat[:args.top_k]
    succ_heads = [(L, H) for L, H, *_ in top_succ]
    succ_spec = to_spec(succ_heads)
    print(f"\nTop-{args.top_k} successor candidates: {succ_heads}")

    # Matched-random in same layers
    rng = np.random.RandomState(args.random_seed)
    rand_spec = {}
    for L, picks in succ_spec.items():
        eligible = [h for h in range(n_head) if h not in picks]
        n_samp = min(len(picks), len(eligible))
        rand_spec[L] = sorted(rng.choice(eligible, size=n_samp, replace=False).tolist())

    # Induction & prev-token circuits (from per-revision final ckpt)
    with open(args.final_mechinterp_json) as f:
        mi = json.load(f)
    all_final_sel = mi["all_head_selectivity"]
    induction_spec = pick_circuit(all_final_sel, "induction", 50.0, require_best_class=False)
    prev_spec = pick_circuit(all_final_sel, "previous-token", 100.0, require_best_class=True)
    print(f"Induction circuit: {sum(len(v) for v in induction_spec.values())}h; prev-token: {sum(len(v) for v in prev_spec.values())}h")

    conditions = [
        ("baseline", {}),
        (f"ablate_top{args.top_k}_succ_heads", succ_spec),
        (f"ablate_matched_random_same_layers ({sum(len(v) for v in rand_spec.values())}h)", rand_spec),
        (f"ablate_induction_circuit ({sum(len(v) for v in induction_spec.values())}h)", induction_spec),
        (f"ablate_prev_token_circuit ({sum(len(v) for v in prev_spec.values())}h)", prev_spec),
    ]
    results = []
    for name, spec in conditions:
        t0 = time.time()
        r = run_condition(model, args.arch, spec, tokens, target_ids, device, head_dim, args.batch_size)
        r["condition"] = name
        r["n_ablated"] = sum(len(v) for v in spec.values())
        r["spec"] = {str(L): list(map(int, hs)) for L, hs in spec.items()}
        r["wall_s"] = time.time() - t0
        results.append(r)
        print(f"  {name}: top1={r['top1_acc']*100:6.2f}%  target_logit={r['target_logit_mean']:+6.2f}  "
              f"target_rank_median={r['target_rank_median']:.1f}  ({r['wall_s']:.0f}s)")

    abl_out = {
        "model": args.model, "arch": args.arch, "revision": args.revision,
        "top_k": args.top_k, "n_examples": n,
        "succ_heads": succ_heads, "conditions": results,
    }
    with open(args.out_ablation, "w") as f:
        json.dump(abl_out, f, indent=2)
    print(f"\nwrote {args.out_ablation}")


if __name__ == "__main__":
    main()
