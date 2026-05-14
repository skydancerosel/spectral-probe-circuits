"""Variable-binding full pipeline per model: baseline + mech-interp + ablations.

Mech-interp screen: attention from the query position (pos 12) to the
*binding-value position* of the queried variable (pos 3 if query=name_A, pos 8
if query=name_B). High selectivity = candidate variable-binding head.

Conditions:
  baseline
  ablate_top5_vb_heads       (binding-value attention top-K)
  ablate_matched_random      (5 random heads in same layers)
  ablate_induction_circuit   (≥50× induction sel from per-revision final)
  ablate_prev_token_circuit  (best-class prev-token, ≥100× sel)
"""
from __future__ import annotations
import argparse, json, time
import numpy as np
import torch
from transformers import AutoTokenizer
from vb_batch import build_vb_batch, evaluate_vb
from prev_token_circuit_ablation import make_pre_hook, get_layer_module, pick_circuit


def run_condition(model, arch, spec, tokens, target_ids, distractor_ids, device, head_dim, batch_size):
    handles = []
    for L, hs in spec.items():
        if not hs: continue
        h = get_layer_module(model, arch, L).register_forward_pre_hook(make_pre_hook(hs, head_dim))
        handles.append(h)
    try:
        return evaluate_vb(model, tokens, target_ids, distractor_ids, device, batch_size)
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
    ap.add_argument("--n-examples", type=int, default=500)
    ap.add_argument("--batch-size", type=int, default=4)
    ap.add_argument("--random-seed", type=int, default=123)
    ap.add_argument("--out-mechinterp", required=True)
    ap.add_argument("--out-ablation", required=True)
    args = ap.parse_args()

    device = "mps" if torch.backends.mps.is_available() else "cpu"
    tk = AutoTokenizer.from_pretrained(args.model)
    tokens, target_ids, distractor_ids, binding_pos, distractor_pos, records = build_vb_batch(tk, n_examples=args.n_examples)
    n, T = tokens.shape
    QUERY_POS = T - 1  # 12
    print(f"vb batch: {n}x{T}; query_pos={QUERY_POS}")

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
    print(f"  loaded in {time.time()-t0:.0f}s; L={n_layer} H={n_head}")

    # Per-(L, H) attention from QUERY_POS to binding-value position (example-specific)
    binding_attn_sum = torch.zeros(n_layer, n_head, dtype=torch.float64)
    distractor_attn_sum = torch.zeros(n_layer, n_head, dtype=torch.float64)
    other_attn_sum = torch.zeros(n_layer, n_head, dtype=torch.float64)
    name_dup_attn_sum = torch.zeros(n_layer, n_head, dtype=torch.float64)  # attn to pos 10 (query-name duplicate)
    n_used = 0

    NAME_DUP_POS = 10

    with torch.no_grad():
        for s in range(0, n, args.batch_size):
            e = min(s + args.batch_size, n)
            tok = tokens[s:e].to(device)
            out = model(tok, output_attentions=True)
            for L, attn in enumerate(out.attentions):
                a_last = attn[:, :, QUERY_POS, :].cpu().double()  # [B, H, T]
                for bi in range(e - s):
                    idx = s + bi
                    bp = int(binding_pos[idx].item())
                    dp = int(distractor_pos[idx].item())
                    binding_attn_sum[L]    += a_last[bi, :, bp]
                    distractor_attn_sum[L] += a_last[bi, :, dp]
                    name_dup_attn_sum[L]   += a_last[bi, :, NAME_DUP_POS]
                    total = a_last[bi].sum(-1)
                    other_attn_sum[L] += total - a_last[bi, :, bp] - a_last[bi, :, dp] - a_last[bi, :, NAME_DUP_POS]
            n_used = e
            del out

    binding_mean = binding_attn_sum / n
    distractor_mean = distractor_attn_sum / n
    name_dup_mean = name_dup_attn_sum / n
    other_per_pos = (other_attn_sum / n) / (T - 3)  # 3 special positions excluded
    eps = 1e-8
    vb_sel = binding_mean / other_per_pos.clamp_min(eps)

    all_sel = {}
    for L in range(n_layer):
        for H in range(n_head):
            all_sel[f"L{L}_H{H}"] = {
                "binding_attn": float(binding_mean[L, H]),
                "distractor_attn": float(distractor_mean[L, H]),
                "name_dup_attn": float(name_dup_mean[L, H]),
                "vb_selectivity": float(vb_sel[L, H]),
            }
    flat = [(L, H, float(vb_sel[L, H]), float(binding_mean[L, H]),
             float(distractor_mean[L, H]), float(name_dup_mean[L, H]))
            for L in range(n_layer) for H in range(n_head)]
    flat.sort(key=lambda x: -x[2])

    print(f"\nTop 15 VB candidates (attention from query pos {QUERY_POS} → binding-value pos):")
    print(f"  {'head':>10}  {'vb_sel':>8}  {'attn_bind':>10}  {'attn_dist':>10}  {'attn_name_dup':>14}")
    for L, H, sel, ba, da, nda in flat[:15]:
        print(f"  L{L}_H{H:<3}  {sel:>8.2f}  {ba:>10.4f}  {da:>10.4f}  {nda:>14.4f}")

    mi_out = {
        "model": args.model, "arch": args.arch, "revision": args.revision,
        "n_examples": n, "n_layer": n_layer, "n_head": n_head,
        "query_pos": QUERY_POS, "name_dup_pos": NAME_DUP_POS,
        "all_head_vb_selectivity": all_sel,
        "top25_vb_candidates": [
            {"L": L, "H": H, "vb_sel": s, "binding_attn": ba, "distractor_attn": da, "name_dup_attn": nda}
            for L, H, s, ba, da, nda in flat[:25]
        ],
    }
    with open(args.out_mechinterp, "w") as f: json.dump(mi_out, f, indent=2)
    print(f"\nwrote {args.out_mechinterp}")

    # Reload without eager attention for faster ablation
    print(f"\nreloading {args.model} for ablation...")
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

    top_vb = flat[:args.top_k]
    vb_heads = [(L, H) for L, H, *_ in top_vb]
    vb_spec = to_spec(vb_heads)
    print(f"\nTop-{args.top_k} VB candidates: {vb_heads}")

    rng = np.random.RandomState(args.random_seed)
    rand_spec = {}
    for L, picks in vb_spec.items():
        eligible = [h for h in range(n_head) if h not in picks]
        n_samp = min(len(picks), len(eligible))
        rand_spec[L] = sorted(rng.choice(eligible, size=n_samp, replace=False).tolist())

    with open(args.final_mechinterp_json) as f: mi = json.load(f)
    all_final = mi["all_head_selectivity"]
    induction_spec = pick_circuit(all_final, "induction", 50.0, require_best_class=False)
    prev_spec = pick_circuit(all_final, "previous-token", 100.0, require_best_class=True)

    conditions = [
        ("baseline", {}),
        (f"ablate_top{args.top_k}_vb_heads", vb_spec),
        (f"ablate_matched_random_same_layers ({sum(len(v) for v in rand_spec.values())}h)", rand_spec),
        (f"ablate_induction_circuit ({sum(len(v) for v in induction_spec.values())}h)", induction_spec),
        (f"ablate_prev_token_circuit ({sum(len(v) for v in prev_spec.values())}h)", prev_spec),
    ]
    results = []
    for name, spec in conditions:
        t0 = time.time()
        r = run_condition(model, args.arch, spec, tokens, target_ids, distractor_ids, device, head_dim, args.batch_size)
        r["condition"] = name
        r["n_ablated"] = sum(len(v) for v in spec.values())
        r["spec"] = {str(L): list(map(int, hs)) for L, hs in spec.items()}
        r["wall_s"] = time.time() - t0
        results.append(r)
        print(f"  {name}: top1={r['top1']*100:6.2f}%  frac(t>d)={r['frac_target_gt_distractor']*100:6.2f}%  "
              f"logit_diff={r['logit_diff_mean']:+6.3f}  ({r['wall_s']:.0f}s)")

    abl_out = {
        "model": args.model, "arch": args.arch, "revision": args.revision,
        "top_k": args.top_k, "n_examples": n,
        "vb_heads": vb_heads, "conditions": results,
    }
    with open(args.out_ablation, "w") as f: json.dump(abl_out, f, indent=2)
    print(f"\nwrote {args.out_ablation}")


if __name__ == "__main__":
    main()
