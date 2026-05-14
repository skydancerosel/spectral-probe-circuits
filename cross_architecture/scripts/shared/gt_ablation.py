"""Greater-than ablations across the three 1B models.

Four conditions:
  baseline
  ablate_top5_gt_heads   (top-5 by gt_selectivity from gt_mechinterp)
  ablate_matched_random  (5 random heads in same layers as top-5)
  ablate_induction_circuit  (heads with induction-sel ≥50× from the final-ckpt
                             per_revision mech-interp — same set as §4.3)
  ablate_prev_token_circuit (heads with best-class=prev-token + sel ≥100× —
                             same set as §5)
"""
from __future__ import annotations
import argparse, json, math, time
import numpy as np
import torch
from transformers import AutoTokenizer
from gt_batch import build_gt_batch, build_two_digit_token_table, evaluate_gt
# Reuse hook helpers from prev_token_circuit_ablation
from prev_token_circuit_ablation import make_pre_hook, get_layer_module, pick_circuit, best_class


def run_condition(model, arch, spec, tokens, decades, two_digit_ids, device, head_dim, batch_size):
    handles = []
    for L, hs in spec.items():
        if not hs: continue
        h = get_layer_module(model, arch, L).register_forward_pre_hook(make_pre_hook(hs, head_dim))
        handles.append(h)
    try:
        return evaluate_gt(model, tokens, decades, two_digit_ids, device, batch_size)
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
    ap.add_argument("--gt-mechinterp-json", required=True)
    ap.add_argument("--final-mechinterp-json", required=True,
                    help="Final-ckpt per_revision_mechinterp JSON for induction + prev-token screens")
    ap.add_argument("--top-k", type=int, default=5)
    ap.add_argument("--n-examples", type=int, default=500)
    ap.add_argument("--batch-size", type=int, default=4)
    ap.add_argument("--random-seed", type=int, default=123)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    device = "mps" if torch.backends.mps.is_available() else "cpu"
    tk = AutoTokenizer.from_pretrained(args.model)
    tokens, decades, records = build_gt_batch(tk, n_examples=args.n_examples)
    two_digit_ids = build_two_digit_token_table(tk)
    print(f"batch: {tuple(tokens.shape)}")

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
    cfg = model.config
    n_layer, n_head = cfg.num_hidden_layers, cfg.num_attention_heads
    head_dim = cfg.hidden_size // n_head
    print(f"  loaded in {time.time()-t0:.0f}s; L={n_layer} H={n_head}")

    # Top-K GT candidates
    with open(args.gt_mechinterp_json) as f:
        gt_mi = json.load(f)
    top_gt = gt_mi["top25_gt_candidates"][:args.top_k]
    gt_heads = [(h["L"], h["H"]) for h in top_gt]
    gt_spec = to_spec(gt_heads)
    print(f"\nTop-{args.top_k} GT candidates: {gt_heads}")

    # Matched-random
    rng = np.random.RandomState(args.random_seed)
    rand_spec = {}
    for L, picks in gt_spec.items():
        eligible = [h for h in range(n_head) if h not in picks]
        n_samp = min(len(picks), len(eligible))
        rand_spec[L] = sorted(rng.choice(eligible, size=n_samp, replace=False).tolist())

    # Induction circuit (≥50× induction sel, no best-class filter — same as §4.3)
    with open(args.final_mechinterp_json) as f:
        mi = json.load(f)
    all_sel = mi["all_head_selectivity"]
    induction_spec = pick_circuit(all_sel, "induction", 50.0, require_best_class=False)
    n_ind = sum(len(v) for v in induction_spec.values())
    print(f"Induction circuit (≥50×): {n_ind} heads, layers {sorted(induction_spec)}")

    # Prev-token circuit (best-class=prev-token, sel ≥100×)
    prev_spec = pick_circuit(all_sel, "previous-token", 100.0, require_best_class=True)
    n_prev = sum(len(v) for v in prev_spec.values())
    print(f"Prev-token circuit: {n_prev} heads")

    conditions = [
        ("baseline", {}),
        (f"ablate_top{args.top_k}_gt_heads", gt_spec),
        (f"ablate_matched_random_same_layers ({sum(len(v) for v in rand_spec.values())}h)", rand_spec),
        (f"ablate_induction_circuit ({n_ind}h)", induction_spec),
        (f"ablate_prev_token_circuit ({n_prev}h)", prev_spec),
    ]
    results = []
    for name, spec in conditions:
        t0 = time.time()
        r = run_condition(model, args.arch, spec, tokens, decades, two_digit_ids, device, head_dim, args.batch_size)
        r["condition"] = name
        r["n_ablated"] = sum(len(v) for v in spec.values())
        r["spec"] = {str(L): list(map(int, hs)) for L, hs in spec.items()}
        r["wall_s"] = time.time() - t0
        results.append(r)
        print(f"  {name}: top1_above={r['top1_above']*100:6.2f}%  "
              f"prob_above={r['prob_above']*100:6.2f}%  "
              f"logit_diff={r['logit_diff_above_below']:+6.3f}  ({r['wall_s']:.0f}s)")

    out_d = {
        "model": args.model, "arch": args.arch, "revision": args.revision,
        "top_k": args.top_k, "n_examples": args.n_examples,
        "gt_heads": gt_heads, "conditions": results,
    }
    with open(args.out, "w") as f:
        json.dump(out_d, f, indent=2)
    print(f"\nwrote {args.out}")


if __name__ == "__main__":
    main()
