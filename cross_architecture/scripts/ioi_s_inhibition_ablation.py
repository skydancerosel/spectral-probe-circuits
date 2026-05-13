"""Ablate top-K subject-attending heads (S-Inhibition candidates) and measure
effect on IOI. Same skeleton as ioi_name_mover_ablation.py but ranks by:

  subj_selectivity = max(subj_first_attn, subj_second_attn) / max(io_attn, eps)

Heads are filtered:
  - subj_max / io_attn >= 2.0 (subject-biased, not mixed-role)
  - subj_max >= 0.1 (substantial absolute attention)

Then top-K by subj_max are ablated.
"""
from __future__ import annotations
import argparse, json, time
import numpy as np
import torch
from transformers import AutoTokenizer
from ioi_batch import build_ioi_batch
from prev_token_circuit_ablation import make_pre_hook, get_layer_module
from ioi_name_mover_ablation import evaluate_ioi, run_condition


def pick_s_inhibition(ioi_mi_path, top_k=5, subj_to_io_min=2.0, subj_abs_min=0.1):
    with open(ioi_mi_path) as f: d = json.load(f)
    sel = d['all_head_ioi_selectivity']
    cands = []
    for lh, vs in sel.items():
        L = int(lh.split('_')[0][1:]); H = int(lh.split('_')[1][1:])
        io_a = vs['io_attn']; sf = vs['subj_first_attn']; ss = vs['subj_second_attn']
        subj_max = max(sf, ss)
        ratio = subj_max / max(io_a, 1e-6)
        if ratio >= subj_to_io_min and subj_max >= subj_abs_min:
            cands.append((L, H, subj_max, ratio, io_a, sf, ss))
    cands.sort(key=lambda x: -x[2])
    picks = cands[:top_k]
    return picks


def to_spec(heads):
    spec = {}
    for tup in heads:
        L, H = tup[0], tup[1]
        spec.setdefault(L, []).append(H)
    return spec


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--arch", required=True, choices=["pythia", "olmo", "olmoe"])
    ap.add_argument("--model", required=True)
    ap.add_argument("--revision", default="main")
    ap.add_argument("--ioi-mechinterp-json", required=True)
    ap.add_argument("--top-k", type=int, default=5)
    ap.add_argument("--n-examples", type=int, default=500)
    ap.add_argument("--batch-size", type=int, default=4)
    ap.add_argument("--random-seed", type=int, default=123)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    device = "mps" if torch.backends.mps.is_available() else "cpu"
    tk = AutoTokenizer.from_pretrained(args.model)
    tokens, targets, distractors, *_ = build_ioi_batch(tk, n_examples=args.n_examples)
    print(f"IOI batch: {tuple(tokens.shape)}")

    print(f"Loading {args.model}@{args.revision} (arch={args.arch})...")
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
    print(f"  loaded in {time.time()-t0:.0f}s  L={n_layer} H={n_head} hd={head_dim}")

    picks = pick_s_inhibition(args.ioi_mechinterp_json, top_k=args.top_k)
    print(f"\nTop-{args.top_k} S-Inhibition candidates (subj/io>=2, subj_max>=0.1):")
    print(f"  {'head':>10}  {'subj_max':>9}  {'subj/io':>8}  {'io_attn':>8}  {'sf':>8}  {'ss':>8}")
    for L, H, sm, r, io_a, sf, ss in picks:
        print(f"  L{L}_H{H:<3}  {sm:>9.4f}  {r:>8.2f}  {io_a:>8.4f}  {sf:>8.4f}  {ss:>8.4f}")
    si_spec = to_spec(picks)
    print(f"  spec: {si_spec}")

    # Matched-random same layers
    rng = np.random.RandomState(args.random_seed)
    rand_spec = {}
    for L, layer_picks in si_spec.items():
        eligible = [h for h in range(n_head) if h not in layer_picks]
        n_samp = min(len(layer_picks), len(eligible))
        rand_spec[L] = sorted(rng.choice(eligible, size=n_samp, replace=False).tolist())

    # Also load name-mover ablation result for combined-circuit test
    # Combined: top-5 name-movers + top-5 S-Inhibition (union, dedup)
    with open(args.ioi_mechinterp_json) as f:
        mi = json.load(f)
    top_nm = mi['top25_name_mover'][:args.top_k]
    nm_heads = [(h['L'], h['H']) for h in top_nm]
    nm_spec = to_spec(nm_heads)
    union = {}
    for spec in [si_spec, nm_spec]:
        for L, hs in spec.items():
            union.setdefault(L, set()).update(hs)
    union = {L: sorted(list(hs)) for L, hs in union.items()}

    conditions = [
        ("baseline", {}),
        (f"ablate_top{args.top_k}_s_inhibition ({sum(len(v) for v in si_spec.values())}h)", si_spec),
        (f"ablate_matched_random_same_layers ({sum(len(v) for v in rand_spec.values())}h)", rand_spec),
        (f"ablate_name_mover+s_inhibition union ({sum(len(v) for v in union.values())}h)", union),
    ]
    results = []
    for name, spec in conditions:
        t0 = time.time()
        r = run_condition(model, args.arch, spec, tokens, targets, distractors, device, head_dim, args.batch_size)
        r["condition"] = name; r["n_ablated"] = sum(len(v) for v in spec.values())
        r["spec"] = {str(L): list(map(int, hs)) for L, hs in spec.items()}
        r["wall_s"] = time.time() - t0
        results.append(r)
        print(f"  {name}: top1={r['top1']*100:6.2f}%  frac(t>d)={r['frac_target_gt_distractor']*100:6.2f}%  "
              f"logit_diff={r['logit_diff_mean']:+6.3f}  ({r['wall_s']:.0f}s)")

    out_d = {
        "model": args.model, "revision": args.revision, "arch": args.arch,
        "top_k": args.top_k, "n_examples": args.n_examples,
        "s_inhibition_heads": [{"L": L, "H": H, "subj_max": sm, "subj_to_io": r, "io_attn": io_a, "sf": sf, "ss": ss}
                                for L, H, sm, r, io_a, sf, ss in picks],
        "name_mover_heads": nm_heads,
        "conditions": results,
    }
    with open(args.out, "w") as f: json.dump(out_d, f, indent=2)
    print(f"\nwrote {args.out}")


if __name__ == "__main__":
    main()
