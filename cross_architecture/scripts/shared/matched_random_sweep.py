"""Re-run the matched-random control for each (model, task, screen) combination
across N seeds. Reports mean / std / [min, max] of Δtop-1 and Δlogit/target_logit.

For each (model, task), we sample N matched-random head sets in the same layers
as the screen-identified heads (no overlap with the screen picks), and ablate.

Loads the existing ablation JSONs to identify:
  - the screen-specific spec (to derive the matching layers)
  - the baseline metric values
and writes a new "matched_random_sweep.json" alongside.
"""
from __future__ import annotations
import argparse, json, time
from pathlib import Path
import numpy as np
import torch
from transformers import AutoTokenizer
from prev_token_circuit_ablation import make_pre_hook, get_layer_module


def run_matched_random_sweep(model, arch, screen_spec, task_eval_fn, tokens, target_ids,
                               device, head_dim, batch_size, n_seeds):
    """For each seed, sample matched-random in the same layers as screen_spec,
    ablate, evaluate, collect metric dict. Returns list of result dicts."""
    n_head = model.config.num_attention_heads
    results = []
    for seed in range(n_seeds):
        rng = np.random.RandomState(seed)
        rand_spec = {}
        for L, picks in screen_spec.items():
            eligible = [h for h in range(n_head) if h not in picks]
            n_samp = min(len(picks), len(eligible))
            rand_spec[L] = sorted(rng.choice(eligible, size=n_samp, replace=False).tolist())
        # Run ablation
        handles = []
        for L, hs in rand_spec.items():
            if not hs: continue
            h = get_layer_module(model, arch, L).register_forward_pre_hook(make_pre_hook(hs, head_dim))
            handles.append(h)
        try:
            r = task_eval_fn(model, tokens, target_ids, device, batch_size)
        finally:
            for h in handles: h.remove()
        r["seed"] = seed
        r["spec"] = {str(L): list(map(int, hs)) for L, hs in rand_spec.items()}
        results.append(r)
    return results


def load_screen_spec_from_ablation(abl_json, task):
    """Parse the screen-spec from an existing ablation JSON.
    task in {ioi_nm, ioi_si, gt, succ} — selects which condition is the 'screen' to match.
    """
    with open(abl_json) as f:
        d = json.load(f)
    # Find the first condition whose name starts with "ablate_top" (the screen-specific)
    for c in d["conditions"]:
        if c["condition"].startswith("ablate_top") and "matched_random" not in c["condition"]:
            spec = {int(L): hs for L, hs in c["spec"].items()}
            return spec, d
    raise ValueError(f"No 'ablate_top...' condition found in {abl_json}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--arch", required=True, choices=["pythia", "olmo", "olmoe"])
    ap.add_argument("--model", required=True)
    ap.add_argument("--task", required=True, choices=["ioi_nm", "ioi_si", "gt", "succ"])
    ap.add_argument("--ablation-json", required=True,
                    help="Existing ablation JSON whose screen spec to match")
    ap.add_argument("--n-seeds", type=int, default=10)
    ap.add_argument("--batch-size", type=int, default=4)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    device = "mps" if torch.backends.mps.is_available() else "cpu"
    tk = AutoTokenizer.from_pretrained(args.model)

    # Build the right batch + eval fn per task
    if args.task in ("ioi_nm", "ioi_si"):
        from ioi_batch import build_ioi_batch
        tokens, target_ids, distractors, _io, _sf, _ss, _recs = build_ioi_batch(tk, n_examples=500)
        from ioi_name_mover_ablation import evaluate_ioi
        def task_eval_fn(model, tok, tgt, dev, bs):
            return evaluate_ioi(model, tok, target_ids, distractors, dev, bs)
        metric_keys = ["top1", "frac_target_gt_distractor", "logit_diff_mean"]
    elif args.task == "gt":
        from gt_batch import build_gt_batch, build_two_digit_token_table, evaluate_gt
        tokens, decades, _recs = build_gt_batch(tk, n_examples=500)
        two_digit_ids = build_two_digit_token_table(tk)
        def task_eval_fn(model, tok, tgt, dev, bs):
            return evaluate_gt(model, tok, decades, two_digit_ids, dev, bs)
        target_ids = decades  # placeholder; eval_fn uses `decades` from closure
        metric_keys = ["top1_above", "prob_above", "logit_diff_above_below"]
    elif args.task == "succ":
        from succ_batch import build_succ_batch, evaluate_succ
        tokens, target_ids, _recs = build_succ_batch(tk)
        def task_eval_fn(model, tok, tgt, dev, bs):
            return evaluate_succ(model, tok, tgt, dev, bs)
        metric_keys = ["top1_acc", "target_logit_mean", "target_rank_median"]
    else:
        raise ValueError(args.task)

    print(f"batch: {tuple(tokens.shape)}; task={args.task}")
    print(f"loading {args.model}...")
    t0 = time.time()
    if args.arch == "pythia":
        from transformers import GPTNeoXForCausalLM
        model = GPTNeoXForCausalLM.from_pretrained(args.model, dtype=torch.float16)
    elif args.arch == "olmo":
        from transformers import OlmoForCausalLM
        model = OlmoForCausalLM.from_pretrained(args.model, dtype=torch.float16)
    else:
        from transformers import OlmoeForCausalLM
        model = OlmoeForCausalLM.from_pretrained(args.model, dtype=torch.float16)
    model = model.to(device).eval()
    head_dim = model.config.hidden_size // model.config.num_attention_heads
    print(f"  loaded in {time.time()-t0:.0f}s")

    screen_spec, abl_d = load_screen_spec_from_ablation(args.ablation_json, args.task)
    print(f"screen spec: {sum(len(v) for v in screen_spec.values())} heads across layers {sorted(screen_spec)}")

    # Sweep
    print(f"\nrunning matched-random sweep across {args.n_seeds} seeds...")
    sweep = run_matched_random_sweep(model, args.arch, screen_spec, task_eval_fn,
                                       tokens, target_ids, device, head_dim,
                                       args.batch_size, args.n_seeds)

    # Aggregate
    agg = {}
    for k in metric_keys:
        vals = np.array([r[k] for r in sweep])
        agg[k] = {
            "mean": float(vals.mean()), "std": float(vals.std(ddof=1)),
            "min": float(vals.min()), "max": float(vals.max()),
            "values": vals.tolist(),
        }
    # Baseline (from first condition of original ablation)
    base = abl_d["conditions"][0]
    # Δ values
    delta_agg = {}
    for k in metric_keys:
        base_val = base[k]
        vals = np.array([r[k] - base_val for r in sweep])
        if k == "top1" or k == "top1_above" or k == "top1_acc":
            vals = vals * 100  # pp
        delta_agg[k] = {
            "mean": float(vals.mean()), "std": float(vals.std(ddof=1)),
            "min": float(vals.min()), "max": float(vals.max()),
        }

    print(f"\nmatched-random sweep results (n={args.n_seeds}):")
    for k in metric_keys:
        a = agg[k]; d = delta_agg[k]
        print(f"  {k:>30}:  abs mean±std = {a['mean']:+8.3f} ± {a['std']:6.3f}   "
              f"Δ mean±std = {d['mean']:+8.3f} ± {d['std']:6.3f}   "
              f"Δ range = [{d['min']:+7.3f}, {d['max']:+7.3f}]")

    out = {
        "model": args.model, "arch": args.arch, "task": args.task,
        "n_seeds": args.n_seeds,
        "screen_spec": {str(L): hs for L, hs in screen_spec.items()},
        "baseline": {k: base[k] for k in metric_keys},
        "absolute": agg,
        "delta": delta_agg,
        "per_seed": sweep,
    }
    with open(args.out, "w") as f:
        json.dump(out, f, indent=2)
    print(f"\nwrote {args.out}")


if __name__ == "__main__":
    main()
