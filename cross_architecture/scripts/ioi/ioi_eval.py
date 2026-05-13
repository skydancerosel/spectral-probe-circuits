"""IOI baseline + ablation eval across the three 1B models.

Eval metric: at the final token position ("to"), compute:
  - top-1 accuracy on target (IO name)
  - logit_diff = logit(target) - logit(distractor)
  - fraction where logit(target) > logit(distractor)
  - mean logit_target, mean logit_distractor

Conditions:
  1. baseline
  2. ablate prev-token circuit (from mech-interp, best-class==previous-token, sel>=100x)
  3. ablate induction circuit (induction sel >=50x)
  4. ablate prev-token + induction (union)
"""
from __future__ import annotations

import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "shared"))  # noqa: E402
import argparse, json, math, time
import numpy as np
import torch
import torch.nn.functional as F
from transformers import AutoTokenizer
from ioi_batch import build_ioi_batch
from prev_token_circuit_ablation import make_pre_hook, get_layer_module, best_class, pick_circuit


def evaluate_ioi(model, tokens, targets, distractors, device, batch_size=4):
    n = tokens.shape[0]; last = tokens.shape[1] - 1
    top1, lt_gt_ld, lt_vals, ld_vals = [], [], [], []
    with torch.no_grad():
        for s in range(0, n, batch_size):
            e = min(s + batch_size, n)
            tok = tokens[s:e].to(device); tgt = targets[s:e].to(device); dis = distractors[s:e].to(device)
            logits = model(tok).logits[:, last, :]
            top1_pred = logits.argmax(-1)
            top1.append((top1_pred == tgt).float().cpu().numpy())
            lt = logits.gather(-1, tgt.unsqueeze(-1)).squeeze(-1)
            ld = logits.gather(-1, dis.unsqueeze(-1)).squeeze(-1)
            lt_gt_ld.append((lt > ld).float().cpu().numpy())
            lt_vals.append(lt.cpu().numpy()); ld_vals.append(ld.cpu().numpy())
    lt = np.concatenate(lt_vals); ld = np.concatenate(ld_vals)
    return {
        "top1": float(np.concatenate(top1).mean()),
        "frac_target_gt_distractor": float(np.concatenate(lt_gt_ld).mean()),
        "logit_diff_mean": float((lt - ld).mean()),
        "logit_diff_median": float(np.median(lt - ld)),
        "logit_target_mean": float(lt.mean()),
        "logit_distractor_mean": float(ld.mean()),
    }


def run_condition(model, arch, spec, tokens, targets, distractors, device, head_dim, batch_size):
    handles = []
    for L, hs in spec.items():
        if not hs: continue
        h = get_layer_module(model, arch, L).register_forward_pre_hook(make_pre_hook(hs, head_dim))
        handles.append(h)
    try:
        return evaluate_ioi(model, tokens, targets, distractors, device, batch_size)
    finally:
        for h in handles: h.remove()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--arch", required=True, choices=["pythia", "olmo", "olmoe"])
    ap.add_argument("--model", required=True)
    ap.add_argument("--revision", default="main")
    ap.add_argument("--mechinterp-json", required=True)
    ap.add_argument("--n-examples", type=int, default=500)
    ap.add_argument("--batch-size", type=int, default=4)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    device = "mps" if torch.backends.mps.is_available() else "cpu"
    print(f"device: {device}")

    tk = AutoTokenizer.from_pretrained(args.model)
    tokens, targets, distractors, records = build_ioi_batch(tk, n_examples=args.n_examples)
    print(f"IOI batch: {tuple(tokens.shape)}, target=IO name, distractor=subject name")

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

    mech = json.load(open(args.mechinterp_json))
    all_sel = mech["all_head_selectivity"]

    prev_circuit = pick_circuit(all_sel, "previous-token", 100.0, require_best_class=True)
    induction_circuit = pick_circuit(all_sel, "induction", 50.0, require_best_class=False)

    # Union of prev + induction
    union = {}
    for spec in [prev_circuit, induction_circuit]:
        for L, hs in spec.items():
            union.setdefault(L, set()).update(hs)
    union = {L: sorted(list(hs)) for L, hs in union.items()}

    print(f"prev-token circuit: {sum(len(v) for v in prev_circuit.values())}h  layers={sorted(prev_circuit)}")
    print(f"induction circuit:  {sum(len(v) for v in induction_circuit.values())}h  layers={sorted(induction_circuit)}")
    print(f"union:              {sum(len(v) for v in union.values())}h  layers={sorted(union)}")

    conditions = [
        ("baseline", {}),
        (f"ablate_prev_token_circuit ({sum(len(v) for v in prev_circuit.values())}h)", prev_circuit),
        (f"ablate_induction_circuit ({sum(len(v) for v in induction_circuit.values())}h)", induction_circuit),
        (f"ablate_prev+induction_union ({sum(len(v) for v in union.values())}h)", union),
    ]
    results = []
    for name, spec in conditions:
        print(f"\n--- {name} ---")
        t0 = time.time()
        r = run_condition(model, args.arch, spec, tokens, targets, distractors, device, head_dim, args.batch_size)
        r["condition"] = name
        r["n_ablated"] = sum(len(v) for v in spec.values())
        r["spec"] = {str(L): list(map(int, hs)) for L, hs in spec.items()}
        r["wall_s"] = time.time() - t0
        results.append(r)
        print(f"  top1={r['top1']*100:.2f}%  frac(t>d)={r['frac_target_gt_distractor']*100:.2f}%  "
              f"logit_diff={r['logit_diff_mean']:+.3f}  ({r['wall_s']:.0f}s)")

    out = {
        "model": args.model, "revision": args.revision, "arch": args.arch,
        "n_examples": args.n_examples, "tokens_shape": list(tokens.shape),
        "n_layer": n_layer, "n_head": n_head, "head_dim": head_dim,
        "conditions": results,
    }
    with open(args.out, "w") as f: json.dump(out, f, indent=2)
    print(f"\nwrote {args.out}")


if __name__ == "__main__":
    main()
