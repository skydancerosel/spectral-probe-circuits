"""Ablate the top-K name-mover candidates (from ioi_mechinterp) and measure
effect on IOI top-1 and logit_diff. Compare to matched-random control.
"""
from __future__ import annotations
import argparse, json, time
import numpy as np
import torch
import torch.nn.functional as F
from transformers import AutoTokenizer
from ioi_batch import build_ioi_batch
from prev_token_circuit_ablation import make_pre_hook, get_layer_module


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


def to_spec(heads):
    spec = {}
    for L, H in heads:
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

    # Load name-mover candidates
    mi = json.load(open(args.ioi_mechinterp_json))
    top_nm = mi["top25_name_mover"][:args.top_k]
    nm_heads = [(h["L"], h["H"]) for h in top_nm]
    nm_spec = to_spec(nm_heads)
    print(f"\nTop-{args.top_k} name-mover candidates: {nm_heads}")
    print(f"  spec: {nm_spec}")

    # Matched-random: same layers, no overlap
    rng = np.random.RandomState(args.random_seed)
    rand_spec = {}
    for L, picks in nm_spec.items():
        eligible = [h for h in range(n_head) if h not in picks]
        n_samp = min(len(picks), len(eligible))
        rand_spec[L] = sorted(rng.choice(eligible, size=n_samp, replace=False).tolist())
    print(f"  matched-random spec: {rand_spec}")

    # All heads in name-mover layers (upper bound)
    upper_spec = {L: list(range(n_head)) for L in nm_spec}

    conditions = [
        ("baseline", {}),
        (f"ablate_top{args.top_k}_name_movers", nm_spec),
        (f"ablate_matched_random_same_layers ({sum(len(v) for v in rand_spec.values())}h)", rand_spec),
        (f"ablate_full_name_mover_layers (upper bound, {sum(len(v) for v in upper_spec.values())}h)", upper_spec),
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
        "name_mover_heads": nm_heads, "conditions": results,
    }
    with open(args.out, "w") as f: json.dump(out_d, f, indent=2)
    print(f"\nwrote {args.out}")


if __name__ == "__main__":
    main()
