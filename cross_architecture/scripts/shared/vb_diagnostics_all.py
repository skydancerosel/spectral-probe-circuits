"""Run the same VB diagnostic on OLMo and OLMoE: individual head ablations of
the top-5 VB-screen heads, plus group ablation of the best-class!=first-token
filtered top-5.
"""
from __future__ import annotations
import argparse, json, math, time
import torch
from transformers import AutoTokenizer
from vb_batch import build_vb_batch, evaluate_vb
from prev_token_circuit_ablation import make_pre_hook, get_layer_module


def best_class(vs):
    bc, bv = None, 0.0
    for c, v in vs.items():
        if v and not (isinstance(v, float) and math.isnan(v)) and v > bv: bc, bv = c, v
    return bc, bv


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--arch", required=True, choices=["olmo", "olmoe"])
    ap.add_argument("--model", required=True)
    ap.add_argument("--vb-mechinterp-json", required=True)
    ap.add_argument("--final-mechinterp-json", required=True)
    ap.add_argument("--batch-size", type=int, default=4)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    device = "mps" if torch.backends.mps.is_available() else "cpu"
    tk = AutoTokenizer.from_pretrained(args.model)
    tokens, target_ids, distractor_ids, *_ = build_vb_batch(tk, n_examples=500)
    print(f"batch: {tuple(tokens.shape)}")

    if args.arch == "olmo":
        from transformers import OlmoForCausalLM as Cls
    else:
        from transformers import OlmoeForCausalLM as Cls
    model = Cls.from_pretrained(args.model, dtype=torch.float16).to(device).eval()
    head_dim = model.config.hidden_size // model.config.num_attention_heads
    print(f"loaded {args.model}")

    def run(spec):
        handles = []
        for L, hs in spec.items():
            if not hs: continue
            h = get_layer_module(model, args.arch, L).register_forward_pre_hook(make_pre_hook(hs, head_dim))
            handles.append(h)
        try:
            return evaluate_vb(model, tokens, target_ids, distractor_ids, device, args.batch_size)
        finally:
            for h in handles: h.remove()

    # Original top-5
    with open(args.vb_mechinterp_json) as f: vb = json.load(f)
    orig_top5 = [(h["L"], h["H"]) for h in vb["top25_vb_candidates"][:5]]
    # Filtered top-5 (best-class != first-token)
    with open(args.final_mechinterp_json) as f: cl = json.load(f)
    cl_sel = cl["all_head_selectivity"]
    ranked = []
    for h in vb["top25_vb_candidates"]:
        lh = f"L{h['L']}_H{h['H']}"
        bc, _ = best_class(cl_sel[lh])
        ranked.append((h["L"], h["H"], h["vb_sel"], bc))
    # Also scan beyond top-25 if needed
    vb_all = vb["all_head_vb_selectivity"]
    extra = []
    for lh, vs in vb_all.items():
        L = int(lh.split("_")[0][1:]); H = int(lh.split("_")[1][1:])
        if any((L, H) == (a, b) for a, b, *_ in ranked): continue
        bc, _ = best_class(cl_sel[lh])
        extra.append((L, H, vs["vb_selectivity"], bc))
    extra.sort(key=lambda x: -x[2])
    full = ranked + extra
    filt = [(L, H, sel, bc) for L, H, sel, bc in full if bc != "first-token"][:5]
    filt_spec = {}
    for L, H, *_ in filt:
        filt_spec.setdefault(L, []).append(H)

    print(f"\noriginal top-5: {orig_top5}")
    print(f"filtered top-5 (non-BOS): {[(L, H, f'{sel:.2f}', bc) for L, H, sel, bc in filt]}")

    conditions = [("baseline", {})]
    for L, H in orig_top5:
        conditions.append((f"ablate_L{L}H{H}_only", {L: [H]}))
    conditions.append((f"ablate_filtered_top5_non-BOS ({sum(len(v) for v in filt_spec.values())}h)", filt_spec))

    results = []
    for name, spec in conditions:
        t0 = time.time()
        r = run(spec)
        r["condition"] = name
        r["spec"] = {str(L): list(map(int, hs)) for L, hs in spec.items()}
        r["wall_s"] = time.time() - t0
        results.append(r)
        print(f"  {name:<60} top1={r['top1']*100:6.2f}%  logit_diff={r['logit_diff_mean']:+6.3f}  ({r['wall_s']:.0f}s)")

    out = {"model": args.model, "arch": args.arch,
           "orig_top5": orig_top5,
           "filtered_top5": [(L, H, sel, bc) for L, H, sel, bc in filt],
           "conditions": results}
    with open(args.out, "w") as f: json.dump(out, f, indent=2)
    print(f"\nwrote {args.out}")


if __name__ == "__main__":
    main()
