"""Diagnostic: probe the Pythia 1B VB result.

The original top-5 VB-screen heads are best-classified as first-token-dominant on
the standard 6-class screen. Two hypotheses:
  (a) The 5 heads are functionally heterogeneous; some interfere with VB, some
      support it; net group-ablation effect is dominated by the interferers.
  (b) All 5 are interferers (BOS-attractor-injecting); ablating any one helps.

Diagnostic conditions:
  - baseline
  - ablate each of original top-5 individually (5 conditions)
  - ablate filtered top-5 (best-class != first-token: L7H2, L4H4, L13H3, L7H0, L0H2)
"""
from __future__ import annotations
import argparse, json, time
import torch
from transformers import AutoTokenizer, GPTNeoXForCausalLM
from vb_batch import build_vb_batch, evaluate_vb
from prev_token_circuit_ablation import make_pre_hook, get_layer_module


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="EleutherAI/pythia-1b")
    ap.add_argument("--batch-size", type=int, default=4)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    device = "mps" if torch.backends.mps.is_available() else "cpu"
    tk = AutoTokenizer.from_pretrained(args.model)
    tokens, target_ids, distractor_ids, *_ = build_vb_batch(tk, n_examples=500)
    print(f"batch: {tuple(tokens.shape)}")

    model = GPTNeoXForCausalLM.from_pretrained(args.model, dtype=torch.float16).to(device).eval()
    cfg = model.config
    head_dim = cfg.hidden_size // cfg.num_attention_heads
    print(f"loaded {args.model}")

    def run(spec):
        handles = []
        for L, hs in spec.items():
            if not hs: continue
            h = get_layer_module(model, "pythia", L).register_forward_pre_hook(make_pre_hook(hs, head_dim))
            handles.append(h)
        try:
            return evaluate_vb(model, tokens, target_ids, distractor_ids, device, args.batch_size)
        finally:
            for h in handles: h.remove()

    # Individual ablations of original top-5
    orig_top5 = [(8, 5), (13, 1), (7, 2), (8, 7), (12, 1)]
    # Filtered top-5 (excluding best-class=first-token, by re-rank)
    filt_top5 = [(7, 2), (4, 4), (13, 3), (7, 0), (0, 2)]

    conditions = [("baseline", {})]
    for L, H in orig_top5:
        conditions.append((f"ablate_L{L}H{H}_only", {L: [H]}))
    conditions.append((f"ablate_filtered_top5 (non-BOS): {filt_top5}",
                       {L: [H for L_, H in filt_top5 if L_ == L] for L in {L for L, _ in filt_top5}}))
    # Make the filtered spec properly
    filt_spec = {}
    for L, H in filt_top5:
        filt_spec.setdefault(L, []).append(H)
    conditions[-1] = (f"ablate_filtered_top5_non-BOS ({len(filt_top5)}h)", filt_spec)

    results = []
    for name, spec in conditions:
        t0 = time.time()
        r = run(spec)
        r["condition"] = name
        r["spec"] = {str(L): list(map(int, hs)) for L, hs in spec.items()}
        r["wall_s"] = time.time() - t0
        results.append(r)
        print(f"  {name:<60} top1={r['top1']*100:6.2f}%  logit_diff={r['logit_diff_mean']:+6.3f}  ({r['wall_s']:.0f}s)")

    out = {"model": args.model, "arch": "pythia", "conditions": results}
    with open(args.out, "w") as f: json.dump(out, f, indent=2)
    print(f"\nwrote {args.out}")


if __name__ == "__main__":
    main()
