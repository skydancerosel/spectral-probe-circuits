"""Prev-token circuit ablation across the three 1B models.

Identifies "prev-token circuit" heads at the final checkpoint:
  best-class == 'previous-token' AND previous-token selectivity >= threshold (default 100x)

Conditions per model:
  1. baseline
  2. ablate_prev_token_circuit (group)
  3. ablate_matched_random_same_layers
  4. ablate_induction_circuit (group of induction-selectivity ≥50x heads — known)

All ablations measured on the synthetic induction batch (seed 42); the question this
answers is the compositional one: does ablating prev-token heads tank induction top-1?
(Prev-token heads compose into induction via the K-vector building step.)

Supports --arch pythia | olmo | olmoe.
"""
from __future__ import annotations
import argparse, json, math, time
import numpy as np
import torch
import torch.nn.functional as F
from mamba2_per_head import build_induction_batch


def make_pre_hook(heads_in_layer, head_dim):
    lo_his = [(h * head_dim, (h + 1) * head_dim) for h in heads_in_layer]
    def pre_hook(_module, ainputs):
        x = ainputs[0].clone()
        for lo, hi in lo_his:
            x[..., lo:hi] = 0
        return (x,) + ainputs[1:]
    return pre_hook


def evaluate(model, tokens, targets, device, batch_size=4):
    n = tokens.shape[0]; last = tokens.shape[1] - 1
    losses, a1, a5 = [], [], []
    s_lb = 0.0
    with torch.no_grad():
        for s in range(0, n, batch_size):
            e = min(s + batch_size, n)
            tok = tokens[s:e].to(device); tgt = targets[s:e].to(device)
            logits = model(tok).logits[:, last, :]
            losses.append(F.cross_entropy(logits, tgt, reduction="none").cpu().numpy())
            a1.append((logits.argmax(-1) == tgt).float().cpu().numpy())
            top5 = logits.topk(5, -1).indices
            a5.append((top5 == tgt.unsqueeze(-1)).any(-1).float().cpu().numpy())
            s_lb += logits.gather(-1, tgt.unsqueeze(-1)).squeeze(-1).sum().item()
    return {
        "loss": float(np.concatenate(losses).mean()),
        "acc_top1": float(np.concatenate(a1).mean()),
        "acc_top5": float(np.concatenate(a5).mean()),
        "mean_logit_B": float(s_lb / n),
    }


def get_layer_module(model, arch, L):
    if arch == "pythia":
        return model.gpt_neox.layers[L].attention.dense
    elif arch in ("olmo", "olmoe"):
        return model.model.layers[L].self_attn.o_proj
    raise ValueError(arch)


def run_condition(model, arch, spec, tokens, targets, device, head_dim, batch_size):
    handles = []
    for L, hs in spec.items():
        if not hs: continue
        h = get_layer_module(model, arch, L).register_forward_pre_hook(make_pre_hook(hs, head_dim))
        handles.append(h)
    try:
        return evaluate(model, tokens, targets, device, batch_size)
    finally:
        for h in handles: h.remove()


def best_class(sels):
    best_c, best_v = None, 0.0
    for c, v in sels.items():
        if v is None or (isinstance(v, float) and math.isnan(v)): continue
        if v > best_v: best_c, best_v = c, v
    return best_c, best_v


def pick_circuit(all_sel, capability, threshold, require_best_class=True):
    """Heads with sels[capability] >= threshold and (optionally) best-class == capability."""
    out = {}
    for k, sels in all_sel.items():
        v = sels.get(capability, 0.0) or 0.0
        if isinstance(v, float) and math.isnan(v): continue
        if v < threshold: continue
        if require_best_class:
            bc, _ = best_class(sels)
            if bc != capability: continue
        L = int(k.split("_")[0][1:]); H = int(k.split("_")[1][1:])
        out.setdefault(L, []).append(H)
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--arch", required=True, choices=["pythia", "olmo", "olmoe"])
    ap.add_argument("--model", required=True, help="HF model id")
    ap.add_argument("--revision", default="main")
    ap.add_argument("--mechinterp-json", required=True, help="Final-ckpt mech-interp JSON with all_head_selectivity")
    ap.add_argument("--prev-token-threshold", type=float, default=100.0)
    ap.add_argument("--induction-threshold", type=float, default=50.0)
    ap.add_argument("--n-examples", type=int, default=2000)
    ap.add_argument("--seq-len", type=int, default=256)
    ap.add_argument("--batch-size", type=int, default=4)
    ap.add_argument("--random-seed", type=int, default=123)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    device = "mps" if torch.backends.mps.is_available() else "cpu"
    rng = np.random.RandomState(42)
    tokens, _, targets = build_induction_batch(args.n_examples, args.seq_len, rng=rng)
    print(f"induction batch: {tuple(tokens.shape)}  device: {device}")

    # Load model
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

    # Load mech-interp
    mech = json.load(open(args.mechinterp_json))
    all_sel = mech["all_head_selectivity"]

    # Prev-token circuit (best-class filter to exclude BOS-dominated heads)
    prev_circuit = pick_circuit(all_sel, "previous-token", args.prev_token_threshold, require_best_class=True)
    n_prev = sum(len(v) for v in prev_circuit.values())
    print(f"\nPrev-token circuit (best-class=previous-token, sel>={args.prev_token_threshold}x): {n_prev} heads")
    for L in sorted(prev_circuit): print(f"  L{L}: {prev_circuit[L]}")

    # Induction circuit (>=50x induction selectivity; no best-class filter — matches existing methodology)
    induction_circuit = pick_circuit(all_sel, "induction", args.induction_threshold, require_best_class=False)
    n_ind = sum(len(v) for v in induction_circuit.values())
    print(f"\nInduction circuit (induction sel >= {args.induction_threshold}x): {n_ind} heads")
    for L in sorted(induction_circuit): print(f"  L{L}: {induction_circuit[L]}")

    # Matched-random for prev-token circuit (same layers, no overlap, cap to eligible)
    rng_c = np.random.RandomState(args.random_seed)
    matched_random = {}
    for L, picks in prev_circuit.items():
        eligible = [h for h in range(n_head) if h not in picks]
        n_samp = min(len(picks), len(eligible))
        matched_random[L] = sorted(rng_c.choice(eligible, size=n_samp, replace=False).tolist())
    n_matched = sum(len(v) for v in matched_random.values())
    print(f"\nMatched-random for prev-token circuit: {n_matched} heads in layers {sorted(matched_random)}")

    # Run conditions
    conditions = [
        ("baseline", {}),
        (f"ablate_prev_token_circuit ({n_prev}h)", prev_circuit),
        (f"ablate_matched_random_same_layers ({n_matched}h)", matched_random),
        (f"ablate_induction_circuit ({n_ind}h)", induction_circuit),
    ]
    results = []
    for name, spec in conditions:
        print(f"\n--- {name} ---")
        t0 = time.time()
        r = run_condition(model, args.arch, spec, tokens, targets, device, head_dim, args.batch_size)
        r["condition"] = name
        r["n_ablated"] = sum(len(v) for v in spec.values())
        r["spec"] = {str(L): list(map(int, hs)) for L, hs in spec.items()}
        r["wall_s"] = time.time() - t0
        results.append(r)
        print(f"  loss={r['loss']:.4f}  top1={r['acc_top1']*100:.2f}%  top5={r['acc_top5']*100:.2f}%  logit_B={r['mean_logit_B']:.2f}  ({r['wall_s']:.0f}s)")

    out = {
        "model": args.model, "revision": args.revision, "arch": args.arch,
        "n_layer": n_layer, "n_head": n_head, "head_dim": head_dim,
        "prev_token_threshold": args.prev_token_threshold,
        "induction_threshold": args.induction_threshold,
        "n_prev_token_circuit": n_prev, "n_induction_circuit": n_ind,
        "n_matched_random": n_matched,
        "conditions": results,
    }
    with open(args.out, "w") as f: json.dump(out, f, indent=2)
    print(f"\nwrote {args.out}")


if __name__ == "__main__":
    main()
