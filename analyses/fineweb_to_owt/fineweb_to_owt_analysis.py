"""
fineweb_to_owt_analysis.py

Mid-training data-distribution-shift test on Karpathy GPT-2 124M.
The model was pretrained on FineWeb-10B through step 17600 and then continued
on OpenWebText starting step 17800. This script runs the per-head spectral
pipeline plus the all-head capability-specific screen at every available
checkpoint in BOTH phases and produces four comparison measurements:

  (M1) Per-head PR continuity at the boundary.
       Plot PR(L, H, t) for the top-30 (by PR-integral on FineWeb) heads
       through both phases, with a vertical marker at the transition.
       A smooth signal across step 17600 -> 17800 supports the claim that
       PR tracks *structural specialization* rather than corpus statistics.
       A discontinuity quantifies the data-specific component.

  (M2) Induction-circuit identity stability.
       At the last FineWeb checkpoint and the last OWT checkpoint, run the
       all-head capability-specific screen at induction-selectivity >= 50x.
       Compare head sets: identity overlap, selectivity-value change, layer
       distribution change.

  (M3) BOS-class fraction across the boundary.
       At every checkpoint in both phases, compute the whole-model BOS-class
       fraction (fraction of heads classified first-token at >= 30x). The
       FineWeb -> OWT transition is a within-model, within-procedure test of
       pure data-distribution effects on the attention-sink phenomenon, which
       isolates the data-contribution to the ~20pp DCLM-vs-Pile gap reported
       in the cross-architecture panel.

  (M4) Capability-selectivity stability for FineWeb-identified heads.
       For each FineWeb-endpoint induction-circuit head (e.g. L8H{8,10,5} on
       Karpathy 124M per INDUCTION_HEADS.md), track its induction-selectivity
       and prev-token-selectivity through the OWT continuation. Do the heads
       remain capability-selective, do they drift to a different selectivity
       class, or do they reset and recompose?

The script is a drop-in alongside the existing analyses/natural_text_124m/
pipeline. It reuses GPT/GPTConfig/load_karpathy_ckpt/build_induction_batch
from induction_heads_per_head_124m.py and the attention-pattern measurement
machinery from induction_heads_mechinterp_124m.py.

Run:

  python fineweb_to_owt_analysis.py \
      --fineweb-ckpt-dir <path>/karpathy_llmc/runs/gpt2_fineweb10B \
      --owt-ckpt-dir     <path>/karpathy_llmc/runs/gpt2_owt_continuation \
      --output-dir       results/fineweb_to_owt \
      [--device mps|cuda|cpu] \
      [--n-examples 2000] [--seq-len 256]

Produces:
  results/fineweb_to_owt/per_head_pr_trajectory.json
  results/fineweb_to_owt/mech_interp_endpoints.json
  results/fineweb_to_owt/bos_fraction_trajectory.json
  results/fineweb_to_owt/circuit_selectivity_trajectory.json
"""

import argparse
import json
import re
import sys
from pathlib import Path

import numpy as np
import torch
from torch.nn import functional as F

# Reuse the existing 124M pipeline machinery.
REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT / "analyses" / "natural_text_124m"))
from induction_heads_per_head_124m import (  # noqa: E402
    GPT, GPTConfig, load_karpathy_ckpt, build_induction_batch,
    compute_pr, per_head_pr_at_positions,
)


# ----------------------- attention-pattern measurement -----------------------

def attention_weights_last_query(model, tokens, n_layer, n_head, head_dim,
                                 device, batch_size=32):
    """Per-head softmaxed attention weights from the last query position to
    every position in the sequence.

    Returns a tensor of shape [N, n_layer, n_head, T].
    """
    n, T = tokens.shape
    out = torch.zeros(n, n_layer, n_head, T, dtype=torch.float32)
    captured = {}

    handles = []
    for L in range(n_layer):
        attn = model.transformer.h[L].attn

        def make_hook(L=L):
            def hook(module, ainputs, output):
                B_, T_, full_C = output.shape
                C = full_C // 3
                q, k, _v = output.split(C, dim=2)
                q = q.view(B_, T_, n_head, head_dim).transpose(1, 2)
                k = k.view(B_, T_, n_head, head_dim).transpose(1, 2)
                q_last = q[:, :, -1:, :]
                scores = (q_last @ k.transpose(-2, -1)) / (head_dim ** 0.5)
                attn_w = F.softmax(scores, dim=-1)
                captured[L] = attn_w[:, :, 0, :].detach()  # [B, H, T]
            return hook

        handles.append(attn.c_attn.register_forward_hook(make_hook()))

    try:
        with torch.no_grad():
            for start in range(0, n, batch_size):
                end = min(start + batch_size, n)
                tok = tokens[start:end].to(device)
                _ = model(tok)
                for L in range(n_layer):
                    out[start:end, L] = captured[L].cpu()
    finally:
        for h in handles:
            h.remove()
    return out


# ----------------------- selectivity & classification -----------------------

def per_head_selectivities(model, tokens, ab_positions, n_layer, n_head, head_dim,
                           device, seq_len, batch_size=32):
    """Compute mean attention from the last query position to each canonical
    target position class, and convert to a selectivity = mean(target) /
    uniform-other-baseline.

    Returns a dict of per-head selectivities for six classes:
      induction, previous_token, duplicate_token, first_token, self, local.
    """
    aw = attention_weights_last_query(model, tokens, n_layer, n_head, head_dim,
                                      device, batch_size=batch_size)
    N = tokens.shape[0]
    T = seq_len
    q = T - 1  # last query position

    # Target positions per example
    first_a = ab_positions  # the index of the first A; B is at first_a + 1
    b_pos = first_a + 1     # the induction target -- attend here to copy B
    prev_pos = torch.full((N,), q - 1, dtype=torch.long)
    self_pos = torch.full((N,), q, dtype=torch.long)
    first_tok = torch.zeros(N, dtype=torch.long)

    # Duplicate-token: position of the earlier occurrence of token at q.
    # In this batch, token at q is the second A; the first A is at first_a.
    duplicate_pos = first_a.clone()

    # Local: mean of attention over positions q-2..q-5
    local_idx = torch.stack([torch.arange(q - 5, q - 1) for _ in range(N)], dim=0)

    # Baseline = 1 / (T - 1) -- uniform attention probability over context.
    baseline = 1.0 / (T - 1)

    def gather_mean(target_idx_1d):
        # target_idx_1d: [N], pick aw[n, :, :, target] for each n
        idx = target_idx_1d.view(N, 1, 1).expand(N, n_layer, n_head)
        gathered = aw.gather(dim=3, index=idx.unsqueeze(-1)).squeeze(-1)
        return gathered.mean(dim=0)  # [n_layer, n_head]

    def gather_mean_local(local_idx_2d):
        # local_idx_2d: [N, K]
        K = local_idx_2d.shape[1]
        idx = local_idx_2d.view(N, 1, 1, K).expand(N, n_layer, n_head, K)
        gathered = aw.gather(dim=3, index=idx)
        return gathered.mean(dim=(0, 3))

    induction = gather_mean(b_pos)
    prev_token = gather_mean(prev_pos)
    duplicate = gather_mean(duplicate_pos)
    first_token = gather_mean(first_tok)
    self_attn = gather_mean(self_pos)
    local = gather_mean_local(local_idx)

    return {
        "induction": (induction / baseline).numpy(),
        "previous_token": (prev_token / baseline).numpy(),
        "duplicate_token": (duplicate / baseline).numpy(),
        "first_token": (first_token / baseline).numpy(),
        "self": (self_attn / baseline).numpy(),
        "local": (local / baseline).numpy(),
    }


def classify_heads(sel, threshold=30.0):
    """For each (L, H), pick the class with max selectivity above threshold;
    otherwise 'unclassified'. Returns dict[(L,H)] -> (class_name, sel)."""
    classes = list(sel.keys())
    n_layer, n_head = sel[classes[0]].shape
    out = {}
    for L in range(n_layer):
        for H in range(n_head):
            best_class, best_val = None, -float("inf")
            for c in classes:
                v = float(sel[c][L, H])
                if v > best_val:
                    best_val, best_class = v, c
            out[(L, H)] = (best_class, best_val) if best_val >= threshold else ("unclassified", best_val)
    return out


def bos_class_fraction(sel, threshold=30.0):
    """Fraction of (L, H) pairs whose best class is 'first_token' at >= threshold."""
    classification = classify_heads(sel, threshold=threshold)
    n = len(classification)
    n_bos = sum(1 for (_, (c, _)) in classification.items() if c == "first_token")
    return n_bos / n


def induction_circuit_at_threshold(sel, threshold=50.0):
    """Return list of (L, H) whose induction selectivity exceeds threshold."""
    ind = sel["induction"]
    n_layer, n_head = ind.shape
    return [(L, H, float(ind[L, H]))
            for L in range(n_layer) for H in range(n_head)
            if ind[L, H] >= threshold]


# ----------------------- main pipeline -----------------------

def discover_checkpoints(ckpt_dir):
    ck_paths = sorted(ckpt_dir.glob("ckpt_*.pt"))
    out = []
    for p in ck_paths:
        m = re.match(r"ckpt_(\d+)\.pt", p.name)
        if m:
            out.append((int(m.group(1)), p))
    return out


def first_a_positions(tokens):
    """For each example in `tokens`, find the first occurrence of the token
    that also appears at position seq_len-1 (which is the second-A token in
    the induction batch). Returns long tensor [N]."""
    N, T = tokens.shape
    last = tokens[:, -1]
    pos = torch.zeros(N, dtype=torch.long)
    for i in range(N):
        a = int(last[i])
        # find earliest position where this token appears (must be < T - 1)
        match = (tokens[i, :T - 1] == a).nonzero(as_tuple=False)
        if match.numel() == 0:
            pos[i] = -1
        else:
            pos[i] = int(match[0].item())
    return pos


def run(args):
    device = args.device
    if device == "auto":
        if torch.cuda.is_available():
            device = "cuda"
        elif torch.backends.mps.is_available():
            device = "mps"
        else:
            device = "cpu"
    print(f"device = {device}")

    cfg = GPTConfig()
    model = GPT(cfg).to(device)
    head_dim = cfg.n_embd // cfg.n_head

    print("Building induction batch...")
    tokens, positions, targets = build_induction_batch(
        n_examples=args.n_examples, seq_len=args.seq_len,
        rng=np.random.RandomState(args.seed)
    )
    print(f"  batch: {tokens.shape}; query position = {positions[0].item()}; "
          f"distinct B targets = {len(set(targets.tolist()))}")
    ab_pos = first_a_positions(tokens)
    if (ab_pos < 0).any():
        bad = int((ab_pos < 0).sum().item())
        raise RuntimeError(f"{bad} examples have no first-A occurrence in the constructed batch")

    fw_ckpts = discover_checkpoints(args.fineweb_ckpt_dir)
    owt_ckpts = discover_checkpoints(args.owt_ckpt_dir)
    print(f"FineWeb checkpoints: {len(fw_ckpts)} "
          f"(steps {fw_ckpts[0][0]}..{fw_ckpts[-1][0]})" if fw_ckpts else "0")
    print(f"OWT checkpoints:     {len(owt_ckpts)} "
          f"(steps {owt_ckpts[0][0]}..{owt_ckpts[-1][0]})" if owt_ckpts else "0")
    if not fw_ckpts:
        raise SystemExit(f"no FineWeb checkpoints found in {args.fineweb_ckpt_dir}")
    if not owt_ckpts:
        raise SystemExit(f"no OWT checkpoints found in {args.owt_ckpt_dir}")

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # ---------- M1 + M3: per-head PR + BOS-fraction at every checkpoint ----------
    print("\n[M1+M3] Per-head PR trajectory + BOS-class fraction across both phases...")
    per_head_keys = [f"L{L}_H{H}"
                     for L in range(cfg.n_layer) for H in range(cfg.n_head)]
    pr_traj = {"phase": [], "step": [], "pr": {k: [] for k in per_head_keys}}
    bos_traj = {"phase": [], "step": [], "bos_fraction": []}

    all_ckpts = [("fineweb", s, p) for (s, p) in fw_ckpts] + \
                [("owt", s, p) for (s, p) in owt_ckpts]
    for i, (phase, step, ckpath) in enumerate(all_ckpts):
        try:
            load_karpathy_ckpt(model, ckpath, device)
        except Exception as e:
            print(f"  [skip] {phase} step={step}: load failed: {e}")
            continue
        model.eval()
        pr_matrix = per_head_pr_at_positions(model, tokens, positions,
                                             cfg.n_layer, cfg.n_head, head_dim, device)
        sel = per_head_selectivities(model, tokens, ab_pos,
                                     cfg.n_layer, cfg.n_head, head_dim, device,
                                     seq_len=args.seq_len)
        bos_frac = bos_class_fraction(sel, threshold=30.0)

        pr_traj["phase"].append(phase)
        pr_traj["step"].append(step)
        for L in range(cfg.n_layer):
            for H in range(cfg.n_head):
                pr_traj["pr"][f"L{L}_H{H}"].append(float(pr_matrix[L, H]))
        bos_traj["phase"].append(phase)
        bos_traj["step"].append(step)
        bos_traj["bos_fraction"].append(float(bos_frac))

        if i % 5 == 0 or i == len(all_ckpts) - 1:
            print(f"  {phase} step={step:>6}  PR max={pr_matrix.max():.2f}  "
                  f"PR mean={pr_matrix.mean():.2f}  BOS-frac={bos_frac:.3f}")

    with open(output_dir / "per_head_pr_trajectory.json", "w") as f:
        json.dump(pr_traj, f)
    with open(output_dir / "bos_fraction_trajectory.json", "w") as f:
        json.dump(bos_traj, f)
    print(f"  wrote per_head_pr_trajectory.json + bos_fraction_trajectory.json")

    # ---------- M2: endpoint comparison ----------
    print("\n[M2] Induction-circuit identity at FineWeb endpoint vs OWT endpoint...")
    endpoint = {}
    for phase, ckpts in [("fineweb", fw_ckpts), ("owt", owt_ckpts)]:
        step, ckpath = ckpts[-1]
        load_karpathy_ckpt(model, ckpath, device)
        model.eval()
        sel = per_head_selectivities(model, tokens, ab_pos,
                                     cfg.n_layer, cfg.n_head, head_dim, device,
                                     seq_len=args.seq_len)
        circuit_50 = induction_circuit_at_threshold(sel, threshold=50.0)
        circuit_100 = induction_circuit_at_threshold(sel, threshold=100.0)
        endpoint[phase] = {
            "step": step,
            "induction_circuit_50x": circuit_50,
            "induction_circuit_100x": circuit_100,
            "selectivities": {
                cls: arr.tolist() for cls, arr in sel.items()
            },
        }
        print(f"  {phase} endpoint step={step}: "
              f"{len(circuit_50)} heads at >=50x, "
              f"{len(circuit_100)} heads at >=100x")

    # Compute set overlap
    fw_set_50 = {(L, H) for (L, H, _) in endpoint["fineweb"]["induction_circuit_50x"]}
    owt_set_50 = {(L, H) for (L, H, _) in endpoint["owt"]["induction_circuit_50x"]}
    overlap_50 = fw_set_50 & owt_set_50
    endpoint["set_comparison"] = {
        "fineweb_only_50x": sorted(list(fw_set_50 - owt_set_50)),
        "owt_only_50x": sorted(list(owt_set_50 - fw_set_50)),
        "shared_50x": sorted(list(overlap_50)),
        "jaccard_50x": (len(overlap_50) / len(fw_set_50 | owt_set_50)
                        if fw_set_50 | owt_set_50 else 1.0),
    }
    print(f"  identity overlap at >=50x: jaccard = "
          f"{endpoint['set_comparison']['jaccard_50x']:.3f} "
          f"(shared {len(overlap_50)}, "
          f"fw-only {len(fw_set_50 - owt_set_50)}, "
          f"owt-only {len(owt_set_50 - fw_set_50)})")
    with open(output_dir / "mech_interp_endpoints.json", "w") as f:
        json.dump(endpoint, f, indent=2)
    print(f"  wrote mech_interp_endpoints.json")

    # ---------- M4: selectivity trajectory for FineWeb-endpoint induction heads ----------
    print("\n[M4] Capability-selectivity trajectory for FineWeb-induction heads through OWT...")
    fw_heads = [(L, H) for (L, H, _) in endpoint["fineweb"]["induction_circuit_50x"]]
    if not fw_heads:
        print("  (FineWeb endpoint produced no >=50x induction heads; skipping M4)")
        return

    print(f"  tracking heads: {fw_heads}")
    sel_traj = {"phase": [], "step": [],
                "induction": {f"L{L}_H{H}": [] for (L, H) in fw_heads},
                "previous_token": {f"L{L}_H{H}": [] for (L, H) in fw_heads},
                "first_token": {f"L{L}_H{H}": [] for (L, H) in fw_heads}}

    for phase, ckpts in [("fineweb", fw_ckpts), ("owt", owt_ckpts)]:
        for (step, ckpath) in ckpts:
            try:
                load_karpathy_ckpt(model, ckpath, device)
            except Exception as e:
                print(f"  [skip] {phase} step={step}: {e}")
                continue
            model.eval()
            sel = per_head_selectivities(model, tokens, ab_pos,
                                         cfg.n_layer, cfg.n_head, head_dim, device,
                                         seq_len=args.seq_len)
            sel_traj["phase"].append(phase)
            sel_traj["step"].append(step)
            for (L, H) in fw_heads:
                sel_traj["induction"][f"L{L}_H{H}"].append(float(sel["induction"][L, H]))
                sel_traj["previous_token"][f"L{L}_H{H}"].append(float(sel["previous_token"][L, H]))
                sel_traj["first_token"][f"L{L}_H{H}"].append(float(sel["first_token"][L, H]))
    with open(output_dir / "circuit_selectivity_trajectory.json", "w") as f:
        json.dump(sel_traj, f)
    print(f"  wrote circuit_selectivity_trajectory.json")

    print("\nDone. Run build_fineweb_to_owt_figure.py to produce the comparison figure.")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--fineweb-ckpt-dir", type=Path, required=True,
                    help="Path to FineWeb-phase checkpoint directory (ckpt_*.pt files)")
    ap.add_argument("--owt-ckpt-dir", type=Path, required=True,
                    help="Path to OWT-continuation checkpoint directory")
    ap.add_argument("--output-dir", type=Path,
                    default=Path("results/fineweb_to_owt"))
    ap.add_argument("--device", default="auto", choices=["auto", "cuda", "mps", "cpu"])
    ap.add_argument("--n-examples", type=int, default=2000)
    ap.add_argument("--seq-len", type=int, default=256)
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()
    run(args)


if __name__ == "__main__":
    main()
