"""
s123_l0_investigation.py

Question: s123 is the only seed (of 6) where ablating L0H{3,6,14,15}
on the full-trained checkpoint has ZERO effect. What is L0 doing on
s123 instead?

Three hypotheses to discriminate:

  H1. L0 is "dead" — no useful computation; the model bypasses L0
  H2. L0 does something else — content-dependent work, not retrieval
  H3. L0 does retrieval but with DIFFERENT heads (not {3,6,14,15})

Tests:

  Test A — Per-head attention pattern (mech-interp): measure attn from
    QUERY-read position back to KEY position for ALL 16 L0 heads on
    s123 at ckpt 10000. Compare to s42 (where L0H{3,6,14,15} have
    selectivity 42-95x).
    - If any L0 heads on s123 have selectivity > 30x: hypothesis H3
      (different specific heads do the work)
    - If all L0 heads on s123 have selectivity < 5x: hypothesis H1
      (L0 dead) or H2 (L0 doing other work)

  Test B — Per-head PR trajectories on L0: load s123's per_head data
    and look at PR over training for all 16 L0 heads.
    - If all L0 heads show flat low PR: H1 confirmed (L0 never
      developed content-dependence)
    - If L0 heads show high PR but not the {3,6,14,15} subset: H3

  Test C — Ablate full L0 on s123 (16 heads at once): if even ablating
    all of L0 has zero effect, then H1 (L0 truly dead). If full-L0
    ablation tanks pin but {3,6,14,15} alone doesn't, then H3 (other
    L0 heads carry the substrate).

Output: analyses/s123_l0_investigation.{json,png}
"""

import json
import sys
from pathlib import Path
import numpy as np
import torch
import torch.nn.functional as F
import matplotlib.pyplot as plt

REPO = Path(__file__).resolve().parent
sys.path.insert(0, str(REPO / "training"))
from config import Config, get_device
from model import GPTModel
from dataset import build_datasets
from pilot import evaluate_probe, evaluate_lm
from torch.utils.data import DataLoader

S123_DIR = REPO / "runs/beta2_ablation/pilot_wd0.5_lr0.001_lp2.0_b20.95_s123"
S42_DIR = REPO / "runs/beta2_ablation/pilot_wd0.5_lr0.001_lp2.0_b20.95_s42"


# ── Test A: per-head L0 attention to KEY ──────────────────────────────

def measure_l0_key_attention(model, run_dir, ckpt_step, n_examples=200):
    """For each of 16 L0 heads, measure attn from query-read to KEY position."""
    device = get_device()
    cfg = Config(seed=42, p_probe=0.10, batch_size=64,
                  n_layer=8, d_model=512, n_head=16, d_ff=2048)
    n_head = cfg.n_head
    head_dim = cfg.d_model // n_head

    # Load codewords for the right run
    cw_path = run_dir / "codewords.json"
    data = build_datasets(cfg, codewords_path=str(cw_path) if cw_path.exists() else None)

    # Use probe_eval_in batch
    probe_in = data["probe_eval_in"]
    probes = probe_in.probes[:n_examples]

    # Load the model weights
    ck = torch.load(run_dir / f"ckpt_{ckpt_step:06d}.pt",
                     map_location=device, weights_only=True)
    model.load_state_dict(ck["model_state_dict"])
    model.eval()

    # For each probe, find KEY position and QUERY-read position
    # Use the existing probe_in's structure: input_ids, key_pos, query_pos
    # build_datasets returns probe_eval_in as a dataset object with .probes

    attn_to_key = np.zeros((n_head,))
    attn_to_baseline = np.zeros((n_head,))
    n_valid = 0

    captured = {}

    def make_hook():
        def hook(module, ainputs, output):
            # output is qkv: [B, T, 3*d_model]
            B, T, _ = output.shape
            C = output.shape[-1] // 3
            q, k, v = output.split(C, dim=2)
            q = q.view(B, T, n_head, head_dim).transpose(1, 2)
            k = k.view(B, T, n_head, head_dim).transpose(1, 2)
            scores = (q @ k.transpose(-2, -1)) / (head_dim ** 0.5)
            # Causal mask
            mask = torch.full((T, T), float("-inf"), device=q.device)
            mask = torch.triu(mask, diagonal=1)
            scores = scores + mask
            captured["w"] = F.softmax(scores, dim=-1).detach()
        return hook

    handle = model.blocks[0].attn.qkv.register_forward_hook(make_hook())
    rng = np.random.RandomState(0)
    try:
        with torch.no_grad():
            for probe in probes:
                input_ids = torch.tensor([probe["input_ids"]]).to(device)
                key_pos = probe["key_pos"]
                query_pos = probe["query_pos"]
                if key_pos < 0 or query_pos < 0 or key_pos >= input_ids.shape[1]:
                    continue
                _ = model(input_ids)
                w = captured["w"][0]  # [n_head, T, T]
                # Attention from query_pos to key_pos
                attn_qp_to_kp = w[:, query_pos, key_pos].cpu().numpy()
                # Baseline: attention to a random position before query_pos, not key_pos
                T = input_ids.shape[1]
                rp = rng.randint(0, query_pos)
                while rp == key_pos:
                    rp = rng.randint(0, query_pos)
                attn_qp_to_rp = w[:, query_pos, rp].cpu().numpy()
                attn_to_key += attn_qp_to_kp
                attn_to_baseline += attn_qp_to_rp
                n_valid += 1
    finally:
        handle.remove()

    if n_valid == 0:
        return None
    attn_to_key /= n_valid
    attn_to_baseline /= n_valid
    selectivity = attn_to_key / np.maximum(attn_to_baseline, 1e-8)
    return {"attn_to_key": attn_to_key.tolist(),
            "attn_to_baseline": attn_to_baseline.tolist(),
            "selectivity": selectivity.tolist(),
            "n_valid": n_valid}


# ── Test B: PR trajectories of L0 heads ──────────────────────────────

def get_l0_pr_features(seed_tag):
    spec = json.load(open(REPO / f"analyses/probe_circuit_per_head_{seed_tag}.json"))
    out = []
    for h in range(16):
        traj = np.array(spec["pr"][f"L0_H{h}"])
        out.append({
            "head": h,
            "min_pr": float(traj.min()),
            "max_pr": float(traj.max()),
            "spread": float(traj.max() - traj.min()),
            "final_pr": float(traj[-1]),
            "trajectory": traj.tolist(),
        })
    return out, spec["ckpt_step"]


# ── Test C: ablate full L0 on s123 ────────────────────────────────────

def make_pre_hook(ablated_heads, n_head, head_dim):
    def pre_hook(_module, ainputs):
        x = ainputs[0]
        B, T, C = x.shape
        xr = x.view(B, T, n_head, head_dim).clone()
        for h in ablated_heads:
            xr[:, :, h, :] = 0.0
        return (xr.view(B, T, C),)
    return pre_hook


def ablate_and_eval(model, ablation_spec, probe_in, probe_ood, val_loader,
                     device, n_head, head_dim):
    handles = []
    for layer_idx, heads in ablation_spec.items():
        if not heads:
            continue
        h = model.blocks[layer_idx].attn.out_proj.register_forward_pre_hook(
            make_pre_hook(heads, n_head, head_dim)
        )
        handles.append(h)
    try:
        pin = evaluate_probe(model, probe_in, device)
        pood = evaluate_probe(model, probe_ood, device)
        vl = evaluate_lm(model, val_loader, device)
    finally:
        for h in handles:
            h.remove()
    return {"probe_in_acc": pin, "probe_ood_acc": pood, "val_loss": vl}


def run_ablation_test_c(ckpt_step):
    device = get_device()
    print(f"\n=== Test C: full-L0 ablation on s123 at step {ckpt_step} ===")

    cfg = Config(seed=42, p_probe=0.10, batch_size=64,
                  n_layer=8, d_model=512, n_head=16, d_ff=2048)
    n_head = cfg.n_head
    head_dim = cfg.d_model // n_head

    cw_path = S123_DIR / "codewords.json"
    data = build_datasets(cfg, codewords_path=str(cw_path) if cw_path.exists() else None)
    vocab_size = len(data["tokenizer"])
    val_loader = DataLoader(data["val_dataset"], batch_size=64,
                              shuffle=False, drop_last=False, num_workers=0)
    probe_in = data["probe_eval_in"]
    probe_ood = data["probe_eval_ood"]

    model = GPTModel(
        vocab_size=vocab_size, seq_len=cfg.seq_len,
        d_model=cfg.d_model, n_layer=cfg.n_layer,
        n_head=cfg.n_head, d_ff=cfg.d_ff, dropout=0.0,
    ).to(device)

    ck = torch.load(S123_DIR / f"ckpt_{ckpt_step:06d}.pt",
                     map_location=device, weights_only=True)
    model.load_state_dict(ck["model_state_dict"])
    model.eval()

    conditions = [
        ("baseline", {}),
        ("ablate_L0H{3,6,14,15} (s42 picks)", {0: [3, 6, 14, 15]}),
        ("ablate_full_L0 (all 16)", {0: list(range(16))}),
        ("ablate_random_4_L0", {0: [0, 1, 5, 7]}),  # control
    ]

    results = {}
    print(f"  {'condition':<40} {'pin':>8} {'pood':>8} {'val':>8}")
    for name, spec in conditions:
        r = ablate_and_eval(model, spec, probe_in, probe_ood, val_loader,
                              device, n_head, head_dim)
        print(f"  {name:<40} {r['probe_in_acc']:>8.4f} "
              f"{r['probe_ood_acc']:>8.4f} {r['val_loss']:>8.4f}")
        results[name] = r
    return results


def main():
    device = get_device()
    print(f"device = {device}")

    # ── Test A: L0 attention selectivity on s123 ──────────────────────
    print("\n=== Test A: L0 attention to KEY (s123 vs s42) ===")

    cfg = Config(seed=42, p_probe=0.10, batch_size=64,
                  n_layer=8, d_model=512, n_head=16, d_ff=2048)
    cw_path = S123_DIR / "codewords.json"
    data = build_datasets(cfg, codewords_path=str(cw_path) if cw_path.exists() else None)
    vocab_size = len(data["tokenizer"])

    model = GPTModel(
        vocab_size=vocab_size, seq_len=cfg.seq_len,
        d_model=cfg.d_model, n_layer=cfg.n_layer,
        n_head=cfg.n_head, d_ff=cfg.d_ff, dropout=0.0,
    ).to(device)

    print("\n-- s123 L0 heads --")
    s123_l0_attn = measure_l0_key_attention(model, S123_DIR, ckpt_step=10000)
    if s123_l0_attn:
        print(f"  {'head':<8} {'attn→KEY':>10} {'attn→rand':>10} {'sel':>8}")
        for h in range(16):
            note = " ← s42-circuit head" if h in [3, 6, 14, 15] else ""
            print(f"  L0H{h:<3} {s123_l0_attn['attn_to_key'][h]:>10.4f} "
                  f"{s123_l0_attn['attn_to_baseline'][h]:>10.4f} "
                  f"{s123_l0_attn['selectivity'][h]:>7.1f}x{note}")

    print("\n-- s42 L0 heads (for comparison) --")
    s42_l0_attn = measure_l0_key_attention(model, S42_DIR, ckpt_step=10000)
    if s42_l0_attn:
        print(f"  {'head':<8} {'attn→KEY':>10} {'attn→rand':>10} {'sel':>8}")
        for h in range(16):
            note = " ← s42-circuit head" if h in [3, 6, 14, 15] else ""
            print(f"  L0H{h:<3} {s42_l0_attn['attn_to_key'][h]:>10.4f} "
                  f"{s42_l0_attn['attn_to_baseline'][h]:>10.4f} "
                  f"{s42_l0_attn['selectivity'][h]:>7.1f}x{note}")

    # ── Test B: PR trajectories of L0 heads ───────────────────────────
    print("\n=== Test B: L0 PR features for s123 vs s42 ===")
    s123_l0_pr, s123_steps = get_l0_pr_features("s123")
    s42_l0_pr, _ = get_l0_pr_features("s42")
    print(f"  {'head':<8} {'s42 spread':>11} {'s42 max':>9} {'s123 spread':>12} {'s123 max':>10}")
    for h in range(16):
        note = " ← s42-circuit head" if h in [3, 6, 14, 15] else ""
        print(f"  L0H{h:<3} "
              f"{s42_l0_pr[h]['spread']:>11.2f} {s42_l0_pr[h]['max_pr']:>9.2f} "
              f"{s123_l0_pr[h]['spread']:>12.2f} {s123_l0_pr[h]['max_pr']:>10.2f}{note}")

    # ── Test C: full-L0 ablation on s123 ──────────────────────────────
    test_c_results = run_ablation_test_c(ckpt_step=10000)

    # Save
    out = {
        "test_A_l0_attn_selectivity": {
            "s123": s123_l0_attn,
            "s42": s42_l0_attn,
        },
        "test_B_l0_pr_features": {
            "s123": s123_l0_pr,
            "s42": s42_l0_pr,
            "ckpt_steps": s123_steps,
        },
        "test_C_full_l0_ablation_s123": test_c_results,
    }
    out_json = REPO / "analyses/s123_l0_investigation.json"
    with open(out_json, "w") as f:
        json.dump(out, f, indent=2)
    print(f"\nwrote {out_json}")


if __name__ == "__main__":
    main()
