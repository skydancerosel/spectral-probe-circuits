"""
Per-head spectral analysis of the probe-retrieval circuit.

Sibling of probe_circuit_spectral.py — same setup, but extracts attention
output PER HEAD (the input to out_proj, reshaped to [B, T, n_head, head_dim]),
rather than the post-block residual stream.

For each (layer, head, ckpt), compute spectral structure of the
[N_probe_examples, head_dim] activation matrix at the query position.

Hypothesis: a small subset of (layer, head) pairs will show sharp PR
transitions LOCALIZED in time (around probe-task emergence step ~400-800),
while most heads remain noisy. Such heads are circuit candidates.

Output:
  analyses/probe_circuit_per_head.json
  analyses/probe_circuit_per_head.png
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt

REPO = Path(__file__).resolve().parent
sys.path.insert(0, str(REPO / "training"))
from config import Config, get_device
from model import GPTModel
from dataset import build_datasets


# Inlined from analyses/forgetting_monitors.py in mini_gpt (the parent
# research repo). These two functions are all we need from there.
def kstar_weighted(sigma: np.ndarray, eps: float = 0.05) -> int:
    """Signal-weighted k* (see thesis Remark after Def kstar):
        k*_w = argmax_{j: sigma_{j+1} >= eps * sigma_1}
                 (sigma_j / sum sigma) * (sigma_j / sigma_{j+1})
    sigma: descending singular values. Returns 1-indexed k*.
    """
    if len(sigma) < 2:
        return 1
    s = sigma.astype(np.float64)
    s_sum = s.sum()
    if s_sum <= 0:
        return 1
    s1 = s[0]
    best_j, best_score = 1, -np.inf
    for j in range(1, len(s) - 1):
        if s[j] < eps * s1:
            break
        if s[j + 1] <= 0:
            continue
        score = (s[j] / s_sum) * (s[j] / s[j + 1])
        if score > best_score:
            best_score = score
            best_j = j
    return best_j + 1


def participation_ratio(sigma: np.ndarray) -> float:
    """PR = exp(spectral entropy of normalized squared singular values).
    Bounded in [1, len(sigma)]. Low PR => energy concentrated.
    """
    s2 = sigma.astype(np.float64) ** 2
    s2_sum = s2.sum()
    if s2_sum <= 0:
        return float("nan")
    p = s2 / s2_sum
    p = p[p > 0]
    H = -float((p * np.log(p)).sum())
    return float(np.exp(H))


import argparse  # noqa: E402
_argp = argparse.ArgumentParser(add_help=False)
_argp.add_argument("--run-dir",
                   default="runs/beta2_ablation/pilot_wd0.5_lr0.001_lp2.0_b20.95_s42")
_argp.add_argument("--tag", default="s42")
_args, _ = _argp.parse_known_args()
PRETRAIN_DIR = REPO / _args.run_dir
TAG = _args.tag


def discover_ckpts(run_dir):
    import re
    out = []
    for p in run_dir.glob("ckpt_*.pt"):
        m = re.match(r"ckpt_(\d+)\.pt", p.name)
        if m:
            out.append((int(m.group(1)), p))
    out.sort(key=lambda x: x[0])
    return out


def collect_per_head_at_query(model, probe_loader, device, n_layer, n_head, head_dim):
    """For each layer, capture out_proj's INPUT (the per-head outputs concatenated).
    Reshape to [B, T, n_head, head_dim] and extract the QUERY-position activation
    of each example into per (layer, head) lists.

    Returns: dict (layer, head) -> [N_examples, head_dim] np.ndarray.
    """
    layer_inputs = {l: [] for l in range(n_layer)}

    def make_pre_hook(l):
        def pre_hook(_module, args):
            # args[0] is the input to out_proj: [B, T, d_model]
            layer_inputs[l].append(args[0].detach().cpu())
        return pre_hook

    handles = [block.attn.out_proj.register_forward_pre_hook(make_pre_hook(l))
               for l, block in enumerate(model.blocks)]

    per_lh = {(l, h): [] for l in range(n_layer) for h in range(n_head)}
    model.eval()
    with torch.no_grad():
        for input_ids, _t, probe_mask in probe_loader:
            input_ids = input_ids.to(device)
            for l in range(n_layer):
                layer_inputs[l].clear()
            _ = model(input_ids)

            pmask = probe_mask.bool()
            valid = pmask.any(dim=1)
            first_pos = pmask.long().argmax(dim=1)

            for l in range(n_layer):
                x = layer_inputs[l][0]  # [B, T, d_model], on cpu
                B, T, C = x.shape
                # Reshape to [B, T, n_head, head_dim]
                xh = x.view(B, T, n_head, head_dim)
                # Gather query position
                xh_at_q = xh[torch.arange(B), first_pos]   # [B, n_head, head_dim]
                xh_at_q = xh_at_q[valid]
                for h in range(n_head):
                    per_lh[(l, h)].append(xh_at_q[:, h, :])

    for hd in handles:
        hd.remove()

    out = {(l, h): torch.cat(per_lh[(l, h)], dim=0).numpy().astype(np.float32)
           for l in range(n_layer) for h in range(n_head)}
    return out


def head_metrics(M):
    sv = np.linalg.svd(M, compute_uv=False).astype(np.float64)
    pr = participation_ratio(sv)
    kstar = kstar_weighted(sv)
    s2 = sv ** 2
    top1 = float(s2[0] / s2.sum()) if s2.sum() > 0 else float("nan")
    return {"pr": float(pr), "kstar_w": int(kstar), "top1_share": top1,
            "norm": float(np.linalg.norm(M))}


def main():
    device = get_device()
    print(f"device = {device}")

    cfg = Config(seed=42, p_probe=0.10, batch_size=64,
                 n_layer=8, d_model=512, n_head=16, d_ff=2048)
    n_layer, n_head = cfg.n_layer, cfg.n_head
    head_dim = cfg.d_model // n_head

    cw_path = PRETRAIN_DIR / "codewords.json"
    print("Loading TS dataset for probe_eval_in...")
    data = build_datasets(cfg, codewords_path=str(cw_path) if cw_path.exists() else None)
    vocab_size = len(data["tokenizer"])
    probe_loader = DataLoader(data["probe_eval_in"], batch_size=64,
                               shuffle=False, drop_last=False, num_workers=0)
    n_examples = len(data["probe_eval_in"])
    print(f"  probe_eval_in: {n_examples} examples; head_dim={head_dim}")

    model = GPTModel(
        vocab_size=vocab_size, seq_len=cfg.seq_len,
        d_model=cfg.d_model, n_layer=cfg.n_layer,
        n_head=cfg.n_head, d_ff=cfg.d_ff, dropout=0.0,
    ).to(device)

    ckpts = discover_ckpts(PRETRAIN_DIR)
    print(f"  {len(ckpts)} checkpoints to process")

    out = {
        "n_layer": n_layer, "n_head": n_head, "head_dim": head_dim,
        "n_examples": n_examples,
        "ckpt_step": [],
        # Per (layer, head): list of metrics per ckpt
        "pr": {f"L{l}_H{h}": [] for l in range(n_layer) for h in range(n_head)},
        "kstar_w": {f"L{l}_H{h}": [] for l in range(n_layer) for h in range(n_head)},
        "top1_share": {f"L{l}_H{h}": [] for l in range(n_layer) for h in range(n_head)},
        "norm": {f"L{l}_H{h}": [] for l in range(n_layer) for h in range(n_head)},
    }

    for ck_idx, (step, ck_path) in enumerate(ckpts):
        ck = torch.load(ck_path, map_location=device, weights_only=True)
        model.load_state_dict(ck["model_state_dict"])

        per_lh = collect_per_head_at_query(model, probe_loader, device,
                                            n_layer, n_head, head_dim)

        out["ckpt_step"].append(int(step))
        for l in range(n_layer):
            for h in range(n_head):
                m = head_metrics(per_lh[(l, h)])
                out["pr"][f"L{l}_H{h}"].append(m["pr"])
                out["kstar_w"][f"L{l}_H{h}"].append(m["kstar_w"])
                out["top1_share"][f"L{l}_H{h}"].append(m["top1_share"])
                out["norm"][f"L{l}_H{h}"].append(m["norm"])

        del per_lh

        if ck_idx % 5 == 0 or ck_idx == len(ckpts) - 1:
            # Show min PR across all heads at this step
            all_pr = [out["pr"][f"L{l}_H{h}"][-1] for l in range(n_layer) for h in range(n_head)]
            print(f"  ckpt {ck_idx+1}/{len(ckpts)} step={step:>5}  "
                  f"PR min={min(all_pr):.2f} max={max(all_pr):.2f} "
                  f"mean={np.mean(all_pr):.2f}")

    # Probe curve for cross-reference
    pm = json.load(open(PRETRAIN_DIR / "pilot_metrics.json"))
    out["probe_curve"] = [(r["step"], r.get("probe_in_acc"), r.get("probe_ood_acc"))
                          for r in pm if "step" in r]

    out_json = REPO / f"analyses/probe_circuit_per_head_{TAG}.json"
    with open(out_json, "w") as f:
        json.dump(out, f, indent=2)
    print(f"\nwrote {out_json}")

    # ── Plot ──────────────────────────────────────────────────────────
    steps = np.array(out["ckpt_step"])
    n_lh = n_layer * n_head

    # Build PR matrix [n_lh, n_steps]
    head_labels = []
    PR = np.zeros((n_lh, len(steps)))
    for i, l in enumerate(range(n_layer)):
        for j, h in enumerate(range(n_head)):
            row = i * n_head + j
            head_labels.append(f"L{l}H{h}")
            PR[row] = out["pr"][f"L{l}_H{h}"]

    fig, axes = plt.subplots(3, 1, figsize=(14, 14),
                              gridspec_kw={"height_ratios": [1, 4, 2]})

    # Panel 0: probe curve
    pc_steps = np.array([s for s, _, _ in out["probe_curve"]])
    pin = np.array([p for _, p, _ in out["probe_curve"]], dtype=float)
    axes[0].plot(pc_steps, pin, color="tab:red", lw=1.5, label="probe_in_acc")
    axes[0].set_ylabel("probe_in_acc")
    axes[0].set_xlim(steps[0], steps[-1])
    axes[0].grid(True, alpha=0.3)
    axes[0].set_title("Per-head spectral structure (PR) at the query position over training")
    axes[0].legend(loc="upper left", fontsize=8)

    # Panel 1: heatmap PR per (layer, head) over ckpts
    im = axes[1].imshow(PR, aspect="auto", cmap="viridis",
                         extent=[steps[0], steps[-1], n_lh - 0.5, -0.5])
    axes[1].set_yticks(range(n_lh))
    axes[1].set_yticklabels(head_labels, fontsize=5)
    axes[1].set_ylabel("(layer, head)")
    plt.colorbar(im, ax=axes[1], label="PR (effective rank, head_dim=32)")

    # Panel 2: Most-transitioning heads — overlay top 6 heads by max-min(PR)
    spread = PR.max(axis=1) - PR.min(axis=1)
    top_idx = np.argsort(spread)[::-1][:6]
    for idx in top_idx:
        axes[2].plot(steps, PR[idx], lw=1.5,
                      label=f"{head_labels[idx]} (Δ={spread[idx]:.1f})")
    axes[2].set_ylabel("PR")
    axes[2].set_xlabel("pretraining step")
    axes[2].grid(True, alpha=0.3)
    axes[2].legend(loc="upper right", fontsize=9)
    axes[2].set_title("Top 6 heads by PR spread (max−min) over training")

    fig.tight_layout()
    out_png = REPO / f"analyses/probe_circuit_per_head_{TAG}.png"
    fig.savefig(out_png, dpi=130, bbox_inches="tight")
    print(f"saved {out_png}")


if __name__ == "__main__":
    main()
