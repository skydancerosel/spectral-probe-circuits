"""
pythia_410m_per_head.py

Same per-head spectral analysis as pythia_per_head.py but on Pythia 410M.
Tests whether the methodology scales up beyond 160M.

Architecture: 24L × 1024d × 16h × head_dim 64 (compared to 160M's 12L × 768d × 12h × 64).

Same induction batch (RNG seed 42) for apples-to-apples comparison
across model sizes.

Output: analyses/pythia_410m_per_head.json
"""

import json
import sys
from pathlib import Path
import numpy as np
import torch

REPO = Path(__file__).resolve().parent
sys.path.insert(0, str(REPO))
from induction_heads_per_head_124m import build_induction_batch, compute_pr
from transformers import GPTNeoXForCausalLM

MODEL_NAME = "EleutherAI/pythia-410m"
CHECKPOINTS = [
    "step1", "step8", "step32", "step128", "step512",
    "step1000", "step1500", "step2000", "step3000",
    "step4000", "step5000", "step7000",
    "step10000", "step15000", "step25000", "step40000",
    "step70000", "step143000",
]


def revision_step(rev):
    return int(rev[len("step"):])


def per_head_pr_at_position(model, tokens, position, n_layer, n_head, head_dim,
                              device, batch_size=16):
    n = tokens.shape[0]
    out = torch.zeros(n, n_layer, n_head, head_dim, dtype=torch.float32)
    captured_mb = {}

    handles = []
    for L in range(n_layer):
        dense = model.gpt_neox.layers[L].attention.dense
        def make_hook(L=L):
            def hook(module, ainputs):
                x = ainputs[0]
                B, T, C = x.shape
                xr = x.view(B, T, n_head, head_dim)
                captured_mb[L] = xr.detach()
            return hook
        handles.append(dense.register_forward_pre_hook(make_hook()))

    try:
        with torch.no_grad():
            for start in range(0, n, batch_size):
                end = min(start + batch_size, n)
                tok = tokens[start:end].to(device)
                _ = model(tok)
                B = end - start
                for L in range(n_layer):
                    xr = captured_mb[L]
                    sel = xr[:, position, :, :]
                    out[start:end, L] = sel.cpu()
    finally:
        for h in handles:
            h.remove()

    pr_matrix = np.zeros((n_layer, n_head))
    for L in range(n_layer):
        for H in range(n_head):
            pr_matrix[L, H] = compute_pr(out[:, L, H, :])
    return pr_matrix


def main():
    device = "mps" if torch.backends.mps.is_available() else "cpu"
    print(f"device = {device}")

    print("Building induction batch (same RNG seed 42 as all prior runs)...")
    rng = np.random.RandomState(42)
    tokens, positions, targets = build_induction_batch(n_examples=2000, seq_len=256,
                                                        rng=rng)
    last_pos = tokens.shape[1] - 1
    print(f"  batch shape: {tokens.shape}, query position: {last_pos}")

    n_layer = 24
    n_head = 16
    head_dim = 64

    out = {
        "model": MODEL_NAME,
        "checkpoints": CHECKPOINTS,
        "n_layer": n_layer,
        "n_head": n_head,
        "head_dim": head_dim,
        "n_examples": int(tokens.shape[0]),
        "ckpt_step": [],
        "pr": {f"L{L}_H{H}": [] for L in range(n_layer) for H in range(n_head)},
    }

    for ck_idx, revision in enumerate(CHECKPOINTS):
        step = revision_step(revision)
        print(f"\n[{ck_idx+1}/{len(CHECKPOINTS)}] Loading {MODEL_NAME} @ {revision} (step {step})...")
        try:
            model = GPTNeoXForCausalLM.from_pretrained(MODEL_NAME, revision=revision)
            model = model.to(device).eval()
        except Exception as e:
            print(f"  [SKIP] failed to load: {e}")
            continue

        # Verify config matches
        cfg = model.config
        assert cfg.num_hidden_layers == n_layer, f"layer mismatch: {cfg.num_hidden_layers}"
        assert cfg.num_attention_heads == n_head, f"head mismatch: {cfg.num_attention_heads}"

        pr_matrix = per_head_pr_at_position(model, tokens, last_pos,
                                             n_layer, n_head, head_dim, device,
                                             batch_size=16)
        out["ckpt_step"].append(step)
        for L in range(n_layer):
            for H in range(n_head):
                out["pr"][f"L{L}_H{H}"].append(float(pr_matrix[L, H]))
        print(f"  step={step}  PR max={pr_matrix.max():.2f}  mean={pr_matrix.mean():.2f}")

        del model
        if device == "mps":
            torch.mps.empty_cache()

    out_json = REPO / "results/pythia_410m_per_head.json"
    with open(out_json, "w") as f:
        json.dump(out, f, indent=2)
    print(f"\nwrote {out_json}")


if __name__ == "__main__":
    main()
