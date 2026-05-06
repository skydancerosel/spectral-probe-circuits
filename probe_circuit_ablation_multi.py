"""
Multi-layer ablation variant of probe_circuit_ablation.py.

Conditions are specified as dicts of {layer_idx: [head_indices]} so we can
ablate multiple heads across multiple layers in a single pass.

Designed for s271 testing where the spectrally-identified circuit candidates
span L6 + L7, not just L0 like in s42.
"""

import json
import sys
from pathlib import Path
import numpy as np
import torch
from torch.utils.data import DataLoader

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO / "training"))
from config import Config, get_device
from model import GPTModel
from dataset import build_datasets
from pilot import evaluate_probe, evaluate_lm

import argparse
ap = argparse.ArgumentParser()
ap.add_argument("--run-dir", required=True)
ap.add_argument("--tag", required=True)
ap.add_argument("--ckpts", default="800,4000,10000")
args = ap.parse_args()

PRETRAIN_DIR = REPO / args.run_dir
TAG = args.tag
TEST_CKPTS = [int(x) for x in args.ckpts.split(",")]


def make_pre_hook(ablated_heads, n_head, head_dim):
    def pre_hook(_module, ainputs):
        x = ainputs[0]
        B, T, C = x.shape
        xr = x.view(B, T, n_head, head_dim).clone()
        for h in ablated_heads:
            xr[:, :, h, :] = 0.0
        return (xr.view(B, T, C),)
    return pre_hook


def run_condition_multi(model, ablation_spec, probe_in, probe_ood, val_loader,
                         device, n_head, head_dim):
    """ablation_spec: dict {layer_idx: [head_indices]} — supports multi-layer.
       Empty dict = baseline (no ablation).
    """
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


def main():
    device = get_device()
    print(f"device = {device}; run_dir = {PRETRAIN_DIR.name}")

    cfg = Config(seed=42, p_probe=0.10, batch_size=64,
                 n_layer=8, d_model=512, n_head=16, d_ff=2048)
    n_head = cfg.n_head
    head_dim = cfg.d_model // n_head

    cw_path = PRETRAIN_DIR / "codewords.json"
    print("Loading TS dataset...")
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

    # s271-specific candidate set: from per-head spectral, top heads by spread
    # during steps 200..2000 (s271 emergence window):
    #   L6H10, L7H9, L6H1, L7H15  (top 4 by spread, spread 9.0–11.0)
    # Plus s42-circuit-on-s271 (L0H{3,6,14,15}) — should NOT cause big effect
    # if seed-specific localization is real.
    rng = np.random.RandomState(42)
    eligible_L6 = [h for h in range(n_head) if h not in (1, 10)]
    eligible_L7 = [h for h in range(n_head) if h not in (9, 15)]
    random_L6 = sorted(rng.choice(eligible_L6, size=2, replace=False).tolist())
    random_L7 = sorted(rng.choice(eligible_L7, size=2, replace=False).tolist())

    conditions = [
        ("baseline",                          {}),
        # Main: ablate s271-identified circuit (L6H{1,10} + L7H{9,15})
        ("ablate_s271_circuit_L6L7",          {6: [1, 10], 7: [9, 15]}),
        # Each circuit head individually
        ("ablate_L6H1",                       {6: [1]}),
        ("ablate_L6H10",                      {6: [10]}),
        ("ablate_L7H9",                       {7: [9]}),
        ("ablate_L7H15",                      {7: [15]}),
        # Per-layer pairs
        ("ablate_L6_pair_only",               {6: [1, 10]}),
        ("ablate_L7_pair_only",               {7: [9, 15]}),
        # Test the s42-circuit hypothesis on s271 (should NOT work if seed-specific)
        ("ablate_s42_circuit_on_s271",        {0: [3, 6, 14, 15]}),
        # Matched control: 2 random heads from L6 + 2 from L7, NOT in circuit
        ("ablate_matched_random_L6L7",        {6: random_L6, 7: random_L7}),
        # Upper bound: ablate ALL of L6 + L7
        ("ablate_full_L6L7",                  {6: list(range(n_head)),
                                                7: list(range(n_head))}),
    ]

    results = {"conditions": [], "test_ckpts": TEST_CKPTS,
               "tag": TAG, "run_dir": str(PRETRAIN_DIR),
               "random_L6": random_L6, "random_L7": random_L7}

    for ckpt_step in TEST_CKPTS:
        print(f"\n{'='*72}\nCheckpoint step={ckpt_step}\n{'='*72}")
        ck_path = PRETRAIN_DIR / f"ckpt_{ckpt_step:06d}.pt"
        if not ck_path.exists():
            print(f"  [SKIP] missing {ck_path.name}")
            continue
        ck = torch.load(ck_path, map_location=device, weights_only=True)
        model.load_state_dict(ck["model_state_dict"])
        del ck

        print(f"  {'condition':<38} {'pin':>7} {'pood':>7} {'val':>7}")
        for name, spec in conditions:
            r = run_condition_multi(model, spec, probe_in, probe_ood,
                                     val_loader, device, n_head, head_dim)
            print(f"  {name:<38} {r['probe_in_acc']:>7.4f} "
                  f"{r['probe_ood_acc']:>7.4f} {r['val_loss']:>7.4f}")
            results["conditions"].append({
                "ckpt_step": ckpt_step,
                "name": name,
                "ablation_spec": {str(k): v for k, v in spec.items()},
                **r,
            })

    out_json = REPO / f"analyses/probe_circuit_ablation_{TAG}.json"
    with open(out_json, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nwrote {out_json}")


if __name__ == "__main__":
    main()
