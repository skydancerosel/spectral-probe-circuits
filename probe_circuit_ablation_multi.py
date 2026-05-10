"""
Multi-layer ablation variant of probe_circuit_ablation.py.

Conditions are specified as dicts of {layer_idx: [head_indices]} so we can
ablate multiple heads across multiple layers in a single pass.

Each --tag selects a hardcoded condition set whose spectral picks come from
probe_circuit_per_head.py (run that first, eyeball the top heads by PR spread).

Tags supported: s42, s271, s149.
"""

import json
import sys
from pathlib import Path
import numpy as np
import torch
from torch.utils.data import DataLoader

REPO = Path(__file__).resolve().parent
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

    # Per-seed circuit-condition sets. Spectral picks were determined offline
    # by probe_circuit_per_head.py and are hardcoded here:
    #   s42:  L0H{3, 6, 14, 15}             (PR spread 19.9–22.8, all L0)
    #   s271: L6H{1, 10} + L7H{9, 15}        (PR spread 9.0–11.0, late layers)
    #   s149: L6H{2, 5, 6, 7} + L7H13        (PR spread 20.4–24.1, late layers)
    #   s256: L5H10 + L6H{2, 4} + L7H{6, 13} (PR spread 19.7–23.4, mid-late layers)
    # s149 and s256 share specific heads L6H2 + L7H13 — first observed cross-seed overlap.
    # Each seed's run includes (i) its own circuit ablation, (ii) the other
    # seeds' circuits as cross-seed checks, (iii) matched random control.
    rng = np.random.RandomState(42)

    s42_circuit  = {0: [3, 6, 14, 15]}
    s271_circuit = {6: [1, 10], 7: [9, 15]}
    s149_circuit = {6: [2, 5, 6, 7], 7: [13]}
    s256_circuit = {5: [10], 6: [2, 4], 7: [6, 13]}
    s123_circuit = {5: [5], 6: [5, 11], 7: [2, 4, 13]}  # truncated at step 5000
    s314_circuit = {5: [7, 14, 15], 7: [0, 5]}  # top 5 by spread

    if TAG == "s271":
        eligible_L6 = [h for h in range(n_head) if h not in (1, 10)]
        eligible_L7 = [h for h in range(n_head) if h not in (9, 15)]
        random_L6 = sorted(rng.choice(eligible_L6, size=2, replace=False).tolist())
        random_L7 = sorted(rng.choice(eligible_L7, size=2, replace=False).tolist())
        conditions = [
            ("baseline",                       {}),
            ("ablate_s271_circuit_L6L7",       s271_circuit),
            ("ablate_L6H1",                    {6: [1]}),
            ("ablate_L6H10",                   {6: [10]}),
            ("ablate_L7H9",                    {7: [9]}),
            ("ablate_L7H15",                   {7: [15]}),
            ("ablate_L6_pair_only",            {6: [1, 10]}),
            ("ablate_L7_pair_only",            {7: [9, 15]}),
            ("ablate_s42_circuit_on_s271",     s42_circuit),
            ("ablate_s149_circuit_on_s271",    s149_circuit),
            ("ablate_s256_circuit_on_s271",    s256_circuit),
            ("ablate_matched_random_L6L7",     {6: random_L6, 7: random_L7}),
            ("ablate_full_L6L7",               {6: list(range(n_head)),
                                                7: list(range(n_head))}),
        ]
    elif TAG == "s149":
        eligible_L6 = [h for h in range(n_head) if h not in (2, 5, 6, 7)]
        eligible_L7 = [h for h in range(n_head) if h != 13]
        random_L6 = sorted(rng.choice(eligible_L6, size=4, replace=False).tolist())
        random_L7 = sorted(rng.choice(eligible_L7, size=1, replace=False).tolist())
        conditions = [
            ("baseline",                       {}),
            ("ablate_s149_circuit_L6L7",       s149_circuit),
            ("ablate_L6H2",                    {6: [2]}),
            ("ablate_L6H5",                    {6: [5]}),
            ("ablate_L6H6",                    {6: [6]}),
            ("ablate_L6H7",                    {6: [7]}),
            ("ablate_L7H13",                   {7: [13]}),
            ("ablate_L6_quad_only",            {6: [2, 5, 6, 7]}),
            ("ablate_s42_circuit_on_s149",     s42_circuit),
            ("ablate_s271_circuit_on_s149",    s271_circuit),
            ("ablate_s256_circuit_on_s149",    s256_circuit),
            ("ablate_s123_circuit_on_s149",    s123_circuit),
            ("ablate_s314_circuit_on_s149",    s314_circuit),
            ("ablate_matched_random_L6L7",     {6: random_L6, 7: random_L7}),
            ("ablate_full_L6L7",               {6: list(range(n_head)),
                                                7: list(range(n_head))}),
        ]
    elif TAG == "s42":
        eligible_L0 = [h for h in range(n_head) if h not in (3, 6, 14, 15)]
        random_L0 = sorted(rng.choice(eligible_L0, size=4, replace=False).tolist())
        random_L6 = []
        random_L7 = []
        conditions = [
            ("baseline",                       {}),
            ("ablate_s42_circuit_L0",          s42_circuit),
            ("ablate_L0H3",                    {0: [3]}),
            ("ablate_L0H6",                    {0: [6]}),
            ("ablate_L0H14",                   {0: [14]}),
            ("ablate_L0H15",                   {0: [15]}),
            ("ablate_s271_circuit_on_s42",     s271_circuit),
            ("ablate_s149_circuit_on_s42",     s149_circuit),
            ("ablate_s256_circuit_on_s42",     s256_circuit),
            ("ablate_matched_random_L0",       {0: random_L0}),
            ("ablate_full_L0",                 {0: list(range(n_head))}),
            ("ablate_full_L6L7",               {6: list(range(n_head)),
                                                7: list(range(n_head))}),
        ]
    elif TAG == "s256":
        eligible_L5 = [h for h in range(n_head) if h != 10]
        eligible_L6 = [h for h in range(n_head) if h not in (2, 4)]
        eligible_L7 = [h for h in range(n_head) if h not in (6, 13)]
        random_L5 = sorted(rng.choice(eligible_L5, size=1, replace=False).tolist())
        random_L6 = sorted(rng.choice(eligible_L6, size=2, replace=False).tolist())
        random_L7 = sorted(rng.choice(eligible_L7, size=2, replace=False).tolist())
        conditions = [
            ("baseline",                       {}),
            ("ablate_s256_circuit_L5L6L7",     s256_circuit),
            ("ablate_L5H10",                   {5: [10]}),
            ("ablate_L6H2",                    {6: [2]}),
            ("ablate_L6H4",                    {6: [4]}),
            ("ablate_L7H6",                    {7: [6]}),
            ("ablate_L7H13",                   {7: [13]}),
            ("ablate_s42_circuit_on_s256",     s42_circuit),
            ("ablate_s271_circuit_on_s256",    s271_circuit),
            ("ablate_s149_circuit_on_s256",    s149_circuit),
            ("ablate_s123_circuit_on_s256",    s123_circuit),
            ("ablate_s314_circuit_on_s256",    s314_circuit),
            ("ablate_matched_random_L5L6L7",   {5: random_L5, 6: random_L6, 7: random_L7}),
            ("ablate_full_L5L6L7",             {5: list(range(n_head)),
                                                6: list(range(n_head)),
                                                7: list(range(n_head))}),
        ]
    elif TAG == "s314":
        eligible_L5 = [h for h in range(n_head) if h not in (7, 14, 15)]
        eligible_L7 = [h for h in range(n_head) if h not in (0, 5)]
        random_L5 = sorted(rng.choice(eligible_L5, size=3, replace=False).tolist())
        random_L7 = sorted(rng.choice(eligible_L7, size=2, replace=False).tolist())
        random_L6 = []
        conditions = [
            ("baseline",                       {}),
            ("ablate_s314_circuit_L5L7",       s314_circuit),
            ("ablate_L5H7",                    {5: [7]}),
            ("ablate_L5H14",                   {5: [14]}),
            ("ablate_L5H15",                   {5: [15]}),
            ("ablate_L7H0",                    {7: [0]}),
            ("ablate_L7H5",                    {7: [5]}),
            ("ablate_s42_circuit_on_s314",     s42_circuit),
            ("ablate_s271_circuit_on_s314",    s271_circuit),
            ("ablate_s149_circuit_on_s314",    s149_circuit),
            ("ablate_s256_circuit_on_s314",    s256_circuit),
            ("ablate_s123_circuit_on_s314",    s123_circuit),
            ("ablate_matched_random_L5L7",     {5: random_L5, 7: random_L7}),
            ("ablate_full_L5L6L7",             {5: list(range(n_head)),
                                                6: list(range(n_head)),
                                                7: list(range(n_head))}),
        ]
    elif TAG == "s123":
        eligible_L5 = [h for h in range(n_head) if h != 5]
        eligible_L6 = [h for h in range(n_head) if h not in (5, 11)]
        eligible_L7 = [h for h in range(n_head) if h not in (2, 4, 13)]
        random_L5 = sorted(rng.choice(eligible_L5, size=1, replace=False).tolist())
        random_L6 = sorted(rng.choice(eligible_L6, size=2, replace=False).tolist())
        random_L7 = sorted(rng.choice(eligible_L7, size=3, replace=False).tolist())
        conditions = [
            ("baseline",                       {}),
            ("ablate_s123_circuit_L5L6L7",     s123_circuit),
            ("ablate_L5H5",                    {5: [5]}),
            ("ablate_L6H5",                    {6: [5]}),
            ("ablate_L6H11",                   {6: [11]}),
            ("ablate_L7H2",                    {7: [2]}),
            ("ablate_L7H4",                    {7: [4]}),
            ("ablate_L7H13",                   {7: [13]}),
            ("ablate_s42_circuit_on_s123",     s42_circuit),
            ("ablate_s271_circuit_on_s123",    s271_circuit),
            ("ablate_s149_circuit_on_s123",    s149_circuit),
            ("ablate_s256_circuit_on_s123",    s256_circuit),
            ("ablate_matched_random_L5L6L7",   {5: random_L5, 6: random_L6, 7: random_L7}),
            ("ablate_full_L5L6L7",             {5: list(range(n_head)),
                                                6: list(range(n_head)),
                                                7: list(range(n_head))}),
        ]
    else:
        raise ValueError(f"Unknown TAG {TAG!r}; expected one of: s42, s271, s149, s256, s123, s314")

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
