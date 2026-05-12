"""
probe_circuit_mechinterp_general.py

Generalized version of probe_circuit_mechinterp.py: measures attention from
the query-read position back to the KEY position for arbitrary
(layer, head) sets, not just L0.

Goal: confirm that the late-layer spectral picks for s271/s149/s256 are
KEY-attending heads (the writeup currently *infers* this from ablation
specificity but doesn't directly measure it).

Usage:
  python probe_circuit_mechinterp_general.py \\
      --run-dir runs/.../pilot_..._s271 --tag s271 --ckpt-step 4000 \\
      --circuit-spec "6:1,10;7:9,15"

The --circuit-spec is "L1:H1a,H1b;L2:H2a,H2b" — each layer's heads
comma-separated, layers semicolon-separated.

Output:
  analyses/probe_circuit_mechinterp_<tag>.{json,png}
"""

import json
import sys
import math
import argparse
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
import matplotlib.pyplot as plt

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO / "training"))
from config import Config, get_device
from model import GPTModel
from dataset import build_datasets


def parse_circuit_spec(s):
    """ '6:1,10;7:9,15' -> {6: [1, 10], 7: [9, 15]} """
    out = {}
    for layer_part in s.split(";"):
        layer_part = layer_part.strip()
        if not layer_part:
            continue
        l_str, hs_str = layer_part.split(":")
        out[int(l_str)] = [int(h) for h in hs_str.split(",")]
    return out


ap = argparse.ArgumentParser()
ap.add_argument("--run-dir", required=True)
ap.add_argument("--tag", required=True)
ap.add_argument("--ckpt-step", type=int, default=4000)
ap.add_argument("--circuit-spec", required=True,
                 help="e.g. '0:3,6,14,15' or '6:1,10;7:9,15'")
ap.add_argument("--n-examples", type=int, default=200)
args = ap.parse_args()

PRETRAIN_DIR = REPO / args.run_dir
CKPT_STEP = args.ckpt_step
TAG = args.tag
N_EXAMPLES = args.n_examples
CIRCUIT_BY_LAYER = parse_circuit_spec(args.circuit_spec)
CIRCUIT_LAYERS = sorted(CIRCUIT_BY_LAYER.keys())


def find_key_position(input_ids_list, code_token_variants, query_pos):
    earliest = None
    for token_seq in code_token_variants:
        L = len(token_seq)
        for i in range(min(query_pos, len(input_ids_list) - L + 1)):
            if input_ids_list[i:i+L] == token_seq:
                if earliest is None or i < earliest:
                    earliest = i
                break
    return earliest


def compute_layer_attention(model, input_ids, target_layer):
    """Forward through blocks 0..target_layer-1 normally, then manually
    replicate the attention computation at block[target_layer] to get
    its attention pattern.

    Returns: att [B, n_head, T, T]
    """
    B, T = input_ids.shape
    device = input_ids.device

    tok = model.tok_emb(input_ids)
    pos = model.pos_emb(torch.arange(T, device=device))
    h = tok + pos.unsqueeze(0)

    # Forward through blocks before target
    for L in range(target_layer):
        h = model.blocks[L](h)

    # At target layer, replicate attention
    block = model.blocks[target_layer]
    x_post_ln = block.ln1(h)
    qkv = block.attn.qkv(x_post_ln)
    q, k, v = qkv.split(block.attn.d_model, dim=2)
    H = block.attn.n_head
    D = block.attn.head_dim
    q = q.view(B, T, H, D).transpose(1, 2)
    k = k.view(B, T, H, D).transpose(1, 2)
    att = (q @ k.transpose(-2, -1)) * (1.0 / math.sqrt(D))
    att = att.masked_fill(block.attn.bias[:, :, :T, :T] == 0, float("-inf"))
    att = F.softmax(att, dim=-1)
    return att  # [B, n_head, T, T]


def main():
    device = get_device()
    print(f"device = {device}")
    print(f"tag={TAG}  ckpt_step={CKPT_STEP}")
    print(f"circuit by layer: {CIRCUIT_BY_LAYER}")

    cfg = Config(seed=42, p_probe=0.10, batch_size=64,
                 n_layer=8, d_model=512, n_head=16, d_ff=2048)
    n_head = cfg.n_head

    cw_path = PRETRAIN_DIR / "codewords.json"
    print("Loading TS dataset...")
    data = build_datasets(cfg, codewords_path=str(cw_path) if cw_path.exists() else None)
    tokenizer = data["tokenizer"]
    vocab_size = len(tokenizer)
    probe_in = data["probe_eval_in"]

    model = GPTModel(
        vocab_size=vocab_size, seq_len=cfg.seq_len,
        d_model=cfg.d_model, n_layer=cfg.n_layer,
        n_head=cfg.n_head, d_ff=cfg.d_ff, dropout=0.0,
    ).to(device)
    ck_path = PRETRAIN_DIR / f"ckpt_{CKPT_STEP:06d}.pt"
    print(f"Loading {ck_path.name}")
    ck = torch.load(ck_path, map_location=device, weights_only=True)
    model.load_state_dict(ck["model_state_dict"])
    model.eval()

    examples = probe_in.examples[:N_EXAMPLES]

    # For each circuit layer, compute attention to KEY for all heads in that layer
    results = {"tag": TAG, "ckpt_step": CKPT_STEP,
                "circuit_by_layer": {str(l): hs for l, hs in CIRCUIT_BY_LAYER.items()},
                "n_examples": 0,
                "per_layer": {}}

    n_valid = 0
    BATCH = 32

    # Precompute (key_pos, query_read_pos, valid_idx) per example so we don't
    # redo for each layer
    example_metadata = []
    for ex in examples:
        code_tokens = ex["code_tokens"]
        codeword = ex["codeword"]
        alt_code_tokens = tokenizer.encode(" " + codeword, add_special_tokens=False)
        variants = [code_tokens, alt_code_tokens] if alt_code_tokens != code_tokens else [code_tokens]
        code_start = ex["code_start"]
        query_read_pos = code_start - 1
        if query_read_pos < 0:
            example_metadata.append(None)
            continue
        input_list = ex["input_ids"].tolist()
        key_pos = find_key_position(input_list, variants, query_read_pos)
        if key_pos is None:
            example_metadata.append(None)
            continue
        example_metadata.append({"key_pos": key_pos,
                                  "query_read_pos": query_read_pos,
                                  "max_key_L": max(len(t) for t in variants)})

    for L in CIRCUIT_LAYERS:
        print(f"\n=== Layer {L}: per-head attention from query→KEY ===")
        attn_to_key = np.zeros(n_head, dtype=np.float64)
        attn_to_self = np.zeros(n_head, dtype=np.float64)
        attn_to_random = np.zeros(n_head, dtype=np.float64)
        layer_n_valid = 0

        with torch.no_grad():
            for batch_start in range(0, len(examples), BATCH):
                batch_examples = examples[batch_start:batch_start + BATCH]
                input_ids_batch = torch.stack(
                    [ex["input_ids"] for ex in batch_examples]).to(device)
                att = compute_layer_attention(model, input_ids_batch, target_layer=L)
                # att: [B, n_head, T, T]
                for ei, ex in enumerate(batch_examples):
                    md = example_metadata[batch_start + ei]
                    if md is None:
                        continue
                    a = att[ei, :, md["query_read_pos"], :].cpu().numpy()
                    key_attention = a[:, md["key_pos"]:md["key_pos"]+md["max_key_L"]].sum(axis=-1)
                    attn_to_key += key_attention
                    attn_to_self += a[:, md["query_read_pos"]]
                    mask = np.ones(att.shape[-1], dtype=bool)
                    mask[md["key_pos"]:md["key_pos"]+md["max_key_L"]] = False
                    if md["query_read_pos"] < att.shape[-1]:
                        mask[md["query_read_pos"]] = False
                    attn_to_random += a[:, mask].mean(axis=-1)
                    layer_n_valid += 1

        attn_to_key /= max(layer_n_valid, 1)
        attn_to_self /= max(layer_n_valid, 1)
        attn_to_random /= max(layer_n_valid, 1)

        circuit_heads_in_L = CIRCUIT_BY_LAYER[L]
        print(f"  N valid examples: {layer_n_valid}")
        print(f"  {'head':<8} {'attn→KEY':>12} {'attn→rand':>12} {'selectivity':>12} note")
        per_head_data = {}
        for h in range(n_head):
            sel = attn_to_key[h] / max(attn_to_random[h], 1e-8)
            note = " ← circuit head" if h in circuit_heads_in_L else ""
            print(f"  L{L}H{h:<3} {attn_to_key[h]:>12.4f} {attn_to_random[h]:>12.4f} "
                  f"{sel:>12.1f}x{note}")
            per_head_data[f"H{h}"] = {
                "attn_to_key": float(attn_to_key[h]),
                "attn_to_self": float(attn_to_self[h]),
                "attn_to_random": float(attn_to_random[h]),
                "selectivity": float(sel),
                "is_circuit": h in circuit_heads_in_L,
            }
        results["per_layer"][str(L)] = per_head_data
        if layer_n_valid > n_valid:
            n_valid = layer_n_valid

    results["n_examples"] = n_valid

    out_json = REPO / f"analyses/probe_circuit_mechinterp_{TAG}.json"
    with open(out_json, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nwrote {out_json}")

    # Plot: one panel per circuit layer
    n_layers = len(CIRCUIT_LAYERS)
    fig, axes = plt.subplots(1, n_layers, figsize=(7 * n_layers, 4.5),
                              squeeze=False)
    axes = axes[0]
    for ax_idx, L in enumerate(CIRCUIT_LAYERS):
        per_head = results["per_layer"][str(L)]
        circuit_heads = CIRCUIT_BY_LAYER[L]
        sel = np.array([per_head[f"H{h}"]["selectivity"] for h in range(n_head)])
        colors = ["tab:red" if h in circuit_heads else "tab:blue" for h in range(n_head)]
        x = np.arange(n_head)
        axes[ax_idx].bar(x, sel, color=colors, edgecolor="k", linewidth=0.5)
        axes[ax_idx].set_xticks(x)
        axes[ax_idx].set_xticklabels([f"H{h}" for h in range(n_head)], fontsize=8)
        axes[ax_idx].set_xlabel(f"Layer-{L} head index")
        axes[ax_idx].set_ylabel("attention(→KEY) / attention(→random) selectivity")
        axes[ax_idx].set_title(f"L{L} per-head KEY-attention selectivity ({TAG}, ckpt {CKPT_STEP})\n"
                                f"red = spectrally-identified circuit head")
        axes[ax_idx].axhline(1, color="k", lw=0.5, ls="--", alpha=0.5)
        axes[ax_idx].grid(True, alpha=0.3, axis="y")

    fig.tight_layout()
    out_png = REPO / f"analyses/probe_circuit_mechinterp_{TAG}.png"
    fig.savefig(out_png, dpi=130, bbox_inches="tight")
    print(f"saved {out_png}")


if __name__ == "__main__":
    main()
