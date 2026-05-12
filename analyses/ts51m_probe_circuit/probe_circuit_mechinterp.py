"""
Mechanistic characterization of the probe-retrieval circuit heads.

For the s42-identified circuit heads L0H{3,6,14,15}, on a batch of probe
examples, we test:

1. ATTENTION PATTERN: at the query "read" position (token before the
   codeword target), do the circuit heads attend back to the KEY position
   (where the codeword was first mentioned in the prefix)?
   Compare to non-circuit L0 heads (e.g., L0H{0,1,5,7}) which should NOT
   show this pattern.

2. OV CIRCUIT: compute W_OV[h] = W_O[h] @ W_V[h] for each head. The unembed
   logits for the codeword token after copy-attention through head h are
   approximately tok_emb[codeword] @ W_OV[h] @ W_unembed (= W_OV[h] applied
   to the embedding, then unembedded). Test whether the circuit heads have
   an OV that approximately preserves codeword identity (high diagonal
   in the codeword-to-codeword matrix).

This is a minimal mech-interp characterization — enough to say what the
circuit DOES, not just WHERE it lives.

Output: analyses/probe_circuit_mechinterp.json + .png
"""

import json
import sys
import math
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

import argparse
_ap = argparse.ArgumentParser(add_help=False)
_ap.add_argument("--run-dir", default="runs/s42",
                  help="path to a trained run dir (contains ckpt_*.pt and codewords.json)")
_ap.add_argument("--ckpt-step", type=int, default=4000)
_ap.add_argument("--n-examples", type=int, default=200)
_args, _ = _ap.parse_known_args()

PRETRAIN_DIR = REPO / _args.run_dir
CKPT_STEP = _args.ckpt_step
N_EXAMPLES = _args.n_examples    # enough for clean per-head averages

# Default circuit picks for the s42 example. For other seeds, edit these or
# generalize the script to take --circuit-heads as CLI arg.
CIRCUIT_HEADS = [3, 6, 14, 15]
CONTROL_HEADS = [0, 1, 5, 7]


def find_key_position(input_ids_list, code_token_variants, query_pos):
    """Return position of the EARLIEST occurrence of any code_token sequence
    BEFORE the query position. Returns None if not found.
    `code_token_variants` is a list of token-id lists (e.g. with and without
    leading-space variants from BPE).
    """
    earliest = None
    for token_seq in code_token_variants:
        L = len(token_seq)
        for i in range(min(query_pos, len(input_ids_list) - L + 1)):
            if input_ids_list[i:i+L] == token_seq:
                if earliest is None or i < earliest:
                    earliest = i
                break
    return earliest


def compute_layer0_attention(model, input_ids):
    """Manually compute layer-0 attention pattern (post-LN1 of block 0).

    Returns:
      attn:    [B, n_head, T, T]  the softmaxed attention pattern
      v:       [B, n_head, T, head_dim]  the value vectors
    """
    block = model.blocks[0]
    # post-LN1 residual = block.ln1(x_input)
    # x_input is the model's residual stream entering block 0:
    #   resid_stream_0 = tok_emb(idx) + pos_emb(0..T)
    B, T = input_ids.shape
    tok = model.tok_emb(input_ids)               # [B, T, d_model]
    pos = model.pos_emb(torch.arange(T, device=input_ids.device))  # [T, d_model]
    x = tok + pos.unsqueeze(0)                    # [B, T, d_model]
    x = block.ln1(x)
    qkv = block.attn.qkv(x)
    q, k, v = qkv.split(block.attn.d_model, dim=2)
    H = block.attn.n_head; D = block.attn.head_dim
    q = q.view(B, T, H, D).transpose(1, 2)
    k = k.view(B, T, H, D).transpose(1, 2)
    v = v.view(B, T, H, D).transpose(1, 2)
    att = (q @ k.transpose(-2, -1)) * (1.0 / math.sqrt(D))
    att = att.masked_fill(block.attn.bias[:, :, :T, :T] == 0, float("-inf"))
    att = F.softmax(att, dim=-1)
    return att, v


def compute_ov_codeword_diagonal(model, codewords, tokenizer):
    """For each head in L0, compute W_OV[h] applied to the codeword embeddings,
    then unembed via tok_emb (tied weight). Returns the average self-similarity
    across codewords:
      score[h] = mean over codewords c: logit_for_c(W_OV[h] @ tok_emb[c]) /
                                         max_alt_logit
    This tests whether OV approximately preserves codeword identity (a
    signature of an "induction-style" copy head).
    """
    block = model.blocks[0]
    H = block.attn.n_head; D = block.attn.head_dim
    d_model = block.attn.d_model
    # W_V is the V part of qkv weight: rows d_model+i for i in [0..d_model)
    W_qkv = block.attn.qkv.weight  # [3*d_model, d_model]
    W_V = W_qkv[2*d_model:3*d_model, :]   # [d_model, d_model]
    # W_O = block.attn.out_proj.weight  # [d_model, d_model]; out = x @ W_O.T
    W_O = block.attn.out_proj.weight

    # Per-head W_V[h]: rows of W_V corresponding to head h's output positions
    # Reshape: W_V is [d_model, d_model]; the OUTPUT is reshaped via
    #   v.view(B, T, H, D).transpose(1, 2) then att @ v -> [B, H, T, D],
    #   then transpose+view back to [B, T, H*D]
    # So W_V[h, i, j] writes to output position (h * D + i) from input j.
    # Per-head V projection: W_V_h = W_V[h*D:(h+1)*D, :]  shape [D, d_model]
    # Per-head O projection: out_proj input is [B, T, H*D], so W_O[i, h*D+j]
    #   for output dim i picks up head h's j-th component.
    # OV[h] (d_model -> d_model) = W_O[:, h*D:(h+1)*D] @ W_V[h*D:(h+1)*D, :]
    scores = []
    # Codeword token IDs
    codeword_ids = []
    for c in codewords:
        ids = tokenizer.encode(c, add_special_tokens=False)
        if len(ids) == 1:
            codeword_ids.append(ids[0])
    codeword_ids = torch.tensor(codeword_ids, device=W_V.device)
    n_cw = len(codeword_ids)
    if n_cw < 10:
        print(f"  WARNING: only {n_cw} single-token codewords; OV analysis weak")
    cw_emb = model.tok_emb(codeword_ids)   # [n_cw, d_model]

    for h in range(H):
        W_V_h = W_V[h*D:(h+1)*D, :]          # [D, d_model]
        W_O_h = W_O[:, h*D:(h+1)*D]          # [d_model, D]
        OV_h = W_O_h @ W_V_h                  # [d_model, d_model]
        # Apply OV to codeword embeddings: out = cw_emb @ OV.T
        copied = cw_emb @ OV_h.T              # [n_cw, d_model]
        # Unembed via tied tok_emb: logits[c, c'] = copied[c] @ tok_emb[c']
        logits = copied @ model.tok_emb.weight.T   # [n_cw, vocab]
        # For each codeword c, what fraction of the time is c the argmax?
        argmax_ids = logits.argmax(dim=-1)
        correct = (argmax_ids == codeword_ids).float().mean().item()
        # Also: average rank (1.0 = perfect copy)
        sorted_ids = logits.argsort(dim=-1, descending=True)
        ranks = []
        for i in range(n_cw):
            ranks.append((sorted_ids[i] == codeword_ids[i]).nonzero(as_tuple=True)[0].item() + 1)
        median_rank = float(np.median(ranks))
        scores.append({
            "head": h,
            "argmax_correct_frac": correct,
            "median_rank_of_target": median_rank,
            "n_codewords_tested": n_cw,
        })
    return scores


def main():
    device = get_device()
    print(f"device = {device}")

    cfg = Config(seed=42, p_probe=0.10, batch_size=64,
                 n_layer=8, d_model=512, n_head=16, d_ff=2048)
    n_head = cfg.n_head

    cw_path = PRETRAIN_DIR / "codewords.json"
    print("Loading TS dataset...")
    data = build_datasets(cfg, codewords_path=str(cw_path) if cw_path.exists() else None)
    tokenizer = data["tokenizer"]
    vocab_size = len(tokenizer)
    probe_in = data["probe_eval_in"]   # has .examples list with metadata

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

    # ── Test 1: Attention pattern from query_read_pos to KEY_pos ────────
    print("\n=== Test 1: per-head attention from query→KEY position (L0) ===")
    examples = probe_in.examples[:N_EXAMPLES]

    # Per-head, accumulate (attention to KEY) and (attention to query_pos itself)
    attn_to_key = np.zeros(n_head, dtype=np.float64)
    attn_to_self = np.zeros(n_head, dtype=np.float64)
    attn_to_random_position = np.zeros(n_head, dtype=np.float64)  # baseline
    n_valid = 0

    BATCH = 32
    with torch.no_grad():
        for batch_start in range(0, len(examples), BATCH):
            batch_examples = examples[batch_start:batch_start + BATCH]
            input_ids_batch = torch.stack([ex["input_ids"] for ex in batch_examples]).to(device)
            att, _v = compute_layer0_attention(model, input_ids_batch)
            # att: [B, n_head, T, T]
            for ei, ex in enumerate(batch_examples):
                code_tokens = ex["code_tokens"]
                # also search for the leading-space variant (BPE quirk):
                # GPT-2 tokenizer encodes "ALPHA" vs " ALPHA" as different tokens
                codeword = ex["codeword"]
                alt_code_tokens = tokenizer.encode(" " + codeword,
                                                    add_special_tokens=False)
                variants = [code_tokens, alt_code_tokens] if alt_code_tokens != code_tokens else [code_tokens]
                code_start = ex["code_start"]
                query_read_pos = code_start - 1
                if query_read_pos < 0:
                    continue
                input_list = ex["input_ids"].tolist()
                key_pos = find_key_position(input_list, variants, query_read_pos)
                if key_pos is None:
                    continue
                # Per-head attention vector at the read position
                a = att[ei, :, query_read_pos, :].cpu().numpy()  # [n_head, T]
                # Sum attention over the longest code-variant span
                max_L = max(len(t) for t in variants)
                key_attention = a[:, key_pos:key_pos+max_L].sum(axis=-1)
                attn_to_key += key_attention
                attn_to_self += a[:, query_read_pos]
                # Baseline: average attention over random non-key non-self positions
                # (uniform over all positions except key and self)
                mask = np.ones(att.shape[-1], dtype=bool)
                mask[key_pos:key_pos+max_L] = False
                if query_read_pos < att.shape[-1]:
                    mask[query_read_pos] = False
                rand_attn = a[:, mask].mean(axis=-1)
                attn_to_random_position += rand_attn
                n_valid += 1

    attn_to_key /= max(n_valid, 1)
    attn_to_self /= max(n_valid, 1)
    attn_to_random_position /= max(n_valid, 1)
    print(f"  N valid examples: {n_valid}")
    print(f"  {'head':<6} {'attn_to_KEY':>12} {'attn_to_self':>13} {'attn_to_rand_pos':>16} note")
    for h in range(n_head):
        note = " ← circuit head" if h in CIRCUIT_HEADS else (" (control)" if h in CONTROL_HEADS else "")
        print(f"  L0H{h:<3} {attn_to_key[h]:>12.4f} {attn_to_self[h]:>13.4f} "
              f"{attn_to_random_position[h]:>16.4f}{note}")

    # ── Test 2: OV circuit (does W_O W_V preserve codeword identity?) ───
    print("\n=== Test 2: per-head OV codeword-copy score (L0) ===")
    codewords = data["codewords"]
    ov_scores = compute_ov_codeword_diagonal(model, codewords, tokenizer)
    print(f"  {'head':<6} {'argmax_correct':>14} {'median_rank':>12} note")
    for s in ov_scores:
        h = s["head"]
        note = " ← circuit head" if h in CIRCUIT_HEADS else (" (control)" if h in CONTROL_HEADS else "")
        print(f"  L0H{h:<3} {s['argmax_correct_frac']:>14.4f} "
              f"{s['median_rank_of_target']:>12.0f}{note}")

    # ── Save ────────────────────────────────────────────────────────────
    out = {
        "ckpt_step": CKPT_STEP, "n_examples": n_valid,
        "circuit_heads": CIRCUIT_HEADS, "control_heads": CONTROL_HEADS,
        "attn_to_key": attn_to_key.tolist(),
        "attn_to_self": attn_to_self.tolist(),
        "attn_to_random_position": attn_to_random_position.tolist(),
        "ov_codeword_score": ov_scores,
    }
    out_json = REPO / "analyses/probe_circuit_mechinterp.json"
    with open(out_json, "w") as f:
        json.dump(out, f, indent=2)
    print(f"\nwrote {out_json}")

    # ── Plot: single headline panel + ratio panel ────────────────────────
    fig, axes = plt.subplots(1, 2, figsize=(15, 5))

    colors = []
    for h in range(n_head):
        if h in CIRCUIT_HEADS:
            colors.append("tab:red")
        elif h in CONTROL_HEADS:
            colors.append("tab:gray")
        else:
            colors.append("tab:blue")

    # Panel 0: attention-to-KEY per head + baseline
    width = 0.4
    x = np.arange(n_head)
    axes[0].bar(x - width/2, attn_to_key, width, color=colors,
                label="attention → KEY position", edgecolor="k", linewidth=0.5)
    axes[0].bar(x + width/2, attn_to_random_position, width,
                color="lightgray", edgecolor="k", linewidth=0.5,
                label="attention → other positions (avg, baseline)")
    axes[0].set_xticks(x)
    axes[0].set_xticklabels([f"H{h}" for h in range(n_head)])
    axes[0].set_xlabel("Layer-0 head index")
    axes[0].set_ylabel("attention from query-read position")
    axes[0].set_title(f"L0 per-head attention back to KEY position\n"
                      f"(red = spectrally-identified circuit head; "
                      f"gray = matched control)\n"
                      f"avg over {n_valid} probe examples, ckpt step={CKPT_STEP}")
    axes[0].axhline(0, color="k", lw=0.5)
    axes[0].grid(True, alpha=0.3, axis="y")
    axes[0].legend(fontsize=9, loc="upper left")

    # Panel 1: ratio = attention(→ KEY) / attention(→ uniform other position)
    # Quantifies how strongly the head selectively attends to the KEY.
    # Equal-uniform weighting would give ratio ≈ 1; selective attention >> 1.
    ratio = attn_to_key / np.maximum(attn_to_random_position, 1e-8)
    axes[1].bar(x, ratio, color=colors, edgecolor="k", linewidth=0.5)
    axes[1].set_xticks(x)
    axes[1].set_xticklabels([f"H{h}" for h in range(n_head)])
    axes[1].set_xlabel("Layer-0 head index")
    axes[1].set_ylabel("attention(→KEY) / attention(→random other) ratio")
    axes[1].set_title(f"Selectivity for KEY position\n"
                      f"(ratio of attention to KEY vs uniform other-position baseline)\n"
                      f"ratio = 1 means no selectivity; circuit heads ≈ 50–100×")
    axes[1].axhline(1, color="k", lw=0.5, ls="--", alpha=0.5)
    axes[1].grid(True, alpha=0.3, axis="y")

    fig.tight_layout()
    out_png = REPO / "analyses/probe_circuit_mechinterp.png"
    fig.savefig(out_png, dpi=130, bbox_inches="tight")
    print(f"saved {out_png}")


if __name__ == "__main__":
    main()
