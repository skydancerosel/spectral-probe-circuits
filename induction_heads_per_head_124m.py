"""
induction_heads_per_head_124m.py

Test the prediction stated in analyses/induction_heads_scope.md:
the per-head participation-ratio (PR) signal — which on TS-51M pre-identifies
the causally-relevant heads of an injected key-retrieval task — should also
pre-identify induction heads in karpathy_llmc GPT-2 124M trained on FineWeb-10B.

Pipeline:
  1. Build a fixed batch of induction-eval sequences (synthetic random tokens
     with planted A,B,...,A patterns; induction target is B at the position
     after the second A).
  2. For each checkpoint in karpathy_llmc/runs/gpt2_fineweb10B/, extract
     per-head attention output at the second-A position over the batch.
  3. SVD → participation ratio.
  4. Track PR(layer, head, ckpt_step). Heads with sharp PR transitions
     during induction-loss emergence are the spectral picks.

Output: analyses/induction_heads_per_head_124m.{json,png}
"""

import json
import sys
import re
import math
from dataclasses import dataclass
from pathlib import Path
import numpy as np
import torch
import torch.nn as nn
from torch.nn import functional as F

REPO = Path(__file__).resolve().parent
KARPATHY_CKPT_DIR = REPO / "karpathy_llmc/runs/gpt2_fineweb10B"

# ───────────────────── Karpathy GPT-2 124M ──────────────────────
# Minimal model that matches the structure of the saved checkpoints.

@dataclass
class GPTConfig:
    block_size: int = 1024
    vocab_size: int = 50257  # raw GPT-2 BPE; karpathy's saved ckpt uses this
    n_layer: int = 12
    n_head: int = 12
    n_embd: int = 768


class CausalSelfAttention(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        assert cfg.n_embd % cfg.n_head == 0
        self.c_attn = nn.Linear(cfg.n_embd, 3 * cfg.n_embd)
        self.c_proj = nn.Linear(cfg.n_embd, cfg.n_embd)
        self.n_head = cfg.n_head
        self.n_embd = cfg.n_embd

    def forward(self, x):
        B, T, C = x.size()
        qkv = self.c_attn(x)
        q, k, v = qkv.split(self.n_embd, dim=2)
        q = q.view(B, T, self.n_head, C // self.n_head).transpose(1, 2)
        k = k.view(B, T, self.n_head, C // self.n_head).transpose(1, 2)
        v = v.view(B, T, self.n_head, C // self.n_head).transpose(1, 2)
        y = F.scaled_dot_product_attention(q, k, v, is_causal=True)
        y = y.transpose(1, 2).contiguous().view(B, T, C)
        y = self.c_proj(y)
        return y


class MLP(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.c_fc = nn.Linear(cfg.n_embd, 4 * cfg.n_embd)
        self.gelu = nn.GELU(approximate="tanh")
        self.c_proj = nn.Linear(4 * cfg.n_embd, cfg.n_embd)

    def forward(self, x):
        return self.c_proj(self.gelu(self.c_fc(x)))


class Block(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.ln_1 = nn.LayerNorm(cfg.n_embd)
        self.attn = CausalSelfAttention(cfg)
        self.ln_2 = nn.LayerNorm(cfg.n_embd)
        self.mlp = MLP(cfg)

    def forward(self, x):
        x = x + self.attn(self.ln_1(x))
        x = x + self.mlp(self.ln_2(x))
        return x


class GPT(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.config = cfg
        self.transformer = nn.ModuleDict(dict(
            wte=nn.Embedding(cfg.vocab_size, cfg.n_embd),
            wpe=nn.Embedding(cfg.block_size, cfg.n_embd),
            h=nn.ModuleList([Block(cfg) for _ in range(cfg.n_layer)]),
            ln_f=nn.LayerNorm(cfg.n_embd),
        ))
        self.lm_head = nn.Linear(cfg.n_embd, cfg.vocab_size, bias=False)
        self.transformer.wte.weight = self.lm_head.weight  # weight tying

    def forward(self, idx):
        B, T = idx.size()
        pos = torch.arange(0, T, dtype=torch.long, device=idx.device)
        x = self.transformer.wte(idx) + self.transformer.wpe(pos)
        for block in self.transformer.h:
            x = block(x)
        x = self.transformer.ln_f(x)
        return self.lm_head(x)


def load_karpathy_ckpt(model, ckpt_path, device):
    ck = torch.load(ckpt_path, map_location=device, weights_only=True)
    sd = ck["model_state_dict"]
    # Strip torch.compile prefix
    sd = {k.replace("_orig_mod.", ""): v for k, v in sd.items()}
    # Drop buffer keys we don't carry (causal mask is implicit via SDPA)
    sd = {k: v for k, v in sd.items() if not k.endswith(".attn.bias")}
    # Drop lm_head.weight if present (we tie via wte)
    sd = {k: v for k, v in sd.items() if k != "lm_head.weight"}
    missing, unexpected = model.load_state_dict(sd, strict=False)
    if unexpected:
        print(f"    [load] unexpected keys (first 3): {list(unexpected)[:3]}")
    return ck["step"]


# ───────────────────── Induction eval batch ──────────────────────

def build_induction_batch(n_examples=2000, seq_len=256, vocab_lo=100, vocab_hi=10000, rng=None):
    """Synthetic induction eval.

    Each sequence has structure:
        [random filler] ... A, B ... [more random filler] ... A
    where A and B are two distinct tokens from [vocab_lo, vocab_hi).

    The 'test position' is the index of the second A. A working induction
    head should attend at that position back to the position after the first
    A (which contains B), causing the V output there to be content-dependent
    on B (which varies across examples).

    Returns:
      tokens: long [n_examples, seq_len]
      test_positions: long [n_examples]
      targets: long [n_examples] — the B token (induction prediction)
    """
    if rng is None:
        rng = np.random.RandomState(0)
    tokens = np.zeros((n_examples, seq_len), dtype=np.int64)
    test_pos = np.zeros(n_examples, dtype=np.int64)
    targets = np.zeros(n_examples, dtype=np.int64)

    for i in range(n_examples):
        seq = rng.randint(vocab_lo, vocab_hi, size=seq_len).astype(np.int64)
        a, b = rng.choice(np.arange(vocab_lo, vocab_hi), size=2, replace=False)
        # Place [A, B] at a random early position
        ab_idx = rng.randint(20, seq_len // 2)
        seq[ab_idx] = a
        seq[ab_idx + 1] = b
        # Place second A late, ensure it doesn't accidentally appear earlier
        # by overwriting any earlier occurrences of `a` (other than position ab_idx)
        # so the only A's are at ab_idx and the late position.
        for k in range(seq_len):
            if seq[k] == a and k != ab_idx:
                # replace with another random token
                seq[k] = rng.randint(vocab_lo, vocab_hi)
        # Ensure b's only "informative" occurrence is right after first A
        for k in range(seq_len):
            if seq[k] == b and k != ab_idx + 1:
                seq[k] = rng.randint(vocab_lo, vocab_hi)
        # Place second A at position seq_len - 1
        seq[-1] = a
        tokens[i] = seq
        test_pos[i] = seq_len - 1
        targets[i] = int(b)

    return (torch.from_numpy(tokens), torch.from_numpy(test_pos), torch.from_numpy(targets))


# ───────────────────── Per-head extraction ──────────────────────

def compute_pr(matrix):
    """Participation ratio of the singular value spectrum.

    matrix: [n, d] tensor on any device.
    """
    if matrix.dim() != 2 or matrix.shape[0] == 0:
        return float("nan")
    matrix = matrix.float()
    try:
        U, S, V = torch.linalg.svd(matrix, full_matrices=False)
    except Exception:
        # SVD can fail on near-degenerate matrices; fall back to CPU
        U, S, V = torch.linalg.svd(matrix.cpu(), full_matrices=False)
    s2 = S * S
    s2_sum = s2.sum().clamp(min=1e-12)
    p = s2 / s2_sum
    H = -(p * torch.log(p + 1e-12)).sum()
    return float(torch.exp(H).item())


def per_head_pr_at_positions(model, tokens, positions, n_layer, n_head, head_dim, device,
                              batch_size=64):
    """For each (layer, head), extract the per-head attention output at each
    sequence's `positions` index, then compute PR over the batch."""
    n = tokens.shape[0]
    # Storage: [n, n_layer, n_head, head_dim]
    out = torch.zeros(n, n_layer, n_head, head_dim, dtype=torch.float32)

    captured_mb = {}  # (layer) -> tensor [B, T, H, D] for current minibatch

    handles = []
    for L in range(n_layer):
        attn = model.transformer.h[L].attn
        def make_hook(L=L):
            def hook(module, ainputs):
                x = ainputs[0]
                B, T, C = x.shape
                xr = x.view(B, T, n_head, head_dim)
                captured_mb[L] = xr.detach()
            return hook
        handles.append(attn.c_proj.register_forward_pre_hook(make_hook()))

    try:
        with torch.no_grad():
            for start in range(0, n, batch_size):
                end = min(start + batch_size, n)
                tok = tokens[start:end].to(device)
                pos = positions[start:end].to(device)
                _ = model(tok)
                B = end - start
                for L in range(n_layer):
                    xr = captured_mb[L]  # [B, T, H, D]
                    # Pick test position per batch element
                    sel = xr[torch.arange(B, device=device), pos]  # [B, H, D]
                    out[start:end, L] = sel.cpu()
    finally:
        for h in handles:
            h.remove()

    # Compute PR per (L, H)
    pr_matrix = np.zeros((n_layer, n_head))
    for L in range(n_layer):
        for H in range(n_head):
            pr_matrix[L, H] = compute_pr(out[:, L, H, :])
    return pr_matrix


# ───────────────────── Main ──────────────────────

def main():
    device = "mps" if torch.backends.mps.is_available() else "cpu"
    print(f"device = {device}")

    cfg = GPTConfig()
    model = GPT(cfg).to(device)

    print("Building induction batch...")
    tokens, positions, targets = build_induction_batch(n_examples=2000, seq_len=256,
                                                        rng=np.random.RandomState(42))
    print(f"  batch: {tokens.shape}, test positions all = {positions[0].item()}, "
          f"distinct targets = {len(set(targets.tolist()))}")

    ckpts = sorted(KARPATHY_CKPT_DIR.glob("ckpt_*.pt"))
    print(f"Found {len(ckpts)} checkpoints in {KARPATHY_CKPT_DIR}")

    out = {
        "model": "karpathy_llmc/runs/gpt2_fineweb10B",
        "n_layer": cfg.n_layer,
        "n_head": cfg.n_head,
        "head_dim": cfg.n_embd // cfg.n_head,
        "n_examples": int(tokens.shape[0]),
        "seq_len": int(tokens.shape[1]),
        "ckpt_step": [],
        "pr": {f"L{l}_H{h}": [] for l in range(cfg.n_layer) for h in range(cfg.n_head)},
    }

    for ck_idx, ck_path in enumerate(ckpts):
        step = int(re.match(r"ckpt_(\d+)\.pt", ck_path.name).group(1))
        try:
            _step = load_karpathy_ckpt(model, ck_path, device)
        except Exception as e:
            print(f"  [skip] {ck_path.name}: load failed: {e}")
            continue
        model.eval()
        pr_matrix = per_head_pr_at_positions(model, tokens, positions,
                                              cfg.n_layer, cfg.n_head,
                                              cfg.n_embd // cfg.n_head, device)
        out["ckpt_step"].append(step)
        for L in range(cfg.n_layer):
            for H in range(cfg.n_head):
                out["pr"][f"L{L}_H{H}"].append(float(pr_matrix[L, H]))
        if ck_idx % 5 == 0 or ck_idx == len(ckpts) - 1:
            top = pr_matrix.max()
            mean = pr_matrix.mean()
            print(f"  ckpt {ck_idx+1}/{len(ckpts)} step={step:>5}  "
                  f"PR max={top:.2f}  mean={mean:.2f}")

    out_json = REPO / "results/induction_heads_per_head_124m.json"
    with open(out_json, "w") as f:
        json.dump(out, f, indent=2)
    print(f"\nwrote {out_json}")


if __name__ == "__main__":
    main()
