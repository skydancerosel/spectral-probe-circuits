"""Successor batch (Gould et al. 2023 style).

Prompts are 5-item ordinal sequences:
  - Days:    "Monday Tuesday Wednesday Thursday Friday" → predict "Saturday"
  - Months:  "January February March April May" → predict "June"
  - Ordinals: "first second third fourth fifth" → predict "sixth"
  - Numbers: "1 2 3 4 5" → predict "6"

All items are single-token in Pythia / OLMo / OLMoE tokenizers (verified).
Every prompt is exactly 5 tokens. Query position = 4 (last item).
Target = the next item in the sequence.

For cycle-based sequences (days, months), starting position wraps around;
for ordinals (10 items) and numbers (1-99), no wrap-around.
"""
from __future__ import annotations
import numpy as np
import torch

DAYS   = ["Monday","Tuesday","Wednesday","Thursday","Friday","Saturday","Sunday"]
MONTHS = ["January","February","March","April","May","June","July","August","September","October","November","December"]
ORDS   = ["first","second","third","fourth","fifth","sixth","seventh","eighth","ninth","tenth"]
NUMS   = [str(i) for i in range(1, 100)]  # 1..99

SEQ_LEN = 5  # number of items shown; target is the next item


def _enumerate_unique_sequences():
    """Return list of (sequence_name, items, target).
    items has length SEQ_LEN; target is the (SEQ_LEN+1)-th item.
    """
    out = []
    # Days — cyclic
    for start in range(len(DAYS)):
        items = [DAYS[(start + i) % len(DAYS)] for i in range(SEQ_LEN)]
        target = DAYS[(start + SEQ_LEN) % len(DAYS)]
        out.append(("days", items, target))
    # Months — cyclic
    for start in range(len(MONTHS)):
        items = [MONTHS[(start + i) % len(MONTHS)] for i in range(SEQ_LEN)]
        target = MONTHS[(start + SEQ_LEN) % len(MONTHS)]
        out.append(("months", items, target))
    # Ordinals — no wrap-around (10 items, so SEQ_LEN=5 means start ∈ [0, 4])
    for start in range(len(ORDS) - SEQ_LEN):
        items = ORDS[start:start + SEQ_LEN]
        target = ORDS[start + SEQ_LEN]
        out.append(("ordinals", items, target))
    # Numbers — no wrap-around (1..99 with SEQ_LEN=5 means start ∈ [0, 93])
    for start in range(len(NUMS) - SEQ_LEN):
        items = NUMS[start:start + SEQ_LEN]
        target = NUMS[start + SEQ_LEN]
        out.append(("numbers", items, target))
    return out


def build_succ_batch(tokenizer, n_examples: int = None, seed: int = 42):
    """Build successor batch. Returns (tokens [n, 5], target_ids [n], records).

    If n_examples is None, uses all unique sequences (~120). If specified and
    > unique count, samples with replacement; otherwise samples without replacement.
    """
    all_seqs = _enumerate_unique_sequences()
    rng = np.random.RandomState(seed)
    if n_examples is None or n_examples >= len(all_seqs):
        rng.shuffle(all_seqs)
        selected = all_seqs if n_examples is None else all_seqs[:n_examples]
    else:
        idx = rng.choice(len(all_seqs), n_examples, replace=False)
        selected = [all_seqs[i] for i in idx]

    encoded = []
    target_ids = []
    records = []
    for seq_name, items, target in selected:
        # Prepend leading space so every item has the leading-space form (avoids
        # sentence-initial-vs-mid tokenization differences for ordinals like "fifth").
        prompt = " " + " ".join(items)  # " Monday Tuesday Wednesday Thursday Friday"
        ids = tokenizer.encode(prompt, add_special_tokens=False)
        if len(ids) != SEQ_LEN:
            raise ValueError(f"Prompt does not tokenize to {SEQ_LEN}: '{prompt}' → {len(ids)}")
        target_id = tokenizer.encode(" " + target, add_special_tokens=False)
        if len(target_id) != 1:
            raise ValueError(f"Target ' {target}' is not single-token: {target_id}")
        encoded.append(ids)
        target_ids.append(target_id[0])
        records.append({"seq": seq_name, "items": items, "target": target, "prompt": prompt})
    tokens = torch.tensor(np.array(encoded, dtype=np.int64))
    target_ids_t = torch.tensor(np.array(target_ids, dtype=np.int64))
    return tokens, target_ids_t, records


def evaluate_succ(model, tokens, target_ids, device, batch_size=4):
    """At query position (last), top-1 accuracy on next item; logit of target."""
    import torch.nn.functional as F
    n = tokens.shape[0]; last = tokens.shape[1] - 1
    top1_correct = np.zeros(n, dtype=np.float64)
    target_logit = np.zeros(n, dtype=np.float64)
    target_logit_rank = np.zeros(n, dtype=np.float64)
    with torch.no_grad():
        for s in range(0, n, batch_size):
            e = min(s + batch_size, n)
            tok = tokens[s:e].to(device)
            tgt = target_ids[s:e].to(device)
            logits = model(tok).logits[:, last, :]  # [B, V]
            top1 = logits.argmax(-1)
            top1_correct[s:e] = (top1 == tgt).float().cpu().numpy()
            target_logit[s:e] = logits.gather(-1, tgt.unsqueeze(-1)).squeeze(-1).float().cpu().numpy()
            # Rank of target among all tokens
            ranks = (logits > logits.gather(-1, tgt.unsqueeze(-1))).sum(-1)
            target_logit_rank[s:e] = ranks.float().cpu().numpy()
    return {
        "top1_acc": float(top1_correct.mean()),
        "target_logit_mean": float(target_logit.mean()),
        "target_rank_mean": float(target_logit_rank.mean()),
        "target_rank_median": float(np.median(target_logit_rank)),
    }


if __name__ == "__main__":
    from transformers import AutoTokenizer
    for name, model in [('pythia', 'EleutherAI/pythia-1b'),
                         ('olmo', 'allenai/OLMo-1B-0724-hf'),
                         ('olmoe', 'allenai/OLMoE-1B-7B-0924')]:
        tk = AutoTokenizer.from_pretrained(model)
        tokens, target_ids, records = build_succ_batch(tk)
        print(f"\n{name}: tokens {tokens.shape}, unique seqs = {len(records)}")
        # Show one from each sequence type
        seen = set()
        for r in records:
            if r['seq'] in seen: continue
            seen.add(r['seq'])
            print(f"  {r['seq']:>8}: {' '.join(r['items'])} → {r['target']!r}")
