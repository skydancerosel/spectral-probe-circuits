"""Greater-than batch (Hanna et al. 2023 style).

Template:
  "The {NOUN} lasted from the year {Y1} to the year {CENTURY}"
where Y1 = {CENTURY}{DD} (e.g. 1737 = 17|37), and the model must complete with
a 2-digit number > DD. With single-token 2-digit BPE (verified for Pythia /
OLMo / OLMoE tokenizers), each prompt has length 12 tokens; the query is at
position 11; the start-year-decade ("DD") position is 7.

Positions (verified deterministic across templates):
  pos 0: "The"
  pos 1-5: " {noun} lasted from the year"  (5 tokens; noun must be single-token)
  pos 6: " {CC}"   (century prefix)
  pos 7: "{DD}"    (start-year decade — THE SCREEN TARGET POSITION)
  pos 8-10: " to the year"
  pos 11: " {CC}"  (query position)

Eval metrics at pos 11:
  - top1_above: top-1 predicted token is a 2-digit number > DD
  - prob_above: total probability mass on 2-digit numbers > DD
  - prob_below: total probability mass on 2-digit numbers <= DD
  - logit_diff_above_below: mean[logits over above] - mean[logits over below]
"""
from __future__ import annotations
import numpy as np
import torch

# Single-token nouns (verified 1-token across Pythia/OLMo/OLMoE tokenizers)
NOUNS = ["war", "drought", "famine", "exile", "voyage", "feast",
         "trial", "siege", "rebellion", "tournament", "plague", "festival",
         "treaty", "blockade", "uprising", "summit", "purge", "revolt",
         "expedition", "campaign", "feud", "boycott", "rally"]


def _valid_year_pairs(tokenizer):
    """Return list of (century_str, decade_int) tuples where " CCDD" tokenizes as 2 tokens
    [" CC", "DD"] in this tokenizer. Many 1900s years BPE-merge into a single token; this
    filters those out."""
    out = []
    for cc in range(14, 20):
        for dd in range(2, 89):
            year = f"{cc}{dd:02d}"
            full_ids = tokenizer.encode(f" {year}", add_special_tokens=False)
            cc_ids = tokenizer.encode(f" {cc}", add_special_tokens=False)
            dd_ids = tokenizer.encode(f"{dd:02d}", add_special_tokens=False)
            if full_ids == cc_ids + dd_ids:
                out.append((str(cc), dd))
    return out


def build_gt_batch(tokenizer, n_examples: int = 500, seed: int = 42):
    """Returns (tokens [n, 12], decade_at_pos7 [n], records).

    Filters (century, decade) pairs that don't tokenize as " CC" + "DD" (some 1900s
    years BPE-merge into single tokens). Uses only single-token nouns.
    Verifies every prompt tokenizes to exactly 12 tokens.
    """
    valid_pairs = _valid_year_pairs(tokenizer)
    if not valid_pairs:
        raise RuntimeError("No valid (century, decade) pairs found for this tokenizer.")
    rng = np.random.RandomState(seed)
    records = []
    encoded = []
    decades = []
    for i in range(n_examples):
        noun = NOUNS[rng.randint(len(NOUNS))]
        century, dd = valid_pairs[rng.randint(len(valid_pairs))]
        prompt = f"The {noun} lasted from the year {century}{dd:02d} to the year {century}"
        ids = tokenizer.encode(prompt, add_special_tokens=False)
        if len(ids) != 12:
            raise ValueError(f"Prompt does not tokenize to 12 tokens: '{prompt}' → {len(ids)}")
        encoded.append(ids)
        decades.append(dd)
        records.append({'noun': noun, 'century': century, 'start_decade': dd, 'prompt': prompt})
    tokens = torch.tensor(np.array(encoded, dtype=np.int64))
    decades_t = torch.tensor(np.array(decades, dtype=np.int64))
    return tokens, decades_t, records


def build_two_digit_token_table(tokenizer) -> dict:
    """Map decade integer (0..99) → token id for the 2-digit string (no leading space)."""
    out = {}
    for dd in range(100):
        s = f"{dd:02d}"
        ids = tokenizer.encode(s, add_special_tokens=False)
        if len(ids) != 1:
            raise ValueError(f"Decade '{s}' is not a single token: {ids}")
        out[dd] = ids[0]
    return out


def evaluate_gt(model, tokens, decades, two_digit_ids, device, batch_size=4):
    """Evaluate the model at pos 11 (last position).
    Returns dict with:
      top1_above, prob_above, prob_below, mean_logit_diff_above_below.
    """
    import torch.nn.functional as F
    n = tokens.shape[0]; last = tokens.shape[1] - 1
    top1_above = np.zeros(n, dtype=np.float64)
    prob_above = np.zeros(n, dtype=np.float64)
    prob_below = np.zeros(n, dtype=np.float64)
    logit_above_mean = np.zeros(n, dtype=np.float64)
    logit_below_mean = np.zeros(n, dtype=np.float64)
    above_token_ids_all = [[two_digit_ids[k] for k in range(dd + 1, 100)] for dd in range(100)]
    below_token_ids_all = [[two_digit_ids[k] for k in range(0, dd + 1)] for dd in range(100)]
    with torch.no_grad():
        for s in range(0, n, batch_size):
            e = min(s + batch_size, n)
            tok = tokens[s:e].to(device)
            logits = model(tok).logits[:, last, :]  # [B, V]
            probs = F.softmax(logits.float(), dim=-1)
            for bi in range(e - s):
                idx = s + bi
                dd = int(decades[idx].item())
                above_ids = above_token_ids_all[dd]
                below_ids = below_token_ids_all[dd]
                # top1
                top1_id = int(logits[bi].argmax().item())
                top1_above[idx] = 1.0 if top1_id in set(above_ids) else 0.0
                prob_above[idx] = float(probs[bi, above_ids].sum().item())
                prob_below[idx] = float(probs[bi, below_ids].sum().item())
                logit_above_mean[idx] = float(logits[bi, above_ids].mean().item())
                logit_below_mean[idx] = float(logits[bi, below_ids].mean().item())
    return {
        "top1_above": float(top1_above.mean()),
        "prob_above": float(prob_above.mean()),
        "prob_below": float(prob_below.mean()),
        "logit_diff_above_below": float((logit_above_mean - logit_below_mean).mean()),
    }


if __name__ == "__main__":
    from transformers import AutoTokenizer
    for name, model in [('pythia', 'EleutherAI/pythia-1b'),
                         ('olmo', 'allenai/OLMo-1B-0724-hf'),
                         ('olmoe', 'allenai/OLMoE-1B-7B-0924')]:
        tk = AutoTokenizer.from_pretrained(model)
        tokens, decades, records = build_gt_batch(tk, n_examples=10)
        print(f'\n{name}: tokens shape {tokens.shape}, first record: {records[0]}')
        # Decode first row to verify positions
        ids = tokens[0].tolist()
        print(f'  decoded: {[tk.decode([i]) for i in ids]}')
        print(f'  pos 7 (start-decade): {tk.decode([ids[7]])!r}; expected {records[0]["start_decade"]:02d}')
        # Verify 2-digit token table builds OK
        table = build_two_digit_token_table(tk)
        print(f'  2-digit token table built OK ({len(table)} entries)')
