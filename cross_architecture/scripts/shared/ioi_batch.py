"""IOI batch builder. Cross-tokenizer compatible (Pythia / OLMo / OLMoE).

Standard IOI template (Wang et al. 2022):
  "When {name_A} and {name_B} went to the {place}, {name_B} gave a {object} to"
Target: " {name_A}"  (the indirect object — appeared first, NOT subject of clause 2)
Distractor: " {name_B}"

All names / places / objects verified to tokenize as 1 token (with leading space)
in all three tokenizers.
"""
from __future__ import annotations
import numpy as np
import torch
from typing import Tuple

NAMES = ['John', 'Mary', 'Alice', 'Bob', 'Anna', 'Tom', 'Lisa', 'Sarah', 'James',
         'David', 'Michael', 'Jessica', 'Emily', 'Daniel', 'Laura', 'Mark', 'Susan',
         'Helen', 'Paul', 'Karen', 'Steven', 'Linda', 'Kevin', 'Donna', 'Brian',
         'Sandra', 'Ruth', 'Frank', 'George', 'Henry', 'Eric', 'Carl', 'Joe',
         'Jane', 'Kate', 'Beth', 'Sam', 'Max', 'Ben', 'Will', 'Adam', 'Eve']
PLACES = ['store', 'park', 'school', 'office', 'house', 'bar']
OBJECTS = ['drink', 'book', 'ball', 'ring', 'phone', 'gift']

# ABBA = "name_A and name_B ... name_B gave ... to <name_A>"  (target appeared first)
# BABA = "name_B and name_A ... name_B gave ... to <name_A>"  (target appeared second)
# Mix both to control for position.
TEMPLATE_ABBA = "When {A} and {B} went to the {place}, {B} gave a {obj} to"
TEMPLATE_BABA = "When {B} and {A} went to the {place}, {B} gave a {obj} to"


def build_ioi_batch(tokenizer, n_examples: int = 500, seed: int = 42):
    """Build IOI batch. Returns (tokens [n, T_pad], target_id [n], distractor_id [n],
    io_pos [n], subj_first_pos [n], subj_second_pos [n], records).

    Tokens are left-padded so the final position is "to". Mixes ABBA and BABA 50/50.
    Position structure (all prompts: 14 tokens, single-token names/places/objects):
      pos 0: "When"
      pos 1: first name  (= A in ABBA, = B in BABA)
      pos 2: " and"
      pos 3: second name (= B in ABBA, = A in BABA)
      pos 4-8: " went to the {place},"
      pos 9: subject (always B)
      pos 10-12: " gave a {obj}"
      pos 13: " to"  (query position)
    """
    rng = np.random.RandomState(seed)
    records, encoded = [], []
    io_pos, subj_first, subj_second = [], [], []
    for i in range(n_examples):
        idx = rng.choice(len(NAMES), 2, replace=False)
        A, B = NAMES[idx[0]], NAMES[idx[1]]
        place = PLACES[rng.randint(len(PLACES))]
        obj = OBJECTS[rng.randint(len(OBJECTS))]
        is_abba = (i % 2 == 0)
        template = TEMPLATE_ABBA if is_abba else TEMPLATE_BABA
        prompt = template.format(A=A, B=B, place=place, obj=obj)
        ids = tokenizer.encode(prompt, add_special_tokens=False)
        tid = tokenizer.encode(" " + A, add_special_tokens=False)[0]
        did = tokenizer.encode(" " + B, add_special_tokens=False)[0]
        # IO = A. ABBA → A at pos 1. BABA → A at pos 3.
        # Subject = B. ABBA → B at pos 3 and 9. BABA → B at pos 1 and 9.
        if is_abba:
            io_p, sf_p, ss_p = 1, 3, 9
        else:
            io_p, sf_p, ss_p = 3, 1, 9
        records.append({'A': A, 'B': B, 'place': place, 'obj': obj,
                        'template': 'ABBA' if is_abba else 'BABA',
                        'prompt': prompt, 'target_id': tid, 'distractor_id': did,
                        'len': len(ids), 'io_pos': io_p, 'subj_first': sf_p, 'subj_second': ss_p})
        encoded.append(ids)
        io_pos.append(io_p); subj_first.append(sf_p); subj_second.append(ss_p)
    max_len = max(len(ids) for ids in encoded)
    pad_id = 0
    tokens = np.full((n_examples, max_len), pad_id, dtype=np.int64)
    pad_offset = []
    for i, ids in enumerate(encoded):
        off = max_len - len(ids)
        tokens[i, off:] = ids
        pad_offset.append(off)
    # Adjust positions by pad offset (with our setup pad_offset is 0 for all, but safe)
    io_pos = np.array(io_pos) + np.array(pad_offset)
    subj_first = np.array(subj_first) + np.array(pad_offset)
    subj_second = np.array(subj_second) + np.array(pad_offset)
    targets = np.array([r['target_id'] for r in records], dtype=np.int64)
    distractors = np.array([r['distractor_id'] for r in records], dtype=np.int64)
    return (torch.from_numpy(tokens), torch.from_numpy(targets), torch.from_numpy(distractors),
            torch.from_numpy(io_pos), torch.from_numpy(subj_first), torch.from_numpy(subj_second),
            records)


if __name__ == "__main__":
    from transformers import AutoTokenizer
    for name, m in [('pythia', 'EleutherAI/pythia-1b'), ('olmo', 'allenai/OLMo-1B-0724-hf'), ('olmoe', 'allenai/OLMoE-1B-7B-0924')]:
        tk = AutoTokenizer.from_pretrained(m)
        tokens, tgt, dis, io_p, sf_p, ss_p, recs = build_ioi_batch(tk, n_examples=10)
        print(f'{name}: tokens shape {tokens.shape}')
        print(f'  example 0 (ABBA): {recs[0]["prompt"]!r} | IO_pos={io_p[0].item()} subj_first={sf_p[0].item()} subj_second={ss_p[0].item()}')
        # Decode the positions to verify
        ids_0 = tokens[0].tolist()
        print(f'  decoded IO ({io_p[0]}): {tk.decode([ids_0[io_p[0]]])!r}')
        print(f'  decoded subj_first ({sf_p[0]}): {tk.decode([ids_0[sf_p[0]]])!r}')
        print(f'  decoded subj_second ({ss_p[0]}): {tk.decode([ids_0[ss_p[0]]])!r}')
        print(f'  example 1 (BABA): {recs[1]["prompt"]!r} | IO_pos={io_p[1].item()} subj_first={sf_p[1].item()} subj_second={ss_p[1].item()}')
        ids_1 = tokens[1].tolist()
        print(f'  decoded IO ({io_p[1]}): {tk.decode([ids_1[io_p[1]]])!r}')
        print(f'  decoded subj_first ({sf_p[1]}): {tk.decode([ids_1[sf_p[1]]])!r}')
