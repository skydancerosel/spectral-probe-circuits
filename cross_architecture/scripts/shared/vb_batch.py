"""Variable-binding batch.

Template:
  " {name_A} lives in {city_A}. {name_B} lives in {city_B}. {query_name} lives in"

Where query_name ∈ {name_A, name_B}, 50/50. The model must dereference
query_name to its bound city and predict that city. Target = city bound to
query_name; distractor = city bound to the other name.

Position structure (verified 13 tokens with all-single-token names/cities):
  pos 0:  " {name_A}"       (first name introduced)
  pos 1:  " lives"
  pos 2:  " in"
  pos 3:  " {city_A}"       (bound value for name_A)
  pos 4:  "."
  pos 5:  " {name_B}"       (second name introduced)
  pos 6:  " lives"
  pos 7:  " in"
  pos 8:  " {city_B}"       (bound value for name_B)
  pos 9:  "."
  pos 10: " {query_name}"   (re-occurrence of the variable to dereference)
  pos 11: " lives"
  pos 12: " in"             (query position; predict next = bound city)
"""
from __future__ import annotations
import numpy as np
import torch

NAMES = ['John','Mary','Alice','Bob','Anna','Tom','Lisa','Sarah','James','David',
         'Michael','Jessica','Emily','Daniel','Laura','Mark','Susan','Helen','Paul',
         'Karen','Steven','Linda','Kevin','Donna','Brian','Sandra','Ruth','Frank',
         'George','Henry','Eric','Carl','Joe','Jane','Kate','Beth','Sam','Max',
         'Ben','Will','Adam','Eve']

CITIES = ['Paris','Rome','Berlin','London','Madrid','Vienna','Boston','Dublin',
          'Sydney','Seoul','Cairo','Moscow','Athens','Tokyo','Mumbai','Beijing',
          'Delhi','Manila','Bangkok','Toronto','Chicago','Miami','Seattle',
          'Houston','Detroit','Denver','Atlanta','Portland','Phoenix','Dallas']

PROMPT_LEN = 13  # tokens per prompt (deterministic given single-token names/cities)


def build_vb_batch(tokenizer, n_examples: int = 500, seed: int = 42):
    """Returns (tokens [n, 13], target_ids [n], distractor_ids [n],
                binding_pos [n], distractor_pos [n], records).

    binding_pos is the position of the bound city for the query variable (3 or 8).
    distractor_pos is the position of the other city (8 or 3).
    """
    rng = np.random.RandomState(seed)
    records, encoded = [], []
    target_ids, distractor_ids = [], []
    binding_positions, distractor_positions = [], []
    for i in range(n_examples):
        # Two distinct names + two distinct cities
        name_idx = rng.choice(len(NAMES), 2, replace=False)
        city_idx = rng.choice(len(CITIES), 2, replace=False)
        name_A, name_B = NAMES[name_idx[0]], NAMES[name_idx[1]]
        city_A, city_B = CITIES[city_idx[0]], CITIES[city_idx[1]]
        # 50/50 which variable is queried
        query_is_A = (i % 2 == 0)
        query_name = name_A if query_is_A else name_B
        target_city = city_A if query_is_A else city_B
        distractor_city = city_B if query_is_A else city_A
        binding_pos = 3 if query_is_A else 8
        distractor_pos = 8 if query_is_A else 3

        # Leading space for first token to keep tokenization clean
        prompt = f" {name_A} lives in {city_A}. {name_B} lives in {city_B}. {query_name} lives in"
        ids = tokenizer.encode(prompt, add_special_tokens=False)
        if len(ids) != PROMPT_LEN:
            raise ValueError(f"Prompt does not tokenize to {PROMPT_LEN} tokens: '{prompt}' → {len(ids)}")
        tid = tokenizer.encode(" " + target_city, add_special_tokens=False)
        did = tokenizer.encode(" " + distractor_city, add_special_tokens=False)
        if len(tid) != 1 or len(did) != 1:
            raise ValueError(f"city not single-token: target={target_city}({tid}) distractor={distractor_city}({did})")

        encoded.append(ids)
        target_ids.append(tid[0])
        distractor_ids.append(did[0])
        binding_positions.append(binding_pos)
        distractor_positions.append(distractor_pos)
        records.append({
            'name_A': name_A, 'name_B': name_B,
            'city_A': city_A, 'city_B': city_B,
            'query_name': query_name, 'query_is_A': query_is_A,
            'target_city': target_city, 'distractor_city': distractor_city,
            'binding_pos': binding_pos, 'distractor_pos': distractor_pos,
            'prompt': prompt,
        })
    tokens = torch.tensor(np.array(encoded, dtype=np.int64))
    target_ids_t = torch.tensor(np.array(target_ids, dtype=np.int64))
    distractor_ids_t = torch.tensor(np.array(distractor_ids, dtype=np.int64))
    binding_pos_t = torch.tensor(np.array(binding_positions, dtype=np.int64))
    distractor_pos_t = torch.tensor(np.array(distractor_positions, dtype=np.int64))
    return tokens, target_ids_t, distractor_ids_t, binding_pos_t, distractor_pos_t, records


def evaluate_vb(model, tokens, target_ids, distractor_ids, device, batch_size=4):
    """At query position (pos 12), top-1 accuracy on target; logit_diff."""
    import torch.nn.functional as F
    n = tokens.shape[0]; last = tokens.shape[1] - 1
    top1, lt_gt_ld, lt_vals, ld_vals = [], [], [], []
    with torch.no_grad():
        for s in range(0, n, batch_size):
            e = min(s + batch_size, n)
            tok = tokens[s:e].to(device)
            tgt = target_ids[s:e].to(device); dis = distractor_ids[s:e].to(device)
            logits = model(tok).logits[:, last, :]
            top1_pred = logits.argmax(-1)
            top1.append((top1_pred == tgt).float().cpu().numpy())
            lt = logits.gather(-1, tgt.unsqueeze(-1)).squeeze(-1)
            ld = logits.gather(-1, dis.unsqueeze(-1)).squeeze(-1)
            lt_gt_ld.append((lt > ld).float().cpu().numpy())
            lt_vals.append(lt.cpu().numpy()); ld_vals.append(ld.cpu().numpy())
    lt = np.concatenate(lt_vals); ld = np.concatenate(ld_vals)
    return {
        "top1": float(np.concatenate(top1).mean()),
        "frac_target_gt_distractor": float(np.concatenate(lt_gt_ld).mean()),
        "logit_diff_mean": float((lt - ld).mean()),
    }


if __name__ == "__main__":
    from transformers import AutoTokenizer
    for name, model in [('pythia', 'EleutherAI/pythia-1b'),
                         ('olmo', 'allenai/OLMo-1B-0724-hf'),
                         ('olmoe', 'allenai/OLMoE-1B-7B-0924')]:
        tk = AutoTokenizer.from_pretrained(model)
        tokens, tgt, dis, bp, dp, recs = build_vb_batch(tk, n_examples=10)
        print(f"\n{name}: tokens {tokens.shape}")
        ids0 = tokens[0].tolist()
        print(f"  ex0: {recs[0]['prompt']!r}")
        print(f"  decoded positions: {[tk.decode([i]) for i in ids0]}")
        print(f"  query='{recs[0]['query_name']}', target='{recs[0]['target_city']}' at pos {recs[0]['binding_pos']}; "
              f"distractor='{recs[0]['distractor_city']}' at pos {recs[0]['distractor_pos']}")
