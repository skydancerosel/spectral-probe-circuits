"""
ioi_sanity_check.py

5-minute test: does the karpathy_llmc 124M final ckpt have working IOI
circuits? Generate ~60 IOI sentences, measure indirect-object prediction
accuracy. If >65% top-1, proceed with full experiment. If <30%, pivot.
"""

import sys
from pathlib import Path
import numpy as np
import torch

REPO = Path(__file__).resolve().parent
sys.path.insert(0, str(REPO))
from induction_heads_per_head_124m import GPT, GPTConfig, load_karpathy_ckpt
from transformers import GPT2Tokenizer

KARPATHY_FINAL = REPO / "karpathy_llmc/runs/gpt2_fineweb10B/ckpt_017600.pt"

# Common short first names — most should be single-token with leading space in GPT-2 BPE
NAMES = ["Mary", "John", "Tom", "Bob", "Alice", "Sarah", "James", "Anna",
         "David", "Emma", "Mike", "Kate", "Paul", "Lisa", "Chris", "Jane",
         "Peter", "Sam", "Mark", "Lucy", "Will", "Ann", "Joe", "Eve"]
PLACES = ["store", "park", "school", "office", "house", "garden"]
ITEMS = ["drink", "book", "ball", "letter", "gift", "key"]
TEMPLATES = [
    "When {a} and {b} went to the {place}, {b} gave a {item} to",
    "After {a} and {b} arrived at the {place}, {b} handed the {item} to",
    "Then, {a} and {b} went to the {place}. {b} gave the {item} to",
    "While {a} and {b} were at the {place}, {b} threw the {item} to",
]


def main():
    device = "mps" if torch.backends.mps.is_available() else "cpu"
    print(f"device = {device}")
    tokenizer = GPT2Tokenizer.from_pretrained("gpt2")

    cfg = GPTConfig()
    model = GPT(cfg).to(device)
    step = load_karpathy_ckpt(model, str(KARPATHY_FINAL), device)
    model.eval()
    print(f"Loaded ckpt step={step}")

    # Verify which names are single-token-with-leading-space
    valid_names = []
    name_token = {}
    for n in NAMES:
        ids = tokenizer.encode(" " + n, add_special_tokens=False)
        if len(ids) == 1:
            valid_names.append(n)
            name_token[n] = ids[0]
    print(f"Valid (single-token w/ leading space) names: {len(valid_names)}: {valid_names}")
    if len(valid_names) < 6:
        print("ERROR: not enough valid names")
        return

    # Generate IOI sentences
    rng = np.random.RandomState(42)
    n_examples = 60
    examples = []
    for i in range(n_examples):
        a, b = rng.choice(valid_names, size=2, replace=False)
        place = rng.choice(PLACES)
        item = rng.choice(ITEMS)
        template = TEMPLATES[i % len(TEMPLATES)]
        text = template.format(a=a, b=b, place=place, item=item)
        examples.append({"text": text, "target": a, "distractor": b})

    # Score
    print("\nScoring...")
    correct_top1 = 0
    correct_top5 = 0
    target_beats_distractor = 0
    target_logits, distractor_logits = [], []

    for ex in examples:
        ids = tokenizer.encode(ex["text"], add_special_tokens=False)
        x = torch.tensor([ids]).to(device)
        with torch.no_grad():
            logits = model(x)
        last_logits = logits[0, -1]  # [V]

        tgt_id = name_token[ex["target"]]
        dist_id = name_token[ex["distractor"]]
        tgt_logit = last_logits[tgt_id].item()
        dist_logit = last_logits[dist_id].item()
        target_logits.append(tgt_logit)
        distractor_logits.append(dist_logit)
        if tgt_logit > dist_logit:
            target_beats_distractor += 1

        top5 = last_logits.topk(5).indices.cpu().tolist()
        if top5[0] == tgt_id:
            correct_top1 += 1
        if tgt_id in top5:
            correct_top5 += 1

    print(f"\nIOI baseline on karpathy_llmc 124M @ step {step}:")
    print(f"  top-1 accuracy:                  {correct_top1}/{n_examples} = "
          f"{100*correct_top1/n_examples:.1f}%")
    print(f"  top-5 accuracy:                  {correct_top5}/{n_examples} = "
          f"{100*correct_top5/n_examples:.1f}%")
    print(f"  target-beats-distractor (key!):  {target_beats_distractor}/{n_examples} = "
          f"{100*target_beats_distractor/n_examples:.1f}%")
    print(f"\n  mean target-logit:    {np.mean(target_logits):.3f}")
    print(f"  mean distractor-logit:{np.mean(distractor_logits):.3f}")
    print(f"  random baseline top-1: {100/len(valid_names):.1f}%")

    # Verdict
    print("\n" + "="*60)
    rate = correct_top1 / n_examples
    rate2 = target_beats_distractor / n_examples
    if rate > 0.50 or rate2 > 0.70:
        print(f"[OK] Model has working IOI signal "
              f"(top-1 {100*rate:.0f}%, target>distractor {100*rate2:.0f}%) "
              f"— proceed with full experiment.")
    elif rate > 0.20 or rate2 > 0.55:
        print(f"[MARGINAL] IOI partially works — may give weak signal.")
    else:
        print(f"[FAIL] IOI likely not learned. Pivot to bigram lookup or previous-token heads.")
    print("="*60)


if __name__ == "__main__":
    main()
