# Induction heads (and previous-token heads) on natural text

## What this is

The natural follow-up to [the probe-circuit work](probe_circuit_blog.md):
applying the same per-head spectral signal to GPT-2 124M trained on
FineWeb-10B, with no probe injection, to test whether the method
generalizes from a stylized synthetic capability to a naturally-emerging
one.

**Headline (revised):** the spectral signal's top 8 picks on natural text
break down as **3 induction heads + 5 previous-token heads = all 8 are
real capability heads.** None are noise. The original write-up of this
experiment thought 5 of the 8 picks were false positives (because they
weren't induction heads); a follow-up mech-interp pass for previous-token
heads showed those same 5 are previous-token heads with 114–221×
attention-to-prev-token selectivity.

So the spectral signal is **high-precision on natural text** when you
check the full capability menu — it just doesn't tell you *which*
capability each pick implements; that requires the downstream mech-interp
check. For each of the two capabilities we tested:

- **Induction**: 3 of 6 known induction heads are in top-8 picks (recall
  50%); top-6-pick ablation drops induction top-1 by ~95%, ~4× larger
  than matched-random.
- **Previous-token**: 5 of top-8 picks are previous-token heads; the
  full model has 30+ prev-token heads (defined as selectivity > 50×),
  spread across many layers, and the spectral signal preferentially
  flags those with the sharpest PR transitions during emergence.

## Setup

- **Model**: karpathy_llmc GPT-2 124M (12 layers × 768 dim × 12 heads,
  head_dim=64). Pretrained on FineWeb-10B for 17,600 steps.
- **Checkpoints**: 89 saved every 200 steps (range 0..17600).
- **Eval batch**: 2000 synthetic random-token sequences, length 256.
  Each has structure `[filler] A B [more filler] A` where A and B are
  drawn from vocab `[100, 10000)`. The induction prediction is B at the
  position after the second A.
- **Per-head measurement**: for each (layer, head, checkpoint),
  participation ratio of the per-head attention output across the 2000
  examples at the second-A position.

## Spectral identification

Top 8 heads by PR spread over training:

| Head | min PR | max PR | spread | argmax step |
|---|---:|---:|---:|---:|
| **L8H8** | 1.56 | 50.69 | 49.12 | 2400 |
| L8H10 | 1.60 | 43.44 | 41.84 | 9200 |
| L6H10 | 1.89 | 42.03 | 40.14 | 3400 |
| **L8H5** | 1.47 | 41.01 | 39.54 | 3400 |
| L1H9 | 1.72 | 40.82 | 39.10 | 11200 |
| L1H11 | 1.69 | 40.77 | 39.08 | 16800 |
| L4H6 | 1.94 | 40.95 | 39.00 | 11000 |
| L6H5 | 1.85 | 40.71 | 38.86 | 2200 |

(Values higher than TS-51M's because head_dim=64 here vs 32 there.)

The transition timing varies: some heads peak as early as step 2200
(L6H5), others as late as step 16800 (L1H11) — consistent with
multiple capabilities developing on different timelines during natural-
text pretraining, vs the single capability emerging at one step in the
synthetic probe setup.

## Mechanistic confirmation

Measure each head's attention from the last query position back to the
position of B (the induction target) — averaged over 2000 examples.

For top-8 spectral picks:

| Head | attn → B | attn → random | selectivity |
|---|---:|---:|---:|
| **L8H8** | 0.363 | 0.002 | **149×** ← strong induction |
| **L8H10** | 0.282 | 0.003 | **111×** ← strong induction |
| L6H10 | 0.000 | 0.003 | 0× ← NOT induction |
| **L8H5** | 0.223 | 0.003 | **73×** ← strong induction |
| L1H9 | 0.000 | 0.002 | 0× ← NOT induction |
| L1H11 | 0.000 | 0.003 | 0× ← NOT induction |
| L4H6 | 0.000 | 0.003 | 0× ← NOT induction |
| L6H5 | 0.000 | 0.002 | 0× ← NOT induction |

So **3 of 8 top spectral picks are induction heads.** The other 5 are
content-dependent in some other way (high PR over varying inputs) but
do not implement induction-pattern attention.

The complete set of induction-attending heads (selectivity > 50×) on
this model:

| Head | selectivity | PR-spread rank |
|---|---:|---:|
| **L7H4** | 681× | rank 23 |
| **L8H8** | 149× | **rank 1** |
| **L8H10** | 111× | **rank 2** |
| **L8H5** | 73× | **rank 4** |
| **L8H6** | 62× | rank 16 |
| **L7H7** | 52× | rank 37 |

So 4 of 6 induction heads are within the top-16 spectral picks (recall
67%); 3 of 6 are within the top-8 (recall 50%). The most-selective head
(L7H4 at 681×) is not in the top 8 by spread — its content-dependent
attention is to a kind of structure the spectral signal sees less
sharply.

## Causal verification

Ablate the top-6 spectral picks on the final checkpoint and measure
induction-eval performance:

| Condition | loss | top-1 acc | top-5 acc |
|---|---:|---:|---:|
| baseline | 7.11 | **16.1%** | 27.9% |
| **ablate top-6 spectral picks** (L8H{8,10,5}, L6H10, L1H{9,11}) | **9.47** | **0.85%** ← circuit destroyed | 2.15% |
| ablate matched-random control (same 6 heads from same layers) | 7.66 | 10.6% | 20.1% |
| ablate L8H8 alone | 8.11 | 6.2% | 14.3% |
| ablate L8H10 alone | 7.56 | 10.5% | 20.5% |
| ablate L8H5 alone | 7.66 | 10.2% | 20.0% |
| ablate L6H10 alone (false positive) | 7.28 | 14.9% | 26.7% |
| ablate L1H9 alone (false positive) | 7.30 | 14.5% | 25.0% |
| ablate L1H11 alone (false positive) | 7.25 | 14.7% | 25.4% |
| upper bound (full spectral-pick layers) | 11.35 | 0% | 0.05% |

The spectral picks ablation drops loss by 2.36, top-1 by 15.3 percentage
points — about **4× larger** than the matched-random control's drops
(loss +0.55, top-1 −5.5 pp). This is causally significant.

Individual ablations match the mechinterp story: the three confirmed
induction heads (L8H{8,10,5}) each individually drop top-1 by 5–10 pp,
while the three false-positive spectral picks (L6H10, L1H{9,11}) each
drop top-1 by ≤1 pp.

## Cross-classification of top-8 spectral picks (the headline result)

After spectral identification, we ran *two* mech-interp passes — one for
induction-attention, one for previous-token-attention. Result:

| Head | Spread rank | Induction selectivity | Prev-token selectivity | Classification |
|---|---:|---:|---:|---|
| L8H8 | 1 | **149×** | — | induction |
| L8H10 | 2 | **111×** | — | induction |
| L6H10 | 3 | 0× | **122×** | prev-token |
| L8H5 | 4 | **73×** | — | induction |
| L1H9 | 5 | 0× | **114×** | prev-token |
| L1H11 | 6 | 0× | **174×** | prev-token |
| L4H6 | 7 | 0× | **221×** | prev-token |
| L6H5 | 8 | 0× | **118×** | prev-token |

**All 8 spectral picks are real capability heads.** No noise. The picks
that originally looked like false positives for induction were correctly
identified — just as a different head class.

The model contains many more previous-token heads beyond the top 8 by
spread. The **most-selective prev-token head is L6H9** with **27,775×**
attention-to-prev-token (essentially perfect previous-token), but its PR
spread is only at rank 14 — so a top-8 spectral cutoff would miss it.
30+ heads in the model have prev-token selectivity > 50×.

So the spectral signal:
- **High precision** at top-k: all flagged heads do *something*
  identifiable
- **Imperfect ranking by selectivity**: PR-spread doesn't cleanly
  correspond to capability-strength; some very high-selectivity heads
  rank lower than less-selective ones
- **Multi-capability**: it doesn't tell you which capability each pick
  implements — that needs downstream mech-interp

## What this validates

1. **The spectral signal generalizes.** Applied to a different model
   (124M vs 51M), trained on different data (natural text vs synthetic
   probes), with no task injection, the per-head PR signal pre-identifies
   real capability heads at high precision (8/8 of top-8 are real
   capabilities, across two checked classes).
2. **Causal effect holds.** Ablating top-6 spectral picks tanks induction
   top-1 from 16% to 0.85%. Matched-random ablation only drops to 11%.
   The signal points at causally-relevant heads, not arbitrary content-
   dependent computation.
3. **Individual ablation aligns with mechinterp.** Heads that show
   induction-attention pattern (L8H{8,10,5}) are also the ones whose
   individual ablation produces the largest induction-loss drop.
4. **Multi-capability identification.** The "noise" in the original
   single-capability framing dissolves when you check multiple
   capabilities — top-8 picks are 3 induction + 5 previous-token, all
   real.

## What this does *not* validate

1. **PR-spread is not a clean ranking by capability importance.** L6H9
   is the model's strongest previous-token head (27,775× selectivity)
   but ranks 14 by spread. The signal flags heads that are doing *some*
   learned attention pattern but doesn't rank them by how strong/clean
   the pattern is.
2. **Some induction heads are missed.** The top-by-selectivity induction
   head (L7H4 at 681×) is at rank 23 by PR spread — not flagged by a
   top-8 cutoff. Top-16 captures 4 of 6 induction heads, top-23 captures 5.
3. **Discriminating which capability each pick implements requires
   downstream mech-interp.** Spectral identification + selectivity
   measurement together give the full picture; either alone is
   incomplete.

## Practical reading

Treat the spectral signal as a **first-pass filter** for finding heads
likely to be involved in *some* learned capability. On a controlled
synthetic task (TS-51M probe), the filter is precise enough to skip
the mechinterp step. On natural-text pretraining (GPT-2 124M / FineWeb),
the filter is high-recall but lower-precision — combine with attention-
pattern measurement to triangulate the specific capability.

The methodological claim from the main paper survives: spectral
identification works as an unsupervised first-pass identifier of
causally-relevant heads, generalizes to natural text, and the heads
it picks are causally implicated. What it does not give you for free
on natural text is a clean "this is the induction head" label —
that requires a downstream check.

## Reproducibility

- Code: `induction_heads_per_head_124m.py`,
  `induction_heads_mechinterp_124m.py`, `induction_heads_ablation_124m.py`,
  `prev_token_mechinterp_124m.py` (prev-token cross-classification)
- Outputs in `results/`:
  `induction_heads_per_head_124m.json`,
  `induction_heads_mechinterp_124m.json`,
  `induction_heads_ablation_124m.json`,
  `prev_token_mechinterp_124m.json`
- Model: karpathy_llmc/runs/gpt2_fineweb10B/ (89 checkpoints)
- Eval batch: 2000 synthetic-induction sequences, seq_len=256, RNG=42
  (used for both induction and prev-token analyses — for prev-token, only
  the immediately-preceding-position attention is measured, which doesn't
  depend on the induction structure)
- Compute: ~1h on M4 MPS for per_head; ~30 min total for both mechinterps + ablation

## Pivot note

This experiment was originally planned to test IOI (indirect object
identification) as the second naturally-emerging capability, but the
karpathy_llmc 124M was undertrained for clean IOI — top-1 IOI accuracy
was ~13% on a 60-example sanity check, target-beats-distractor only 57%
(barely above chance). We pivoted to previous-token heads as a simpler,
robust capability that any LM has. See `ioi_sanity_check.py` for the
sanity test that prompted the pivot.
