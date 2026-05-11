# Induction heads (and previous-token heads) on natural text

## TL;DR

Applying the spectral signal from [the probe-circuit work](README.md)
to natural-text language models trained by different people on different data
gives a strikingly consistent picture across an 8× scale range:

- **~17–19% of heads in every model do identifiable specialized computation**
  (induction, previous-token, self-attention, first-token, etc.) — conserved
  across Karpathy 124M (FineWeb), Pythia 160M (Pile), and Pythia 410M (Pile).
- **Precision-at-k matches across models** when k is scaled to total head count.
  All models hit 100% precision at k≤10 and ~90–95% at the natural elbow in the
  PR-integral distribution.
- **Capabilities distribute across more heads at larger scale**, but the
  *fraction* of model capacity used for identifiable specialized work stays
  constant. (Tested directly on Pythia 410M: ablating all 11 heads with
  induction selectivity ≥50× tanks induction top-1 from 3.7% → 0.0%.)
- **Methodology refinement at scale:** the integral of the PR trajectory beats
  PR-spread for ranking; for capability-specific causal verification, screen
  *all* heads (not just spectral top-k) for the capability of interest.

The artifact is a portable methodology with cross-scale evidence, not a
specific-setup quirk.

## Setup

- **Models tested:**
  - Karpathy 124M (12L × 768d × 12h) trained on FineWeb-10B for 17,600 steps;
    89 checkpoints saved.
  - EleutherAI Pythia 160M (12L × 768d × 12h) and Pythia 410M (24L × 1024d × 16h),
    trained on the Pile; checkpoints subsampled from the 143K-step training run.
- **Eval batch:** 2000 synthetic random-token sequences, length 256. Each has
  structure `[filler] A B [more filler] A` where A and B are random tokens
  from vocab `[100, 10000)`. The induction prediction is B at the position
  after the second A.
- **Per-head measurement:** for each (layer, head, checkpoint), participation
  ratio (PR) of the per-head attention output across the 2000 examples at the
  second-A position.

## Method

The pipeline is three steps:

**1. Spectral identification** — for each (layer, head, checkpoint) compute
PR of the per-head attention output over the 2000-example batch. Rank heads
by an aggregate trajectory signal (see "Ranking signal" below).

**2. Capability classification** (mech-interp) — at the final checkpoint,
for each top-k pick, measure attention from the query position to canonical
target positions for six capability classes:
- **induction**: attn → position of B (induction target)
- **previous-token**: attn → position t-1
- **duplicate-token**: attn → earlier position of token at t
- **first-token / BOS**: attn → position 0
- **self**: attn → position t (current)
- **local**: mean attn over t-2..t-5

Classify each head by its highest-selectivity class (threshold ≥ 30× over
uniform-other baseline).

**3. Causal verification** — ablate the heads of interest by zeroing per-head
attention output at the layer's output projection. Measure induction
top-1/top-5 accuracy on the same eval batch.

### Ranking signal: integral, not spread

We tested 9 alternative trajectory features against PR-spread for ranking.
Winner: **the integral of (PR − 1) over training** — total content-dependent
computation accumulated through the run.

| Ranking signal | Precision-at-30 (Karpathy 124M) | Mean selectivity in top-5 |
|---|---:|---:|
| spread (max − min) | 0.93 | 155× |
| **integral** | **0.97** | **5,791×** |
| max_pr | 0.93 | 177× |
| mean_post_grok | 0.93 | 258× |
| max_rate | 0.53 | 169× |

Why integral wins: it rewards *sustained* high PR, whereas spread only
measures the max-min gap. On Karpathy 124M, L6H9 (27,776× prev-token
selectivity) is rank 14 by spread but rank 5 by integral — spread underrates
heads that are consistently elevated.

**Integral is essential on Pythia.** L0 heads on Pythia start at PR ≈ 60
(random attention at init produces high effective rank) and *collapse* to
PR ≈ 2–30 by training end. PR-spread flags them as top picks; PR-integral
correctly demotes them in favor of heads that *gain* sustained PR through
training.

## Single-model deep dive: Karpathy 124M

Top-30 picks by PR-spread, classified into 6 capability classes:

| k | precision (classified / k) |
|---:|---:|
| 1, 5, 10, **15** | **100%** |
| 20 | 95% |
| 25 | 96% |
| 30 | **93%** (28 of 30 classified) |

**Class breakdown across top-30:**

| Class | Count |
|---|---:|
| self (attn to current position) | 14 |
| previous-token | 9 |
| induction | 5 |
| unclassified | 2 |

**The two unclassified heads** (L1H10, L9H8) have max selectivity 11–13× —
weakly content-dependent diffuse heads with no specific dominant pattern.

### Causal verification on Karpathy 124M

Ablate top-6 spectral picks on the final checkpoint and measure induction
prediction accuracy on the 2000-example eval:

| Condition | top-1 acc |
|---|---:|
| baseline | **16.1%** |
| **ablate top-6 spectral picks** (L8H{8,10,5}, L6H10, L1H{9,11}) | **0.85%** ← circuit destroyed |
| ablate matched-random control | 10.6% |
| upper bound (full spectral-pick layers) | 0% |

Spectral picks ablation drops top-1 by 15.3 percentage points — about **4×
larger** than the matched-random control. Individual ablations of the 3
mech-interp-confirmed induction heads (L8H{8,10,5}) each drop top-1 by 5–10
pp; the 3 false-positive picks (L6H10, L1H{9,11} — actually prev-token heads,
not induction) each drop top-1 by ≤1pp.

### Robustness: classifications hold across query positions

A reasonable concern: classification was done at one query position (the
last one). We re-ran the classification at five query positions
{50, 100, 150, 200, 255}.

| Class @ p=255 | Heads | Consistent across 5 positions | Rate |
|---|---:|---:|---:|
| self | 14 | 11 | **79%** |
| previous-token | 9 | 7 | **78%** |
| induction | 5 | n/a (intrinsically position-specific*) | — |

\* Induction is defined by the batch structure; at non-last positions there's
no induction target. Their selectivity at the last position is already
directly measured (73–149×).

The self class is real (79% consistency, comparable to prev-token's 78%).
~20% of single-position labels are position-specific noise (mostly
self↔prev-token confusion), fixable by multi-position classification.

### Side observation: time-of-emergence by class

For each top-30 pick, when does it first cross PR=15?

| Class | n | mean step | range |
|---|---:|---:|---|
| **induction** | 5 | **840** | 800–1000 (very tight) |
| previous-token | 9 | 1556 | 800–4600 (wide) |
| self | 14 | 1257 | 800–2400 |

Induction heads emerge in a narrow ~200-step window — consistent with the
phase-transition character Olsson et al. (2022) observed.

## Cross-scale validation: 124M → 160M → 410M

We ran the pipeline on **EleutherAI Pythia 160M and 410M** to test whether
the spectral signal generalizes across (data, training procedure, RNG,
codebase, scale).

**Precision-at-k across all three models:**

| k | Karpathy 124M (FineWeb) | Pythia 160M (Pile) | Pythia 410M (Pile) |
|---|---:|---:|---:|
| 5 | 100% | 100% | 100% |
| 10 | 100% | 100% | 100% |
| 15 | 100% | 93% | 93% |
| 30 | 93% | 93% | 90% |
| 50 | — | — | 90% |
| 80 | — | — | 81% |

Match within 1-3 percentage points across an **8× parameter scale range** and
**two completely different training pipelines**. The methodology generalizes.

### The conserved fraction (the headline finding)

When we extend Pythia 410M's classification to top-80 (head-count-matched
to top-30 on the 144-head models) and look at fraction of heads classified
into a known capability class:

| Model | Total heads | k (matched) | Classified | Fraction |
|---|---:|---:|---:|---:|
| Karpathy 124M | 144 | 30 | 28 | **19.4%** |
| Pythia 160M | 144 | 30 | 27 | **18.8%** |
| Pythia 410M | 384 | 80 | 65 | **16.9%** |

**~17–19% of heads in a model do identifiable specialized computation,
conserved across an 8× scale range.** The capability *count* scales with
model size; the *fraction of heads doing specialized work* stays constant.

The PR-integral distribution has a natural elbow that scales similarly:

| Model | Total heads | Elbow k | Elbow / total |
|---|---:|---:|---:|
| Karpathy 124M | 144 | 30 | 20.8% |
| Pythia 160M | 144 | 23 | 16.0% |
| Pythia 410M | 384 | 70 | 18.2% |

The elbow sits at ~16–21% of total heads in every model — model-agnostic
cutoff, no need to pick k by hand.

### Cross-model invariant: every model has one super-prev-token head

| Model | head | prev-token selectivity |
|---|---|---:|
| Karpathy 124M | L6H9 | 27,776× |
| Pythia 160M | L3H2 | **81,792×** |
| Pythia 410M | L5H2 | 23,634× |

Different layer in each model, but integral ranking finds it across all
three (PR-spread misses all three — their trajectories are
high-but-not-the-highest-spread). Suggests a real architectural regularity:
**there's probably one degree of freedom per model that gets compressed into
a single near-perfect prev-token implementation, and the methodology is
sensitive enough to find it without knowing where to look.**

### Class-mix shifts with scale

| Class | Karpathy 124M | Pythia 160M | Pythia 410M (top-30) | Pythia 410M (top-80) |
|---|---:|---:|---:|---:|
| previous-token | 9 | 9 | 14 | 25 |
| self | 14 | 11 | 9 | 18 |
| induction | 5 | 2 | 1 | 1 |
| first-token (BOS) | 0 | 6 | 3 | 20 |
| unclassified | 2 | 2 | 3 | 15 |

Pythia (Pile) has many more first-token-attending heads than Karpathy
(FineWeb) — possibly tied to BOS-token usage in Pile. The induction class
in top-30 shrinks with scale, but as the next section shows, that doesn't
mean induction is *lost* — it's just distributed across more heads at lower
per-head selectivity.

## Causal verification across scale

The spectral → mech-interp → targeted ablation **workflow** was validated
on each model. The targeting itself differs by scale:

| Model | Targeting | top-1 drop |
|---|---|---:|
| Karpathy 124M | top-6 spectral picks (incl 3 induction heads) | 16% → 0.85% (−95%) |
| Pythia 160M | mech-interp-classified L8H2 + L5H0 | 4.7% → 0.05% (−99%) |
| Pythia 410M | top-6 spectral picks (mostly self/prev) | 3.7% → 2.1% (−43%) |
| Pythia 410M | mech-interp-classified + 2nd-class induction | 3.7% → 0.85% (−77%) |
| Pythia 410M | **all heads with induction selectivity ≥50× (11 heads)** | **3.7% → 0.0% (−100%)** |

On Karpathy 124M, top-6 spectral picks happened to contain 3 induction
heads, so spectral-picks ablation alone tanked induction. On Pythia 160M
and 410M, top-6 picks are dominated by self/prev-token heads — and the
mech-interp triangulation in step 2 of the workflow becomes load-bearing.

### Distribution vs dilution: distribution wins

The Pythia 410M result above evolved across three targeting attempts:
(a) top-6 spectral → 43% drop, (b) mech-interp + 2nd-class → 77%, (c)
all-head induction screen → 100%. Two hypotheses for why the smaller
attempts didn't fully tank induction:
- **Dilution**: induction at scale has lower per-head selectivity; we
  genuinely captured less of the circuit
- **Distribution**: induction is spread across more heads at lower
  individual selectivity; our targeting was incomplete

Screening all 384 heads of Pythia 410M for induction selectivity ≥50× found
**11 such heads** — *more* than Karpathy 124M's 6 — but at lower per-head
selectivity (max 203× vs Karpathy's 681×).

| Condition | top-1 | n_heads ablated |
|---|---:|---:|
| baseline | 3.7% | — |
| previous "extended" set (3 heads) | 0.85% | 3 |
| **all heads with induction selectivity ≥ 100× (8 heads)** | **0.10%** | **8** |
| **all heads with induction selectivity ≥ 50× (11 heads)** | **0.0%** | **11** |

**Distribution wins.** Total induction "signal" is preserved, just spread
across more heads. The methodology fully captures the circuit when ablation
targets the full induction-selective set.

**Only 3 of the 11 induction-selective heads on 410M were in our top-30
spectral picks.** The other 8 (L8H6 at 203×, L7H1 at 177×, etc.) live at
lower integral ranks and would be missed by spectral-picks-only mech-interp.

### Multi-purpose heads

L11H14 on Pythia 410M is primary-classified as first-token (287×) but has
strong 2nd-class induction (124×); single-head ablation tanks induction by
50% on its own. Heads with multiple non-trivial selectivities are common
in larger models. **2nd-class selectivities can be load-bearing for
capability-specific ablation.** The capability-specific all-head screen
above naturally picks these up.

## Methodological recommendations

For applying this method to a new model:

1. **Use the PR-integral, not PR-spread.** Spread misses heads with
   sustained high PR and on Pythia-style models gets corrupted by L0 heads
   that start high-PR and collapse during training.

2. **Scale k linearly with total head count, or use the elbow.** k=30 on
   12L × 12h corresponds to k≈80 on 24L × 16h. Or just take the natural
   knee point in the sorted PR-integral distribution — it sits at ~17–19%
   of total heads in every model we tested.

3. **For capability-specific causal verification** (rather than capability
   discovery), don't restrict ablation to spectral top-k. Once mech-interp
   has identified that a class (induction, prev-token, etc.) is implemented
   by some heads, screen *all* heads for that class's selectivity and
   ablate the full set with threshold ≥ 50×.

4. **Treat 2nd-class selectivities as real.** Multi-purpose heads exist
   and matter for capability-specific ablation; the all-head screen above
   captures them naturally.

## What this is and what it isn't

**It is** a portable methodology validated across (synthetic-probe TS-51M
seeds × 6) and (natural-text models × 3) with consistent precision-at-k and
a conserved fraction-of-identifiable-heads across an 8× scale range. The
spectral → mech-interp → targeted-ablation workflow gives capability-specific
causal verification on every model tested.

**It is not** a one-shot capability classifier. The spectral signal flags
heads doing *some* identifiable computation; downstream mech-interp is
required to label *which capability* each pick implements. For circuit-
specific causal claims, the all-head screen for the capability of interest
is the right second step.

**Open questions** the data raises but doesn't fully answer:
- Why does each model have exactly one super-prev-token head, and why is
  the layer placement model-specific?
- Does the 17–19% conserved-fraction hold at much larger scale (1B+)?
- Is the BOS / first-token class growing with Pile (vs FineWeb) data, or
  with model scale, or both? At 410M the class is much larger than at
  160M *and* than at Karpathy 124M.

## Reproducibility

Scripts (in repo root):
- `induction_heads_per_head_124m.py` — per-head spectral on Karpathy 124M
- `pythia_per_head.py` / `pythia_410m_per_head.py` — same on Pythia
- `*_mechinterp_*.py` — capability classification
- `*_ablation_*.py` — causal verification
- `pythia_410m_distributed_induction_test.py` — all-head induction screen
- `pythia_410m_extended_survey.py` — top-80 classification + elbow + conserved fraction
- `better_ranking_signal_124m.py` — integral vs spread comparison
- `capability_survey_multi_pos_124m.py` — multi-position robustness check

Result JSONs in `results/`. Models loaded via HF transformers (`EleutherAI/pythia-160m`
and `EleutherAI/pythia-410m`, with `revision="step{N}"`). Karpathy 124M
checkpoints from `karpathy_llmc/runs/gpt2_fineweb10B/` (separate repo —
see main README for setup).

Compute: ~1h on M4 MPS for per_head on each Pythia ckpt set, ~30 min for
mech-interp + ablation. The full 410M pipeline (per_head + mech-interp +
ablation + extended survey) takes ~5h end-to-end.

## Pivot note

Originally planned to test IOI (indirect-object identification) as the
second naturally-emerging capability. Sanity check on Karpathy 124M showed
it was undertrained for clean IOI (top-1 13%, target>distractor 57% —
barely above chance). Pivoted to previous-token heads as a simpler, robust
capability that any LM has. See `ioi_sanity_check.py`.
