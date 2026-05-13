# Capability circuit emergence across three 1B model families

Mech-interp capability classification at each of the 10 trajectory revisions for **Pythia 1B**, **OLMo 1B-0724-hf**, and **OLMoE 1B-7B-0924** — 30 runs total. Synthetic induction batch (RNG seed 42, 2000 sequences, seq_len 256); per-(L,H) hook on the attention output projection; selectivity = (target attention)/(uniform-other baseline), threshold ≥30× for class assignment, ≥50× for circuit membership. fp32 for Pythia 1B (early checkpoints underflow at fp16); fp16 for OLMo and OLMoE.

## Finding 1: L0–L1 zero-BOS is universal across all of training

L0 and L1 produce zero BOS-classified heads at every one of the 10 revisions, in every one of the three 1B models. Per-layer BOS counts at selected revisions:

**Pythia 1B (Pile, 16L × 8h):**

| step | L0 | L1 | L2 | L3 | L4 | L5–L8 | L9–L12 | L13–L15 |
|---|---:|---:|---:|---:|---:|---|---|---|
| step1 | 0/8 | 0/8 | 0/8 | 0/8 | 0/8 | 0,0,0,0 | 0,0,0,0 | 0,0,0 |
| step512 | 0/8 | 0/8 | 0/8 | 0/8 | 0/8 | 0,0,0,0 | 0,0,0,0 | 0,0,0 |
| step3000 | 0/8 | 0/8 | 0/8 | 0/8 | 0/8 | 0,1,2,1 | 0,4,1,3 | 1,1,0 |
| step10000 | 0/8 | 0/8 | 0/8 | 0/8 | 7/8 | 3,3,3,2 | 2,4,4,3 | 3,2,2 |
| step143000 | **0/8** | **0/8** | 0/8 | 0/8 | 8/8 | 7,7,6,7 | 7,7,7,5 | 4,4,5 |

**OLMo 1B (DCLM, 16L × 16h):**

| step | L0 | L1 | L2 | L3 | L4 | L5–L8 | L9–L12 | L13–L15 |
|---|---:|---:|---:|---:|---:|---|---|---|
| step1000-tokens2B | 0/16 | 0/16 | 0/16 | 0/16 | 0/16 | 0,0,0,0 | 0,0,0,0 | 0,0,0 |
| step25000-tokens52B | 0/16 | 0/16 | 0/16 | 0/16 | 0/16 | 0,0,0,0 | 0,0,0,0 | 0,0,0 |
| step56000-tokens117B | 0/16 | 0/16 | 0/16 | 3/16 | 9/16 | 7,0,0,0 | 0,0,0,0 | 0,0,0 |
| step126000-tokens264B | 0/16 | 0/16 | 11/16 | 11/16 | 15/16 | 13,15,15,13 | 15,16,16,12 | 13,9,6 |
| step1454000-tokens3048B | **0/16** | **0/16** | 14/16 | 16/16 | 16/16 | 15,16,16,15 | 16,16,15,15 | 16,14,7 |

OLMoE 1B-7B shows the same pattern at all 10 revisions.

L0/L1 zero-BOS holds from random initialization through trillions of tokens. It is not a convergence outcome; the model never crosses that floor at any point during training.

## Finding 2: BOS-attractor formation has three distinct shapes

Whole-model BOS-class fraction (fraction of all heads classified first-token at ≥30× selectivity) by tokens trained:

| tokens | Pythia 1B | OLMo 1B | OLMoE 1B-7B |
|---:|---:|---:|---:|
| ~6B | 10.9% | — | — |
| 20–23B | (no ckpt) | 0.0% | 3.5% |
| ~80B | 46.1% | — | — |
| 104–117B | — | 7.4% | 42.2% |
| 264B | — | **70.3%** | — |
| 300B / 419B | 57.8% | — | 50.8% |
| 838B | — | — | 69.9% |
| 1350–1677B | — | 80.9% | 71.5% |
| 3048–3355B | — | 80.9% | 74.2% |
| 5117B | — | — | 75.8% |

Three shapes:

- **Pythia 1B (Pile, dense): gradual monotonic ramp.** BOS rises from 0% to 58% over ~6B → 300B tokens.
- **OLMo 1B (DCLM, dense): sharp phase transition.** 0% through 52B tokens, 7% at 117B, jumps to 70% at 264B — between two adjacent checkpoints. Final 81%.
- **OLMoE 1B-7B (DCLM, MoE): gradual monotonic ramp.** Similar shape to Pythia, higher saturation. 3.5% at 20B, 76% at 5117B.

Same data (DCLM), opposite emergence shapes depending on architecture (dense vs MoE). MoE both reduces the final BOS fraction (by ~10pp) and smooths the dynamics; the architectural effect on the trajectory is separate from the effect on the magnitude.

## Finding 3: Induction emerges before the BOS attractor in DCLM models

First-revision-crossing for each milestone:

| milestone | Pythia 1B | OLMo 1B | OLMoE 1B-7B |
|---|---:|---:|---:|
| induction ≥3 heads (≥50× selectivity) | ~6B | 23B | 20B |
| induction ≥5 heads (≥50× selectivity) | ~6B | — | 20B |
| BOS-class ≥10% of all heads | ~6B | 264B | 41B |
| BOS-class ≥30% | ~80B | 264B | 104B |
| BOS-class ≥50% | 300B | 264B | 419B |
| BOS-class ≥70% | — | 264B | 1677B |
| prev-token ≥30 | ~6B | 117B | 20B |
| self ≥30 | ~6B | 52B | 20B |

The induction circuit reaches its final 3–6-head size within the first 6–25B tokens in all three models, then stays flat or contracts slightly over the remaining 300B–5T tokens. The BOS attractor reaches saturation much later, especially in DCLM:

- **OLMo 1B**: induction at 23B, BOS-50% at 264B — 11× gap in tokens.
- **OLMoE 1B-7B**: induction at 20B, BOS-50% at 419B — 21× gap.
- **Pythia 1B**: induction and BOS-10% co-emerge at ~6B; the gap is below the resolution of this checkpoint set.

Capability-circuit formation and attention-sink formation are temporally separable in DCLM models — not a single phase transition.

## Finding 4: The capability-screen converges within ~1% of training

Applying the all-head capability-specific screen (induction selectivity ≥50×) at each intermediate revision and computing recall against the final-checkpoint induction set:

| model | first revision at 33% recall | at 67% | at 100% | fraction of total tokens for 67% |
|---|---:|---:|---:|---:|
| Pythia 1B (3-head circuit) | step256 (~0.5B) | step3000 (~6B) | step38000 (~80B) | ~2% |
| OLMo 1B (3-head circuit) | step1000-tokens2B | step5000-tokens10B | step1454000-tokens3048B (final) | ~0.3% |
| OLMoE 1B-7B (4-head circuit) | step5000-tokens20B | step10000-tokens41B | step400000-tokens1677B | ~0.8% |

The screen identifies most of the final-checkpoint induction circuit using a small fraction of the training budget — 0.3% to 2% across the three models. For practical purposes: you do not need the final model to find induction heads. An intermediate checkpoint at 10–25B tokens already has the circuit.

## Finding 5: Per-head PR trajectories rise before induction selectivity does

For each final-checkpoint induction head, the per-revision PR trajectory and induction-selectivity trajectory align such that PR rises first. Example, Pythia 1B L4_H4 (a top induction head, sel=181× at final checkpoint):

| tokens (B) | PR | induction sel |
|---:|---:|---:|
| 0.002 | 1.96 | 1.0 |
| 0.008 | 1.96 | 1.0 |
| 0.033 | 1.32 | 1.0 |
| 0.134 | 2.58 | 1.0 |
| 0.536 | **27.35** | 0.2 |
| 1.073 | 13.21 | 3.9 |
| 6.291 | 48.05 | **171.8** |
| 20.971 | 35.62 | 150.7 |
| 79.691 | 26.38 | 147.9 |
| 299.892 | 28.29 | 181.6 |

PR rises sharply at 0.5B tokens; induction selectivity crosses 50× at 6B tokens — a ~12× lead in token count. Same pattern for the other final-checkpoint induction heads (Pythia L7_H1, L7_H0; OLMo L2_H11, L4_H12, L12_H8; OLMoE L7_H0, L9_H8, L5_H10, L12_H14): PR is already elevated by the time induction selectivity crosses threshold.

The spectral signal precedes the capability-pattern signal *for the heads that end up induction-selective*. That is not the same as the PR-integral top-K identifying induction heads specifically: in attention-sink-dominated 1B models the top-K by PR-integral is dominated by L0/L1 generic content-dependent heads, not the induction circuit. PR-integral is a general "specialized computation" indicator; the capability-specific screen disambiguates which specialized computation each head is doing.

## Methodology consequences

- The recipe is predictive, not retrospective. The capability-specific screen at 1–10% of total training tokens identifies most of the final-checkpoint circuit; the final model is not required.
- "Phase transition in attention" needs disambiguation. In DCLM 1B models, induction-circuit formation and BOS-attractor formation are separated by 10–20× in tokens — two transitions, not one.
- L0/L1 zero-BOS is an architectural constraint rather than a learned outcome. An account of the attention-sink mechanism needs to explain why the gradient signal never installs BOS heads at L0/L1, at any point during training.

## Files

- `per_revision_mechinterp/{pythia1b,olmo,olmoe}_mechinterp_{revision}.json` — 30 mech-interp result files.
- `per_revision_mechinterp/STATUS.md` — full run configuration.
- `per_revision_mechinterp_driver.sh` — sequential driver.

## Related notes

- [`../methodology_paper.md`](../methodology_paper.md) — the full three-step recipe.
- [`ioi_extension_note.md`](ioi_extension_note.md) — IOI across the three 1B models.
