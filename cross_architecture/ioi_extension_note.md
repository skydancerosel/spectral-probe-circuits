# IOI across three 1B model families: same task, three mechanistic implementations

Indirect Object Identification (IOI) — "When Mary and John went to the store, John gave a drink to ___" → " Mary" — is a composed task that requires resolving which of two introduced names is the *indirect* object (mentioned first, not the subject of the giving clause). Classical mechanistic-interpretability work on GPT-2-small (Wang et al. 2022) identified a circuit with name-mover heads, S-Inhibition heads, prev-token heads, induction heads, and duplicate-token heads.

This note applies the spectral-probe-circuit methodology to IOI across **Pythia 1B (Pile, dense)**, **OLMo 1B-0724-hf (DCLM, dense)**, and **OLMoE 1B-7B-0924 (DCLM, MoE)**. The result: all three models solve IOI at baseline, but route the task through three different attention patterns.

## Setup

- **Batch**: 500 prompts, mixing ABBA template (target appears first) and BABA template (target appears second), 50/50. Cross-tokenizer-compatible: all names, places, objects are single tokens in all three tokenizers (Pythia GPT-NeoX, OLMo, OLMoE).
- **Names**: 42 single-token names (John, Mary, Alice, Bob, Anna, Tom, …).
- **Places**: store, park, school, office, house, bar.
- **Objects**: drink, book, ball, ring, phone, gift.
- **Evaluation**: at the final position ("to"), top-1 accuracy on the IO name, fraction where logit(IO) > logit(subject), mean logit_diff = logit(IO) − logit(subject).

## Baseline IOI capability

| Model | top-1 | frac(IO > subject) | logit_diff |
|---|---:|---:|---:|
| Pythia 1B | 90.4% | 99.8% | +4.00 |
| OLMo 1B | 87.8% | 100.0% | +3.64 |
| OLMoE 1B-7B | 58.6% | 99.6% | +3.95 |

All three 1B models solve IOI in the sense that the IO logit exceeds the subject logit on ~100% of prompts. Pythia and OLMo also produce the IO name as the top-1 prediction on ~90% of prompts. OLMoE's frac(IO > subject) is also 99.6%, but on ~41% of prompts a non-name token (function word or punctuation) outranks both names — its IO/subject preference is just as strong as the dense models', but its absolute top-1 accuracy is lower because the names aren't always the argmax.

GPT-2 124M (Karpathy / FineWeb-10B) does not solve IOI at this scale — top-1 13%, target > distractor 57% — so this is the first capability test in our panel where the small models can't be used as a sanity check.

## Capability-screen ablation (prev-token circuit, induction circuit)

Using the all-head capability screens defined in [`../methodology_paper.md`](../methodology_paper.md):

| ablation target | Pythia 1B | OLMo 1B | OLMoE 1B-7B |
|---|---:|---:|---:|
| baseline IOI top-1 | 90.4% | 87.8% | 58.6% |
| ablate prev-token circuit (best-class, sel ≥100×) | **8.4% (Δ−82)** | 94.0% (Δ+6) | 85.0% (Δ+26) |
| ablate induction circuit (sel ≥50×) | 89.8% (Δ−0.6) | 84.6% (Δ−3) | 57.6% (Δ−1) |
| ablate prev-token + induction (union) | 17.6% (Δ−73) | 92.8% (Δ+5) | 86.6% (Δ+28) |

The prev-token-circuit screen — the same screen with the same selectivity definition and threshold — produces opposite causal effects across the three models. In Pythia 1B it destroys IOI (90% → 8.4%). In OLMo and OLMoE, ablating the same screen *improves* IOI by 6 and 26 percentage points. Induction-circuit ablation has near-zero effect across the panel (Δ between −3 and −1pp).

## IOI-specific screen: name-mover candidates

The IOI task structure suggests an obvious task-specific screen: heads that, at the final query position, attend strongly to the IO-name position relative to the subject-name positions. This is the attention signature of a *name-mover* head — copying the IO name into the residual stream.

Selectivity: `nm_sel = mean_attn(query → IO_pos) / max(mean_attn(query → subj_first), mean_attn(query → subj_second))`, averaged across the 500 prompts.

Top-5 name-mover candidates per model (all are in late layers, L9–L14):

| Model | Top-5 (L_H, nm_sel) |
|---|---|
| Pythia 1B | L13_H1 (19.93), L10_H6 (9.09), L10_H5 (7.25), L10_H4 (5.66), L11_H7 (4.66) |
| OLMo 1B | L12_H9 (4.25), L12_H8 (3.75), L12_H4 (2.78), L14_H1 (2.33), L13_H10 (2.25) |
| OLMoE 1B-7B | L12_H14 (7.66), L12_H3 (4.88), L12_H10 (2.33), L11_H8 (2.20), L9_H1 (2.17) |

Pythia has one dominant name-mover (L13_H1, nm_sel ≈ 20 — much sharper than any head in OLMo or OLMoE, where top nm_sel is 4–8 and the signal is distributed across several heads). OLMo's L12_H8 and OLMoE's L12_H14 are also in the *induction* circuit — multi-role late-layer heads.

### Causal ablation

| ablation target | Pythia 1B | OLMo 1B | OLMoE 1B-7B |
|---|---:|---:|---:|
| top-5 name-movers | 97.6% (Δ+7) | 70.6% (Δ−17) | **40.8% (Δ−18)** |
| matched-random same layers (5h) | 86.8% (Δ−4) | 76.4% (Δ−11) | 58.0% (Δ−0.6) |
| full late-layer ablation (upper bound) | 98.0% (Δ+8) | 8.4% (Δ−79) | 9.0% (Δ−50) |

- **Pythia 1B**: ablating name-movers *helps* IOI (Δ+7), as does the full late-layer upper bound (Δ+8). These heads are correlates of IOI, not causes — Pythia's IOI is built up in mid-layers from prev-token computation, and by L9+ the answer is already in the residual stream.
- **OLMoE 1B-7B**: name-mover screen drops top-1 by 18pp; matched-random in the same layers drops it by 0.6pp. **Differential is 30×** — the screen cleanly identifies the OLMoE IOI circuit.
- **OLMo 1B**: name-mover screen drops top-1 by 17pp; matched-random drops it by 11pp. Differential is 1.5×. OLMo's circuit is captured *partially* by the name-mover screen but is more distributed across late-layer heads.

## IOI-specific screen: S-Inhibition candidates

A second IOI-relevant attention pattern: heads at the final query position that attend strongly to the **subject** positions (the duplicated name's first or second occurrence). The IOI circuit on GPT-2-small (Wang et al. 2022) uses S-Inhibition heads to read the subject and write a suppressive signal that reduces its output logit.

Selectivity: `subj_sel = max(subj_first, subj_second) / max(io_attn, ε)`. Filter heads with `subj_sel ≥ 2` (subject-biased, not mixed-role) and `subj_max ≥ 0.1` (substantial absolute attention), then rank by `subj_max`.

Top-5 candidates per model:

| Model | Top-5 (L_H, subj_max, subj/io) |
|---|---|
| Pythia 1B | L12_H1 (0.50, 3.7×), L9_H1 (0.39, 5.1×), L8_H7 (0.32, 13.0×), L12_H4 (0.26, 6.2×), L9_H6 (0.20, 6.4×) |
| OLMo 1B | L13_H3 (0.47, 6.5×), L11_H5 (0.29, 9.8×), L11_H8 (0.24, 2.4×), L12_H1 (0.19, 5.0×), L10_H15 (0.17, 2.5×) |
| OLMoE 1B-7B | L13_H2 (0.55, 8.1×), L8_H2 (0.45, 19.0×), L13_H5 (0.43, 6.0×), L13_H1 (0.32, 4.7×), L11_H13 (0.26, 3.5×) |

### Causal ablation

| ablation target | Pythia 1B | OLMo 1B | OLMoE 1B-7B |
|---|---:|---:|---:|
| top-5 S-Inhibition | 57.8% (Δ−33) | **56.0% (Δ−32)** | 59.2% (Δ+0.6) |
| matched-random same layers (5h) | 84.2% (Δ−6) | 92.6% (Δ+5) | 57.0% (Δ−1.6) |
| name-mover + S-Inhibition union (10h) | 37.6% (Δ−53), logit_diff **−0.47** | 54.0% (Δ−34) | 42.2% (Δ−16) |

- **OLMo 1B**: S-Inhibition is the primary IOI screen for OLMo. Top-5 ablation drops top-1 by 32pp; matched-random in the same layers actually *helps* by 5pp. The differential is essentially infinite. This is the screen the prev-token and name-mover screens both missed.
- **Pythia 1B**: S-Inhibition is a real secondary mechanism. Top-5 ablation drops top-1 by 33pp (matched-random −6pp; differential 5×). The name-mover + S-Inhibition union flips logit_diff to −0.47 — the model now prefers the subject. Pythia's IOI is prev-token-primary (Δ−82), but late-layer S-Inhibition heads carry enough signal that combining their ablation with the name-movers reverses the prediction.
- **OLMoE 1B-7B**: S-Inhibition has no top-1 effect (Δ+0.6), but logit_diff drops from +3.95 to +0.95 — strong margin effect, no argmax effect. OLMoE's top-1 is name-mover-driven; S-Inhibition only modulates the margin.

## Three mechanistic implementations of the same task

| Model | Primary | Secondary | Wrong screen |
|---|---|---|---|
| Pythia 1B (Pile, dense) | Prev-token-circuit (Δ−82) | S-Inhibition (Δ−33, 5× differential) | Name-mover (correlate, Δ+7) |
| OLMo 1B (DCLM, dense) | S-Inhibition (Δ−32; matched-random *helps* +5) | Name-mover (Δ−17, 1.5× differential) | Prev-token (Δ+6) |
| OLMoE 1B-7B (DCLM, MoE) | Name-mover (Δ−18, 30× differential) | none for top-1; S-Inhibition shifts logit_diff only | Prev-token (Δ+26) |

Same task, same family of capability-pattern types, four candidate screens (prev-token, induction, name-mover, S-Inhibition). Each model uses a different *primary* screen, and no two models use the same combination.

## Methodological implication

To find the IOI circuit on a given model, you need a *family* of candidate screens — each derived from the task structure (which attention pattern would solve this?) — and each validated by ablation against matched-random. The four screens above cover IOI in all three 1B models, but which one is *primary* is a per-model empirical question.

The recipe ports as a recipe (spectral signal → task-pattern screen → causal verification). The specific screen does not port across model families for composed tasks. Circuit-level mechanistic findings on one model are not safe to assume on another, even at the same scale.

## Files

- `ioi_batch.py` — shared cross-tokenizer IOI batch (500 prompts).
- `ioi_eval.py` — capability-screen ablation (prev-token + induction).
- `ioi/{pythia1b,olmo,olmoe}_ioi.json` — capability-screen ablation results.
- `ioi_mechinterp.py` — per-head IOI-specific name-mover selectivity.
- `ioi/{pythia1b,olmo,olmoe}_ioi_mechinterp.json` — full IOI selectivity matrices.
- `ioi_name_mover_ablation.py` — name-mover-circuit ablation.
- `ioi/{pythia1b,olmo,olmoe}_ioi_nm_ablation.json` — name-mover ablation results.
- `ioi_s_inhibition_ablation.py` — S-Inhibition-circuit ablation (subject-attending heads, with name-mover union condition).
- `ioi/{pythia1b,olmo,olmoe}_ioi_si_ablation.json` — S-Inhibition ablation results.

## Related notes

- [`../methodology_paper.md`](../methodology_paper.md) — the full recipe + scope statement.
- [`developmental_note.md`](developmental_note.md) — per-revision capability emergence.
