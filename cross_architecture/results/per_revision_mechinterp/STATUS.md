# Per-revision mech-interp sweep — 1B models

Started: 2026-05-12 (Yongzhong machine, MPS)

## Goal

For each of the three 1B models, run the mech-interp capability-classification step
at *each* of the 10 trajectory revisions (same as Phase 1 PR-trajectory revisions),
not just the final checkpoint. This lets us track:
- emergence time of BOS-class heads (do they appear early or late in training?)
- emergence time of induction-selective heads (≥30× and ≥50× thresholds)
- whether the universal L0/L1 zero-BOS pattern holds at all checkpoints or develops over training

Open question explicitly flagged in cross_architecture/README.md:
> "When during training do the BOS heads emerge? ... A per-revision mech-interp
>  re-classification would resolve emergence timing more directly. **Open
>  (would be the next phase).**"

## Models × revisions

**Pythia 1B (10 revisions, fp32 — early ckpt baseline-attention underflow on fp16 has hit smaller Pythias before, fp32 for safety):**
step1, step4, step16, step64, step256, step512, step3000, step10000, step38000, step143000

**OLMo 1B (10 revisions, fp16 — earliest is step1000-tokens2B, well past random init):**
step1000-tokens2B, step2000-tokens4B, step5000-tokens10B, step11000-tokens23B,
step25000-tokens52B, step56000-tokens117B, step126000-tokens264B,
step285000-tokens597B, step644000-tokens1350B, step1454000-tokens3048B

**OLMoE 1B-7B (10 revisions, fp16 — earliest is step5000-tokens20B):**
step5000-tokens20B, step10000-tokens41B, step25000-tokens104B, step50000-tokens209B,
step100000-tokens418B, step200000-tokens836B, step400000-tokens1672B,
step600000-tokens2508B, step1000000-tokens4181B, step1220000-tokens5117B

## Eval setup (matches final-checkpoint runs exactly)

- Synthetic induction batch (RNG seed 42): 2000 sequences, seq_len=256, vocab [100, 10000),
  structure `[filler] A B [filler] A` with target B at second-A position
- Per-(L,H) attention from query position to canonical targets for 6 classes:
  induction, previous-token, duplicate-token, first-token (BOS), self, local
- Selectivity = (target attn) / (uniform-other baseline)
- Classify each head by best-class at threshold ≥ 30×
- Save full all-head selectivity matrix (not just top-K) for emergence-curve plotting

## Output

`per_revision_mechinterp/{pythia1b,olmo,olmoe}_mechinterp_{revision}.json` — 30 files total.

## Driver

`per_revision_mechinterp_driver.sh` — sequential loop, one GPU job at a time.
Resume-safe: skips outputs that already exist.

## What to analyze when done

1. **BOS-class fraction over time** per model (line plot, x=tokens, y=% heads classified BOS).
   Does it grow monotonically? Is there a knee? Does it differ between Pile / DCLM-dense / DCLM-MoE?
2. **Induction-selective head count over time** per model (≥30× and ≥50× thresholds).
   When does the 3-4 head circuit crystallize?
3. **L0/L1 zero-BOS over time** — universal floor at final ckpt, but is it universal at all revisions?
4. **Per-head selectivity trajectory** for the 3-4 final-checkpoint induction heads — do they emerge sharply or smoothly?
