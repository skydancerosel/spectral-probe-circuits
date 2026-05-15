# FineWeb $\to$ OWT distribution-shift analysis

Karpathy GPT-2 124M was pretrained on FineWeb-10B through step 17600 (89
checkpoints) and then continued on OpenWebText starting step 17800. This
gives a single-model, single-procedure, single-architecture, single-optimizer
test of how attention-head circuits respond to a *pure data distribution
shift* mid-training. Almost no other publicly-available LM training run has
this structure, so it is an opportunity to isolate the data component of
several effects that the cross-model panel can only see confounded.

## What the script measures

`fineweb_to_owt_analysis.py` runs the per-head spectral pipeline plus the
all-head capability-specific screen at every checkpoint in both phases and
produces four files in `<output-dir>`:

| File | Contents |
|------|----------|
| `per_head_pr_trajectory.json` | PR$(L,H,t)$ at every checkpoint, both phases, with a `phase` label per step. |
| `bos_fraction_trajectory.json` | Whole-model BOS-class fraction (heads classified first-token at $\geq 30\times$) at every checkpoint. |
| `mech_interp_endpoints.json` | Full selectivity matrices at the last FineWeb checkpoint and the last OWT checkpoint, plus the $\geq 50\times$ induction-circuit set for each phase and their set intersection / Jaccard / per-phase-only heads. |
| `circuit_selectivity_trajectory.json` | Induction / prev-token / first-token selectivity at every checkpoint for the FineWeb-endpoint induction circuit heads. |

`build_fineweb_to_owt_figure.py` consumes those four files and produces a
four-panel comparison figure (PR continuity, BOS-fraction trajectory,
endpoint-vs-endpoint scatter, FineWeb-heads-through-OWT selectivity).

## Running it

You need both phases of checkpoints. From the methodology paper appendix and
existing scripts, FineWeb checkpoints live in
`karpathy_llmc/runs/gpt2_fineweb10B/`. The OWT continuation lives in a
sibling directory (e.g. `karpathy_llmc/runs/gpt2_owt_continuation/` or the
`karpathy-2` directory referenced in the project notes).

```
# from the repo root, on the machine where the checkpoints are
python analyses/fineweb_to_owt/fineweb_to_owt_analysis.py \
    --fineweb-ckpt-dir /path/to/karpathy/runs/gpt2_fineweb10B \
    --owt-ckpt-dir     /path/to/karpathy-2/runs/gpt2_owt_continuation \
    --output-dir       results/fineweb_to_owt \
    --device           mps     # or cuda / cpu

# Then build the figure:
python analyses/fineweb_to_owt/build_fineweb_to_owt_figure.py \
    --input-dir  results/fineweb_to_owt \
    --output     figures/fineweb_to_owt_panel.pdf
```

Approximate runtime: ~5 sec per checkpoint on M4 MPS for PR + screen (200
checkpoints across both phases $\approx$ 15--20 min total). Multiply by 2 if
the selectivity-trajectory step (M4 in the script) is also enabled.

## Outcomes and what each means

The four measurements give four possible signal patterns:

**(M1) PR continuity at the boundary.**
- *Smooth across the boundary:* the spectral signal is tracking
  structural specialization. The methodology recipe is robust to data
  shift -- a strong claim that lands in Paper 1.
- *Visible discontinuity for some heads, smooth for others:* the
  heads-that-jump are the data-coupled heads (likely BOS-class, possibly
  prev-token); the smooth heads carry structural roles. A useful refinement
  of the methodology.
- *Discontinuity for all heads:* PR is more data-sensitive than the cross-
  scale invariants in Paper 1 currently suggest; a caveat for the recipe.

**(M2) Induction-circuit identity stability (endpoint scatter).**
- *Heads stay near the diagonal:* same identities, similar selectivity.
  Strong stability claim.
- *Heads stay near the diagonal but shift down:* same circuit, slightly
  weaker on OWT. Quantifies a data-specific selectivity component.
- *Heads scatter off the diagonal:* identity instability. Different OWT
  heads end up induction-selective than FineWeb heads. This is a major
  refinement of the "circuit identity is a model property" framing and
  would belong as its own section.

**(M3) BOS-fraction across the boundary.**
- *Same BOS fraction in both phases:* attention-sink formation is
  independent of training data distribution (within these two corpora,
  which are similar in style but not identical). Confirms architecture as
  the dominant driver.
- *Different BOS fraction:* quantifies the data-contribution to the
  $\sim 20\,$pp DCLM-vs-Pile gap reported in the cross-architecture panel.
  Even a $5\,$pp shift over 1B tokens of OWT training is enough to attribute
  a substantial fraction of that 20pp to corpus distribution.

**(M4) Selectivity through OWT for FineWeb-endpoint induction heads.**
- *Selectivities flat through OWT:* once formed, induction heads are
  data-distribution-robust.
- *Selectivities decay through OWT:* the FineWeb-formed circuit is
  partially "rewritten" by OWT training. Tests whether continued
  pretraining on similar-but-different data degrades pre-formed circuits.
- *Selectivities decay, but other heads rise to take their place:*
  consistent with circuit relocation -- the function is preserved, the
  implementation changes.

Each of the four outcomes is publishable; the script gives you the result
either way.

## Outputs to look at first

After the run finishes, the four numbers that matter most:

1. `mech_interp_endpoints.json -> set_comparison.jaccard_50x` -- the
   identity-stability headline. $0.8$+ is "stable"; $<0.3$ is "rewritten";
   in between is "partial overlap."

2. `mech_interp_endpoints.json -> fineweb.induction_circuit_50x`
   vs `owt.induction_circuit_50x` -- the actual head lists, by phase.

3. `bos_fraction_trajectory.json -> bos_fraction` at the last FineWeb step
   vs at the last OWT step -- the BOS-shift headline.

4. `per_head_pr_trajectory.json -> pr[L8_H8]` etc. (or whatever the FineWeb
   endpoint induction heads turned out to be) -- visually inspect the
   trajectory across the boundary.

## How this lands in the trilogy

- *Paper 1 (methodology):* a one-page section "robustness to mid-training
  data shift" with Panel A + Panel C, used to support the "PR tracks
  structural specialization rather than corpus statistics" claim that is
  currently inferred rather than directly tested.

- *Paper 2 (developmental):* a small section augmenting the L0/L1
  zero-BOS finding with a within-model data-shift control. If the
  zero-BOS floor holds through the OWT transition too, the architectural-
  invariant claim hardens.

- *Paper 3 (cross-arch):* the BOS-fraction shift (Panel B) directly
  decomposes the cross-model BOS variance into data and architecture
  components, which the current paper can only co-vary.

A single small experiment ($\sim$15 minutes of MPS compute) touches all
three papers.
