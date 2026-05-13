# Probe-circuits across architectures: what stays the same, what differs, and what the "Pythia 5% BOS" number really means

A short summary of the May 2026 cross-architecture extension. Full methodology and results: [`README.md`](README.md).

## The setup

We took the spectral → mech-interp → causal pipeline that worked across TS-51M (six seeds) and three natural-text dense transformers (GPT-2 124M, Pythia 160M, Pythia 410M), and asked: **does it port to (a) 1B-class scale, (b) a different architecture (MoE), and (c) a different training distribution (DCLM)?**

Five new test models, all run with the *same synthetic induction batch (RNG seed 42)* the writeup uses and the *same all-head capability-specific screen at induction-selectivity ≥ 50×*:

| Model | Architecture | Pretraining data | Vendor |
|-------|-------------|------------------|--------|
| Pythia 160M | dense | The Pile | EleutherAI |
| Pythia 410M | dense | The Pile | EleutherAI |
| **Pythia 1B** | dense | The Pile | EleutherAI |
| **OLMo-1B-0724-hf** | dense | DCLM-aligned | AllenAI |
| **OLMoE-1B-7B-0924** | MoE (64 experts, top-8) | DCLM | AllenAI |

## What ports cleanly (the headline)

**A small 3–4 head capability-specific screen, at the same induction-selectivity threshold, identifies an induction circuit in every model tested.**

| Model | Induction heads ≥ 50× | Synthetic top-1 baseline → ablated |
|-------|----------------------:|------------------------------------|
| Pythia 1B | 3 (L4H4, L7H0, L7H1) | 4.05% → 0.25% |
| OLMoE 1B-7B | 4 (L5H10, L7H0, L9H8, L12H14) | 4.80% → 0.00% |
| OLMo 1B | 3 (L2H11, L4H12, L12H8) | 1.00% → 0.05% |

On natural-text contexts the same heads produce a 5–7× differential effect on the target-token logit over matched-random ablations in the same layers. The capability circuit is real, replicates the writeup's 410M finding, and survives the MoE / DCLM perturbations.

## A statistic that makes cross-arch comparison clean

`INDUCTION_HEADS.md` reports things like "Pythia 410M: 20 of 384 = 5.2% first-token." That number is a precision-at-K measurement: of the heads picked by integral-top-K, what fraction classified as first-token? It's the right thing to look at for "did the spectral signal pick up a first-token head?" — but it's not the same as "what fraction of *all heads* in the model classify as first-token at the same selectivity threshold?"

For comparing across architectures, the whole-model fraction is the cleaner statistic. Re-running mech-interp with `all_head_selectivity` saved (fp32, to avoid baseline attention underflow on the smaller Pythia checkpoints) gives:

| Model | BOS % synthetic | BOS % natural-text |
|-------|----------------:|-------------------:|
| Pythia 160M | 43.1% | 15.3% |
| Pythia 410M | 58.1% | 69.0% |
| Pythia 1B | 53.9% | 68.0% |
| OLMoE 1B-7B | 68.0% | 73.4% |
| OLMo 1B | 78.1% | 84.0% |

On this metric Pythia 410M's natural-text BOS (69.0%) sits very close to OLMoE's (73.4%) — the "MoE is uniquely BOS-dominated vs transformers" reading isn't supported by the whole-model number.

## Three things that move the BOS fraction

1. **Scale grows BOS within the Pythia family.** 43% (160M) → 58% (410M) → saturates near 54% at 1B. Most of the growth happens between 160M and 410M.
2. **DCLM data adds ~20pp** over Pile at the same scale+architecture (OLMo 1B dense DCLM 78% vs Pythia 1B dense Pile 54%).
3. **MoE *reduces* BOS by ~10pp** vs dense at the same scale+data (OLMoE 1B-7B MoE DCLM 68% vs OLMo 1B dense DCLM 78%). MoE doesn't cause attention sinks; if anything it suppresses them.

## What's universal across all 5 models (the surprise)

Per-layer BOS-class fraction, all five models on synthetic:

| Layer | Pythia 160M | Pythia 410M | Pythia 1B | OLMoE | OLMo |
|------:|------------:|------------:|----------:|------:|-----:|
| **L0** | **0%** | **0%** | **0%** | **0%** | **0%** |
| **L1** | **0%** | **0%** | **0%** | **0%** | **0%** |
| L2 | 0% | 0% | 0% | 38% | 88% |
| L3 | 0% | 0% | 0% | 69% | 94% |
| L4 | 83% | 0% | 88% | 75% | 100% |
| L6 | 58% | 88% | 75% | 75% | 100% |
| L7+ | 50–67% | 70–100% | 50–90% | 80–100% | 80–100% |

**L0 and L1 have zero BOS-classified heads across every model.** This is a universal architectural property of decoder-only LMs at 100M+ scale: the first two layers do diverse general-purpose computation (high-PR, content-dependent throughout training), and the BOS attractor only kicks in from L2 onward. **Where it kicks in is data-dependent** — DCLM-trained models transition at L2 (early), Pile-trained models at L4–L6 (later) — but the L0/L1 zero-BOS *floor* is the same in every model we tested.

That floor is the cleanest cross-architecture finding in this extension. It says that whatever an attention sink "is," it isn't something the model can or wants to do at L0–L1.

## What the cross-arch data narrows down

Two readings from the original writeup look different once we have whole-model BOS-class fractions across all five models:

- "Pythia is mostly content-driven; OLMoE is uniquely BOS-dominated" — on the whole-model statistic, Pythia 410M and 1B sit at 54–69% and OLMoE at 68–73%, so the gap is much smaller than the precision-at-K comparison suggested. OLMo 1B dense (78–84%) is actually the most BOS-dominated of the five.
- "Best-class mech-interp classification surfaces capability heads" — when ~70%+ of heads classify as `first-token`, the integral-top-K-by-best-class doesn't reliably surface induction or prev-token heads. The **all-head capability-specific screen** (measure capability-X selectivity in isolation, classify by capability rather than by best-class) was always the writeup's prescription for "distribution wins" cases like Pythia 410M; the cross-arch extension makes it the default rather than the exception.

## Methodology consequences

- **The capability circuit (3–4 heads, induction-selective ≥ 50×) is the load-bearing object, not the integral-top-K.** Integral-top-K is good for the "are there real circuit heads in this model" surface question; for the actual *causal* claim, screen all heads for the capability of interest and ablate that set.
- **For attention-sink-dominated architectures, `first-token` is a class to *exclude* from the rank-ordered analysis, not the dominant capability to track.** The non-BOS conserved-fraction is more like 12% (excluding first-token-class) — slightly below the original writeup's 17–19% transformer baseline, but still real.
- **The natural-text vs synthetic BOS amplification is itself a scale-related signal.** Pythia 160M is the only model where natural text *reduces* BOS (43% → 15%) — content-driven attention overrides the default. From 410M onward, real text *amplifies* BOS by 5–14pp — content-driven attention fails to override the default. There's a behavioral phase transition between 160M and 410M dense Pythia where attention organization flips from "content-driven" to "sink-dominated."

## Reproduce

- All scripts: [`scripts/`](scripts/) (22 files — per-head, mech-interp, ablation for each model; shared induction-batch helpers; tier-A cross-architecture layer-BOS analysis)
- All result JSONs: [`results/`](results/) (31 files spanning Phase 1 PR trajectories, Phase 2 mech-interp synthetic + natural, Phase 3 ablation)
- Full methodology writeup: [`README.md`](README.md)

The 22 scripts copy the existing-repo pattern (extracting per-head attention output, computing PR, identifying capability heads via all-head screen, ablating with matched-random control) and just swap in the new model classes / HF revisions. Nothing exotic in the methodology — the extension is the cross-arch panel itself.
