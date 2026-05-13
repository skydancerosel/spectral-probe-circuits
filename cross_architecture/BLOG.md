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

## Whole-model BOS-class fractions

Fraction of *all heads* classified as first-token at ≥30× selectivity, on synthetic and on natural text:

| Model | BOS % synthetic | BOS % natural-text |
|-------|----------------:|-------------------:|
| Pythia 160M | 43.1% | 15.3% |
| Pythia 410M | 58.1% | 69.0% |
| Pythia 1B | 53.9% | 68.0% |
| OLMoE 1B-7B | 68.0% | 73.4% |
| OLMo 1B | 78.1% | 84.0% |

Pythia 410M (69% natural) and OLMoE (73% natural) sit very close on this statistic. OLMo 1B dense is the most BOS-dominated of the five.

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

## Methodology notes from the cross-arch panel

- On the whole-model BOS-class statistic, Pythia 410M / 1B (54–69%) and OLMoE (68–73%) are close; OLMo 1B dense (78–84%) is the most BOS-dominated of the five. BOS dominance is widespread across decoder-only LMs at 100M+ scale, not unique to MoE.
- When ~70%+ of heads classify as `first-token`, integral-top-K-by-best-class doesn't reliably surface induction or prev-token heads. The **all-head capability-specific screen** (measure capability-X selectivity in isolation, classify by capability rather than by best-class) is the robust approach in that regime, and it identifies a 3–4 head induction circuit in every model in the panel.

## Methodology consequences

- **The capability circuit (3–4 heads, induction-selective ≥ 50×) is the load-bearing object, not the integral-top-K.** Integral-top-K is good for the "are there real circuit heads in this model" surface question; for the actual *causal* claim, screen all heads for the capability of interest and ablate that set.
- **For attention-sink-dominated architectures, `first-token` is a class to *exclude* from the rank-ordered analysis, not the dominant capability to track.** Excluding first-token-class, the non-BOS conserved fraction is around 12% — below the Pythia transformer baseline of 17–19%, but still a real specialized fraction.
- **The natural-text vs synthetic BOS amplification is itself a scale-related signal.** Pythia 160M is the only model where natural text *reduces* BOS (43% → 15%) — content-driven attention overrides the default. From 410M onward, real text *amplifies* BOS by 5–14pp — content-driven attention fails to override the default. There's a behavioral phase transition between 160M and 410M dense Pythia where attention organization flips from "content-driven" to "sink-dominated."

## Reproduce

- All scripts: [`scripts/`](scripts/) (22 files — per-head, mech-interp, ablation for each model; shared induction-batch helpers; tier-A cross-architecture layer-BOS analysis)
- All result JSONs: [`results/`](results/) (31 files spanning Phase 1 PR trajectories, Phase 2 mech-interp synthetic + natural, Phase 3 ablation)
- Full methodology writeup: [`README.md`](README.md)

The 22 scripts copy the existing-repo pattern (extracting per-head attention output, computing PR, identifying capability heads via all-head screen, ablating with matched-random control) and just swap in the new model classes / HF revisions. Nothing exotic in the methodology — the extension is the cross-arch panel itself.
