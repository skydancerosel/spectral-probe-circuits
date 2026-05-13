# Scripts layout

```
scripts/
├── README.md                              # this file
│
│   # Shared utilities (importable by everything below)
├── mamba2_per_head.py                     # build_induction_batch + compute_pr (synthetic induction batch)
├── ioi_batch.py                           # build_ioi_batch (cross-tokenizer IOI prompt builder)
├── prev_token_circuit_ablation.py         # make_pre_hook, get_layer_module, pick_circuit + prev-token ablation main
├── ioi_name_mover_ablation.py             # evaluate_ioi, run_condition + name-mover ablation main
├── build_natural_induction_batch.py       # OpenWebText natural-text induction batch builder
│
│   # Per-model-family scripts (per-head trajectory, mech-interp, ablation)
├── pythia/      ← pythia_per_head.py, pythia_mechinterp.py, pythia_ablation.py,
│                  pythia_mechinterp_naturaltext.py, pythia_ablation_naturaltext.py,
│                  pythia_full_pipeline.sh, pythia_smaller_fp32.sh
├── olmo/        ← olmo_per_head.py, olmo_mechinterp.py, olmo_ablation.py,
│                  olmo_mechinterp_naturaltext.py, olmo_ablation_naturaltext.py,
│                  olmo_full_pipeline.sh, olmo_pipeline_resume.sh
├── olmoe/       ← olmoe_per_head.py, olmoe_mechinterp.py, olmoe_ablation.py,
│                  olmoe_mechinterp_naturaltext.py, olmoe_ablation_naturaltext.py,
│                  olmoe_baseline.py, olmoe_tier1_analysis.py
│
│   # Per-experiment drivers
├── per_revision/    ← per_revision_mechinterp_driver.sh
├── prev_token/      ← prev_token_ablation_driver.sh
├── ioi/             ← ioi_mechinterp.py, ioi_eval.py, ioi_s_inhibition_ablation.py
│                     + 4 drivers: ioi_driver.sh (capability-screen ablation),
│                                  ioi_full_driver.sh (per-head IOI selectivity),
│                                  ioi_nm_driver.sh (name-mover ablation),
│                                  ioi_si_driver.sh (S-Inhibition ablation)
├── tier_a/          ← tier_a_layer_bos_analysis.py, tier_a_chain.sh (Pythia 160M+410M natural text)
└── misc/            ← bench_mamba2_mps.py, bench_olmoe_mps.py (early MPS benchmarks)
```

## Cross-script imports

Each Python script in a subdirectory prepends the `scripts/` root to `sys.path` so that imports of the shared utilities (e.g., `from mamba2_per_head import build_induction_batch`) resolve correctly when the script is run as `python3 pythia/pythia_mechinterp.py ...` from anywhere.

The stanza is two lines, placed after `from __future__ import annotations` and before any other import:

```python
import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
```

## Running

All drivers `cd` to the appropriate working directory and reference results via absolute paths under `../results/`. From the repo root:

```bash
# Per-revision mech-interp sweep (30 runs across the 3 1B models)
bash cross_architecture/scripts/per_revision/per_revision_mechinterp_driver.sh

# Prev-token circuit ablation (3 runs)
bash cross_architecture/scripts/prev_token/prev_token_ablation_driver.sh

# IOI evaluation (capability screen + name-mover + S-Inhibition; ~12 runs across 3 models)
bash cross_architecture/scripts/ioi/ioi_driver.sh
bash cross_architecture/scripts/ioi/ioi_full_driver.sh
bash cross_architecture/scripts/ioi/ioi_nm_driver.sh
bash cross_architecture/scripts/ioi/ioi_si_driver.sh

# Tier-A: Pythia 160M/410M natural-text mech-interp + cross-arch layer-BOS analysis
bash cross_architecture/scripts/tier_a/tier_a_chain.sh
```

Per-model-family pipelines (`pythia_full_pipeline.sh`, `olmo_full_pipeline.sh`, etc.) chain Phase 1 → 2 → 3 for a single model; they `cd` into their own subdirectory and write intermediate artifacts there.

## Reproducibility

All synthetic batches use RNG seed 42; matched-random controls use seed 123. Phase 1 features files (`pythia_1b_phase1_features.json`, `olmo_phase1_features.json`, `olmoe_phase1_features.json`) are committed under `../results/` and are inputs to the per-revision and prev-token drivers.
