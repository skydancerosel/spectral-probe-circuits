# Scripts layout

```
scripts/
├── README.md                              # this file
├── shared/                                # importable utility modules
│   ├── induction_utils.py                 #   build_induction_batch, compute_pr
│   ├── ioi_batch.py                       #   build_ioi_batch (cross-tokenizer IOI prompts)
│   ├── prev_token_circuit_ablation.py     #   hook helpers + prev-token ablation main
│   ├── ioi_name_mover_ablation.py         #   evaluate_ioi, run_condition + name-mover ablation main
│   └── natural_induction_batch.py         #   OpenWebText natural-text induction batch builder
│
│                                          # Per-model-family scripts
├── pythia/    ← per_head, mechinterp, ablation (+ naturaltext variants); 2 pipelines
├── olmo/      ← per_head, mechinterp, ablation (+ naturaltext variants); 2 pipelines
├── olmoe/     ← per_head, mechinterp, ablation (+ naturaltext variants); baseline; tier1_analysis
│
│                                          # Per-experiment drivers
├── per_revision/  ← per_revision_mechinterp_driver.sh
├── prev_token/    ← prev_token_ablation_driver.sh
├── ioi/           ← ioi_mechinterp.py, ioi_eval.py, ioi_s_inhibition_ablation.py
│                    + 4 drivers (capability-screen, mechinterp, name-mover, S-Inhibition)
└── tier_a/        ← tier_a_layer_bos_analysis.py, tier_a_chain.sh
```

## Cross-script imports

Each Python script in a subdirectory adds `scripts/shared/` to `sys.path` via a two-line stanza placed after `from __future__ import annotations`:

```python
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "shared"))
```

This lets the script import shared utilities (`from induction_utils import build_induction_batch`, etc.) regardless of cwd.

## Running

From the repo root:

```bash
# Per-revision mech-interp sweep (30 runs across the 3 1B models)
bash cross_architecture/scripts/per_revision/per_revision_mechinterp_driver.sh

# Prev-token circuit ablation (3 runs)
bash cross_architecture/scripts/prev_token/prev_token_ablation_driver.sh

# IOI four-screen pipeline (3 models × 4 conditions = 12 ablation runs + 3 mech-interp runs)
bash cross_architecture/scripts/ioi/ioi_full_driver.sh         # per-head IOI selectivity
bash cross_architecture/scripts/ioi/ioi_driver.sh              # capability-screen ablation
bash cross_architecture/scripts/ioi/ioi_nm_driver.sh           # name-mover ablation
bash cross_architecture/scripts/ioi/ioi_si_driver.sh           # S-Inhibition ablation

# Tier-A: Pythia 160M/410M natural-text mech-interp + cross-arch layer-BOS analysis
bash cross_architecture/scripts/tier_a/tier_a_chain.sh

# Per-model-family Phase 1→2→3 pipelines
bash cross_architecture/scripts/pythia/pythia_full_pipeline.sh
bash cross_architecture/scripts/olmo/olmo_full_pipeline.sh
```

All drivers anchor to a known directory and write outputs under `../results/`. Phase 1 features files (`{pythia_1b,olmo,olmoe}_phase1_features.json`) are committed under `cross_architecture/results/` and are read as inputs to the per-revision and prev-token drivers.

## Reproducibility

Synthetic batches use RNG seed 42; matched-random controls use seed 123.
