"""Experiment A: per-layer sink-vs-capability decomposition.

For each model in the panel, aggregate all_head_selectivity by layer and
report, per layer:
  - n_sink_heads (first-token >= 30x)
  - n_induction_heads (induction >= 30x)
  - n_prev_token_heads (previous-token >= 30x)
  - max first-token, max induction selectivity in the layer
  - "overlap" flag when both sink_heads >= 1 AND induction_heads >= 1

Hypothesis (motivated by Paper 1 §6.3 BOS-suppression observation):
in Pythia 1B, the layers where matched-random ablation releases induction
(L4-L8) should be exactly the layers where sink heads and capability
heads co-occur. If true, this localizes the sink-vs-capability
competition to specific layers rather than treating it as a model-wide
phenomenon.

Output: cross_architecture/results/sink_vs_capability_layers.json
"""
from __future__ import annotations

import json
from pathlib import Path

REPO = Path(__file__).resolve().parents[3]
RESULTS = REPO / "cross_architecture" / "results"
OUT = RESULTS / "sink_vs_capability_layers.json"

MODELS = [
    ("Pythia 160M", "pythia_160m_mechinterp_fp32.json"),
    ("Pythia 410M", "pythia_410m_mechinterp_fp32.json"),
    ("Pythia 1B",   "pythia_1b_mechinterp.json"),
    ("OLMo 1B",     "olmo_mechinterp.json"),
    ("OLMoE 1B-7B", "olmoe_mechinterp.json"),
]

THRESHOLD = 30.0  # class-assignment threshold (matches Paper 1 §3.3)


def parse_head_key(k):
    L = int(k.split("_")[0][1:])
    H = int(k.split("_")[1][1:])
    return L, H


def analyze_model(name, path):
    d = json.load(open(path))
    ahs = d["all_head_selectivity"]
    n_layer = d.get("n_layer") or max(parse_head_key(k)[0] for k in ahs) + 1
    n_head_per_layer = d.get("n_head_per_layer") or max(parse_head_key(k)[1] for k in ahs) + 1

    per_layer = []
    for L in range(n_layer):
        layer_heads = [(parse_head_key(k)[1], v) for k, v in ahs.items()
                       if parse_head_key(k)[0] == L]
        sink_count = sum(1 for _, v in layer_heads if v.get("first-token", 0) >= THRESHOLD)
        ind_count  = sum(1 for _, v in layer_heads if v.get("induction", 0) >= THRESHOLD)
        prev_count = sum(1 for _, v in layer_heads if v.get("previous-token", 0) >= THRESHOLD)
        max_sink   = max((v.get("first-token", 0)    for _, v in layer_heads), default=0.0)
        max_ind    = max((v.get("induction", 0)      for _, v in layer_heads), default=0.0)
        max_prev   = max((v.get("previous-token", 0) for _, v in layer_heads), default=0.0)

        per_layer.append({
            "layer": L,
            "n_heads_in_layer": len(layer_heads),
            "n_sink_heads_geq30x": sink_count,
            "n_induction_heads_geq30x": ind_count,
            "n_prev_token_heads_geq30x": prev_count,
            "max_first_token_sel": float(max_sink),
            "max_induction_sel": float(max_ind),
            "max_prev_token_sel": float(max_prev),
            "overlap_sink_and_induction": (sink_count >= 1) and (ind_count >= 1),
        })

    # Summary stats
    overlap_layers = [r["layer"] for r in per_layer if r["overlap_sink_and_induction"]]
    pure_sink_layers = [r["layer"] for r in per_layer
                        if r["n_sink_heads_geq30x"] >= 1 and r["n_induction_heads_geq30x"] == 0]
    pure_induction_layers = [r["layer"] for r in per_layer
                              if r["n_induction_heads_geq30x"] >= 1 and r["n_sink_heads_geq30x"] == 0]
    zero_bos_layers = [r["layer"] for r in per_layer if r["n_sink_heads_geq30x"] == 0]
    total_sink_heads = sum(r["n_sink_heads_geq30x"] for r in per_layer)
    total_induction_heads = sum(r["n_induction_heads_geq30x"] for r in per_layer)

    return {
        "model": name,
        "n_layer": n_layer,
        "n_head_per_layer": n_head_per_layer,
        "n_heads_total": n_layer * n_head_per_layer,
        "total_sink_heads_geq30x": total_sink_heads,
        "total_induction_heads_geq30x": total_induction_heads,
        "sink_fraction": total_sink_heads / (n_layer * n_head_per_layer),
        "zero_bos_layers": zero_bos_layers,
        "pure_sink_layers": pure_sink_layers,
        "pure_induction_layers": pure_induction_layers,
        "overlap_layers (sink + induction)": overlap_layers,
        "per_layer": per_layer,
    }


def main():
    summary = {"threshold_x": THRESHOLD, "models": {}}
    for name, fname in MODELS:
        path = RESULTS / fname
        if not path.exists():
            print(f"  [skip] {name}: {fname} not found")
            continue
        r = analyze_model(name, path)
        summary["models"][name] = r
        print(f"\n=== {name} ({r['n_layer']}L x {r['n_head_per_layer']}H = {r['n_heads_total']} heads) ===")
        print(f"  Sink fraction (heads with first-token >= {THRESHOLD:.0f}x): "
              f"{r['total_sink_heads_geq30x']}/{r['n_heads_total']} ({r['sink_fraction']:.1%})")
        print(f"  Induction heads (>= {THRESHOLD:.0f}x): {r['total_induction_heads_geq30x']}")
        print(f"  Zero-BOS layers: {r['zero_bos_layers']}")
        print(f"  Sink-only layers (sink>=1, induction=0): {r['pure_sink_layers']}")
        print(f"  Induction-only layers (induction>=1, sink=0): {r['pure_induction_layers']}")
        print(f"  OVERLAP layers (sink + induction both >=1): {r['overlap_layers (sink + induction)']}")
        print(f"  {'L':>3} {'n_sink':>6} {'n_ind':>5} {'n_prev':>6} {'max_ft':>8} {'max_ind':>8} {'overlap?':>9}")
        for row in r["per_layer"]:
            overlap_marker = "<-- HERE" if row["overlap_sink_and_induction"] else ""
            print(f"  {row['layer']:>3} {row['n_sink_heads_geq30x']:>6} "
                  f"{row['n_induction_heads_geq30x']:>5} {row['n_prev_token_heads_geq30x']:>6} "
                  f"{row['max_first_token_sel']:>8.1f} {row['max_induction_sel']:>8.1f}  "
                  f"{overlap_marker}")

    with open(OUT, "w") as f:
        json.dump(summary, f, indent=2)
    print(f"\nwrote {OUT}")


if __name__ == "__main__":
    main()
