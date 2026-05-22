"""Experiment C: developmental emergence of L0/L1 zero-BOS vs L2+ sink ascent.

Question: at what training fraction does the L2+ ascent (sinks forming above
the L0/L1 floor) happen? Is the L0/L1 zero-BOS pattern present from random
initialization, or does training establish it? When do L2+ layers first
become sink-active?

Per-revision mech-interp data for all 3 1B models lives at
cross_architecture/results/per_revision_mechinterp/. Each file contains
all_head_selectivity for one revision. We aggregate by layer and track:
  - n_sink_heads_per_layer at each revision
  - first revision where layer L has any sink head (>= 30x first-token)
  - whether L0 and L1 ever cross the threshold (Paper 2 Finding 1 claim)

Outputs:
  cross_architecture/results/sink_emergence_developmental.json
"""
from __future__ import annotations

import json
import re
from pathlib import Path

REPO = Path(__file__).resolve().parents[3]
RESULTS = REPO / "cross_architecture" / "results"
PER_REV = RESULTS / "per_revision_mechinterp"
OUT = RESULTS / "sink_emergence_developmental.json"

THRESHOLD = 30.0


def parse_head_key(k):
    L = int(k.split("_")[0][1:])
    H = int(k.split("_")[1][1:])
    return L, H


def parse_revision_filename(fname):
    """Parse Pythia / OLMo / OLMoE revision filenames into (model_key, sort_key, label, tokens_B).

    Pythia:  pythia1b_mechinterp_step143000.json     -> ("pythia_1b", 143000, "step143000", None)
    OLMo:    olmo_mechinterp_step1000-tokens2B.json  -> ("olmo_1b", 1000, "step1000", 2.0)
    OLMoE:   olmoe_mechinterp_step5000-tokens20B.json -> ("olmoe_1b_7b", 5000, "step5000", 20.0)
    """
    name = fname.replace(".json", "")
    if name.startswith("pythia1b_mechinterp_"):
        rest = name.replace("pythia1b_mechinterp_", "")
        m = re.match(r"step(\d+)", rest)
        if not m: return None
        step = int(m.group(1))
        return ("pythia_1b", step, f"step{step}", None)
    if name.startswith("olmoe_mechinterp_"):
        rest = name.replace("olmoe_mechinterp_", "")
        m = re.match(r"step(\d+)-tokens(\d+)B", rest)
        if not m: return None
        step, toks = int(m.group(1)), float(m.group(2))
        return ("olmoe_1b_7b", step, f"step{step}", toks)
    if name.startswith("olmo_mechinterp_"):
        rest = name.replace("olmo_mechinterp_", "")
        m = re.match(r"step(\d+)-tokens(\d+)B", rest)
        if not m: return None
        step, toks = int(m.group(1)), float(m.group(2))
        return ("olmo_1b", step, f"step{step}", toks)
    return None


# Total tokens per model (for normalising to training fraction)
TOTAL_TOKENS_B = {
    "pythia_1b":   299.892,  # Pythia 1B trained on ~300B
    "olmo_1b":     3048.0,   # OLMo 1B-0724 — final revision is at 3048B
    "olmoe_1b_7b": 5117.0,   # OLMoE final at 5117B
}
# Pythia uses step -> tokens conversion (2.1M tokens/step)
PYTHIA_TOKENS_PER_STEP = 2.097152  # in millions


def pythia_tokens_B(step):
    return step * PYTHIA_TOKENS_PER_STEP / 1000  # convert M to B


def main():
    out = {"threshold_x": THRESHOLD, "models": {}}

    # Group files by model
    by_model = {}
    for f in sorted(PER_REV.glob("*.json")):
        parsed = parse_revision_filename(f.name)
        if parsed is None:
            continue
        model_key, step, label, toks = parsed
        by_model.setdefault(model_key, []).append((step, label, toks, f))

    for model_key, files in by_model.items():
        files.sort(key=lambda x: x[0])
        per_revision = []
        n_layer = n_head_per_layer = None

        for step, label, toks, path in files:
            d = json.load(open(path))
            ahs = d["all_head_selectivity"]
            if n_layer is None:
                n_layer = d.get("n_layer") or max(parse_head_key(k)[0] for k in ahs) + 1
                n_head_per_layer = d.get("n_head_per_layer") or max(parse_head_key(k)[1] for k in ahs) + 1
            per_layer_sink = [0] * n_layer
            per_layer_induction = [0] * n_layer
            for k, v in ahs.items():
                L, _ = parse_head_key(k)
                if v.get("first-token", 0) >= THRESHOLD:
                    per_layer_sink[L] += 1
                if v.get("induction", 0) >= THRESHOLD:
                    per_layer_induction[L] += 1

            if toks is None and model_key == "pythia_1b":
                toks = pythia_tokens_B(step)
            train_frac = (toks / TOTAL_TOKENS_B[model_key]) if toks is not None else None

            per_revision.append({
                "step": step,
                "label": label,
                "tokens_B": toks,
                "training_fraction": train_frac,
                "n_sink_per_layer": per_layer_sink,
                "n_induction_per_layer": per_layer_induction,
                "total_sink": sum(per_layer_sink),
                "total_induction": sum(per_layer_induction),
                "L0_sink": per_layer_sink[0],
                "L1_sink": per_layer_sink[1],
                "L2_sink": per_layer_sink[2] if n_layer > 2 else None,
            })

        # Find first revision where each layer has >=1 sink head
        first_sink_per_layer = []
        for L in range(n_layer):
            first = None
            for row in per_revision:
                if row["n_sink_per_layer"][L] >= 1:
                    first = {"step": row["step"], "label": row["label"],
                             "tokens_B": row["tokens_B"], "training_fraction": row["training_fraction"]}
                    break
            first_sink_per_layer.append({"layer": L, "first_sink_revision": first})

        L0_ever_sink = any(r["L0_sink"] >= 1 for r in per_revision)
        L1_ever_sink = any(r["L1_sink"] >= 1 for r in per_revision)

        out["models"][model_key] = {
            "n_layer": n_layer,
            "n_head_per_layer": n_head_per_layer,
            "n_revisions": len(per_revision),
            "L0_ever_has_sink": L0_ever_sink,
            "L1_ever_has_sink": L1_ever_sink,
            "first_sink_per_layer": first_sink_per_layer,
            "per_revision_summary": [
                {k: v for k, v in r.items() if k not in {"n_sink_per_layer", "n_induction_per_layer"}}
                for r in per_revision
            ],
            "per_revision_full": per_revision,
        }

        print(f"\n=== {model_key} ({n_layer}L x {n_head_per_layer}H, {len(per_revision)} revisions) ===")
        print(f"  L0 EVER has sink head (>= {THRESHOLD}x first-token)?  {L0_ever_sink}")
        print(f"  L1 EVER has sink head?                                {L1_ever_sink}")
        print(f"\n  First revision where each layer gets its first sink head:")
        print(f"    {'L':>3}  {'first sink revision':<22}  tokens_B   train_frac")
        for r in first_sink_per_layer:
            f = r["first_sink_revision"]
            if f is None:
                print(f"    {r['layer']:>3}  {'(never)':<22}")
            else:
                tb = f"{f['tokens_B']:.1f}" if f["tokens_B"] is not None else "?"
                tf = f"{f['training_fraction']*100:.2f}%" if f["training_fraction"] is not None else "?"
                print(f"    {r['layer']:>3}  {f['label']:<22}  {tb:>8}   {tf:>10}")
        print(f"\n  Total sink-head count over training:")
        print(f"    {'step':<14} {'tokens_B':>10} {'L0':>3} {'L1':>3} {'L2':>3} {'total':>6}")
        for row in per_revision:
            tb = f"{row['tokens_B']:.1f}" if row["tokens_B"] is not None else "?"
            l2 = row["L2_sink"] if row["L2_sink"] is not None else "-"
            print(f"    {row['label']:<14} {tb:>10} {row['L0_sink']:>3} {row['L1_sink']:>3} {l2:>3} {row['total_sink']:>6}")

    with open(OUT, "w") as f:
        json.dump(out, f, indent=2)
    print(f"\nwrote {OUT}")


if __name__ == "__main__":
    main()
