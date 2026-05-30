"""B2: parameter-space bridge.

Tests whether the activation-space per-head spectral signal (PR of the
attention output across the eval batch) relates to parameter-space
per-head weight structure (the singular spectrum of the head's output
projection W_O slice). This is the bridge the developmental paper's
Discussion promises between the activation-space probe-circuit program
and the parameter-space spectral-edge program.

For each model, at the final checkpoint, per head:
  weight side (from W_O output-projection slice [head_dim x hidden]):
    - stable_rank = ||W||_F^2 / ||W||_2^2
    - weight_pr   = participation ratio of the singular values
    - sigma1      = top singular value
    - sigma_gap   = (sigma1 - sigma2) / sigma1
  activation side (from existing phase1_features.json):
    - final_pr    = PR of attention output at the final checkpoint
    - integral    = PR-integral ranking statistic
  capability (from existing mechinterp.json all_head_selectivity):
    - induction, first-token, previous-token selectivities

Reports Pearson + Spearman correlations of each weight metric against
activation final_pr and against the capability selectivities, per model
and pooled.

Output: cross_architecture/results/param_space_bridge.json
"""
from __future__ import annotations

import json
import os
from pathlib import Path

import numpy as np
import torch

REPO = Path(__file__).resolve().parents[3]
RESULTS = REPO / "cross_architecture" / "results"
OUT = RESULTS / "param_space_bridge.json"

MODELS = [
    ("Pythia 1B",   "pythia", "EleutherAI/pythia-1b",       "step143000",
     "pythia_1b_phase1_features.json", "pythia_1b_mechinterp.json"),
    ("OLMo 1B",     "olmo",   "allenai/OLMo-1B-0724-hf",     "main",
     "olmo_phase1_features.json", "olmo_mechinterp.json"),
    ("OLMoE 1B-7B", "olmoe",  "allenai/OLMoE-1B-7B-0924",    "main",
     "olmoe_phase1_features.json", "olmoe_mechinterp.json"),
]


def spearman(x, y):
    x, y = np.asarray(x), np.asarray(y)
    rx = np.argsort(np.argsort(x))
    ry = np.argsort(np.argsort(y))
    return float(np.corrcoef(rx, ry)[0, 1])


def pearson(x, y):
    x, y = np.asarray(x), np.asarray(y)
    if x.std() == 0 or y.std() == 0:
        return float("nan")
    return float(np.corrcoef(x, y)[0, 1])


def wo_slices(model, family, n_layer, n_head, head_dim):
    """Return dict (L,H) -> W_O slice [head_dim, hidden] for that head."""
    slices = {}
    for L in range(n_layer):
        if family == "pythia":
            W_O = model.gpt_neox.layers[L].attention.dense.weight.detach()  # [hidden, hidden]
        else:
            W_O = model.model.layers[L].self_attn.o_proj.weight.detach()    # [hidden, hidden]
        # o_proj maps concatenated head outputs [n_head*head_dim] -> hidden.
        # Column block [h*head_dim:(h+1)*head_dim] is head h's contribution.
        for H in range(n_head):
            sl = W_O[:, H * head_dim:(H + 1) * head_dim].float().cpu()  # [hidden, head_dim]
            slices[(L, H)] = sl
    return slices


def head_weight_metrics(W):
    """W: [hidden, head_dim] slice. Return spectral metrics."""
    # singular values
    try:
        s = torch.linalg.svdvals(W).numpy()
    except Exception:
        s = torch.linalg.svdvals(W.double()).float().numpy()
    s = np.sort(s)[::-1]
    fro2 = float((s ** 2).sum())
    spec2 = float(s[0] ** 2) if len(s) else 0.0
    stable_rank = fro2 / spec2 if spec2 > 0 else float("nan")
    p = (s ** 2) / max((s ** 2).sum(), 1e-12)
    weight_pr = float(np.exp(-(p * np.log(p + 1e-12)).sum()))
    sigma1 = float(s[0]) if len(s) else 0.0
    sigma2 = float(s[1]) if len(s) > 1 else 0.0
    sigma_gap = (sigma1 - sigma2) / sigma1 if sigma1 > 0 else float("nan")
    return {"stable_rank": stable_rank, "weight_pr": weight_pr,
            "sigma1": sigma1, "sigma_gap": sigma_gap}


def main():
    summary = {"models": {}, "pooled": {}}
    pooled = {k: [] for k in
              ["stable_rank", "weight_pr", "sigma1", "sigma_gap",
               "act_final_pr", "act_integral", "induction", "first_token", "prev_token"]}

    for name, family, model_id, revision, feat_json, mech_json in MODELS:
        print(f"\n=== {name} ({model_id}@{revision}) ===")
        feat = json.load(open(RESULTS / feat_json))["features"]
        mech = json.load(open(RESULTS / mech_json))["all_head_selectivity"]

        print(f"  loading weights...")
        if family == "pythia":
            from transformers import GPTNeoXForCausalLM as M
        elif family == "olmo":
            from transformers import OlmoForCausalLM as M
        else:
            from transformers import OlmoeForCausalLM as M
        model = M.from_pretrained(model_id, revision=revision, dtype=torch.float32).eval()
        cfg = model.config
        n_layer, n_head = cfg.num_hidden_layers, cfg.num_attention_heads
        head_dim = cfg.hidden_size // n_head
        print(f"  L={n_layer} H={n_head} hd={head_dim}")

        slices = wo_slices(model, family, n_layer, n_head, head_dim)
        del model

        rows = []
        for (L, H), W in slices.items():
            key = f"L{L}_H{H}"
            if key not in feat or key not in mech:
                continue
            wm = head_weight_metrics(W)
            row = {
                "head": key, "layer": L,
                **wm,
                "act_final_pr": float(feat[key].get("final_pr", float("nan"))),
                "act_integral": float(feat[key].get("integral", float("nan"))),
                "induction": float(mech[key].get("induction", 0.0)),
                "first_token": float(mech[key].get("first-token", 0.0)),
                "prev_token": float(mech[key].get("previous-token", 0.0)),
            }
            rows.append(row)
            for k in pooled:
                pooled[k].append(row[k])

        # per-model correlations
        def corr_block(xkey):
            xs = [r[xkey] for r in rows]
            out = {}
            for wk in ["stable_rank", "weight_pr", "sigma1", "sigma_gap"]:
                ws = [r[wk] for r in rows]
                out[wk] = {"pearson": pearson(ws, xs), "spearman": spearman(ws, xs)}
            return out

        model_summary = {
            "n_heads": len(rows),
            "weight_vs_act_final_pr": corr_block("act_final_pr"),
            "weight_vs_induction":    corr_block("induction"),
            "weight_vs_first_token":  corr_block("first_token"),
            "rows": rows,
        }
        summary["models"][name] = model_summary

        print(f"  n_heads analyzed: {len(rows)}")
        print(f"  weight-metric vs activation final_pr (Spearman):")
        for wk in ["stable_rank", "weight_pr", "sigma1", "sigma_gap"]:
            sp = model_summary["weight_vs_act_final_pr"][wk]["spearman"]
            print(f"    {wk:<12} rho = {sp:+.3f}")
        print(f"  weight-metric vs induction-selectivity (Spearman):")
        for wk in ["stable_rank", "weight_pr", "sigma1", "sigma_gap"]:
            sp = model_summary["weight_vs_induction"][wk]["spearman"]
            print(f"    {wk:<12} rho = {sp:+.3f}")

        with open(OUT, "w") as f:
            json.dump(summary, f, indent=2)

    # pooled correlations
    def pooled_corr(xkey):
        xs = pooled[xkey]
        out = {}
        for wk in ["stable_rank", "weight_pr", "sigma1", "sigma_gap"]:
            out[wk] = {"pearson": pearson(pooled[wk], xs), "spearman": spearman(pooled[wk], xs)}
        return out
    summary["pooled"] = {
        "n_heads": len(pooled["act_final_pr"]),
        "weight_vs_act_final_pr": pooled_corr("act_final_pr"),
        "weight_vs_induction":    pooled_corr("induction"),
        "weight_vs_first_token":  pooled_corr("first_token"),
    }
    print(f"\n=== POOLED ({len(pooled['act_final_pr'])} heads across 3 models) ===")
    print(f"  weight-metric vs activation final_pr (Spearman):")
    for wk in ["stable_rank", "weight_pr", "sigma1", "sigma_gap"]:
        sp = summary["pooled"]["weight_vs_act_final_pr"][wk]["spearman"]
        print(f"    {wk:<12} rho = {sp:+.3f}")

    with open(OUT, "w") as f:
        json.dump(summary, f, indent=2)
    print(f"\nwrote {OUT}")


if __name__ == "__main__":
    main()
