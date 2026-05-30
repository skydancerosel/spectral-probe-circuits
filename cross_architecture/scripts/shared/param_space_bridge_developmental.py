"""B2-developmental: do per-head activation-PR and weight-spectral structure
CO-EVOLVE over training?

The static bridge (param_space_bridge.py) correlates final-checkpoint
activation PR with final-checkpoint weight spectra. The SED program is
about training trajectories, so the faithful test is developmental: at
each of 10 log-spaced revisions, compute per-head weight metrics, and
relate their trajectories to the per-head activation-PR trajectories
(already measured, in {model}_phase1_trajectory.json -> 'pr').

Three views, increasingly non-trivial:
  (1) Per-head trajectory correlation: for each head, Spearman between its
      10-point weight-metric series and its 10-point activation-PR series.
      Report the distribution. CAVEAT: both tend to rise from init, so a
      positive median partly reflects shared monotonic trend.
  (2) Cross-sectional per revision: at each revision, across-head Spearman
      (weight-metric vs activation-PR). Does the static bridge hold and
      strengthen over training?
  (3) Growth co-evolution (the non-trivial one): across heads, correlate
      Delta-activation-PR (final - initial) with Delta-weight-metric. Do
      heads that gain MORE activation PR also gain more weight rank?

Output: cross_architecture/results/param_space_bridge_developmental.json
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import torch

REPO = Path(__file__).resolve().parents[3]
RESULTS = REPO / "cross_architecture" / "results"
OUT = RESULTS / "param_space_bridge_developmental.json"

MODELS = [
    ("Pythia 1B",   "pythia", "EleutherAI/pythia-1b",    "pythia_1b_phase1_trajectory.json"),
    ("OLMo 1B",     "olmo",   "allenai/OLMo-1B-0724-hf", "olmo_phase1_trajectory.json"),
    ("OLMoE 1B-7B", "olmoe",  "allenai/OLMoE-1B-7B-0924","olmoe_phase1_trajectory.json"),
]


def spearman(x, y):
    x, y = np.asarray(x, float), np.asarray(y, float)
    if len(x) < 3 or np.std(x) == 0 or np.std(y) == 0:
        return float("nan")
    rx = np.argsort(np.argsort(x)); ry = np.argsort(np.argsort(y))
    return float(np.corrcoef(rx, ry)[0, 1])


def head_weight_metrics(W):
    try:
        s = torch.linalg.svdvals(W).numpy()
    except Exception:
        s = torch.linalg.svdvals(W.double()).float().numpy()
    s = np.sort(s)[::-1]
    fro2 = float((s ** 2).sum()); spec2 = float(s[0] ** 2) if len(s) else 0.0
    stable_rank = fro2 / spec2 if spec2 > 0 else float("nan")
    p = (s ** 2) / max((s ** 2).sum(), 1e-12)
    weight_pr = float(np.exp(-(p * np.log(p + 1e-12)).sum()))
    sigma1 = float(s[0]) if len(s) else 0.0
    return {"stable_rank": stable_rank, "weight_pr": weight_pr, "sigma1": sigma1}


def wo_slice(model, family, L, H, head_dim):
    if family == "pythia":
        W_O = model.gpt_neox.layers[L].attention.dense.weight.detach()
    else:
        W_O = model.model.layers[L].self_attn.o_proj.weight.detach()
    return W_O[:, H * head_dim:(H + 1) * head_dim].float().cpu()


def main():
    summary = {"models": {}}
    WMETRICS = ["stable_rank", "weight_pr", "sigma1"]

    for name, family, model_id, traj_json in MODELS:
        print(f"\n=== {name} ({model_id}) ===")
        traj = json.load(open(RESULTS / traj_json))
        revisions = traj["revisions"]
        act_pr = traj["pr"]  # {L#_H#: [10 PR values]}
        n_layer = traj["n_layer"]; n_head = traj["num_heads"]
        head_dim = None

        if family == "pythia":
            from transformers import GPTNeoXForCausalLM as M
        elif family == "olmo":
            from transformers import OlmoForCausalLM as M
        else:
            from transformers import OlmoeForCausalLM as M

        # weight-metric trajectories: {metric: {head: [per-revision]}}
        wtraj = {m: {f"L{L}_H{H}": [] for L in range(n_layer) for H in range(n_head)}
                 for m in WMETRICS}

        for ri, rev in enumerate(revisions):
            print(f"  [{ri+1}/{len(revisions)}] loading {rev} ...", flush=True)
            model = M.from_pretrained(model_id, revision=rev, dtype=torch.float32).eval()
            if head_dim is None:
                head_dim = model.config.hidden_size // n_head
            for L in range(n_layer):
                for H in range(n_head):
                    wm = head_weight_metrics(wo_slice(model, family, L, H, head_dim))
                    for m in WMETRICS:
                        wtraj[m][f"L{L}_H{H}"].append(wm[m])
            del model

        # (1) per-head trajectory correlation distribution
        # (3) growth co-evolution
        per_head_traj_corr = {m: [] for m in WMETRICS}
        growth_act = []; growth_w = {m: [] for m in WMETRICS}
        heads = [f"L{L}_H{H}" for L in range(n_layer) for H in range(n_head)]
        valid_heads = [h for h in heads if h in act_pr]
        for h in valid_heads:
            a = act_pr[h]
            growth_act.append(a[-1] - a[0])
            for m in WMETRICS:
                w = wtraj[m][h]
                per_head_traj_corr[m].append(spearman(w, a))
                growth_w[m].append(w[-1] - w[0])

        # (2) cross-sectional per revision
        cross_sectional = {m: [] for m in WMETRICS}
        for ri in range(len(revisions)):
            a_r = [act_pr[h][ri] for h in valid_heads]
            for m in WMETRICS:
                w_r = [wtraj[m][h][ri] for h in valid_heads]
                cross_sectional[m].append(spearman(w_r, a_r))

        model_summary = {"revisions": revisions, "n_heads": len(valid_heads),
                         "per_head_traj_corr": {}, "cross_sectional_by_revision": {},
                         "growth_coevolution": {}}
        print(f"  n_heads: {len(valid_heads)}")
        for m in WMETRICS:
            phc = np.array([c for c in per_head_traj_corr[m] if c == c])
            gco = spearman(growth_w[m], growth_act)
            model_summary["per_head_traj_corr"][m] = {
                "mean": float(np.mean(phc)), "median": float(np.median(phc)),
                "frac_positive": float((phc > 0).mean())}
            model_summary["cross_sectional_by_revision"][m] = cross_sectional[m]
            model_summary["growth_coevolution"][m] = gco
            print(f"  [{m}] per-head-traj median rho={np.median(phc):+.3f} "
                  f"(frac+ {float((phc>0).mean()):.2f}) | "
                  f"cross-sectional first->last {cross_sectional[m][0]:+.2f}->{cross_sectional[m][-1]:+.2f} | "
                  f"growth-coevolution rho={gco:+.3f}")

        summary["models"][name] = model_summary
        with open(OUT, "w") as f:
            json.dump(summary, f, indent=2)

    with open(OUT, "w") as f:
        json.dump(summary, f, indent=2)
    print(f"\nwrote {OUT}")


if __name__ == "__main__":
    main()
