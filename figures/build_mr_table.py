"""Aggregate matched-random sweeps across 4 tasks × 3 models × 10 seeds
into a single Markdown table for the methodology paper.
"""
import json
from pathlib import Path

HERE = Path(__file__).parent
WORKTREE = HERE.parent

MODELS = [("Pythia 1B", "pythia1b"), ("OLMo 1B", "olmo"), ("OLMoE 1B-7B", "olmoe")]
TASKS = [
    ("IOI name-mover",       "ioi_nm",  "top1",        "logit_diff_mean"),
    ("IOI S-Inhibition",     "ioi_si",  "top1",        "logit_diff_mean"),
    ("Greater-than",         "gt",      "top1_above",  "logit_diff_above_below"),
    ("Successor",            "succ",    "top1_acc",    "target_logit_mean"),
]


def load_sweep(short, task):
    p = WORKTREE / f"matched_random_sweep/{short}_{task}_mr_sweep.json"
    if not p.exists():
        return None
    with open(p) as f: return json.load(f)


def screen_delta(short, task):
    """Return the screen-specific ablation Δ from the original ablation JSON."""
    if task in ("ioi_nm",):
        p = WORKTREE / f"ioi/{short}_ioi_nm_ablation.json"
    elif task == "ioi_si":
        p = WORKTREE / f"ioi/{short}_ioi_si_ablation.json"
    elif task == "gt":
        p = WORKTREE / f"gt/{short}_gt_ablation.json"
    elif task == "succ":
        p = WORKTREE / f"succ/{short}_succ_ablation.json"
    if not p.exists(): return None
    with open(p) as f: d = json.load(f)
    base = d["conditions"][0]
    for c in d["conditions"][1:]:
        if c["condition"].startswith("ablate_top"):
            return c, base
    return None


print("# Matched-random sweep summary (n=10 seeds)\n")
print("Per-model, per-task: screen-specific ablation Δ vs matched-random Δ (mean ± std over 10 random seeds).\n")
for task_label, task, top1_key, logit_key in TASKS:
    print(f"\n## {task_label}\n")
    print("| Model | Screen Δtop-1 | Matched-random Δtop-1 (mean ± std) | Screen Δlogit | MR Δlogit (mean ± std) | Spec. ratio (top-1) |")
    print("|---|---:|---:|---:|---:|---:|")
    for model_label, short in MODELS:
        sw = load_sweep(short, task)
        sd = screen_delta(short, task)
        if sw is None or sd is None:
            print(f"| {model_label} | (pending) | (pending) | (pending) | (pending) | — |")
            continue
        screen_c, base = sd
        # Compute screen Δ
        b1 = base[top1_key]; bl = base[logit_key]
        screen_top1 = (screen_c[top1_key] - b1) * 100
        screen_logit = screen_c[logit_key] - bl
        # MR deltas from sweep
        mr_top1_mean = sw["delta"][top1_key]["mean"]
        mr_top1_std = sw["delta"][top1_key]["std"]
        mr_logit_mean = sw["delta"][logit_key]["mean"]
        mr_logit_std = sw["delta"][logit_key]["std"]
        # Specificity ratio (top-1)
        if abs(mr_top1_mean) < 0.5:
            ratio = "∞" if abs(screen_top1) > 1 else "—"
        else:
            ratio = f"{screen_top1 / mr_top1_mean:.1f}×"
        print(f"| {model_label} | {screen_top1:+.1f}pp | {mr_top1_mean:+.1f} ± {mr_top1_std:.1f}pp | "
              f"{screen_logit:+.2f} | {mr_logit_mean:+.2f} ± {mr_logit_std:.2f} | {ratio} |")

# Also save raw json
out = {}
for task_label, task, top1_key, logit_key in TASKS:
    out[task] = {"label": task_label}
    for model_label, short in MODELS:
        sw = load_sweep(short, task)
        sd = screen_delta(short, task)
        if sw is None or sd is None:
            out[task][short] = None; continue
        screen_c, base = sd
        b1 = base[top1_key]; bl = base[logit_key]
        out[task][short] = {
            "screen_top1_delta_pp": (screen_c[top1_key] - b1) * 100,
            "screen_logit_delta": screen_c[logit_key] - bl,
            "mr_top1_delta_mean_pp": sw["delta"][top1_key]["mean"],
            "mr_top1_delta_std_pp": sw["delta"][top1_key]["std"],
            "mr_logit_delta_mean": sw["delta"][logit_key]["mean"],
            "mr_logit_delta_std": sw["delta"][logit_key]["std"],
            "n_seeds": sw["n_seeds"],
        }
out_path = HERE / "mr_sweep_summary.json"
with open(out_path, "w") as f:
    json.dump(out, f, indent=2)
print(f"\n\n(saved aggregated summary to {out_path})")
