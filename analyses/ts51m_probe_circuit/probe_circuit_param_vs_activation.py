"""
probe_circuit_param_vs_activation.py

Test #3 from the "what's next" list:
  Does the activation-space PR signal (which we use to identify circuit
  heads) correlate temporally with the parameter-space gap signal that
  the parent spectral-edge program tracks?

If both signals transition at the same checkpoint step, that's evidence
that activation-space and parameter-space are two windows on the same
underlying spectral structure — and connects this work to the broader
spectral-edge framework as a coherent two-surface claim instead of two
separate observations.

Method (s42 / TS-51M / canonical):

  1. Load consecutive checkpoints over training (step 0 .. 2000 — covers
     L0 PR transition at steps 400-1000).
  2. For each circuit head h ∈ {3, 6, 14, 15}, extract its L0 Q/K/V
     parameter rows from `blocks[0].attn.qkv.weight`.
  3. Compute ΔW_i = W_i - W_{i-1} (per-step parameter updates).
  4. For a rolling window of W=10 consecutive deltas, stack into a
     [10, P] matrix and compute the [10, 10] Gram. Take its eigenvalue
     spectrum, identify k* by the signal-weighted definition, compute
     the spectral gap g_λ = λ_{k*}² - λ_{k*+1}².
  5. Compare the gap-signal trajectory to the activation-space PR
     trajectory (already computed by probe_circuit_per_head.py) for
     the same checkpoints. Overlay with probe_in_acc.

If the gap signal collapses (or peaks) at the same step PR transitions,
correlation supports the two-surfaces claim.

Output: analyses/probe_circuit_param_vs_activation_s42.{json,png}
"""

import json
import sys
import re
from pathlib import Path
import numpy as np
import torch
import matplotlib.pyplot as plt

REPO = Path(__file__).resolve().parent
sys.path.insert(0, str(REPO / "training"))
from config import Config
from model import GPTModel

PRETRAIN_DIR = REPO / "runs/beta2_ablation/pilot_wd0.5_lr0.001_lp2.0_b20.95_s42"
PER_HEAD_JSON = REPO / "results/probe_circuit_per_head.json"

CIRCUIT_HEADS_L0 = [3, 6, 14, 15]
CONTROL_HEADS_L0 = [0, 1, 5, 7]
WINDOW = 10
STEP_LIMIT = 2000  # focus on the emergence window
HEAD_DIM = 32
D_MODEL = 512


def kstar_weighted(sigma, eps=0.05):
    """Same as in probe_circuit_per_head: signal-weighted k*."""
    if len(sigma) < 2:
        return 1
    s = sigma.astype(np.float64)
    s_sum = s.sum()
    if s_sum <= 0:
        return 1
    s1 = s[0]
    best_j, best_score = 1, -np.inf
    for j in range(1, len(s) - 1):
        if s[j] < eps * s1:
            break
        if s[j + 1] <= 0:
            continue
        score = (s[j] / s_sum) * (s[j] / s[j + 1])
        if score > best_score:
            best_score = score
            best_j = j
    return best_j + 1


def extract_circuit_params(state_dict, head_indices):
    """Extract the L0 attention Q/K/V weight rows for the specified heads.

    Returns a flat 1-D tensor concatenating the Q, K, V slices for all
    specified heads.
    """
    qkv = state_dict["blocks.0.attn.qkv.weight"]  # [3*d_model, d_model]
    pieces = []
    for h in head_indices:
        # Q: rows h*HEAD_DIM .. (h+1)*HEAD_DIM
        pieces.append(qkv[h*HEAD_DIM:(h+1)*HEAD_DIM, :].flatten())
        # K
        pieces.append(qkv[D_MODEL + h*HEAD_DIM:D_MODEL + (h+1)*HEAD_DIM, :].flatten())
        # V
        pieces.append(qkv[2*D_MODEL + h*HEAD_DIM:2*D_MODEL + (h+1)*HEAD_DIM, :].flatten())
    return torch.cat(pieces)  # [n_heads * 3 * HEAD_DIM * d_model] = [4*3*32*512] = 196608


def main():
    print(f"Run dir: {PRETRAIN_DIR.name}")

    # Discover checkpoints up to STEP_LIMIT, ordered by step
    ckpt_paths = []
    for p in sorted(PRETRAIN_DIR.glob("ckpt_*.pt")):
        m = re.match(r"ckpt_(\d+)\.pt", p.name)
        if m:
            step = int(m.group(1))
            if step <= STEP_LIMIT:
                ckpt_paths.append((step, p))
    ckpt_paths.sort()
    steps = [s for s, _ in ckpt_paths]
    print(f"Loaded {len(ckpt_paths)} ckpts in step range [0, {STEP_LIMIT}]")
    print(f"  steps: {steps[:5]} ... {steps[-3:]}")

    # Load each ckpt's circuit params + matched-control params
    print("Extracting circuit + control params from each ckpt...")
    circuit_params_per_ckpt = []   # list of 1-D tensors
    control_params_per_ckpt = []
    for step, ck_path in ckpt_paths:
        ck = torch.load(ck_path, map_location="cpu", weights_only=True)
        sd = ck["model_state_dict"]
        circuit_params_per_ckpt.append(extract_circuit_params(sd, CIRCUIT_HEADS_L0))
        control_params_per_ckpt.append(extract_circuit_params(sd, CONTROL_HEADS_L0))
    P = circuit_params_per_ckpt[0].numel()
    print(f"  param vector size per ckpt: {P}")

    # Compute per-step deltas: ΔW_i = W_{i+1} - W_i (between consecutive ckpts)
    # Note: consecutive ckpts have varying step gaps (1->200, 200->400, ...,
    # 600->650, 650->700, etc.). We just use the delta between adjacent
    # *saved* ckpts as a proxy for the parameter motion in that interval.
    circuit_deltas = []
    control_deltas = []
    delta_steps = []  # step at which the delta "ends"
    for i in range(1, len(circuit_params_per_ckpt)):
        circuit_deltas.append(circuit_params_per_ckpt[i] - circuit_params_per_ckpt[i-1])
        control_deltas.append(control_params_per_ckpt[i] - control_params_per_ckpt[i-1])
        delta_steps.append(steps[i])
    print(f"  built {len(circuit_deltas)} consecutive deltas")

    # Rolling window: for each window of W=WINDOW consecutive deltas, build
    # the Gram matrix and compute spectral gap.
    print(f"\nComputing rolling-window Gram-matrix gap signal (W={WINDOW})...")
    window_steps = []
    circuit_kstar = []
    circuit_gap = []
    circuit_lam1 = []
    control_kstar = []
    control_gap = []
    control_lam1 = []
    for i in range(WINDOW - 1, len(circuit_deltas)):
        # Window = deltas[i-W+1 .. i]
        window_circuit = torch.stack(circuit_deltas[i-WINDOW+1:i+1])  # [W, P]
        window_control = torch.stack(control_deltas[i-WINDOW+1:i+1])

        # Gram: [W, W] = W_window @ W_window.T
        G_circ = window_circuit @ window_circuit.T
        G_ctrl = window_control @ window_control.T

        # Eigenvalues of Gram (sorted descending)
        eig_c = torch.linalg.eigvalsh(G_circ).flip(0).cpu().numpy()
        eig_t = torch.linalg.eigvalsh(G_ctrl).flip(0).cpu().numpy()
        eig_c = np.clip(eig_c, 0, None)
        eig_t = np.clip(eig_t, 0, None)

        # Singular values of W_window are sqrt(eigvals of W_window @ W_window.T)
        sigma_c = np.sqrt(eig_c)
        sigma_t = np.sqrt(eig_t)

        ks_c = kstar_weighted(sigma_c)
        ks_t = kstar_weighted(sigma_t)
        # gap_lambda = sigma_{k*}^2 - sigma_{k*+1}^2 (eigenvalues, not sigmas)
        # NOTE eigenvalues already squared (eig of A A^T = sigma^2)
        if ks_c < len(eig_c):
            gap_c = eig_c[ks_c - 1] - eig_c[ks_c]  # 0-indexed: eig[ks-1] is the k*-th
        else:
            gap_c = float("nan")
        if ks_t < len(eig_t):
            gap_t = eig_t[ks_t - 1] - eig_t[ks_t]
        else:
            gap_t = float("nan")

        circuit_kstar.append(int(ks_c))
        control_kstar.append(int(ks_t))
        circuit_gap.append(float(gap_c))
        control_gap.append(float(gap_t))
        circuit_lam1.append(float(eig_c[0]))
        control_lam1.append(float(eig_t[0]))
        window_steps.append(delta_steps[i])

    print(f"  computed {len(window_steps)} window points")

    # Load activation-space PR data
    print("\nLoading activation-space PR data...")
    spec = json.load(open(PER_HEAD_JSON))
    spec_steps = np.array(spec["ckpt_step"])
    pr_circuit = np.array([spec["pr"][f"L0_H{h}"] for h in CIRCUIT_HEADS_L0]).mean(axis=0)
    pr_control = np.array([spec["pr"][f"L0_H{h}"] for h in CONTROL_HEADS_L0]).mean(axis=0)
    # filter to our step range
    in_range = spec_steps <= STEP_LIMIT
    spec_steps = spec_steps[in_range]
    pr_circuit = pr_circuit[in_range]
    pr_control = pr_control[in_range]

    # Probe accuracy
    pc = spec["probe_curve"]
    pc_steps = np.array([row[0] for row in pc])
    pc_pin = np.array([row[1] for row in pc], dtype=float)
    in_range_pc = pc_steps <= STEP_LIMIT
    pc_steps = pc_steps[in_range_pc]
    pc_pin = pc_pin[in_range_pc]

    # Save JSON
    out = {
        "run_dir": str(PRETRAIN_DIR),
        "circuit_heads": CIRCUIT_HEADS_L0,
        "control_heads": CONTROL_HEADS_L0,
        "window_size": WINDOW,
        "window_steps": window_steps,
        "circuit_param_kstar": circuit_kstar,
        "control_param_kstar": control_kstar,
        "circuit_param_gap": circuit_gap,
        "control_param_gap": control_gap,
        "circuit_param_lam1": circuit_lam1,
        "control_param_lam1": control_lam1,
        "act_pr_steps": spec_steps.tolist(),
        "act_pr_circuit_mean": pr_circuit.tolist(),
        "act_pr_control_mean": pr_control.tolist(),
        "probe_curve_steps": pc_steps.tolist(),
        "probe_curve_pin": pc_pin.tolist(),
    }
    out_json = REPO / "results/probe_circuit_param_vs_activation_s42.json"
    with open(out_json, "w") as f:
        json.dump(out, f, indent=2)
    print(f"\nwrote {out_json}")

    # Plot: 3-panel overlay
    fig, axes = plt.subplots(3, 1, figsize=(11, 10), sharex=True)

    # Panel 1: probe accuracy
    ax = axes[0]
    ax.plot(pc_steps, pc_pin, color="tab:red", lw=2, label="probe_in_acc")
    ax.set_ylabel("probe_in_acc", fontsize=11)
    ax.set_ylim(-0.05, 1.05)
    ax.grid(True, alpha=0.3)
    ax.set_title("(A) Behavioral emergence: probe accuracy 0 → 1 around step 800")
    ax.legend(loc="upper left", fontsize=10)

    # Panel 2: activation-space PR (mean over circuit / control heads)
    ax = axes[1]
    ax.plot(spec_steps, pr_circuit, color="tab:red", lw=2,
            label=f"mean PR over circuit heads L0H{CIRCUIT_HEADS_L0}")
    ax.plot(spec_steps, pr_control, color="tab:gray", lw=1.5, ls="--",
            label=f"mean PR over control heads L0H{CONTROL_HEADS_L0}")
    ax.set_ylabel("activation-space PR\n(per-head, head_dim=32)", fontsize=11)
    ax.grid(True, alpha=0.3)
    ax.set_title("(B) Activation-space spectral signal: PR of per-head outputs")
    ax.legend(loc="upper left", fontsize=10)

    # Panel 3: parameter-space gap signal (rolling window over consecutive ckpt deltas)
    ax = axes[2]
    # Normalize each gap series by its lam1 to make scales comparable
    gap_c_norm = np.array(circuit_gap) / np.array(circuit_lam1)
    gap_t_norm = np.array(control_gap) / np.array(control_lam1)
    ax.plot(window_steps, gap_c_norm, color="tab:red", lw=2,
            label=f"normalized gap, circuit head params (W={WINDOW} rolling window)")
    ax.plot(window_steps, gap_t_norm, color="tab:gray", lw=1.5, ls="--",
            label=f"normalized gap, control head params")
    ax.set_xlabel("pretraining step", fontsize=11)
    ax.set_ylabel("parameter-space gap (normalized)\nλ_{k*}² − λ_{k*+1}² over λ_1²",
                  fontsize=11)
    ax.grid(True, alpha=0.3)
    ax.set_title("(C) Parameter-space spectral gap: rolling-window Gram-matrix eigenvalue gap")
    ax.legend(loc="upper right", fontsize=10)

    fig.suptitle("Activation-space PR vs parameter-space gap signal — s42, L0 circuit\n"
                  "do they transition at the same training step?",
                  fontsize=12, weight="bold", y=0.998)
    fig.tight_layout()
    out_png = REPO / "results/probe_circuit_param_vs_activation_s42.png"
    fig.savefig(out_png, dpi=130, bbox_inches="tight")
    print(f"saved {out_png}")


if __name__ == "__main__":
    main()
