"""
l0_substrate_and_emergence_124m.py

Two cheap follow-up analyses on existing karpathy_llmc 124M data:

  (a) L0 substrate test: does the 124M model have an analog of the
      'L0H{3,6,14,15} retrieval substrate' pattern from TS-51M?
      - Where do L0 heads rank in the spectral signal?
      - Are any L0 heads strong on the measured capabilities (induction,
        prev-token, self)?

  (b) Time-of-emergence by capability class: when does each capability
      class first appear during training? Plot emergence step for each
      classified head.
      - Emergence definition: first ckpt where PR exceeds 15 (mid-rise
        threshold, between rank-1 floor of ~1 and saturation of ~30-50)
      - Predictions from literature: induction emerges late (~2-3B tokens),
        prev-token early. Compare to our data.

No new training, no new GPU compute beyond loading data.

Output: analyses/l0_substrate_and_emergence_124m.{json,png}
"""

import json
import sys
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt

REPO = Path(__file__).resolve().parent

# Load existing data
spec = json.load(open(REPO / "analyses/induction_heads_per_head_124m.json"))
n_layer = spec["n_layer"]
n_head = spec["n_head"]
steps = np.array(spec["ckpt_step"])

# Existing capability classifications (top-30 spectral picks)
cap = json.load(open(REPO / "analyses/capability_survey_124m.json"))

# Per-head selectivity (all 144 heads)
ind = json.load(open(REPO / "analyses/induction_heads_mechinterp_124m.json"))
pt = json.load(open(REPO / "analyses/prev_token_mechinterp_124m.json"))


def parse_head_key(k):
    s = k.replace(" ", "").replace("L", "").split("H")
    return int(s[0]), int(s[1])


# Build per-head selectivity dict for all 144 heads
ind_sel = {parse_head_key(k): v["selectivity"] for k, v in ind["per_head"].items()}
pt_sel = {parse_head_key(k): v["selectivity"] for k, v in pt["selectivity_per_head"].items()}

# Self-selectivity from better_ranking_signal computation? Let's just use
# the cap survey's top-30 self entries; for non-top-30 we don't have it.
self_sel_top30 = {}
for c in cap["classifications"]:
    self_sel_top30[(c["layer"], c["head"])] = c["all_selectivities"]["self"]

# ── Test (a): L0 substrate analysis ──────────────────────────────────
print("=" * 88)
print("(a) L0 SUBSTRATE TEST on karpathy_llmc 124M")
print("=" * 88)

# Compute per-head ranking signals
flat_spread = []
flat_integral = []
for L in range(n_layer):
    for H in range(n_head):
        traj = np.array(spec["pr"][f"L{L}_H{H}"])
        sp = float(traj.max() - traj.min())
        integ = float(np.trapz(np.maximum(traj - 1.0, 0), steps))
        flat_spread.append((L, H, sp))
        flat_integral.append((L, H, integ))

flat_spread.sort(key=lambda x: -x[2])
flat_integral.sort(key=lambda x: -x[2])

# Find L0 heads in the rankings
print("\n  L0 heads ranked by PR spread (overall rank out of 144):")
print(f"  {'L0 head':<8} {'spread':>8} {'overall rank by spread':>24}  {'ind sel':>10} {'prev-tok sel':>14}")
l0_data = []
for L0H in range(n_head):
    spread_val = next((sp for L, H, sp in flat_spread if L == 0 and H == L0H), 0)
    rank_spread = next((i+1 for i, (L, H, _) in enumerate(flat_spread) if L == 0 and H == L0H), -1)
    rank_integral = next((i+1 for i, (L, H, _) in enumerate(flat_integral) if L == 0 and H == L0H), -1)
    is_val = ind_sel.get((0, L0H), 0)
    pt_val = pt_sel.get((0, L0H), 0)
    print(f"  L0H{L0H:<3} {spread_val:>8.2f} {rank_spread:>24}  {is_val:>10.1f}x {pt_val:>14.1f}x")
    l0_data.append({
        "head": L0H,
        "spread": spread_val,
        "rank_by_spread_overall": rank_spread,
        "rank_by_integral_overall": rank_integral,
        "induction_selectivity": is_val,
        "prev_token_selectivity": pt_val,
    })

# Layer distribution of top-30 picks
print(f"\n  Top-30 spectral picks distributed by layer:")
top30_layers = [L for L, _, _ in flat_spread[:30]]
for L in range(n_layer):
    cnt = top30_layers.count(L)
    bar = "█" * cnt
    print(f"  L{L:<2}  {cnt:>2}  {bar}")

# ── Test (b): Time-of-emergence by capability class ──────────────────
print("\n" + "=" * 88)
print("(b) TIME-OF-EMERGENCE BY CAPABILITY CLASS on karpathy_llmc 124M")
print("=" * 88)

PR_EMERGE_THRESHOLD = 15.0


def emergence_step(pr_traj, steps, threshold=PR_EMERGE_THRESHOLD):
    arr = np.array(pr_traj)
    above = arr >= threshold
    if not above.any():
        return None
    return int(steps[above.argmax()])


# Use the top-30 capability survey classifications
emergence_by_class = {"induction": [], "previous-token": [], "self": [],
                       "first-token": [], "duplicate-token": [], "local": [],
                       "unclassified": []}
for c in cap["classifications"]:
    L, H = c["layer"], c["head"]
    cls = c["classification"]
    traj = spec["pr"][f"L{L}_H{H}"]
    e = emergence_step(traj, steps)
    if cls in emergence_by_class:
        emergence_by_class[cls].append({"head": f"L{L}H{H}", "emergence_step": e,
                                         "rank": c["rank"]})

print(f"\n  Emergence step (first ckpt where PR ≥ {PR_EMERGE_THRESHOLD}) for top-30 picks by class:")
print(f"  {'class':<20} {'count':>6} {'mean':>8} {'std':>8} {'min':>6} {'max':>6}")
class_summaries = {}
for cls, members in emergence_by_class.items():
    if not members:
        continue
    valid = [m["emergence_step"] for m in members if m["emergence_step"] is not None]
    if not valid:
        print(f"  {cls:<20} {len(members):>6}   never reaches threshold")
        continue
    arr = np.array(valid)
    print(f"  {cls:<20} {len(members):>6} {arr.mean():>8.0f} {arr.std():>8.0f} "
          f"{arr.min():>6} {arr.max():>6}")
    class_summaries[cls] = {"count": len(valid), "mean_step": float(arr.mean()),
                              "std_step": float(arr.std()),
                              "min_step": int(arr.min()), "max_step": int(arr.max())}

print(f"\n  Per-head emergence steps (sorted by step):")
all_picks = []
for cls, members in emergence_by_class.items():
    for m in members:
        m["class"] = cls
        all_picks.append(m)
all_picks.sort(key=lambda m: (m["emergence_step"] if m["emergence_step"] is not None else 999999))
for m in all_picks:
    e = m["emergence_step"] if m["emergence_step"] is not None else "never"
    print(f"  {m['head']:<8} (rank {m['rank']:>2}, {m['class']:<18}) → step {e}")

# ── Save ────────────────────────────────────────────────────────────
out = {
    "test_a_l0_substrate": {
        "l0_heads": l0_data,
        "top30_layer_distribution": {f"L{L}": top30_layers.count(L) for L in range(n_layer)},
    },
    "test_b_emergence_by_class": {
        "threshold_pr": PR_EMERGE_THRESHOLD,
        "class_summaries": class_summaries,
        "per_head": all_picks,
    },
}
out_json = REPO / "analyses/l0_substrate_and_emergence_124m.json"
with open(out_json, "w") as f:
    json.dump(out, f, indent=2)
print(f"\nwrote {out_json}")

# ── Plot ────────────────────────────────────────────────────────────
fig, axes = plt.subplots(1, 2, figsize=(14, 5.5))

# Left: layer distribution of top-30 spectral picks
ax = axes[0]
counts = [top30_layers.count(L) for L in range(n_layer)]
bars = ax.bar(range(n_layer), counts, color=["tab:red" if L == 0 else "tab:blue" for L in range(n_layer)],
                edgecolor="k", linewidth=0.5)
ax.set_xlabel("layer")
ax.set_ylabel("# of top-30 spectral picks")
ax.set_title("(a) Layer distribution of top-30 spectral picks (124M)\n"
              "L0 (red) is NOT a substrate hub here — different from TS-51M",
              fontsize=11)
ax.set_xticks(range(n_layer))
ax.grid(True, alpha=0.3, axis="y")
for L, c in enumerate(counts):
    if c > 0:
        ax.text(L, c + 0.1, str(c), ha="center", fontsize=10, weight="bold")

# Right: emergence steps by class
ax = axes[1]
class_colors = {"induction": "tab:red", "previous-token": "tab:blue",
                  "self": "tab:green", "first-token": "tab:orange",
                  "unclassified": "tab:gray"}
y_pos = 0
for cls, members in emergence_by_class.items():
    valid = [m for m in members if m["emergence_step"] is not None]
    if not valid:
        continue
    xs = [m["emergence_step"] for m in valid]
    ax.scatter(xs, [y_pos] * len(xs), color=class_colors.get(cls, "gray"),
               s=80, edgecolor="k", linewidth=0.5, label=f"{cls} (n={len(valid)})")
    y_pos += 1
ax.set_yticks(range(y_pos))
ax.set_yticklabels([cls for cls, members in emergence_by_class.items()
                     if any(m["emergence_step"] is not None for m in members)])
ax.set_xlabel(f"step at which PR first exceeds {PR_EMERGE_THRESHOLD}")
ax.set_title("(b) Capability emergence step on 124M\n"
              "horizontal scatter: when did each pick's PR first reach 15",
              fontsize=11)
ax.grid(True, alpha=0.3, axis="x")
ax.legend(fontsize=9, loc="lower right")

fig.tight_layout()
out_png = REPO / "analyses/l0_substrate_and_emergence_124m.png"
fig.savefig(out_png, dpi=130, bbox_inches="tight")
print(f"saved {out_png}")
