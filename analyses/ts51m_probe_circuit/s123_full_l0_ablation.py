"""Quick test: full-L0 ablation on s123 at step 10000."""
import sys, json
from pathlib import Path
import torch
from torch.utils.data import DataLoader

REPO = Path(__file__).resolve().parent
sys.path.insert(0, str(REPO / "training"))
from config import Config, get_device
from model import GPTModel
from dataset import build_datasets
from pilot import evaluate_probe, evaluate_lm

S123_DIR = REPO / "runs/beta2_ablation/pilot_wd0.5_lr0.001_lp2.0_b20.95_s123"


def make_pre_hook(ablated_heads, n_head, head_dim):
    def pre_hook(_module, ainputs):
        x = ainputs[0]
        B, T, C = x.shape
        xr = x.view(B, T, n_head, head_dim).clone()
        for h in ablated_heads:
            xr[:, :, h, :] = 0.0
        return (xr.view(B, T, C),)
    return pre_hook


def main():
    device = get_device()
    cfg = Config(seed=42, p_probe=0.10, batch_size=64,
                  n_layer=8, d_model=512, n_head=16, d_ff=2048)
    n_head = cfg.n_head
    head_dim = cfg.d_model // n_head

    cw_path = S123_DIR / "codewords.json"
    data = build_datasets(cfg, codewords_path=str(cw_path))
    vocab_size = len(data["tokenizer"])
    val_loader = DataLoader(data["val_dataset"], batch_size=64,
                              shuffle=False, drop_last=False, num_workers=0)
    probe_in = data["probe_eval_in"]
    probe_ood = data["probe_eval_ood"]

    model = GPTModel(
        vocab_size=vocab_size, seq_len=cfg.seq_len,
        d_model=cfg.d_model, n_layer=cfg.n_layer,
        n_head=cfg.n_head, d_ff=cfg.d_ff, dropout=0.0,
    ).to(device)

    ck = torch.load(S123_DIR / "ckpt_010000.pt", map_location=device, weights_only=True)
    model.load_state_dict(ck["model_state_dict"])
    model.eval()

    conditions = [
        ("baseline", []),
        ("ablate_L0H{3,6,14,15} (s42 picks)", [3, 6, 14, 15]),
        ("ablate_full_L0 (all 16)", list(range(16))),
        ("ablate_random_4_L0 (control: H{0,1,5,7})", [0, 1, 5, 7]),
    ]

    results = {}
    print(f"  {'condition':<48} {'pin':>8} {'pood':>8} {'val':>8}")
    for name, heads in conditions:
        if heads:
            handle = model.blocks[0].attn.out_proj.register_forward_pre_hook(
                make_pre_hook(heads, n_head, head_dim)
            )
        try:
            pin = evaluate_probe(model, probe_in, device)
            pood = evaluate_probe(model, probe_ood, device)
            vl = evaluate_lm(model, val_loader, device)
            print(f"  {name:<48} {pin:>8.4f} {pood:>8.4f} {vl:>8.4f}")
            results[name] = {"pin": pin, "pood": pood, "val": vl}
        finally:
            if heads:
                handle.remove()

    out_json = REPO / "analyses/s123_full_l0_ablation.json"
    with open(out_json, "w") as f:
        json.dump({"step": 10000, "results": results}, f, indent=2)
    print(f"\nwrote {out_json}")


if __name__ == "__main__":
    main()
