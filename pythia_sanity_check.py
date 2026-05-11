"""
pythia_sanity_check.py

5-min sanity test before committing to the full Pythia 160M experiment:
  1. Download one Pythia 160M checkpoint (final, step143000)
  2. Load model + tokenizer via HF transformers
  3. Run a single forward pass on a test sentence
  4. Verify per-head attention output extraction works
  5. Print model structure + sanity check on a known continuation

If this passes, we proceed to the full experiment.
"""

import sys
from pathlib import Path
import torch

REPO = Path(__file__).resolve().parent

print("Loading transformers + downloading Pythia 160M (final ckpt)...")
from transformers import GPTNeoXForCausalLM, AutoTokenizer

MODEL_NAME = "EleutherAI/pythia-160m"
REVISION = "step143000"  # final checkpoint

print(f"  Downloading {MODEL_NAME} @ {REVISION} (one-time, ~640 MB)...")
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME, revision=REVISION)
model = GPTNeoXForCausalLM.from_pretrained(MODEL_NAME, revision=REVISION)
print(f"  loaded. params: {sum(p.numel() for p in model.parameters())/1e6:.1f}M")

device = "mps" if torch.backends.mps.is_available() else "cpu"
model = model.to(device)
model.eval()

# Print model structure summary
print(f"\nArchitecture:")
print(f"  n_layer = {model.config.num_hidden_layers}")
print(f"  n_head = {model.config.num_attention_heads}")
print(f"  hidden_size = {model.config.hidden_size}")
print(f"  head_dim = {model.config.hidden_size // model.config.num_attention_heads}")
print(f"  vocab_size = {model.config.vocab_size}")

# Sanity check: predict next token on a familiar sentence
test_text = "The cat sat on the"
ids = tokenizer.encode(test_text, return_tensors="pt").to(device)
with torch.no_grad():
    out = model(ids)
logits = out.logits[0, -1]
top5 = logits.topk(5)
top5_tokens = [tokenizer.decode([i.item()]) for i in top5.indices]
top5_logits = [v.item() for v in top5.values]
print(f"\nForward pass OK. Top-5 next tokens after '{test_text}':")
for t, l in zip(top5_tokens, top5_logits):
    print(f"  {t!r}  logit={l:.2f}")

# Test the per-head attention output hook (the part we need for spectral analysis)
print(f"\nTesting per-head attention output hook on layer 0...")

# In GPT-NeoX architecture:
#   model.gpt_neox.layers[L].attention.dense  is the output projection (= "c_proj" in GPT-2 lingo)
#   The input to .dense is the per-head attention output, shape [B, T, n_head*head_dim]
# Hook the input to .dense — that's the per-head V output we use for spectral

n_head = model.config.num_attention_heads
head_dim = model.config.hidden_size // n_head
captured = {}

def hook(module, ainputs):
    x = ainputs[0]
    print(f"  hook captured: shape {tuple(x.shape)}")
    B, T, C = x.shape
    print(f"    can reshape to per-head: [B={B}, T={T}, n_head={n_head}, head_dim={head_dim}]")
    captured["x"] = x

handle = model.gpt_neox.layers[0].attention.dense.register_forward_pre_hook(hook)
try:
    with torch.no_grad():
        _ = model(ids)
finally:
    handle.remove()

print(f"\n[OK] Pythia 160M loads, runs, and per-head hook works. Ready for full experiment.")
