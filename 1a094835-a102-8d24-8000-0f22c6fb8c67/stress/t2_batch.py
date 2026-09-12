"""T2: batch boundary tests (batch=1 vs 8, all-EOS-immediate batch, etc.)."""
import sys
sys.path.insert(0, "/mnt/agents/output")
import torch
from helioslm_v5.configs.config_v5 import HeliosLMv5Config
from helioslm_v5.src.model_v5 import HeliosLMv5

torch.manual_seed(0)
cfg = HeliosLMv5Config(size="lite")
model = HeliosLMv5(cfg).eval()

def banner(s): print(f"\n=== {s} ===")

# --- batch invariance: row of a batched forward == solo forward ---
banner("batch invariance of forward")
ids = torch.randint(0, 1024, (8, 9))
with torch.no_grad():
    logits_b, _, past_b = model(ids, use_cache=True)
    max_diff = 0.0
    for i in range(8):
        logits_1, _, _ = model(ids[i:i+1], use_cache=True)
        d = (logits_b[i] - logits_1[0]).abs().max().item()
        max_diff = max(max_diff, d)
print(f"max |batched - solo| logit diff over 8 rows: {max_diff:.3e}")
assert max_diff < 1e-5

# --- batched generate batch=1 vs batch=8 greedy equality (same rows) ---
banner("generate batch=1 vs batch=8")
torch.manual_seed(2)
ids = torch.randint(0, 1024, (8, 9))
out8 = model.generate(ids, max_new_tokens=8, temperature=0.0)
same = True
for i in range(8):
    out1 = model.generate(ids[i:i+1], max_new_tokens=8, temperature=0.0)
    if not torch.equal(out8[i:i+1], out1):
        same = False
        print(f"row {i} DIFFERS:\n  batch: {out8[i].tolist()}\n  solo : {out1[0].tolist()}")
print("batched greedy == solo greedy:", same)

# --- batch with per-row attention mask (left padding) ---
banner("generate with left-padded batch (attention_mask)")
prompts = [[5, 6, 7], [9]]
pad = cfg.pad_token_id
maxlen = max(len(p) for p in prompts)
ids = torch.tensor([[pad] * (maxlen - len(p)) + p for p in prompts])
mask = torch.tensor([[0] * (maxlen - len(p)) + [1] * len(p) for p in prompts])
out = model.generate(ids, max_new_tokens=4, temperature=0.0, attention_mask=mask)
solo = model.generate(torch.tensor([[9]]), max_new_tokens=4, temperature=0.0)
print("padded row tokens:", out[1].tolist())
print("solo row tokens:  ", solo[0].tolist())
print("padded row matches solo:", out[1, maxlen:].tolist() == solo[0, 1:].tolist())

# --- all-EOS-immediate batch via forced-EOS forward patch (test-only) ---
banner("all rows emit EOS on first step (forced)")
orig_forward = model.forward
def forced_forward(input_ids, **kw):
    logits, h, past = orig_forward(input_ids, **kw)
    logits = logits.clone()
    logits[..., cfg.eos_token_id] = 1e6   # force argmax/multinomial -> EOS
    return logits, h, past
model.forward = forced_forward
ids = torch.randint(0, 1024, (4, 5))
out = model.generate(ids, max_new_tokens=10, temperature=0.0)
newpart = out[:, 5:]
print("generated part:", newpart.tolist())
ok = bool((newpart[:, 0] == cfg.eos_token_id).all()) and bool((newpart[:, 1:] == cfg.pad_token_id).all())
print("all rows: first token EOS then frozen PAD:", ok)
# MTP path with forced EOS
try:
    out_mtp = model.generate(ids, max_new_tokens=10, temperature=0.0, use_mtp=True)
    print("MTP forced-EOS out:", out_mtp[:, 5:].tolist())
except Exception as e:
    print(f"MTP forced-EOS EXC: {type(e).__name__}: {e}")
model.forward = orig_forward

# --- batch=8 sampling with extreme temps ---
banner("batch=8 extreme sampling")
torch.manual_seed(3)
ids = torch.randint(0, 1024, (8, 6))
for temp, tp in [(1e-8, 1.0), (100.0, 1.0), (0.7, 0.0), (0.7, 0.5)]:
    out = model.generate(ids, max_new_tokens=4, temperature=temp, top_p=tp)
    print(f"temp={temp} top_p={tp}: shape {tuple(out.shape)} finite={torch.isfinite(out.float()).all().item()}")
print("DONE")
