"""T3: cache/generation consistency.
- cached incremental decode vs full-sequence forward logits, various prompt lens
- MTP (batch 1/2/4) greedy == non-MTP greedy
- MTP sampling vs non-MTP sampling distribution coarse check
"""
import sys
sys.path.insert(0, "/mnt/agents/output")
import torch
from helioslm_v5.configs.config_v5 import HeliosLMv5Config
from helioslm_v5.src.model_v5 import HeliosLMv5

torch.manual_seed(0)
cfg = HeliosLMv5Config(size="lite")
model = HeliosLMv5(cfg).eval()

def banner(s): print(f"\n=== {s} ===")

# ---------- cached vs full-sequence logits ----------
banner("cached decode vs full forward (prompt len 1,2,7,64 x decode 8)")
for plen in (1, 2, 7, 64):
    torch.manual_seed(plen)
    ids = torch.randint(0, 1024, (1, plen))
    # full forward
    with torch.no_grad():
        logits_full, _, _ = model(ids)
        # incremental: prefill prompt[:1]? do prefill of whole prompt then decode tokens one by one from a continuation
        cont = torch.randint(0, 1024, (1, 8))
        full = torch.cat([ids, cont], dim=1)
        logits_ff, _, _ = model(full)
        # cached path: prefill prompt, then feed cont token-by-token
        _, _, past = model(ids, use_cache=True)
        cached_logits = []
        for t in range(cont.shape[1]):
            lg, _, past = model(cont[:, t:t+1], past_key_values=past, use_cache=True)
            cached_logits.append(lg[:, -1])
        cached = torch.stack(cached_logits, dim=1)  # [1, 8, V]
        d = (cached - logits_ff[:, plen:]).abs().max().item()
    print(f"prompt={plen:3d}: max|cached-full| = {d:.3e} {'OK' if d < 1e-4 else 'FAIL'}")

# also with partial prefill: prefill prompt[:k], decode rest of prompt + cont
banner("split prefill (prefill k of prompt, decode remainder)")
for plen, k in [(7, 1), (7, 3), (64, 63)]:
    torch.manual_seed(100 + plen + k)
    ids = torch.randint(0, 1024, (1, plen))
    with torch.no_grad():
        logits_ff, _, _ = model(ids)
        _, _, past = model(ids[:, :k], use_cache=True)
        outs = []
        for t in range(k, plen):
            lg, _, past = model(ids[:, t:t+1], past_key_values=past, use_cache=True)
            outs.append(lg[:, -1])
        d = (torch.stack(outs, 1) - logits_ff[:, k:]).abs().max().item()
    print(f"plen={plen} prefill={k}: max diff {d:.3e} {'OK' if d < 1e-4 else 'FAIL'}")

# ---------- MTP greedy == non-MTP greedy ----------
banner("MTP greedy equivalence, batch 1/2/4")
for B in (1, 2, 4):
    torch.manual_seed(10 + B)
    ids = torch.randint(0, 1024, (B, 7))
    ref = model.generate(ids, max_new_tokens=12, temperature=0.0)
    mtp = model.generate(ids, max_new_tokens=12, temperature=0.0, use_mtp=True)
    eq = torch.equal(ref, mtp)
    print(f"B={B}: equal={eq} shapes {tuple(ref.shape)}/{tuple(mtp.shape)}")
    if not eq:
        for i in range(B):
            if not torch.equal(ref[i], mtp[i]):
                print(f"  row {i}: ref {ref[i].tolist()}\n         mtp {mtp[i].tolist()}")

# MTP greedy with len-1 prompt
torch.manual_seed(99)
ids = torch.randint(0, 1024, (2, 1))
ref = model.generate(ids, max_new_tokens=8, temperature=0.0)
mtp = model.generate(ids, max_new_tokens=8, temperature=0.0, use_mtp=True)
print("len-1 prompt MTP greedy equal:", torch.equal(ref, mtp))

# ---------- MTP sampling distribution vs non-MTP (coarse) ----------
banner("MTP sampling distribution ~ non-MTP (temperature=1.0, first 3 new tokens)")
from collections import Counter
PROMPT = torch.tensor([[11, 22, 33, 44]])
def collect(use_mtp, n):
    c = [Counter(), Counter(), Counter()]
    for s in range(n):
        torch.manual_seed(10_000 + s)
        out = model.generate(PROMPT, max_new_tokens=3, temperature=1.0, top_p=1.0,
                             use_mtp=use_mtp)
        if out.shape[1] < PROMPT.shape[1] + 3:
            continue  # early EOS stop; skip
        for j in range(3):
            c[j][int(out[0, PROMPT.shape[1] + j])] += 1
    return c
N = 400
c_ref = collect(False, N)
c_mtp = collect(True, N)
for j in range(3):
    keys = set(c_ref[j]) | set(c_mtp[j])
    l1 = sum(abs(c_ref[j][k] - c_mtp[j][k]) for k in keys) / (2 * N)  # TV distance
    print(f"pos {j}: TV distance = {l1:.4f} (n={N}) {'OK' if l1 < 0.10 else 'SUSPECT'}")
print("DONE")
