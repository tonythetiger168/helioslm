"""Self-test for MLA weight absorption (helioslm v5.1 DEFERRED item).

Run from /mnt/agents/output:  python selftest_absorption.py

Checks:
 1. Same MLA instance, absorbed vs non-absorbed: prefill + 3 decode steps,
    outputs match at atol 1e-4.
 2. Absorbed cached decode == full-sequence forward (cache correctness).
 3. Cache memory per token: absorbed vs expanded vs MHA baseline.
 4. dim-2 cache truncation (MTP rollback simulation) keeps decoding.
 5. Full lite model with absorption ON: forward triple, deterministic
    batch=2 greedy, use_mtp=True == greedy.
"""
import sys
import torch

from helioslm_v5.configs.config_v5 import HeliosLMv5Config
from helioslm_v5.src.attention.mla import MLA
from helioslm_v5.src.model_v5 import HeliosLMv5

torch.manual_seed(0)
ok = True


def check(name, cond, detail=""):
    global ok
    print(f"[{'PASS' if cond else 'FAIL'}] {name}" + (f" — {detail}" if detail else ""))
    ok = ok and cond


# ---------------------------------------------------------------- 1 & 2
config = HeliosLMv5Config(size="lite")
assert config.attention.use_absorption is True, "lite default must be ON"
torch.manual_seed(7)
mla = MLA(config).eval()
B, L = 2, 8
h = torch.randn(B, L, config.hidden_size)
toks = [torch.randn(B, 1, config.hidden_size) for _ in range(3)]


def run(mode):
    mla.use_absorption = mode
    with torch.no_grad():
        out, past = mla(h, use_cache=True)
        for t in toks:
            o, past = mla(t, past_key_value=past, use_cache=True)
            out = torch.cat([out, o], dim=1)
    return out, past


out_abs, past_abs = run(True)
out_exp, past_exp = run(False)
d1 = (out_abs - out_exp).abs().max().item()
check("1. absorbed vs non-absorbed (prefill+3 decode)", d1 < 1e-4,
      f"max|diff|={d1:.3e}")

# 2. absorbed cached decode vs full forward over the whole sequence
mla.use_absorption = True
with torch.no_grad():
    full, _ = mla(torch.cat([h] + toks, dim=1), use_cache=False)
d2 = (full - out_abs).abs().max().item()
check("2. absorbed cached decode == full forward", d2 < 1e-4,
      f"max|diff|={d2:.3e}")

# ---------------------------------------------------------------- 3
a = config.attention
per_abs = a.kv_latent_dim + a.rope_head_dim
per_exp = a.num_attention_heads * (a.no_rope_head_dim + a.v_head_dim) + a.rope_head_dim
per_mha = 2 * a.num_attention_heads * a.head_dim
actual_abs = sum(t.shape[1] * t.shape[3] for t in past_abs)
actual_exp = sum(t.shape[1] * t.shape[3] for t in past_exp)
info = mla.get_kv_cache_size(1)
print(f"    per-token values: absorbed={actual_abs} expanded={actual_exp} "
      f"MHA={per_mha} | saving vs MHA: absorbed "
      f"{(1 - actual_abs / per_mha) * 100:.1f}%, expanded "
      f"{(1 - actual_exp / per_mha) * 100:.1f}%")
check("3. cache memory: absorbed << expanded << MHA",
      actual_abs == per_abs and actual_exp == per_exp
      and info["mla"] == per_abs and actual_abs < actual_exp < per_mha,
      f"absorbed/expanded = {actual_abs / actual_exp:.2%} of expanded, "
      f"{actual_abs / per_mha:.2%} of MHA")

# ---------------------------------------------------------------- 4
keep = L + 1  # truncate away the last 2 decode steps
trunc_a = tuple(t[:, :, :keep] for t in past_abs)
trunc_e = tuple(t[:, :, :keep] for t in past_exp)
new_tok = torch.randn(B, 1, config.hidden_size)
with torch.no_grad():
    mla.use_absorption = True
    o_ta, p_ta = mla(new_tok, past_key_value=trunc_a, use_cache=True)
    mla.use_absorption = False
    o_te, _ = mla(new_tok, past_key_value=trunc_e, use_cache=True)
d4 = (o_ta - o_te).abs().max().item()
check("4. dim-2 truncated cache decode (MTP rollback)",
      d4 < 1e-4 and p_ta[0].shape[2] == keep + 1 and torch.isfinite(o_ta).all(),
      f"post-truncation cross-mode max|diff|={d4:.3e}, cache len {p_ta[0].shape[2]}")

# ---------------------------------------------------------------- 5
torch.manual_seed(42)
model = HeliosLMv5(HeliosLMv5Config(size="lite")).eval()
assert model.config.attention.use_absorption
ids = torch.randint(3, model.config.vocab_size, (2, 8))
with torch.no_grad():
    logits, hidden, past = model(ids, use_cache=True)
triple_ok = (logits.shape == (2, 8, model.config.vocab_size)
             and hidden.shape == (2, 8, model.config.hidden_size)
             and len(past) == model.config.num_hidden_layers
             and len(past[0]) == 2 and torch.isfinite(logits).all())
g1 = model.generate(ids, max_new_tokens=5, temperature=0)
g2 = model.generate(ids, max_new_tokens=5, temperature=0)
det = torch.equal(g1, g2)
plain = model.generate(ids[:1], max_new_tokens=6, temperature=0)
mtp = model.generate(ids[:1], max_new_tokens=6, temperature=0, use_mtp=True)
n = mtp.shape[1]
mtp_ok = n <= plain.shape[1] and torch.equal(plain[:, :n], mtp)
check("5. lite model w/ absorption: triple + greedy det + MTP==greedy",
      triple_ok and det and mtp_ok,
      f"triple_ok={triple_ok} deterministic={det} mtp==greedy={mtp_ok} (len {n})")

print("=" * 60)
print("ALL PASS" if ok else "SOME CHECKS FAILED")
sys.exit(0 if ok else 1)
