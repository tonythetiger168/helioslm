"""T1: boundary inputs + extreme sampling params on HeliosLMv5 lite."""
import sys, traceback
sys.path.insert(0, "/mnt/agents/output")
import torch
from helioslm_v5.configs.config_v5 import HeliosLMv5Config
from helioslm_v5.src.model_v5 import HeliosLMv5

torch.manual_seed(0)
cfg = HeliosLMv5Config(size="lite")
model = HeliosLMv5(cfg).eval()
V = cfg.vocab_size

results = []
def check(name, fn):
    try:
        r = fn()
        results.append((name, "OK", str(r)))
        print(f"[OK] {name}: {r}")
    except Exception as e:
        results.append((name, "EXC", f"{type(e).__name__}: {e}"))
        print(f"[EXC] {name}: {type(e).__name__}: {e}")

# 1. empty sequence forward
def t_empty_forward():
    logits, h, past = model(torch.zeros(1, 0, dtype=torch.long))
    return f"logits {tuple(logits.shape)} finite={torch.isfinite(logits).all().item() if logits.numel() else 'n/a'}"
check("forward empty seq [1,0]", t_empty_forward)

# 2. generate with empty prompt
def t_gen_empty():
    return model.generate(torch.zeros(1, 0, dtype=torch.long), max_new_tokens=2)
check("generate empty prompt", t_gen_empty)

# 3. length-1 prompt
def t_len1():
    out = model.generate(torch.tensor([[5]]), max_new_tokens=4, temperature=0.0)
    return f"out {tuple(out.shape)}"
check("generate len-1 prompt", t_len1)

# 4. max_new_tokens=0
def t_mn0():
    ids = torch.tensor([[1, 2, 3]])
    out = model.generate(ids, max_new_tokens=0)
    assert torch.equal(out, ids), f"expected unchanged, got {out.shape}"
    return "unchanged"
check("generate max_new=0", t_mn0)

# 5. max_new_tokens=1
def t_mn1():
    out = model.generate(torch.tensor([[1, 2, 3]]), max_new_tokens=1, temperature=0.0)
    assert out.shape == (1, 4)
    return "shape ok"
check("generate max_new=1", t_mn1)

# 6. all-same token
def t_same():
    out = model.generate(torch.full((2, 6), 7), max_new_tokens=5, temperature=0.0)
    assert out.shape == (2, 11) and torch.isfinite(out.float()).all()
    return "ok"
check("generate all-same-token", t_same)

# 7. vocab boundary ids
def t_vocab_edge():
    out = model.generate(torch.tensor([[0], [V - 1]]), max_new_tokens=3, temperature=0.0)
    return f"ok {tuple(out.shape)}"
check("generate vocab boundary ids (0, V-1)", t_vocab_edge)

# 8. illegal id >= vocab
def t_bad_id():
    model.generate(torch.tensor([[V]]), max_new_tokens=1)
check("generate id >= vocab (expect loud error)", t_bad_id)

# 9. negative id
def t_neg_id():
    model.generate(torch.tensor([[-1]]), max_new_tokens=1)
check("generate negative id (expect loud error)", t_neg_id)

# 10. extreme temperature 1e-8 (should behave ~greedy, must not NaN/crash)
def t_temp_tiny():
    torch.manual_seed(1)
    out = model.generate(torch.tensor([[3, 4, 5]]), max_new_tokens=5, temperature=1e-8, top_p=1.0)
    g = model.generate(torch.tensor([[3, 4, 5]]), max_new_tokens=5, temperature=0.0)
    return f"finite={torch.isfinite(out.float()).all().item()} matches_greedy={torch.equal(out, g)}"
check("generate temperature=1e-8", t_temp_tiny)

# 11. temperature=100
def t_temp_100():
    torch.manual_seed(1)
    out = model.generate(torch.tensor([[3, 4, 5]]), max_new_tokens=5, temperature=100.0, top_p=1.0)
    return f"finite={torch.isfinite(out.float()).all().item()}"
check("generate temperature=100", t_temp_100)

# 12. top_p=0 (must keep top token -> greedy-like)
def t_topp0():
    torch.manual_seed(1)
    out = model.generate(torch.tensor([[3, 4, 5]]), max_new_tokens=5, temperature=0.8, top_p=0.0)
    g = model.generate(torch.tensor([[3, 4, 5]]), max_new_tokens=5, temperature=0.0)
    return f"finite={torch.isfinite(out.float()).all().item()} matches_greedy={torch.equal(out, g)}"
check("generate top_p=0", t_topp0)

# 13. top_p=1.0 sampling
def t_topp1():
    torch.manual_seed(1)
    out = model.generate(torch.tensor([[3, 4, 5]]), max_new_tokens=5, temperature=0.8, top_p=1.0)
    return f"finite={torch.isfinite(out.float()).all().item()}"
check("generate top_p=1.0", t_topp1)

# 14. forward with seq longer than... prefill 64 then decode consistency done in T3.
# 15. BOS=EOS corner: prompt that is only EOS token
def t_prompt_is_eos():
    out = model.generate(torch.tensor([[cfg.eos_token_id]]), max_new_tokens=4, temperature=0.0)
    return f"out len {out.shape[1]} (prompt 1 + up to 4)"
check("generate prompt=[EOS]", t_prompt_is_eos)

print("\n=== SUMMARY ===")
for n, s, d in results:
    print(f"{s:4s} {n}: {d[:120]}")
