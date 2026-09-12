import sys, torch, warnings
sys.path.insert(0, "/mnt/agents/output")
torch.manual_seed(0)
from helioslm_v5.configs.config_v5 import HeliosLMv5Config
from helioslm_v5.src.attention.mla import MLA, RotaryEmbedding

cfg = HeliosLMv5Config(size="lite")
mla = MLA(cfg).eval()

# 3) RoPE persistent=False -> state_dict has no cos/sin
sd_keys = list(mla.state_dict().keys())
print("rope in state_dict:", [k for k in sd_keys if "cos_cached" in k or "sin_cached" in k])
# long-seq growth then strict load into fresh model
x = torch.randn(1, 3000, cfg.hidden_size)
with torch.no_grad():
    mla(x)
mla2 = MLA(cfg)
mla2.load_state_dict(mla.state_dict(), strict=True)
print("strict load after rope growth: OK")

# 4) non-monotonic (packed) position_ids at prefill: mask must respect position space
B, L = 1, 6
pos = torch.tensor([[0,1,2,0,1,2]])  # two packed seqs
with torch.no_grad():
    out_packed, _ = mla(torch.randn(B,L,cfg.hidden_size), position_ids=pos)
    # reference: two separate 3-token forwards
    h = torch.randn(B,L,cfg.hidden_size)
# causal check: token at packed position 0 (idx 3) must NOT attend to idx 0..2
# do it directly via mask builder
mask = mla._build_attn_mask(pos, L, None, 0, L, B, True)
row3 = mask[0,0,3]
print("packed mask row idx3 (pos 0): attends", row3.nonzero().flatten().tolist(), "(expect [3])")
row5 = mask[0,0,5]
print("packed mask row idx5 (pos 2): attends", row5.nonzero().flatten().tolist(), "(expect [3,4,5])")

# decode with non-default custom positions must raise
past = (torch.randn(1,1,4,cfg.attention.kv_latent_dim), torch.randn(1,1,4,cfg.attention.rope_head_dim))
try:
    mla(torch.randn(1,1,cfg.hidden_size), past_key_value=past, position_ids=torch.tensor([[0]]))
    print("decode custom pos: NO RAISE (bug)")
except ValueError as e:
    print("decode custom pos raises ValueError: OK")
# decode with default-equivalent custom positions must NOT raise
try:
    mla(torch.randn(1,1,cfg.hidden_size), past_key_value=past, position_ids=torch.tensor([[4]]))
    print("decode default-eq custom pos: OK no raise")
except ValueError as e:
    print("decode default-eq custom pos RAISED (bug):", e)

# negative position_ids silently wrap? 
try:
    cos, sin = mla.rope(torch.tensor([[-1, 0, 1]]))
    print("negative position_ids: no error (wraps index)")
except Exception as e:
    print("negative position_ids raises:", type(e).__name__)

# int32 position_ids at decode default-eq
try:
    mla(torch.randn(1,1,cfg.hidden_size), past_key_value=past, position_ids=torch.tensor([[4]], dtype=torch.int32))
    print("int32 position_ids decode: OK no raise")
except ValueError as e:
    print("int32 position_ids decode RAISED (spurious):", str(e)[:80])
