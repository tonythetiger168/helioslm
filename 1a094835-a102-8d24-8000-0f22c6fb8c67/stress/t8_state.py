"""T8: state-leakage tests between consecutive generates/forwards."""
import sys
sys.path.insert(0, "/mnt/agents/output")
import torch
from helioslm_v5.configs.config_v5 import HeliosLMv5Config
from helioslm_v5.src.model_v5 import HeliosLMv5

def banner(s): print(f"\n=== {s} ===")

torch.manual_seed(0)
cfg = HeliosLMv5Config(size="lite")
model = HeliosLMv5(cfg)

banner("1. generate() flips model to eval and never restores (M-T1 known)")
model.train()
assert model.training
model.generate(torch.tensor([[1, 2, 3]]), max_new_tokens=2, temperature=0.0)
print("model.training after generate:", model.training, "(leak if False and caller had train mode)")

banner("2. second generate identical to first (greedy, no cache residue)")
model.eval()
ids = torch.tensor([[3, 4, 5], [6, 7, 8]])
torch.manual_seed(42)
a = model.generate(ids, max_new_tokens=6, temperature=0.0)
torch.manual_seed(42)
b = model.generate(ids, max_new_tokens=6, temperature=0.0)
print("repeat greedy identical:", torch.equal(a, b))
# interleave different batch in between
c = model.generate(torch.tensor([[100]]*3), max_new_tokens=6, temperature=0.0)
d = model.generate(ids, max_new_tokens=6, temperature=0.0)
print("after interleaved generate, still identical:", torch.equal(a, d))

banner("3. MoE expert_load polluted by training forward but not eval/generate")
moe = model.layers[0].moe
moe.expert_load.zero_()
model.train()
x = torch.randint(0, 1024, (2, 5))
out = model(x)  # no_grad not set: training forward
load_after_train = moe.expert_load.sum().item()
model.eval()
with torch.no_grad():
    model(x)
load_after_eval = moe.expert_load.sum().item()
model.generate(torch.tensor([[1, 2]]), max_new_tokens=3, temperature=0.0)
load_after_gen = moe.expert_load.sum().item()
print(f"load train={load_after_train} eval={load_after_eval} gen={load_after_gen}",
      "OK" if load_after_eval == load_after_train == load_after_gen else "note")

banner("4. MTP generate leaves model/mtp in eval")
model.train()
model.generate(torch.tensor([[1, 2, 3]]), max_new_tokens=2, temperature=0.0, use_mtp=True)
print("after MTP generate: model.training =", model.training,
      " mtp.training =", model.mtp_modules[0].training)

banner("5. sampling generate twice with same seed -> identical (no hidden RNG state)")
model.eval()
ids = torch.tensor([[9, 9, 9]])
torch.manual_seed(123)
s1 = model.generate(ids, max_new_tokens=6, temperature=0.8, top_p=0.9)
torch.manual_seed(123)
s2 = model.generate(ids, max_new_tokens=6, temperature=0.8, top_p=0.9)
print("sampling reproducible:", torch.equal(s1, s2))

banner("6. MTP generate then plain generate: no cross-contamination")
torch.manual_seed(1)
p = model.generate(ids, max_new_tokens=6, temperature=0.0)
m1 = model.generate(ids, max_new_tokens=6, temperature=0.0, use_mtp=True)
p2 = model.generate(ids, max_new_tokens=6, temperature=0.0)
print("plain == MTP == plain-after:", torch.equal(p, m1) and torch.equal(p, p2))

banner("7. RoPE buffer growth is stable (no drift)")
rope = model.layers[0].attention.rope
n0 = rope.max_seq_len
model.generate(torch.randint(0, 1024, (1, 200)), max_new_tokens=2, temperature=0.0)
n1 = rope.max_seq_len
l1, _, _ = model(torch.tensor([[1, 2, 3]]))
l2, _, _ = model(torch.tensor([[1, 2, 3]]))
print(f"rope cache {n0}->{n1}; forward deterministic after growth:",
      torch.equal(l1, l2))

banner("8. multimodal: audio encoder state leaks between model.forward calls")
cfg2 = HeliosLMv5Config(size="lite")
cfg2.multimodal.enabled = True
cfg2.multimodal.vision_patch_size = 4
cfg2.multimodal.vision_hidden_size = 32
cfg2.multimodal.vision_num_layers = 1
cfg2.multimodal.vision_num_heads = 4
cfg2.multimodal.vision_max_grid = 8
cfg2.multimodal.audio_n_mels = 16
cfg2.multimodal.audio_hidden_size = 32
cfg2.multimodal.audio_num_layers = 1
torch.manual_seed(0)
mm = HeliosLMv5(cfg2).eval()
ids = torch.tensor([[3, 4, 5]])
aud = torch.randn(1, 16, 20)
with torch.no_grad():
    la, _, _ = mm(ids, audio_features=aud)
    lb, _, _ = mm(ids, audio_features=aud)  # same call again, NO reset
d = (la[:, -1] - lb[:, -1]).abs().max().item()
print(f"same audio forward twice, text logit diff: {d:.4f}",
      "STATE LEAK (audio encoder carries conv/memory state)" if d > 1e-5 else "no leak")
# vision encoder is stateless:
img = torch.randn(1, 3, 8, 8)
with torch.no_grad():
    va, _, _ = mm(ids, images=img)
    vb, _, _ = mm(ids, images=img)
dv = (va[:, -1] - vb[:, -1]).abs().max().item()
print(f"vision forward twice diff: {dv:.2e}", "OK" if dv < 1e-6 else "LEAK")
print("DONE")
