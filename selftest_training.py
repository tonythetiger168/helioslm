"""Self-tests for training fixes: dualpipe.py, fp8_trainer.py, grpo.py."""
import sys, copy
sys.path.insert(0, "/mnt/agents/output/helioslm_v5/src")

import torch
import torch.nn as nn
import torch.nn.functional as F
from types import SimpleNamespace

torch.manual_seed(0)
results = []

def check(name, cond, detail=""):
    results.append((name, bool(cond)))
    print(f"[{'PASS' if cond else 'FAIL'}] {name} {detail}")

# =====================================================================
# 1. DualPipe
# =====================================================================
from training.dualpipe import DualPipeStage, DualPipeScheduler, ExpertParallelism

torch.manual_seed(42)
stages = [DualPipeStage(nn.ModuleList([nn.Linear(8, 8), nn.Tanh()])) for _ in range(3)]
M = 4
inputs = [torch.randn(2, 8) for _ in range(M)]
loss_fn = lambda o: (o ** 2).mean()

sched = DualPipeScheduler(stages, num_micro_batches=M)
total = sched.run_dual([x.clone() for x in inputs], loss_fn)

# 1a. params have non-None, non-zero grads
grads_ok = all(
    p.grad is not None and p.grad.abs().sum() > 0
    for st in stages for p in st.parameters()
)
check("DualPipe: stage params have non-None nonzero grads", grads_ok)

# 1b. gradient values match direct full forward+backward (atol=1e-5)
ref_stages = [DualPipeStage(nn.ModuleList([nn.Linear(8, 8), nn.Tanh()])) for _ in range(3)]
with torch.no_grad():
    for rs, s in zip(ref_stages, stages):
        for rp, p in zip(rs.parameters(), s.parameters()):
            rp.copy_(p)
ref_losses = []
for x in inputs:
    h = x.clone()
    for st in ref_stages:
        h = st(h)
    ref_losses.append(loss_fn(h) / M)
ref_total = sum(ref_losses)
ref_total.backward()
max_diff = max(
    (p.grad - rp.grad).abs().max().item()
    for st, rst in zip(stages, ref_stages)
    for p, rp in zip(st.parameters(), rst.parameters())
)
check("DualPipe: grads match direct forward+backward", max_diff < 1e-5,
      f"max_diff={max_diff:.2e}")
check("DualPipe: loss is microbatch mean", abs(total - ref_total.item()) < 1e-5,
      f"run_dual={total:.6f} ref={ref_total.item():.6f}")

# 1c. F/B interleaving order via trace: F0, (F1,B0), (F2,B1), (F3,B2), B3
expected = [("F", 0)]
for k in range(1, M):
    expected += [("F", k), ("B", k - 1)]
expected += [("B", M - 1)]
check("DualPipe: 1F1B interleaved schedule trace", sched.trace == expected,
      f"trace={sched.trace}")

# 1d. ExpertParallelism minor fixes
try:
    ExpertParallelism(num_experts=7, num_devices=4)
    check("ExpertParallelism: non-divisible raises ValueError", False)
except ValueError:
    check("ExpertParallelism: non-divisible raises ValueError", True)
ep = ExpertParallelism(num_experts=8, num_devices=4)
routed = ep.route_to_device(torch.tensor([[0, 3], [4, 7]]))
check("ExpertParallelism: vectorized route_to_device",
      routed.tolist() == [[0, 1], [2, 3]] and routed.shape == (2, 2))
try:
    ep.all_to_all([torch.zeros(2)], 0, 1)
    check("ExpertParallelism: all_to_all raises w/o dist", False)
except NotImplementedError:
    check("ExpertParallelism: all_to_all raises w/o dist", True)

# =====================================================================
# 2. FP8Trainer
# =====================================================================
from training.fp8_trainer import FP8Trainer, FP8Linear

class TinyNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.l1 = nn.Linear(16, 8)
        self.l2 = nn.Linear(8, 4)
    def forward(self, x, targets):
        h = F.relu(self.l1(x))
        return {"loss": F.mse_loss(self.l2(h), targets)}

torch.manual_seed(1)
model = TinyNet()
config = SimpleNamespace(lr=5e-2)
trainer = FP8Trainer(model, config)

bx = torch.randn(16, 16)
by = torch.randn(16, 4)
batch = {"x": bx, "targets": by}

# 2a. loss decreasing over 3 steps
losses = [trainer.train_step(batch) for _ in range(3)]
check("FP8: loss decreasing over 3 steps", losses[2] < losses[0],
      f"losses={[round(l,4) for l in losses]}")

# 2b. grads do not accumulate across steps (tiny lr so weights ~constant)
grad_norms = []
for p in trainer.model.parameters():
    if p.grad is not None:
        p.register_hook(lambda g: grad_norms.append(g.norm().item()) or g)
config2 = SimpleNamespace(lr=1e-6)
model2 = TinyNet()
with torch.no_grad():  # same init as trainer's model start is fine; just need consistency
    pass
trainer2 = FP8Trainer(model2, config2)
p0 = next(trainer2.model.parameters())
p0.register_hook(lambda g: grad_norms.append(g.norm().item()) or g)
trainer2.train_step(batch)
trainer2.train_step(batch)
check("FP8: no cross-step grad accumulation",
      len(grad_norms) == 2 and 0.5 < grad_norms[1] / max(grad_norms[0], 1e-12) < 1.5,
      f"norms={grad_norms}")

# 2c. _quantize_to_fp8 is real: non-identity, bounded round-trip error
lin = FP8Linear(16, 8)
x = torch.randn(64, 16)
scale = x.abs().max() / 448.0
q = lin._quantize_to_fp8(x, scale, "e4m3")
err = (q - x).abs().max().item()
amax = x.abs().max().item()
check("FP8: quantization is non-identity", not torch.allclose(q, x),
      f"max_err={err:.3e}")
# e4m3 round-to-nearest: rel err <= 2^-4 within binade; allow 2^-3 bound rel to amax
check("FP8: quantization error bounded", err <= 0.04 * amax,
      f"err/amax={err/amax:.4f}")

# 2d. all-zero batch doesn't crash (buffer TypeError fix), scalar history_idx buffer
q0 = lin._quantize_to_fp8(torch.zeros(4, 16), torch.tensor(0.0), "e4m3")
check("FP8: zero scale/batch safe", torch.isfinite(q0).all().item())
sd = lin.state_dict()
check("FP8: history_idx is a registered buffer", "history_idx" in sd)

# =====================================================================
# 3. GRPO
# =====================================================================
from training.grpo import GRPOTrainer

VOCAB = 260
class DummyLM(nn.Module):
    def __init__(self):
        super().__init__()
        self.embed = nn.Embedding(VOCAB, 32)
        self.head = nn.Linear(32, VOCAB)
    def forward(self, input_ids, attention_mask=None, position_ids=None,
                past_key_values=None, use_cache=False, images=None, audio_features=None):
        h = self.embed(input_ids)
        logits = self.head(h)
        return logits, h, None
    @torch.no_grad()
    def generate(self, input_ids, max_new_tokens=8, temperature=1.0, eos_token_id=None):
        ids = input_ids
        for _ in range(max_new_tokens):
            logits, _, _ = self.forward(ids)
            probs = F.softmax(logits[:, -1] / temperature, dim=-1)
            nxt = torch.multinomial(probs, 1)
            ids = torch.cat([ids, nxt], dim=1)
        return ids

torch.manual_seed(7)
policy = DummyLM()
ref = DummyLM()
ref.load_state_dict(policy.state_dict())
cfg = SimpleNamespace(grpo=SimpleNamespace(group_size=3, epsilon=0.2,
                                           kl_coef=0.1, lr=1e-3))
trainer3 = GRPOTrainer(policy, ref, cfg)

# answer alignment verification: record which question produced each response
resp2q = {}
orig_sample = trainer3._sample_response
def spy_sample(q):
    r = orig_sample(q)
    resp2q[r[0]] = q
    return r
trainer3._sample_response = spy_sample
q2a = {"2+2?": "4", "1+1?": "2"}
alignment_ok = []
orig_rewards = trainer3.compute_rewards
def spy_rewards(responses, answers):
    alignment_ok.append(
        all(q2a[resp2q[r]] == a for r, a in zip(responses, answers))
        and list(answers) == ["4", "4", "4", "2", "2", "2"]
    )
    return orig_rewards(responses, answers)
trainer3.compute_rewards = spy_rewards

# capture loss.requires_grad via backward spy
captured = {}
orig_backward = torch.Tensor.backward
def spy_backward(self, *a, **k):
    captured["requires_grad"] = self.requires_grad
    return orig_backward(self, *a, **k)
torch.Tensor.backward = spy_backward
try:
    out = trainer3.train_step(["2+2?", "1+1?"], ["4", "2"])
finally:
    torch.Tensor.backward = orig_backward

check("GRPO: train_step runs end-to-end", isinstance(out, dict) and "loss" in out,
      f"out={ {k: round(v,4) for k,v in out.items()} }")
check("GRPO: loss.requires_grad == True", captured.get("requires_grad") is True)
param_grads = [p.grad for p in policy.parameters()]
check("GRPO: model params have non-None nonzero grads after step",
      all(g is not None for g in param_grads) and
      any(g.abs().sum() > 0 for g in param_grads))
check("GRPO: answers aligned per group (batch=2, G=3)", all(alignment_ok))
check("GRPO: KL >= -1e-6", out["kl_penalty"] >= -1e-6,
      f"kl={out['kl_penalty']:.6f}")

# KL stays non-negative across a few more steps
kls = [trainer3.train_step(["2+2?", "1+1?"], ["4", "2"])["kl_penalty"] for _ in range(3)]
check("GRPO: KL >= -1e-6 across steps", all(k >= -1e-6 for k in kls),
      f"kls={[round(k,5) for k in kls]}")

# ref model frozen
check("GRPO: ref_model frozen + eval",
      all(not p.requires_grad for p in ref.parameters()) and not ref.training)

print("\n==== SUMMARY ====")
n_pass = sum(1 for _, c in results if c)
print(f"{n_pass}/{len(results)} passed")
sys.exit(0 if n_pass == len(results) else 1)
