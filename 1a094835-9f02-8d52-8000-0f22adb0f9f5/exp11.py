import sys, torch
sys.path.insert(0, "/mnt/agents/output")
torch.manual_seed(0)
from helioslm_v5.configs.config_v5 import HeliosLMv5Config
import helioslm_v5.src.model_v5 as m5
from helioslm_v5.src.training.grpo import GRPOTrainer

cfg = HeliosLMv5Config(size="lite")
model = m5.HeliosLMv5(cfg)
ref = m5.HeliosLMv5(cfg)
ref.load_state_dict(model.state_dict())

# per-sample backward (v5.3) vs batched-mean backward on identical fake data
class FakeTok:
    def encode(self, t): return [ord(c) % 200 + 3 for c in t]
    def decode(self, ids): return "".join(chr((i-3) % 90 + 33) for i in ids)

tr = GRPOTrainer(model, ref, cfg, tokenizer=FakeTok())
# Monkeypatch sampling to fixed fake samples so we can compare gradient paths
samples = []
torch.manual_seed(7)
for q in ["1+1", "2+2"]:
    for _ in range(cfg.grpo.group_size):
        pl = 3
        full = torch.randint(3, cfg.vocab_size, (1, pl + 4))
        with torch.no_grad():
            old_lp = GRPOTrainer._sequence_logprob(model, full, pl).detach()
        samples.append({"response": "x2", "old_logprob": old_lp, "full_ids": full, "prompt_len": pl})

import torch as T
rewards = T.tensor([1.0,0.0,0.5,0.2,0.9,0.1,0.3,0.8, 0.4,0.6,0.7,0.05,0.15,0.95,0.25,0.35]).view(2, 8)
mean_r = rewards.mean(dim=1, keepdim=True)
std_r = rewards.std(dim=1, unbiased=False, keepdim=True).clamp(min=1e-8)
adv = ((rewards - mean_r) / std_r).reshape(-1)
with torch.no_grad():
    ref_lps = T.stack([GRPOTrainer._sequence_logprob(ref, s["full_ids"], s["prompt_len"]) for s in samples])

def run(mode):
    for p in model.parameters(): p.grad = None
    N = len(samples)
    if mode == "batched":
        losses = []
        for i, s in enumerate(samples):
            new_lp = GRPOTrainer._sequence_logprob(model, s["full_ids"], s["prompt_len"])
            log_ratio = (new_lp - s["old_logprob"]).clamp(-20, 20)
            ratio = torch.exp(log_ratio)
            surr = torch.min(ratio*adv[i], torch.clamp(ratio, 0.8, 1.2)*adv[i])
            logr = (ref_lps[i] - new_lp).clamp(max=60.0)
            kl = cfg.grpo.kl_coef * (torch.expm1(logr) - logr)
            losses.append((-surr + kl) / N)
        torch.stack(losses).sum().backward()
    else:
        for i, s in enumerate(samples):
            new_lp = GRPOTrainer._sequence_logprob(model, s["full_ids"], s["prompt_len"])
            log_ratio = (new_lp - s["old_logprob"]).clamp(-20, 20)
            ratio = torch.exp(log_ratio)
            surr = torch.min(ratio*adv[i], torch.clamp(ratio, 0.8, 1.2)*adv[i])
            logr = (ref_lps[i] - new_lp).clamp(max=60.0)
            kl = cfg.grpo.kl_coef * (torch.expm1(logr) - logr)
            ((-surr + kl) / N).backward()
    return torch.cat([p.grad.flatten() for p in model.parameters() if p.grad is not None])

g1 = run("batched"); g2 = run("persample")
print("per-sample vs batched grad max diff:", (g1-g2).abs().max().item())

# k3 expm1 vs naive near-zero logr
logr = torch.tensor(1e-7, dtype=torch.float64)
naive = torch.exp(logr) - logr - 1
exact = torch.expm1(logr) - logr
print(f"k3 @logr=1e-7 (fp64): naive={naive:.3e} expm1={exact:.3e}")
logr32 = torch.tensor(1e-7, dtype=torch.float32)
print(f"fp32: naive={torch.exp(logr32)-logr32-1:.3e} expm1={torch.expm1(logr32)-logr32:.3e}")
