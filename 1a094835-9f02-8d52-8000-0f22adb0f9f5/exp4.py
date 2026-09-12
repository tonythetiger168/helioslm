import sys, torch, warnings
sys.path.insert(0, "/mnt/agents/output")
torch.manual_seed(0)
from helioslm_v5.configs.config_v5 import HeliosLMv5Config
from helioslm_v5.src.moe.sigmoid_moe import DeviceLimitedMoE

cfg = HeliosLMv5Config(size="lite")
moe = DeviceLimitedMoE(cfg)
x = torch.randn(2, 5, cfg.hidden_size)

# M-C2: eval/no-grad must NOT accumulate expert_load
moe.eval()
with torch.no_grad():
    moe(x)
print("eval load sum:", moe.expert_load.sum().item(), "(expect 0)")
moe.eval(); moe(x)  # eval WITH grad enabled
print("eval+grad load sum:", moe.expert_load.sum().item(), "(expect 0)")
moe.train(); moe(x)
print("train load sum:", moe.expert_load.sum().item(), "(expect 2*5*top_k=%d)" % (2*5*cfg.moe.num_activated_experts))

# update_bias consumes and resets
before = moe.route_bias.clone()
moe.update_bias()
print("bias changed:", not torch.equal(before, moe.route_bias), "| load reset:", moe.expert_load.sum().item()==0)
# repeated update with no new stats: load all zero -> mean 0 -> sign(0)=0 -> no change
b2 = moe.route_bias.clone()
moe.update_bias()
print("idempotent update with zero load:", torch.equal(b2, moe.route_bias))
