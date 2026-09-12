"""T7: training boundary tests (GRPO, FP8, DualPipe)."""
import sys, warnings
sys.path.insert(0, "/mnt/agents/output")
import torch, torch.nn as nn
from helioslm_v5.configs.config_v5 import HeliosLMv5Config
from helioslm_v5.src.model_v5 import HeliosLMv5
from helioslm_v5.src.training.grpo import GRPOTrainer
from helioslm_v5.src.training.fp8_trainer import FP8Trainer, FP8Linear
from helioslm_v5.src.training.dualpipe import DualPipeStage, DualPipeScheduler

def banner(s): print(f"\n=== {s} ===")

torch.manual_seed(0)
cfg = HeliosLMv5Config(size="lite")

banner("1. GRPO group_size=1, batch=1")
cfg.grpo.group_size = 1
cfg.grpo.max_new_tokens = 4
model = HeliosLMv5(cfg)
ref = HeliosLMv5(cfg)
ref.load_state_dict(model.state_dict())
tr = GRPOTrainer(model, ref, cfg)
try:
    r = tr.train_step(["1+1="], ["2"])
    print("train_step ok:", {k: round(v, 4) for k, v in r.items()})
    print("model.training restored:", model.training)
except Exception as e:
    print(f"EXC: {type(e).__name__}: {e}")

banner("2. GRPO group_size=2, batch=2 (normal)")
cfg.grpo.group_size = 2
model = HeliosLMv5(cfg); ref = HeliosLMv5(cfg); ref.load_state_dict(model.state_dict())
tr = GRPOTrainer(model, ref, cfg)
try:
    r = tr.train_step(["1+1=", "2+2="], ["2", "4"])
    print("ok:", {k: round(v, 4) for k, v in r.items()})
except Exception as e:
    print(f"EXC: {type(e).__name__}: {e}")

banner("3. GRPO on model in eval mode (mode restore)")
model.eval()
try:
    r = tr.train_step(["hi"], ["hi"])
    print("ok; model.training after:", model.training, "(expect False)")
except Exception as e:
    print(f"EXC: {type(e).__name__}: {e}")

banner("4. FP8Trainer: all-zero input, single-parameter model")
class Tiny(nn.Module):
    def __init__(self):
        super().__init__()
        self.lin = nn.Linear(4, 4)
    def forward(self, x, y):
        return ((self.lin(x) - y) ** 2).mean()
class TinyCfg: fp8_format = "e4m3"; lr = 1e-3
tiny = Tiny()
tr8 = FP8Trainer(tiny, TinyCfg())
try:
    for i in range(3):
        loss = tr8.train_step({"x": torch.zeros(2, 4), "y": torch.zeros(2, 4)})
    print("all-zero steps ok, last loss:", loss)
    print("converted:", type(tiny.lin).__name__)
    # grads flow?
    loss = tr8.train_step({"x": torch.randn(2, 4), "y": torch.randn(2, 4)})
    print("normal step loss:", loss)
except Exception as e:
    print(f"EXC: {type(e).__name__}: {e}")

banner("5. FP8Trainer on full lite model (mutation-during-iteration check)")
m = HeliosLMv5(HeliosLMv5Config(size="lite")).eval()
try:
    tr8b = FP8Trainer(m, TinyCfg())
    n_fp8 = sum(1 for x in m.modules() if isinstance(x, FP8Linear))
    n_lin = sum(1 for x in m.modules() if isinstance(x, nn.Linear))
    print(f"converted: FP8Linear={n_fp8}, remaining nn.Linear={n_lin}")
except Exception as e:
    print(f"EXC: {type(e).__name__}: {e}")

banner("6. FP8Linear with NaN input (M-T2 guard)")
fl = FP8Linear(4, 4)
try:
    out = fl(torch.full((2, 4), float("nan")))
    print("NaN input -> finite out:", torch.isfinite(out).all().item())
    hmax = fl.amax_history.max().item()
    print("history contaminated by NaN:", bool(hmax != hmax), f"max={hmax}")
except Exception as e:
    print(f"EXC: {type(e).__name__}: {e}")

banner("7. DualPipe M=1")
stages = [DualPipeStage(nn.ModuleList([nn.Linear(8, 8)])),
          DualPipeStage(nn.ModuleList([nn.Linear(8, 8)]))]
sch = DualPipeScheduler(stages, num_micro_batches=1)
x = torch.randn(2, 8)
try:
    loss = sch.run_dual([x], lambda o: (o ** 2).mean())
    print("M=1 loss:", round(loss, 4), "trace:", sch.trace)
    g = stages[0].layers[0].weight.grad
    print("stage0 grad exists:", g is not None and torch.isfinite(g).all().item())
except Exception as e:
    print(f"EXC: {type(e).__name__}: {e}")

banner("8. DualPipe M=1 grad equals direct backprop")
torch.manual_seed(7)
st1 = [DualPipeStage(nn.ModuleList([nn.Linear(8, 8)])), DualPipeStage(nn.ModuleList([nn.Linear(8, 8)]))]
sch1 = DualPipeScheduler(st1, num_micro_batches=1)
x = torch.randn(2, 8)
l1 = sch1.run_dual([x], lambda o: (o ** 2).mean())
# direct
torch.manual_seed(7)
ref_stages = [DualPipeStage(nn.ModuleList([nn.Linear(8, 8)])), DualPipeStage(nn.ModuleList([nn.Linear(8, 8)]))]
out = ref_stages[1](ref_stages[0](x))
l2 = (out ** 2).mean()
l2.backward()
d = (st1[0].layers[0].weight.grad - ref_stages[0].layers[0].weight.grad).abs().max().item()
print(f"loss diff {abs(l1 - l2.item()):.2e}, grad diff {d:.2e} {'OK' if d < 1e-5 else 'FAIL'}")

banner("9. DualPipe M=3 grad vs direct")
torch.manual_seed(8)
st = [DualPipeStage(nn.ModuleList([nn.Linear(8, 8)])), DualPipeStage(nn.ModuleList([nn.Linear(8, 8)]))]
sch3 = DualPipeScheduler(st, num_micro_batches=3)
xs = [torch.randn(2, 8) for _ in range(3)]
l = sch3.run_dual(xs, lambda o: (o ** 2).mean())
torch.manual_seed(8)
rs = [DualPipeStage(nn.ModuleList([nn.Linear(8, 8)])), DualPipeStage(nn.ModuleList([nn.Linear(8, 8)]))]
lr = sum((rs[1](rs[0](xx)) ** 2).mean() for xx in xs) / 3
lr.backward()
d = (st[0].layers[0].weight.grad - rs[0].layers[0].weight.grad).abs().max().item()
print(f"M=3 grad diff {d:.2e} {'OK' if d < 1e-5 else 'FAIL'}")

banner("10. DualPipe M=0 rejected")
sch0 = DualPipeScheduler(st, num_micro_batches=1)
try:
    sch0.run_dual([], lambda o: o.mean())
    print("M=0 NOT rejected (unexpected)")
except ValueError as e:
    print("M=0 ValueError (expected)")
try:
    DualPipeScheduler(st, num_micro_batches=0)
    print("ctor M=0 NOT rejected (unexpected)")
except ValueError:
    print("ctor num_micro_batches=0 ValueError (expected)")
print("DONE")
