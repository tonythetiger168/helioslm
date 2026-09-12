import sys, torch
sys.path.insert(0, "/mnt/agents/output")
import torch.nn as nn
from helioslm_v5.src.training.dualpipe import DualPipeStage, DualPipeScheduler

def make(seed):
    torch.manual_seed(seed)
    return nn.Sequential(nn.Linear(16,16), nn.Dropout(0.5), nn.Linear(16,16))
m1, m2 = make(1), make(2)
stages = [DualPipeStage(nn.ModuleList([m1])), DualPipeStage(nn.ModuleList([m2]))]
sched = DualPipeScheduler(stages, num_micro_batches=4)
xs = [torch.randn(3,16) for _ in range(4)]

# check recompute self-consistency directly
x = xs[0]
leaf = x.detach().requires_grad_(True)
st = sched._capture_rng_state()
with torch.no_grad():
    out_orig = stages[0](leaf)
# recompute under captured state
saved = sched._capture_rng_state()
sched._restore_rng_state(st)
out_re = stages[0](leaf)
sched._restore_rng_state(saved)
print("recompute matches original fwd:", torch.equal(out_orig, out_re.detach()))

# full pipeline vs direct with identical seed state at start of both runs
state0 = torch.random.get_rng_state()
loss = sched.run_dual(xs, lambda o: (o**2).mean())
g_pipe = [p.grad.clone() for p in list(m1.parameters())+list(m2.parameters())]
for p in list(m1.parameters())+list(m2.parameters()): p.grad = None

torch.random.set_rng_state(state0)
tot = 0.0
for x in xs:
    l = (m2(m1(x))**2).mean()/4
    l.backward(); tot += l.item()
g_ref = [p.grad.clone() for p in list(m1.parameters())+list(m2.parameters())]
md = max((a-b).abs().max().item() for a,b in zip(g_pipe,g_ref))
print(f"same-seed: loss diff={abs(loss-tot):.3e} grad max diff={md:.3e}")
