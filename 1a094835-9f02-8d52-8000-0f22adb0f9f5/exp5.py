import sys, torch, warnings
sys.path.insert(0, "/mnt/agents/output")
torch.manual_seed(0)
import torch.nn as nn
from helioslm_v5.src.training.dualpipe import DualPipeStage, DualPipeScheduler

# DualPipe RNG fork: dropout=0.5, grads must equal direct fwd+bwd
def make(seed):
    torch.manual_seed(seed)
    return nn.Sequential(nn.Linear(16,16), nn.Dropout(0.5), nn.Linear(16,16))
m1, m2 = make(1), make(2)
stages = [DualPipeStage(nn.ModuleList([m1])), DualPipeStage(nn.ModuleList([m2]))]
sched = DualPipeScheduler(stages, num_micro_batches=4)
xs = [torch.randn(3,16) for _ in range(4)]
loss = sched.run_dual(xs, lambda o: (o**2).mean())
g_pipe = [p.grad.clone() for p in list(m1.parameters())+list(m2.parameters())]

torch.manual_seed(1234)  # different global seed: dropout masks must still match (forked RNG)
m1b, m2b = make(1), make(2)
tot = 0.0
for x in xs:
    l = (m2b(m1b(x))**2).mean()/4
    l.backward(); tot += l.item()
g_ref = [p.grad.clone() for p in list(m1b.parameters())+list(m2b.parameters())]
md = max((a-b).abs().max().item() for a,b in zip(g_pipe,g_ref))
print(f"dualpipe dropout0.5: loss diff={abs(loss-tot):.2e} grad max diff={md:.2e} (expect 0)")

# run_dual mismatch validation
try:
    sched.run_dual([torch.randn(3,16)], lambda o: o.mean())
    print("M mismatch: NO RAISE (bug)")
except ValueError:
    print("run_dual num_micro_batches mismatch raises: OK")
try:
    DualPipeScheduler(stages, num_micro_batches=0)
    print("nmb=0: NO RAISE (bug)")
except ValueError:
    print("num_micro_batches=0 raises: OK")
