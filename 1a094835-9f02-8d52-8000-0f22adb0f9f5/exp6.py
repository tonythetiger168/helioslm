import sys, torch, warnings
sys.path.insert(0, "/mnt/agents/output")
torch.manual_seed(0)
import torch.nn as nn
from helioslm_v5.src.training.fp8_trainer import FP8Trainer

class M(nn.Module):
    def __init__(s):
        super().__init__(); s.a = nn.Linear(8, 8); s.b = nn.Linear(8, 4)
    def forward(s, x): return s.b(s.a(x))

mm = M()
tr = FP8Trainer(mm, type("C", (), {"fp8_format": "e4m3"})())
w0 = mm.a.weight.detach().clone()
with warnings.catch_warnings(record=True) as ws:
    warnings.simplefilter("always")
    l = tr.train_step({"x": torch.full((2, 8), float("nan"))})
    print("nan loss returned:", l, "| warned:", any("non-finite" in str(w.message) for w in ws), "| skipped:", tr.skipped_steps)
print("weights unchanged after NaN step:", torch.equal(w0, mm.a.weight.detach()))
print("amax history finite:", torch.isfinite(mm.a.amax_history).all().item())
for _ in range(3):
    l = tr.train_step({"x": torch.randn(2, 8)})
print("recovered loss:", round(l,4), "| weights changed:", not torch.equal(w0, mm.a.weight.detach()))

# grad NaN with finite loss: inject NaN grad via inf activations in backward only
class M3(nn.Module):
    def __init__(s):
        super().__init__(); s.a = nn.Linear(4, 4)
    def forward(s, x):
        y = s.a(x)
        return y.sum() + (y * 0).sum() / (x.sum()*0 + 1)  # finite
m3 = M3(); tr3 = FP8Trainer(m3, type("C", (), {"fp8_format": "e4m3"})())
w3 = m3.a.weight.detach().clone()
l = tr3.train_step({"x": torch.randn(2, 4)})
print("normal step loss:", round(l, 4))
