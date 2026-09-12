import sys, torch
sys.path.insert(0, "/mnt/agents/output")
torch.manual_seed(0)
from helioslm_v5.src.inference.mtp import MTPDecoder, _filter_top_p

# _residual_sample distribution: emitted dist of (accept min(1,p/q) + residual) == p
p = torch.tensor([0.5, 0.3, 0.15, 0.05])
q = torch.tensor([0.25, 0.25, 0.25, 0.25])
import random
random.seed(0); torch.manual_seed(0)
N = 200000
counts = torch.zeros(4)
for _ in range(N):
    tok = torch.multinomial(q, 1).item()
    if random.random() < min(1.0, (p[tok]/q[tok]).item()):
        counts[tok] += 1
    else:
        counts[MTPDecoder._residual_sample(p, q)] += 1
emp = counts / N
print("spec-sampling empirical:", [round(v,4) for v in emp.tolist()], "target:", p.tolist(),
      "max dev:", (emp-p).abs().max().item())

# _filter_top_p sanity
probs = torch.tensor([0.4, 0.3, 0.2, 0.1])
print("top_p=0.7 ->", _filter_top_p(probs, 0.7).tolist())
print("top_p=1.0 -> identity:", torch.equal(_filter_top_p(probs, 1.0), probs))
# degenerate: top_p tiny keeps top-1
print("top_p=0.01 ->", _filter_top_p(probs, 0.01).tolist())
