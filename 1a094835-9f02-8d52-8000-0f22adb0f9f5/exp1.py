import sys, types, torch
sys.path.insert(0, "/mnt/agents/output")
torch.manual_seed(0)
import torch.nn as nn
from helioslm_v5.src.quantization.standard_quant import GPTQLinear, AWQLinear

# 1) odd-dim quant roundtrip
lin = nn.Linear(47, 33, bias=True)
x = torch.randn(5, 47); cal = torch.randn(128, 47)
for tag, m in [("gptq-rtn", GPTQLinear.from_linear(lin, group_size=16)),
               ("gptq-cal", GPTQLinear.from_linear(lin, group_size=16, calibration_data=cal)),
               ("awq-rtn", AWQLinear.from_linear(lin, group_size=16)),
               ("awq-cal", AWQLinear.from_linear(lin, group_size=16, activations=cal))]:
    w = m.weight
    d = (m(x) - nn.functional.linear(x, w, m.bias)).abs().max().item()
    print(tag, "shape", tuple(w.shape), "fwd-vs-weight", f"{d:.2e}", "finite", torch.isfinite(m(x)).all().item())

# fp16/bf16 input cast (M-Q1)
lin2 = nn.Linear(32, 16, bias=True)
g = GPTQLinear.from_linear(lin2, group_size=16)
xh = torch.randn(4, 32, dtype=torch.float16)
print("gptq fp16 fwd finite:", torch.isfinite(g(xh)).all().item(), g(xh).dtype)

# 2) GRPO answer extraction
from helioslm_v5.src.training.grpo import GRPOTrainer
d = types.SimpleNamespace()
chk = types.MethodType(GRPOTrainer._check_correctness, d)
cases = [("The answer is 25","2",False),("The answer is 2","2",True),
         (r"... \boxed{42}","42",True),(r"... \boxed{42}","43",False),
         ("result: 42.0","42",True),("<think>blah 99</think> Final: 7","7",True),
         ("","2",False),("Answer: $1,000.","1000",True),
         ("thinking 25 then 3","3",True), ("x = -4","-4",True)]
for resp, gold, expect in cases:
    got = chk(resp, gold)
    print("OK" if got==expect else "FAIL", repr(resp), gold, "->", got)
