"""T5: quantization boundary tests."""
import sys
sys.path.insert(0, "/mnt/agents/output")
import torch, torch.nn as nn
from helioslm_v5.configs.config_v5 import HeliosLMv5Config
from helioslm_v5.src.model_v5 import HeliosLMv5
from helioslm_v5.src.quantization.standard_quant import (
    QuantizationManager, AWQLinear, GPTQLinear, FP8Linear)

def banner(s): print(f"\n=== {s} ===")

banner("1. AWQ-quantized lite model can generate; MTP rebinding intact")
torch.manual_seed(0)
cfg = HeliosLMv5Config(size="lite")
model = HeliosLMv5(cfg).eval()
QuantizationManager("awq").quantize_model(model, group_size=32)
ids = torch.tensor([[3, 4, 5], [9, 8, 7]])
out = model.generate(ids, max_new_tokens=6, temperature=0.0)
print("generate ok:", tuple(out.shape))
out_mtp = model.generate(ids, max_new_tokens=6, temperature=0.0, use_mtp=True)
print("MTP generate ok:", tuple(out_mtp.shape))
# weight tying: mtp lm_head IS the model's (quantized) lm_head
print("MTP lm_head is model.lm_head:", model.mtp_modules[0].lm_head is model.lm_head)
print("MTP embed is model.embed_tokens:", model.mtp_modules[0].embed_tokens is model.embed_tokens)

banner("2. quantize model with a single Linear (nn.Sequential)")
lin_model = nn.Sequential(nn.Linear(16, 8))
QuantizationManager("awq").quantize_model(lin_model, group_size=8)
print("type after:", type(lin_model[0]).__name__)
x = torch.randn(2, 16)
print("forward ok:", tuple(lin_model(x).shape))

banner("2b. quantize a BARE nn.Linear as the model (name=='')")
bare = nn.Linear(16, 8)
QuantizationManager("awq").quantize_model(bare, group_size=8)
print("type after:", type(bare).__name__, "(still nn.Linear => silently NOT quantized)" )

banner("3. odd dims: in=33, out=7, group 8")
lin = nn.Linear(33, 7)
x = torch.randn(4, 33)
ref = lin(x)
q = AWQLinear.from_linear(lin, group_size=8)
err = (q(x) - ref).abs().max().item()
rel = err / ref.abs().max().item()
print(f"AWQ odd-dim max abs err {err:.3e} (rel {rel:.3%})")
g = GPTQLinear.from_linear(lin, group_size=8)
err = (g(x) - ref).abs().max().item()
print(f"GPTQ odd-dim max abs err {err:.3e} (rel {err/ref.abs().max().item():.3%})")

banner("4. group_size > in_features (gs=128, in=16)")
lin = nn.Linear(16, 8)
x = torch.randn(4, 16)
ref = lin(x)
q = AWQLinear.from_linear(lin, group_size=128)
print("AWQ  gs>in ok, rel err {:.3%}".format(((q(x)-ref).abs().max()/ref.abs().max()).item()))
g = GPTQLinear.from_linear(lin, group_size=128)
print("GPTQ gs>in ok, rel err {:.3%}".format(((g(x)-ref).abs().max()/ref.abs().max()).item()))

banner("5. idempotency: quantize_model twice")
torch.manual_seed(0)
cfg2 = HeliosLMv5Config(size="lite")
m2 = HeliosLMv5(cfg2).eval()
qm = QuantizationManager("awq")
qm.quantize_model(m2, group_size=32)
snap = {n: b.clone() for n, b in m2.named_buffers() if b.dtype == torch.uint8}
out1 = m2.generate(torch.tensor([[1, 2, 3]]), max_new_tokens=4, temperature=0.0)
qm.quantize_model(m2, group_size=32)   # second pass
same_buffers = all(torch.equal(snap[n], b) for n, b in m2.named_buffers() if n in snap)
out2 = m2.generate(torch.tensor([[1, 2, 3]]), max_new_tokens=4, temperature=0.0)
print("packed buffers unchanged after 2nd pass:", same_buffers)
print("generate identical after 2nd pass:", torch.equal(out1, out2))
n_lin = sum(1 for m in m2.modules() if isinstance(m, nn.Linear))
n_q = sum(1 for m in m2.modules() if isinstance(m, (AWQLinear, GPTQLinear)))
print(f"remaining nn.Linear={n_lin}, quantized={n_q}")

banner("6. GPTQ with calibration, tiny dims (in=16 < blocksize=128)")
lin = nn.Linear(16, 8)
calib = torch.randn(64, 16)
g = GPTQLinear.from_linear(lin, group_size=8, calibration_data=calib)
x = torch.randn(4, 16)
print("GPTQ calibrated forward ok, rel err {:.3%}".format(
    ((g(x)-lin(x)).abs().max()/lin(x).abs().max()).item()))

banner("7. FP8Linear (standard_quant) all-zero weight")
if FP8Linear is not None:
    lin = nn.Linear(16, 8)
    with torch.no_grad(): lin.weight.zero_()
    f = FP8Linear.from_linear(lin)
    x = torch.randn(4, 16)
    print("zero-weight FP8 forward finite:", torch.isfinite(f(x)).all().item())
else:
    print("FP8Linear unavailable on this torch")

banner("8. GPTQ calibration with zero/degenerate activations")
lin = nn.Linear(16, 8)
try:
    g = GPTQLinear.from_linear(lin, group_size=8, calibration_data=torch.zeros(64, 16))
    x = torch.randn(4, 16)
    print("zero-calib GPTQ forward finite:", torch.isfinite(g(x)).all().item())
except Exception as e:
    print(f"zero-calib GPTQ EXC: {type(e).__name__}: {e}")

banner("9. quantized model state_dict round-trip")
try:
    sd = model.state_dict()
    cfg3 = HeliosLMv5Config(size="lite")
    m3 = HeliosLMv5(cfg3)
    QuantizationManager("awq").quantize_model(m3, group_size=32)
    m3.load_state_dict(sd)
    print("quantized state_dict round-trip: OK")
except Exception as e:
    print(f"state_dict round-trip EXC: {type(e).__name__}: {e}")
print("DONE")
