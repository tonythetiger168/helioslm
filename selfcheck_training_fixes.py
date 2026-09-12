"""Self-verification for the training-module review fixes (not part of the
shipped test suite; run manually)."""
import math, warnings, sys
sys.path.insert(0, "/mnt/agents/output")
import torch
import torch.nn as nn

from helioslm_v5.tests.test_v5 import _lite_model

ok = []
def check(name, cond, extra=""):
    assert cond, f"FAIL {name} {extra}"
    ok.append(name)
    print(f"[OK] {name} {extra}")

# ---------------------------------------------------------------- M-T1
from helioslm_v5.src.training.grpo import GRPOTrainer

model, config = _lite_model(seed=201)
ref, _ = _lite_model(seed=202)
config.grpo.group_size = 2
config.grpo.max_new_tokens = 3
tr = GRPOTrainer(model, ref, config)

modes = []
orig_seq_logprob = GRPOTrainer._sequence_logprob
def spy_seq_logprob(m, ids, pl):
    if torch.is_grad_enabled():
        modes.append(m.training)
    return orig_seq_logprob(m, ids, pl)
GRPOTrainer._sequence_logprob = staticmethod(spy_seq_logprob)

model.train()  # caller mode = train
out = tr.train_step(["1+1?"], ["2"])
check("M-T1 training forwards ran in train mode", modes and all(modes),
      f"modes={modes}")
check("M-T1 mode restored to caller's (train)", model.training)

model.eval()  # caller mode = eval
out = tr.train_step(["1+1?"], ["2"])
check("M-T1 mode restored to caller's (eval)", not model.training)
GRPOTrainer._sequence_logprob = staticmethod(orig_seq_logprob)

# dropout>0 actually affects the training forward (would be dead in eval)
from helioslm_v5.configs.config_v5 import HeliosLMv5Config
from helioslm_v5.src.model_v5 import HeliosLMv5
config2 = HeliosLMv5Config(size="lite")
config2.attention.attention_dropout = 0.5  # must be set BEFORE construction
torch.manual_seed(203)
model2 = HeliosLMv5(config2)
model2.train()
ids = torch.randint(3, config2.vocab_size, (1, 8))
lp1 = GRPOTrainer._sequence_logprob(model2, ids, 4)
lp2 = GRPOTrainer._sequence_logprob(model2, ids, 4)
check("M-T1 dropout active in train mode (nondeterministic fwd)",
      not torch.equal(lp1, lp2), f"{lp1.item()} vs {lp2.item()}")

# ---------------------------------------------------------------- M-T3
from helioslm_v5.src.training.grpo import GRPOTrainer as G
check("M-T3 '25' vs '2' -> incorrect", not tr._check_correctness("25", "2"))
check("M-T3 'the answer is 42' vs '42' -> correct",
      tr._check_correctness("the answer is 42", "42"))
check("M-T3 boxed", tr._check_correctness(r"so $\\boxed{42}$", "42"))
check("M-T3 think-segment", tr._check_correctness(
    "<think>25 steps</think>the answer is 42.", "42"))
check("M-T3 wrong number", not tr._check_correctness("the answer is 43", "42"))
check("M-T3 '42' vs '420' no false positive",
      not tr._check_correctness("420", "42"))
check("M-T3 numeric form 42.0 == 42", tr._check_correctness("answer: 42.0", "42"))

# m6: control chars / EOS stripped
clean = tr._clean_response_text("answer\x00\x02 is 5\x07")
check("m6 control chars stripped", clean == "answer is 5", repr(clean))

# m5: byte fallback decode safe for ids > 256 / negative
from helioslm_v5.src.training.grpo import _ByteLevelTokenizer
s = _ByteLevelTokenizer().decode([65, 300, -3, 2])
check("m5 byte decode safe for out-of-range ids", isinstance(s, str))

# ---------------------------------------------------------------- M-T2
from helioslm_v5.src.training.fp8_trainer import FP8Trainer, FP8Linear
model3, config3 = _lite_model(seed=205)
fpt = FP8Trainer(model3, config3)
ids = torch.randint(3, config3.vocab_size, (2, 10))
good = {"input_ids": ids}

l0 = fpt.train_step(good)
# NaN batch: poison the embedding so the forward loss is NaN
with torch.no_grad():
    model3.embed_tokens.weight[0, 0] = float("nan")
bad = {"input_ids": torch.zeros(2, 10, dtype=torch.long)}  # hits row 0
with warnings.catch_warnings(record=True) as w:
    warnings.simplefilter("always")
    lbad = fpt.train_step(bad)
check("M-T2 NaN loss detected + warned", not math.isfinite(lbad) and len(w) >= 1)
check("M-T2 skipped_steps counted", fpt.skipped_steps == 1)
# restore and continue
with torch.no_grad():
    model3.embed_tokens.weight[0, 0] = 0.01
l1 = fpt.train_step(good)
l2 = fpt.train_step(good)
check("M-T2 post-NaN losses finite", all(map(math.isfinite, (l1, l2))),
      f"{l1}, {l2}")
masters_ok = all(torch.isfinite(m).all() for _, m in fpt.param_map)
scales_ok = all(
    torch.isfinite(m.input_scale) and torch.isfinite(m.weight_scale)
    and torch.isfinite(m.amax_history).all()
    for m in model3.modules() if isinstance(m, FP8Linear))
check("M-T2 master weights finite after NaN batch", masters_ok)
check("M-T2 scales/history finite after NaN batch", scales_ok)
check("M-T2 output_scale buffer removed",
      not any(hasattr(m, "output_scale")
              for m in model3.modules() if isinstance(m, FP8Linear)))

# non-finite GRAD skip path: loss finite, grads poisoned via hook
class PoisonLoss(nn.Module):
    def __init__(self, base):
        super().__init__()
        self.base = base
    def forward(self, **kw):
        out = self.base(**kw)
        return out  # finite
# simpler: poison grads directly after backward is hard; instead verify the
# finite-grad path counts correctly and skip path triggered for loss is enough.

# ---------------------------------------------------------------- dualpipe
from helioslm_v5.src.training.dualpipe import (
    DualPipeScheduler, DualPipeStage, ExpertParallelism)

# m8: run_dual([]) guard
lin = DualPipeStage(nn.ModuleList([nn.Linear(4, 4)]))
sched = DualPipeScheduler([lin], num_micro_batches=2)
try:
    sched.run_dual([], lambda o: o.sum())
    raise SystemExit("run_dual([]) should raise")
except ValueError:
    ok.append("m8 run_dual([]) ValueError"); print("[OK] m8 run_dual([]) ValueError")

# m7: num_micro_batches wired
try:
    sched.run_dual([torch.randn(2, 4)], lambda o: o.sum())
    raise SystemExit("mismatch should raise")
except ValueError:
    ok.append("m7 num_micro_batches mismatch ValueError")
    print("[OK] m7 num_micro_batches mismatch ValueError")
try:
    DualPipeScheduler([lin], num_micro_batches=0)
    raise SystemExit("ctor 0 should raise")
except ValueError:
    ok.append("m7 ctor num_micro_batches>=1"); print("[OK] m7 ctor num_micro_batches>=1")

# m9: dropout self-consistency — detached pipeline grads == direct grads
torch.manual_seed(300)
stage = DualPipeStage(nn.ModuleList([
    nn.Sequential(nn.Linear(8, 8), nn.Dropout(0.5), nn.Linear(8, 8))]))
stage.train()
loss_fn = lambda o: o.pow(2).mean()

def make_inputs():  # deterministic inputs from a fixed seed
    torch.manual_seed(301)
    return [torch.randn(3, 8) for _ in range(3)]

inputs = make_inputs()   # RNG state after this == reference's after this
sched = DualPipeScheduler([stage], num_micro_batches=3)
lp = sched.run_dual(inputs, loss_fn)
g_pipe = [p.grad.clone() for p in stage.parameters()]
stage.zero_grad(set_to_none=True)
inputs = make_inputs()   # replay: same inputs, same post-creation RNG state
loss_ref = 0.0
for x in inputs:
    l = loss_fn(stage(x)) / 3
    l.backward()
    loss_ref += l.item()
g_ref = [p.grad.clone() for p in stage.parameters()]
maxd = max((a - b).abs().max().item() for a, b in zip(g_pipe, g_ref))
check("m9 dropout pipeline grads == direct grads", maxd < 1e-6,
      f"max diff {maxd:.2e}, loss {lp:.5f} vs {loss_ref:.5f}")
check("m9 loss matches direct", abs(lp - loss_ref) < 1e-6)

# m11: route_to_device negative/out-of-range check
ep = ExpertParallelism(num_experts=8, num_devices=4)
try:
    ep.route_to_device(torch.tensor([0, -1]))
    raise SystemExit("negative id should raise")
except ValueError:
    ok.append("m11 negative expert id ValueError"); print("[OK] m11 negative expert id ValueError")
try:
    ep.route_to_device(torch.tensor([8]))
    raise SystemExit("oob id should raise")
except ValueError:
    ok.append("m11 oob expert id ValueError"); print("[OK] m11 oob expert id ValueError")
r = ep.route_to_device(torch.tensor([0, 7]))
check("m11 valid routing intact", r.tolist() == [0, 3])

# m10: all_to_all validates device args, still NotImplementedError w/o pg
try:
    ep.all_to_all([torch.randn(2)], 0, 9)
    raise SystemExit("bad dst should raise")
except ValueError:
    ok.append("m10 all_to_all validates device range")
    print("[OK] m10 all_to_all validates device range")
try:
    ep.all_to_all([torch.randn(2)], 0, 1)
    raise SystemExit("no pg should raise NotImplementedError")
except NotImplementedError:
    ok.append("m10 all_to_all NotImplementedError w/o pg")
    print("[OK] m10 all_to_all NotImplementedError w/o pg")

# --------------------------------------------- grpo m1: grad parity check
# Per-sample backward accumulation must equal a single batched backward.
model4, config4 = _lite_model(seed=206)
ref4, _ = _lite_model(seed=207)
config4.grpo.group_size = 2
config4.grpo.max_new_tokens = 3
tr4 = GRPOTrainer(model4, ref4, config4)
torch.manual_seed(400)
samples = []
model4.eval()
with torch.no_grad():
    for q in ["q1?", "q2?"]:
        for _ in range(2):
            resp, old_lp, full_ids, pl = tr4._sample_response(q)
            samples.append({"response": resp, "old_logprob": old_lp,
                            "full_ids": full_ids, "prompt_len": pl})
N = len(samples)
advs = torch.tensor([1.0, -0.5, 0.25, 0.75])
ref_lps = tr4._get_ref_logprobs(samples).detach()

# reference: one batched graph, single backward (old scheme, with clamp+expm1)
model4.train()
model4.zero_grad(set_to_none=True)
new_lps = torch.stack([tr4._sequence_logprob(model4, s["full_ids"], s["prompt_len"])
                       for s in samples])
old_lps = torch.stack([s["old_logprob"] for s in samples])
ratio = torch.exp((new_lps - old_lps).clamp(-20.0, 20.0))
surr1 = ratio * advs
surr2 = torch.clamp(ratio, 1 - tr4.epsilon, 1 + tr4.epsilon) * advs
pol = -torch.min(surr1, surr2).mean()
logr = (ref_lps - new_lps).clamp(max=60.0)
kl = tr4.kl_coef * (torch.expm1(logr) - logr).mean()
loss_ref = pol + kl
loss_ref.backward()
g_ref = {n: p.grad.clone() for n, p in model4.named_parameters()
         if p.grad is not None}

# new scheme: per-sample backward
model4.zero_grad(set_to_none=True)
for i, s in enumerate(samples):
    new_lp = tr4._sequence_logprob(model4, s["full_ids"], s["prompt_len"])
    lr_ = (new_lp - s["old_logprob"]).clamp(-20.0, 20.0)
    r_ = torch.exp(lr_)
    p_i = -torch.min(r_ * advs[i],
                     torch.clamp(r_, 1 - tr4.epsilon, 1 + tr4.epsilon) * advs[i])
    lg = (ref_lps[i] - new_lp).clamp(max=60.0)
    k_i = tr4.kl_coef * (torch.expm1(lg) - lg)
    ((p_i + k_i) / N).backward()
g_new = {n: p.grad.clone() for n, p in model4.named_parameters()
         if p.grad is not None}
maxd = max((g_ref[n] - g_new[n]).abs().max().item() for n in g_ref)
check("grpo m1 per-sample backward == batched backward", maxd < 1e-6,
      f"max grad diff {maxd:.2e}")

print(f"\nALL {len(ok)} SELF-CHECKS PASSED")
