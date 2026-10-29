"""T152 — MoE layers under torch.compile, v5.158.

The first L4 compile run (torch 2.8) failed the MoE pilot: dynamo raised
GuardOnDataDependentSymNode on the per-expert count loop of the batched
dispatch. Both dispatch paths sync per-expert counts to the host and
build data-dependent shapes, so they are excluded from compilation
(``torch.compiler.disable``) and the rest of the layer is compiled.

v5.162: on CUDA, torch 2.8's inductor fused "fp32 routed output + bf16
shared-expert down-projection" into one addmm with mixed dtypes (v5.159's
explicit autocast did not avoid it); the shared output is now cast to the
promoted dtype before the add, as eager promotes it.

Pins: the dispatch methods carry the disable marker, and a compiled
model (dynamo with the eager backend, no inductor needed) gives the eager
model's loss and gradients for every dispatch mode.

Run from repo root: python3 -m pytest helioslm_v5/tests/test_moe_compile.py
"""
import copy
import sys
from pathlib import Path

try:
    import pytest
except ImportError:
    # This file is also executed directly by the daily TESTS suite
    # (case_upstream_suites), where pytest is not installed. The shim
    # covers the only pytest feature used below: mark.parametrize
    # (collection-time only; a no-op when the functions are called
    # directly with explicit arguments). Same pattern as
    # test_api_accounting.py (v5.155).
    class _Mark:
        @staticmethod
        def parametrize(*_args, **_kwargs):
            return lambda f: f

    class _PytestShim:
        mark = _Mark()

    pytest = _PytestShim()

import torch

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from helioslm_v5.src.moe.sigmoid_moe import DeviceLimitedMoE  # noqa: E402
from helioslm_v5.src.training.pretrain import HeliosLMv5, pilot_config  # noqa: E402


def test_dispatch_is_not_compiled():
    for name in ("_dispatch_blocked", "_dispatch_batched"):
        assert getattr(getattr(DeviceLimitedMoE, name), "_torchdynamo_disable", False), name


def _loss(model, ids):
    out = model(ids)
    logits = out["logits"] if isinstance(out, dict) else out[0]
    return torch.nn.functional.cross_entropy(
        logits[:, :-1].reshape(-1, logits.shape[-1]).float(), ids[:, 1:].reshape(-1))


@pytest.mark.parametrize("dispatch", ["blocked", "batched", "grouped"])
def test_compiled_equals_eager(dispatch):
    torch._dynamo.reset()
    cfg = pilot_config("moe", vocab_size=512, hidden_size=128, num_layers=2,
                       intermediate_size=256, max_seq=128, moe_dispatch=dispatch)
    torch.manual_seed(0)
    eager = HeliosLMv5(cfg).train()
    comp = copy.deepcopy(eager)
    fwd = torch.compile(comp, backend="eager")
    ids = torch.randint(0, 512, (2, 48), generator=torch.Generator().manual_seed(1))
    le, lc = _loss(eager, ids), _loss(fwd, ids)
    le.backward()
    lc.backward()
    assert torch.allclose(le, lc, atol=1e-6)
    for (n, a), (_, b) in zip(eager.named_parameters(), comp.named_parameters()):
        if a.grad is None:
            assert b.grad is None, n
        else:
            assert torch.allclose(a.grad, b.grad, atol=1e-6), n


def test_compiled_under_bf16_autocast():
    # v5.159 / v5.162: compiled and eager agree under bf16 autocast.
    torch._dynamo.reset()
    cfg = pilot_config("moe", vocab_size=512, hidden_size=128, num_layers=2,
                       intermediate_size=256, max_seq=128, moe_dispatch="batched")
    torch.manual_seed(0)
    eager = HeliosLMv5(cfg).train()
    fwd = torch.compile(copy.deepcopy(eager), backend="eager")
    ids = torch.randint(0, 512, (2, 48), generator=torch.Generator().manual_seed(1))
    with torch.autocast("cpu", dtype=torch.bfloat16):
        le, lc = _loss(eager, ids), _loss(fwd, ids)
    assert torch.allclose(le, lc, atol=1e-6)
