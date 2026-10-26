"""T152 — MoE layers under torch.compile, v5.158.

The first L4 compile run (torch 2.8) failed the MoE pilot: dynamo raised
GuardOnDataDependentSymNode on the per-expert count loop of the batched
dispatch. Both dispatch paths sync per-expert counts to the host and
build data-dependent shapes, so they are excluded from compilation
(``torch.compiler.disable``) and the rest of the layer is compiled.

Pins: the dispatch methods carry the disable marker, and a compiled
model (dynamo with the eager backend, no inductor needed) gives the eager
model's loss and gradients for every dispatch mode.

Run from repo root: python3 -m pytest evastar/tests/test_moe_compile.py
"""
import copy
import sys
from pathlib import Path

import pytest
import torch

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from evastar.src.moe.sigmoid_moe import DeviceLimitedMoE  # noqa: E402
from evastar.src.training.pretrain import EvastarV5, pilot_config  # noqa: E402


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
    eager = EvastarV5(cfg).train()
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
