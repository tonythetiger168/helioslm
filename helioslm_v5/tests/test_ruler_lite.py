"""OS7: RULER-lite tasks (eval/ruler_lite.py) and the probe's arms.

Pins:
1. every task builds sequences of exactly the requested length ending
   ``[QUERY k]``, deterministic per generator seed;
2. the answer is recoverable from the sequence by the task's rule
   (needle lookup; for vartrack, following ARROW links back to a value)
   and the queried key's needle is present exactly once;
3. too-short lengths are refused;
4. the four probe arms build, share width/depth, and run one forward and
   backward at a short length; ``accuracy`` returns a fraction.

Run from repo root: python3 -m pytest helioslm_v5/tests/test_ruler_lite.py
"""
import sys
from pathlib import Path

try:
    import pytest
except ImportError:
    # This file is also executed directly by the daily TESTS suite
    # (case_upstream_suites), where pytest is not installed. The shim
    # covers exactly the two pytest features used below: raises (no
    # match= arguments are passed) and mark.parametrize
    # (collection-time only; the wrapper calls test_tasks once per task
    # with explicit arguments). Same pattern as test_hsh_tools.py.
    import contextlib as _contextlib
    import re as _re

    class _Mark:
        @staticmethod
        def parametrize(*_args, **_kwargs):
            return lambda f: f

    class _PytestShim:
        mark = _Mark()

        @staticmethod
        @_contextlib.contextmanager
        def raises(exc, match=None):
            try:
                yield
            except exc as e:
                if match is not None and not _re.search(match, str(e)):
                    raise AssertionError(
                        f"raised {e!r}, which does not match /{match}/"
                    ) from None
                return
            raise AssertionError(f"{exc.__name__} was not raised")

    pytest = _PytestShim()

import torch

_REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(_REPO))
sys.path.insert(0, str(_REPO / "examples"))

from helioslm_v5.eval import ruler_lite as rl  # noqa: E402


def _solve(ids):
    """Answer by the task rule, reading the sequence only."""
    assert ids[-2] == rl.QUERY
    binds = {}
    for i, t in enumerate(ids[:-2]):
        if t == rl.NEEDLE:
            k = ids[i + 1]
            assert k not in binds, "key bound twice"
            binds[k] = ("ref", ids[i + 3]) if ids[i + 2] == rl.ARROW \
                else ("val", ids[i + 2])
    k = ids[-1]
    for _ in range(10):
        kind, x = binds[k]
        if kind == "val":
            return x
        k = x
    raise AssertionError("cycle")


@pytest.mark.parametrize("task", rl.TASKS)
def test_tasks(task):
    gen = torch.Generator().manual_seed(0)
    for length in (32, 100, 257):
        ids, ans = rl.make_batch(task, 16, length, gen)
        assert ids.shape == (16, length)
        assert ((ans >= rl.VAL0) & (ans < rl.VAL0 + rl.N_VALUES)).all()
        for row, a in zip(ids.tolist(), ans.tolist()):
            assert _solve(row) == a
            assert all(0 < t < rl.VOCAB for t in row)
    a1 = rl.make_batch(task, 4, 64, torch.Generator().manual_seed(5))
    a2 = rl.make_batch(task, 4, 64, torch.Generator().manual_seed(5))
    assert torch.equal(a1[0], a2[0]) and torch.equal(a1[1], a2[1])
    with pytest.raises(ValueError):
        rl.sample(task, 4, gen)


def test_probe_arms_run():
    import longctx_arch_probe as probe
    sizes = set()
    for arm in probe.ARMS:
        cfg = probe.arm_config(arm, hidden=64, window=8, topk=16)
        from helioslm_v5.src.model_v5 import HeliosLMv5
        torch.manual_seed(0)
        m = HeliosLMv5(cfg)
        sizes.add((cfg.num_hidden_layers, cfg.hidden_size))
        ids, ans = rl.make_batch("vartrack", 2, 48, torch.Generator().manual_seed(1))
        loss = torch.nn.functional.cross_entropy(m(ids)[0][:, -1], ans)
        loss.backward()
        assert torch.isfinite(loss)
        acc = rl.accuracy(m, "niah_single", 48, 1, 2)
        assert 0.0 <= acc <= 1.0
    assert len(sizes) == 1
