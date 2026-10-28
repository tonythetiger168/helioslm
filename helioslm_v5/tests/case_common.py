"""Shared helpers for the split HeliosLM v5 test suite.

Everything here lived at the head of the monolithic test_v5.py before
the 2026-10-11 split: the repo-root bootstrap, the RESULTS accumulator,
the pass/fail/expect helpers, the lite-model factory, and the two
module-level DualPipe stage builders (module-level so multiprocessing
spawn can pickle them by qualified name).
"""
import sys
import traceback
from pathlib import Path

import torch
import torch.nn as nn

from helioslm_v5.configs.config_v5 import HeliosLMv5Config

# Derive the repo root from this file: <root>/helioslm_v5/tests/case_common.py
_REPO_ROOT = Path(__file__).resolve().parents[2]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

RESULTS = []


def _pass(name, detail=""):
    RESULTS.append((name, True))
    print(f"[PASS] {name}" + (f" — {detail}" if detail else ""))


def _fail(name):
    RESULTS.append((name, False))
    print(f"[FAIL] {name}")
    traceback.print_exc()


def _expect_raises(exc_type, fn, what):
    try:
        fn()
    except exc_type:
        return
    raise AssertionError(f"expected {exc_type.__name__} from {what}")


def _lite_model(seed):
    from helioslm_v5.src.model_v5 import HeliosLMv5
    torch.manual_seed(seed)
    config = HeliosLMv5Config(size="lite")
    return HeliosLMv5(config).eval(), config


def _mpdp_stage_builder(rank: int):
    """Deterministic per-rank stage builder (module level: spawn pickling)."""
    from helioslm_v5.src.training.dualpipe import DualPipeStage
    torch.manual_seed(1234 + rank)
    return DualPipeStage(nn.ModuleList([
        nn.Linear(32, 32), nn.ReLU(),
        nn.Linear(32, 32 if rank == 0 else 16),
    ]))


def _mpdp_attnres_stage_builder(rank: int):
    """Attn-res threading stage builder (module level: spawn pickling)."""
    from helioslm_v5.src.model_v5 import HeliosLMv5
    from helioslm_v5.src.training.dualpipe import DualPipeStage, LayerWrap
    torch.manual_seed(4321 + rank)
    config = HeliosLMv5Config(size="lite")
    config.use_attention_residuals = True
    model = HeliosLMv5(config)
    return DualPipeStage(nn.ModuleList([LayerWrap(model.layers[rank])]))
