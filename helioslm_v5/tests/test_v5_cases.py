"""The test_v5.py suite as pytest cases (tech-debt Phase 3, v5.79).

test_v5.py runs its 93 case functions one after another in one process,
which pytest-xdist cannot spread over workers. This module parametrizes
over the SAME list (test_v5.TESTS), so ``pytest -n 4`` runs them in
parallel with the rest of the suite. Every case raises on failure (none
records a failure without raising), so calling it is the whole check.
``python helioslm_v5/tests/test_v5.py`` keeps working unchanged.
"""
import os
import sys
from pathlib import Path

import pytest

_REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(_REPO))

from helioslm_v5.tests import test_v5  # noqa: E402

if os.environ.get("PYTEST_XDIST_WORKER"):
    # test_v5 pins 2 threads on CI at import; one per xdist worker instead
    import torch

    torch.set_num_threads(1)


def test_case_list_is_unique():
    names = [t.__name__ for t in test_v5.TESTS]
    assert len(names) == len(set(names)) == 93


@pytest.mark.parametrize("case", test_v5.TESTS,
                         ids=[t.__name__ for t in test_v5.TESTS])
def test_v5_case(case):
    case()
