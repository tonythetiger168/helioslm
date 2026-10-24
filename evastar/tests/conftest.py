"""pytest configuration for evastar/tests (tech-debt Phase 3, v5.79).

Under pytest-xdist every worker is its own process with its own torch
thread pool; pin each to ONE thread so ``-n 4`` on a 4-vCPU runner does
not oversubscribe the cores (16 threads) — the same failure mode that made
test_v5.py pin 2 threads on CI (PR #16). Runs without xdist are untouched.
"""
import os

if os.environ.get("PYTEST_XDIST_WORKER"):
    import torch

    torch.set_num_threads(1)
