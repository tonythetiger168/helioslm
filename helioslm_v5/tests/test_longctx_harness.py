"""v5.42 — long-context passkey harness oracles (Phase 2.2).

Proves the harness measures retrieval: a stub that boosts seen tokens must
score 100%, a uniform stub must score chance, and the REAL untrained lite
model must run end-to-end (its number is harness smoke, NOT a capability
claim — the untrained model is expected at chance).

Run from repo root: python helioslm_v5/tests/test_longctx_harness.py
"""
import sys
from pathlib import Path

import torch
import torch.nn as nn

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

from helioslm_v5.configs.config_v5 import HeliosLMv5Config
from helioslm_v5.eval.longctx import (KEY_MARKER, QUERY_MARKER,
                                      candidate_logprob, make_candidates,
                                      make_task, run_passkey_eval)
from helioslm_v5.src.model_v5 import HeliosLMv5


class _SeenTokenBooster(nn.Module):
    """Logits 10 for every token already seen in the row, else 0 — a
    perfect retrieval stub. Vocab fixed small for speed."""

    def __init__(self, vocab=1024):
        super().__init__()
        self.vocab = vocab
        self.config = type("C", (), {"max_position_embeddings": 10 ** 9})()

    def forward(self, ids):
        B, T = ids.shape
        seen = torch.zeros(B, T, self.vocab)
        onehot = torch.nn.functional.one_hot(ids, self.vocab).float()
        seen = onehot.cumsum(dim=1).clamp(max=1) * 10.0
        return seen, None, None


class _UniformStub(nn.Module):
    def __init__(self, vocab=1024):
        super().__init__()
        self.vocab = vocab
        self.config = type("C", (), {"max_position_embeddings": 10 ** 9})()

    def forward(self, ids):
        B, T = ids.shape
        return torch.zeros(B, T, self.vocab), None, None


def test_task_construction():
    ids = make_task(100, key_id=15, pos_frac=0.3, seed=1)
    assert len(ids) == 100
    ki = ids.index(KEY_MARKER)
    assert ids[ki + 1] == 15
    assert ids[-1] == QUERY_MARKER
    assert 15 not in ids[:ki] + ids[ki + 2:-1], "key leaked into filler"
    cands = make_candidates(15, 8, seed=2)
    assert len(cands) == 8 and len(set(cands)) == 8 and 15 in cands
    for bad in (lambda: make_task(100, 60, 0.5, 0),      # key in filler rng
                lambda: make_task(100, 15, 1.7, 0)):     # bad pos_frac
        try:
            bad()
        except ValueError:
            pass
        else:
            raise AssertionError("invalid task config accepted")
    print("PASS test_task_construction length/markers/absent-key/no-leak")


def test_perfect_stub_scores_100():
    stub = _SeenTokenBooster()
    r = run_passkey_eval(stub, lengths=(128, 256), trials=2,
                         n_candidates=8, pos_fracs=(0.2, 0.8),
                         model_tag="seen-booster-stub")
    for L, row in r["lengths"].items():
        assert row["accuracy"] == 1.0, (L, row)
    print("PASS test_perfect_stub_scores_100 at 128/256, all positions")


def test_uniform_stub_scores_chance():
    stub = _UniformStub()
    r = run_passkey_eval(stub, lengths=(128,), trials=50,
                         n_candidates=4, pos_fracs=(0.5,),
                         model_tag="uniform-stub")
    acc = r["lengths"][128]["accuracy"]
    # 50 trials, p=0.25: 3-sigma band ~ [0.25 +- 0.18]; assert sane band
    assert 0.05 <= acc <= 0.50, f"uniform stub acc {acc} far from chance"
    print(f"PASS test_uniform_stub_scores_chance acc={acc:.2f} "
          f"(chance 0.25, n={r['lengths'][128]['n']})")


def test_untrained_lite_model_runs_end_to_end():
    torch.manual_seed(0)
    model = HeliosLMv5(HeliosLMv5Config(size="lite"))
    r = run_passkey_eval(model, lengths=(256, 512), trials=1,
                         n_candidates=4, pos_fracs=(0.5,),
                         model_tag="HeliosLMv5-lite-untrained")
    for L, row in r["lengths"].items():
        assert 0.0 <= row["accuracy"] <= 1.0 and row["n"] > 0
    # honesty contract: provenance is recorded with the numbers
    assert r["model"] == "HeliosLMv5-lite-untrained"
    assert r["max_position_embeddings"] == 2048
    print(f"PASS test_untrained_lite_model_runs_end_to_end "
          f"acc256={r['lengths'][256]['accuracy']} "
          f"acc512={r['lengths'][512]['accuracy']} (untrained ~ chance; "
          f"harness smoke only, NOT a capability claim)")


def test_beyond_native_fails_loudly():
    torch.manual_seed(0)
    model = HeliosLMv5(HeliosLMv5Config(size="lite"))
    try:
        run_passkey_eval(model, lengths=(4096,), trials=1,
                         n_candidates=4, pos_fracs=(0.5,))
    except ValueError as e:
        assert "rope_scaling" in str(e)
    else:
        raise AssertionError("silent rotary extrapolation accepted")
    print("PASS test_beyond_native_fails_loudly 4096 > 2048 needs "
          "explicit rope_scaling")


if __name__ == "__main__":
    test_task_construction()
    test_perfect_stub_scores_100()
    test_uniform_stub_scores_chance()
    test_untrained_lite_model_runs_end_to_end()
    test_beyond_native_fails_loudly()
    print("\n5/5 longctx harness tests passed")
