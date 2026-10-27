"""T88 — roadmap Phase A: calibration leaderboard oracles.

Pins:

1. auroc / brier against hand-computed values, including ties and the
   undefined single-class case (None, never a fake 0.5 or 1.0);
2. input validation is loud (non-bool correct, out-of-range conf, empty
   results, all-null conf, byte-identical duplicate inputs);
3. null-conf rows count toward accuracy but not calibration metrics;
4. ranking is by Brier with a deterministic tie-break;
5. the committed board and its .md companion rebuild byte-for-byte from
   the committed inputs, input hashes match, and the board is registered
   in the verification-tier registry.

Run from repo root: python3 helioslm_v5/tests/test_calibration_board.py
"""
import json
import subprocess
import sys
import tempfile
from pathlib import Path

_REPO = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(_REPO))

from helioslm_v5.eval.artifact_tiers import load_registry, tier_of  # noqa: E402
from helioslm_v5.eval.calibration_board import (  # noqa: E402
    SCHEMA_ID, BoardSchemaError, auroc, brier, build_board,
    canonical_sha256, score_artifact, sha256_file)

_BOARD = _REPO / "benchmarks" / "calibration_board_2026-10-15.json"
_SCRIPT = _REPO / "examples" / "calibration_board.py"


def _write(td, name, rows, **extra):
    p = Path(td) / name
    p.write_text(json.dumps({"results": rows, **extra}))
    return p


def _expect_loud(fn, needle):
    try:
        fn()
    except BoardSchemaError as e:
        assert needle in str(e), f"{needle!r} not in {e}"
        return
    raise AssertionError(f"expected BoardSchemaError containing {needle!r}")


def test_metric_oracles():
    # perfect separation, inverted separation, all ties
    assert auroc([0.9, 0.8, 0.2, 0.1], [True, True, False, False]) == 1.0
    assert auroc([0.1, 0.2, 0.8, 0.9], [True, True, False, False]) == 0.0
    assert auroc([0.5, 0.5, 0.5], [True, False, False]) == 0.5
    # 2 pos x 2 neg: (0.9>0.3)+(0.9>0.6)+(0.4>0.3)+(0.4<0.6) = 3/4
    assert auroc([0.9, 0.4, 0.3, 0.6], [True, True, False, False]) == 0.75
    # one class only -> undefined
    assert auroc([0.9, 0.8], [True, True]) is None
    assert auroc([0.9, 0.8], [False, False]) is None
    # brier: ((1-1)^2 + (0.5-0)^2 + (1-0)^2) / 3
    assert abs(brier([1.0, 0.5, 1.0], [True, False, False]) - 1.25 / 3) < 1e-12
    print("PASS test_metric_oracles")


def test_loud_validation():
    with tempfile.TemporaryDirectory() as td:
        bad_flag = _write(td, "a.json", [{"correct": "yes", "conf": 0.5}])
        _expect_loud(lambda: score_artifact(bad_flag), "correct must be a bool")
        for conf in (1.5, -0.1, "0.5", True):
            p = _write(td, "b.json", [{"correct": True, "conf": conf}])
            _expect_loud(lambda: score_artifact(p), "conf must be a number")
        empty = _write(td, "c.json", [])
        _expect_loud(lambda: score_artifact(empty), "non-empty list")
        nulls = _write(td, "d.json", [{"correct": True, "conf": None}])
        _expect_loud(lambda: score_artifact(nulls), "no row carries a conf")
        _expect_loud(lambda: score_artifact(Path(td) / "missing.json"),
                     "not found")
        _expect_loud(lambda: build_board([], "x"), "no artifacts")
        ok = _write(td, "e.json", [{"correct": True, "conf": 0.7}])
        dup = _write(td, "f.json", [{"correct": True, "conf": 0.7}])
        _expect_loud(lambda: build_board([ok, dup], "x"), "duplicate input")
    print("PASS test_loud_validation")


def test_null_conf_rows_and_ranking():
    with tempfile.TemporaryDirectory() as td:
        a = _write(td, "a.json", [
            {"correct": True, "conf": 0.9},
            {"correct": False, "conf": 0.95},
            {"correct": False, "conf": None},
        ], model="model-a")
        b = _write(td, "b.json", [
            {"correct": True, "conf": 0.8},
            {"correct": False, "conf": 0.2},
        ])
        row = score_artifact(a)
        assert row["model"] == "model-a"
        assert row["n"] == 3 and row["n_scored"] == 2
        assert row["accuracy"] == round(1 / 3, 4)   # null row counts here
        assert row["auroc"] == 0.0                   # wrong > right
        assert row["overconfident_wrong"] == 1 and row["n_wrong_scored"] == 1
        board = build_board([a, b], "2026-10-15")
        assert [r["artifact"] for r in board["board"]] == ["b.json", "a.json"]
        assert [r["rank"] for r in board["board"]] == [1, 2]
        assert board["results_sha256"] == canonical_sha256(board["board"])
        # identical metrics tie-break on artifact name
        c = _write(td, "c.json", json.loads(b.read_text())["results"],
                   model="same-metrics")
        names = [r["artifact"] for r in build_board([c, b], "x")["board"]]
        assert names == ["b.json", "c.json"]
    print("PASS test_null_conf_rows_and_ranking")


def test_committed_board_reproduces():
    board = json.loads(_BOARD.read_text(encoding="utf-8"))
    assert board["schema"] == SCHEMA_ID
    assert board["results_sha256"] == canonical_sha256(board["board"])
    for rel, digest in board["inputs"].items():
        assert sha256_file(_REPO / rel) == digest, f"input drifted: {rel}"
    reg = load_registry(_REPO / "benchmarks" / "verification_tiers.json")
    assert tier_of(_BOARD.name, reg) == "measured"
    r = subprocess.run([sys.executable, str(_SCRIPT), "--check"],
                       cwd=_REPO, capture_output=True, text=True, timeout=120)
    assert r.returncode == 0, r.stdout + r.stderr
    print("PASS test_committed_board_reproduces")


if __name__ == "__main__":
    test_metric_oracles()
    test_loud_validation()
    test_null_conf_rows_and_ranking()
    test_committed_board_reproduces()
