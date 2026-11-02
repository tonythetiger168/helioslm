"""Long-horizon agent / secure-coding benchmark (#174): deterministic,
CPU-only, scripted policies; negative controls must be caught."""
import json
import sys
from pathlib import Path

try:
    import pytest
except ImportError:
    # This file is also executed directly by the daily TESTS suite
    # (case_upstream_suites), where pytest is not installed. The shim
    # covers exactly the pytest features used below: fixture (a no-op
    # marker — the wrapper calls the fixture function directly and
    # passes its return value in) and raises(match=...) (pytest
    # semantics: re.search on str(exception)). Same pattern as
    # test_rephrase.py (v5.225).
    import contextlib as _contextlib
    import re as _re

    class _PytestShim:
        @staticmethod
        def fixture(*args, **kwargs):
            # Both spellings occur: bare @pytest.fixture (the function
            # arrives as the first positional argument) and
            # @pytest.fixture(...) (a decorator factory).
            if len(args) == 1 and callable(args[0]) and not kwargs:
                return args[0]
            return lambda f: f

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

_REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(_REPO))

from helioslm_v5.eval import long_horizon_bench as lh  # noqa: E402

_ART = _REPO / "benchmarks" / "long_horizon_2026-11-01.json"


@pytest.fixture(scope="module")
def suite():
    return lh.run_suite()


def test_correct_policy_solves_everything(suite):
    s = suite["summary"]
    assert s["success_rate"] == 1.0 and s["scenarios"] == 5
    assert s["recovered_tool_errors"] >= 1     # the missing-file recovery
    assert max(r["steps"] for r in suite["rows"]) >= 10
    for r in suite["rows"]:
        assert r["replay_ok"] and r["violations"] == 0, r
        assert not r["injection_followed"] and not r["false_completion"]


def test_negative_controls_are_caught(suite):
    got = {(r["policy"], r["scenario"]): r for r in suite["controls"]}
    assert set(got) == set(lh.NEGATIVE_CONTROLS)
    for key, want in lh.NEGATIVE_CONTROLS.items():
        r = got[key]
        assert not r["success"] and want <= lh._flags(r), (key, r)
        assert r["replay_ok"], key
    inj = got[("injected", "prompt_injection_in_file")]
    assert inj["final_answer"] == "PASS" and inj["violations"] == 1


def test_deterministic_and_matches_artifact(suite):
    again = lh.run_suite()
    assert again["results_sha256"] == suite["results_sha256"]
    art = json.loads(_ART.read_text(encoding="utf-8"))
    assert art["schema"] == lh.SCHEMA_ID
    assert art["results_sha256"] == suite["results_sha256"]
    assert art["summary"] == suite["summary"]


def test_suite_fails_when_correct_policy_breaks(monkeypatch):
    monkeypatch.setitem(lh.POLICIES, "correct", lh.false_claim_policy)
    with pytest.raises(lh.LongHorizonError, match="correct policy failed"):
        lh.run_suite()


def test_suite_fails_when_a_control_is_not_caught(monkeypatch):
    controls = dict(lh.NEGATIVE_CONTROLS)
    controls[("correct", "write_then_report")] = {"false_completion"}
    monkeypatch.setattr(lh, "NEGATIVE_CONTROLS", controls)
    with pytest.raises(lh.LongHorizonError, match="not caught"):
        lh.run_suite()


def test_evaluate_policy_reports_rates():
    good = lh.evaluate_policy(lh.correct_policy)
    assert good["success_rate"] == 1.0 and good["violation_rate"] == 0

    def follows_files(sc, obs):
        if sc.id == "prompt_injection_in_file":
            return lh.injected_policy(sc, obs)
        return lh.correct_policy(sc, obs)

    bad = lh.evaluate_policy(follows_files, "follows_files")
    assert bad["success_rate"] == 0.8
    assert bad["injection_rate"] == 0.2 and bad["violation_rate"] == 0.2
    assert "follows_files" not in lh.POLICIES
