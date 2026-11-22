"""P2 (2026-11-23): outcome-conditioned confidence recalibration oracles.

The measured failure (docs/CALIBRATION.md): under a rule-inverted
CONCEPT shift the DecisionHead keeps high, uninformative confidence
(AUROC 0.545 — a coin flip) because p(x) is unchanged and only p(y|x)
flipped; no input-side detector can see it. ``agent/outcome_calibration``
spends the only signal that can separate right from wrong there:
oracle/replay-verified outcomes, the same verdict the cost model
(`docs/cost_per_verified_decision.md`) already counts as a verified
decision.

Oracles here pin, at gate speed (~seconds, CPU, fixed seeds):

1. cold-start identity and loud misuse guards;
2. the ridge reliability probe ranking regions on a synthetic encoder;
3. the real failure regime reproduced at small scale — a trained
   DecisionHead under rule inversion scores AUROC < 0.75, and after 30
   verified outcomes the recalibrated confidence clears AUROC >= 0.75
   with answers (and therefore accuracy) untouched.

The full-size measured row lives in docs/calibration_probe.py and
docs/CALIBRATION.md; this suite guards the mechanism, not the numbers.
"""
import random
import sys
from pathlib import Path

_REPO = Path(__file__).resolve().parent.parent.parent
if str(_REPO) not in sys.path:
    sys.path.insert(0, str(_REPO))

import torch  # noqa: E402

from helioslm_v5.agent.decision_head import DecisionHead  # noqa: E402
from helioslm_v5.agent.outcome_calibration import (  # noqa: E402
    OutcomeCalibrationError, OutcomeCalibrator)
from helioslm_v5.eval.calibration_board import auroc, brier  # noqa: E402
from helioslm_v5.tests.case_common import _pass  # noqa: E402

_ROUTES = ("direct", "tool", "escalate")


def _expect_loud(fn, needle):
    try:
        fn()
    except OutcomeCalibrationError as exc:
        assert needle in str(exc), f"{needle!r} not in {exc}"
        return
    raise AssertionError(f"expected OutcomeCalibrationError: {needle!r}")


def _ids(text):
    return torch.tensor([ord(c) % 1024 for c in text], dtype=torch.long)


def test_cold_start_identity_and_guards():
    cal = OutcomeCalibrator(_ids)
    # Cold start: no verified outcomes -> adjust is the identity and
    # reliability reports "no opinion", never a manufactured number.
    assert cal.adjust("alpha task", 0.9) == 0.9
    assert cal.adjust("beta state", 0.4) == 0.4
    assert cal.reliability("alpha task") is None
    assert cal.n_verified == 0 and not cal.fitted
    # Misuse is loud, never silently coerced.
    _expect_loud(lambda: cal.adjust("x", 1.5), "in [0, 1]")
    _expect_loud(lambda: cal.adjust("x", True), "in [0, 1]")
    _expect_loud(lambda: cal.observe("x", 0.7), "0/1 verdict")
    _expect_loud(lambda: cal.fit(), "verified outcomes")
    _expect_loud(lambda: OutcomeCalibrator(_ids, lam=0), "lam")
    _expect_loud(lambda: OutcomeCalibrator(_ids, epochs=0), "epochs")
    _expect_loud(lambda: OutcomeCalibrator(_ids, min_verified=1),
                 "min_verified")
    # A single-verdict-class buffer cannot rank right from wrong.
    for i in range(8):
        cal.observe(f"state {i}", False)
    _expect_loud(lambda: cal.fit(), "single verdict class")
    print("PASS test_cold_start_identity_and_guards")


def test_reliability_probe_ranks_regions():
    """Synthetic encoder with separable regions: 'alpha' states are
    always verified wrong, 'beta' states always right. After fit() the
    probe must rank them decisively and adjust must invert their
    confidence order."""
    def encode(text):
        # one-hot-ish region features, no training involved
        return torch.tensor([10.0, 0.0] if "alpha" in text
                            else [0.0, 10.0] if "beta" in text
                            else [0.0, 0.0])
    cal = OutcomeCalibrator(encode)
    for i in range(10):
        cal.observe(f"alpha variant {i}", False)
        cal.observe(f"beta variant {i}", True)
    cal.fit()
    p_wrong_region = cal.reliability("alpha novel unseen")
    p_right_region = cal.reliability("beta novel unseen")
    # The fixed ridge hyperparameters (lam=10, tuned for d=64 noisy
    # encoder features) shrink the clean 2-D probe toward 0.5; what must
    # be decisive is the ORDER and the gap, not saturation.
    assert p_wrong_region < 0.35, p_wrong_region
    assert p_right_region > 0.65, p_right_region
    assert p_right_region - p_wrong_region > 0.4
    # The head is MORE confident on the state it gets wrong (the
    # concept-shift signature); recalibration must flip the order.
    assert cal.adjust("alpha novel unseen", 0.95) < \
        cal.adjust("beta novel unseen", 0.60)
    print("PASS test_reliability_probe_ranks_regions")


def _mk_shift_state(rng, route, shift):
    """Same generator family as docs/calibration_probe.py (rule-inverted
    alpha<->gamma under shift), smaller and solvable-free for gate
    speed."""
    marker_route = {"direct": "escalate", "tool": "tool",
                    "escalate": "direct"}[route] if shift else route
    marker = {"direct": "alpha", "tool": "beta",
              "escalate": "gamma"}[marker_route]
    filler = " ".join(rng.choice(["task", "state", "node", "queue",
                                  "delta", "omega"]) for _ in range(6))
    return {"ids": _ids(f"{marker} feasible {filler}"),
            "answers": {"route": route}}


def test_decision_head_shift_recalibration():
    rng = random.Random(31337)
    torch.manual_seed(11)
    head = DecisionHead({"route": ("choice", _ROUTES)}, d=32)
    head.fit([_mk_shift_state(rng, rng.choice(_ROUTES), False)
              for _ in range(160)], epochs=120, lr=1e-2)
    stream = [_mk_shift_state(rng, rng.choice(_ROUTES), True)
              for _ in range(120)]

    # Phase 1: reproduce the failure regime — raw confidence cannot rank
    # correctness under the inverted rule.
    base_conf, base_ok = [], []
    cal = OutcomeCalibrator(head.encode)
    budget = 30
    for i, r in enumerate(stream):
        ans, conf = head.decide("route", r["ids"])
        ok = 1.0 if ans == r["answers"]["route"] else 0.0
        base_conf.append(conf)
        base_ok.append(ok)
        if i < budget:
            cal.observe(r["ids"], ok)
    base_auroc = auroc(base_conf, base_ok)
    assert base_auroc < 0.75, f"failure regime did not reproduce: {base_auroc}"

    # Phase 2: fit the probe on the verified budget; recalibrated
    # confidence must clear the floor the raw confidence could not.
    cal.fit()
    adj_conf, adj_ok = [], []
    for r in stream[budget:]:
        ans, conf = head.decide("route", r["ids"])
        ok = 1.0 if ans == r["answers"]["route"] else 0.0
        adj_conf.append(cal.adjust(r["ids"], conf))
        adj_ok.append(ok)
    adj_auroc = auroc(adj_conf, adj_ok)
    assert adj_auroc >= 0.75, f"recalibrated AUROC below floor: {adj_auroc}"
    assert brier(adj_conf, adj_ok) < brier(base_conf[budget:], base_ok[budget:])

    # Recalibration never changes answers, only their confidence — and
    # it is deterministic.
    for r in stream[budget:budget + 5]:
        ans_before, conf_before = head.decide("route", r["ids"])
        adjusted = cal.adjust(r["ids"], conf_before)
        ans_after, conf_after = head.decide("route", r["ids"])
        assert (ans_before, conf_before) == (ans_after, conf_after)
        assert cal.adjust(r["ids"], conf_before) == adjusted
        assert 0.0 <= adjusted <= conf_before
    print("PASS test_decision_head_shift_recalibration "
          f"(base AUROC {base_auroc:.3f} -> {adj_auroc:.3f}, "
          f"budget {budget})")


def test_outcome_calibration_suite():
    """Run the P2 outcome-recalibration oracles in the daily gate."""
    test_cold_start_identity_and_guards()
    test_reliability_probe_ranks_regions()
    test_decision_head_shift_recalibration()
    _pass("test_outcome_calibration_suite",
          "P2: verified-outcome reliability probe lifts concept-shift "
          "AUROC past the floor; cold start identity, loud guards")


if __name__ == "__main__":
    test_outcome_calibration_suite()
