"""Outcome-conditioned confidence recalibration (roadmap P2).

The failure this addresses is measured in ``docs/CALIBRATION.md``: under
a CONCEPT shift (p(y|x) flips while p(x) is unchanged — the probe's
rule-inverted split reuses the exact same state words), DecisionHead
accuracy collapses to 0.34 while confidence stays high and uninformative
(AUROC 0.545, a coin flip). No input-side mechanism can see that regime:
the inputs are in-distribution by construction, so density/OOD scores,
ensembles of identically-trained heads, and split-conformal all fire
blind. The only signal that can separate "right" from "wrong" there is
an OUTCOME — and the repo's verified-decision engine already defines
one: a decision counts as verified when an oracle or replay check backs
it (``docs/cost_per_verified_decision.md``).

This module spends that signal honestly. Each verified outcome
(``observe(ids, correct)``) is stored as the head's OWN encoder features
plus the 0/1 verdict; ``fit()`` trains a small ridge-logistic
reliability probe ``P(correct | z)`` on those verified records, and
``adjust(ids, conf)`` rescales reported confidence by it. The answer is
never changed — only the confidence — so accuracy is untouched by
construction and every point of AUROC lift comes from ranking.

Hard boundaries, kept loud:

- cold start / unfitted: ``adjust`` is the identity (never silently
  degrades an uninstrumented deployment);
- ``fit()`` refuses a single-class buffer (a probe trained on "always
  wrong" or "always right" cannot rank, and returning something anyway
  would manufacture confidence);
- the calibrator knows nothing about tasks, markers, or the shift — it
  sees encoder features and verdicts only.

Determinism: fixed-length projected gradient descent on CPU, no
sampling, no data-dependent control flow beyond the buffer itself.
"""
import torch

_DEFAULT_LAM = 10.0
_DEFAULT_EPOCHS = 300
_DEFAULT_LR = 0.5
_MIN_VERIFIED = 8


class OutcomeCalibrationError(ValueError):
    """Misuse that must never pass silently."""


class OutcomeCalibrator:
    """Ridge-logistic reliability probe over verified outcomes."""

    def __init__(self, encode, *, lam=_DEFAULT_LAM, epochs=_DEFAULT_EPOCHS,
                 lr=_DEFAULT_LR, min_verified=_MIN_VERIFIED):
        """encode: ids -> 1-D feature tensor (e.g. DecisionHead.encode).
        Frozen hyperparameters are recorded for provenance; they are not
        tuned per deployment."""
        if lam <= 0:
            raise OutcomeCalibrationError("lam must be positive")
        if epochs <= 0 or lr <= 0:
            raise OutcomeCalibrationError("epochs and lr must be positive")
        if min_verified < 2:
            raise OutcomeCalibrationError("min_verified must be >= 2")
        self._encode = encode
        self.lam = float(lam)
        self.epochs = int(epochs)
        self.lr = float(lr)
        self.min_verified = int(min_verified)
        self._buffer = []          # list of (z detached, correct 0.0/1.0)
        self._w = None             # probe weights over [standardized z, 1]
        self._mu = None
        self._sd = None

    @property
    def n_verified(self):
        return len(self._buffer)

    @property
    def fitted(self):
        return self._w is not None

    def observe(self, ids, correct):
        """Record one oracle/replay-verified outcome for a state."""
        if not isinstance(correct, (bool, int, float)) \
                or float(correct) not in (0.0, 1.0):
            raise OutcomeCalibrationError(
                "correct must be a 0/1 verdict, not a score")
        with torch.no_grad():
            z = self._encode(ids).detach().flatten().float()
        self._buffer.append((z, float(correct)))
        return self

    def fit(self):
        """Train the reliability probe on the verified buffer.

        Refuses two degenerate buffers loudly: too few verdicts (the
        probe would be noise fitted as signal) and a single verdict
        class (nothing to rank — a probe that always predicts one class
        rescales every confidence by the same factor, which is a
        no-op for AUROC and a lie for Brier)."""
        if len(self._buffer) < self.min_verified:
            raise OutcomeCalibrationError(
                f"need >= {self.min_verified} verified outcomes, "
                f"have {len(self._buffer)}")
        z = torch.stack([z for z, _ in self._buffer])
        y = torch.tensor([c for _, c in self._buffer])
        if float(y.min()) == float(y.max()):
            raise OutcomeCalibrationError(
                "verified buffer has a single verdict class; a probe "
                "trained on it cannot rank right from wrong")
        self._mu = z.mean(0)
        self._sd = z.std(0) + 1e-6
        x = torch.cat([(z - self._mu) / self._sd,
                       torch.ones(len(z), 1)], dim=1)
        w = torch.zeros(x.shape[1])
        for _ in range(self.epochs):
            p = torch.sigmoid(x @ w)
            grad = x.T @ (p - y) / len(y) + self.lam * w / len(y)
            w = w - self.lr * grad
        self._w = w
        return self

    def reliability(self, ids):
        """P(correct | state) from the fitted probe; None when unfitted."""
        if not self.fitted:
            return None
        with torch.no_grad():
            z = self._encode(ids).detach().flatten().float()
            x = torch.cat([(z - self._mu) / self._sd, torch.ones(1)])
            return float(torch.sigmoid(x @ self._w))

    def adjust(self, ids, conf):
        """Rescale a reported confidence by the fitted reliability.

        Identity when unfitted (cold start must never degrade a
        deployment that has no verified outcomes yet)."""
        if isinstance(conf, bool) or not isinstance(conf, (int, float)) \
                or not 0.0 <= conf <= 1.0:
            raise OutcomeCalibrationError(
                "conf must be a probability in [0, 1]")
        p = self.reliability(ids)
        if p is None:
            return float(conf)
        return float(conf) * p
