"""HeliosLM v5.32 P3 — DecisionHead-backed Gate (one-pass routing).

Closes the System-One loop: the gate's routing decisions are served by a
trained non-autoregressive DecisionHead instead of serial heuristics.

- decide() maps the head's route Choice back to the Route enum.
- ask_batch() answers the FULL pi-warden preset in ONE forward pass
  (the K-questions-for-the-price-of-one property; T30 counts passes).
- escalate() stays an oracle-only path — a trained head never fabricates
  oracle observations (recorded: escalate remains NotImplementedError,
  same discipline as FixedGate).
- Sub-floor uncertainty (v5.32.1 finding): route Choice confidence has
  a hard floor at 1/K = 0.5. Register a `certainty` Score question —
  sigmoid heads express [0,1] freely — and threshold THAT for
  sub-50% abstention. This is exactly why Jev ships three primitives.
"""
import torch

try:
    from .decision import Choice, Noul, Score
    from .decision_head import DecisionHead
    from .gate import Gate, Route
    from .trajectory import text_to_ids
except ImportError:
    from decision import Choice, Noul, Score
    from decision_head import DecisionHead
    from gate import Gate, Route
    from trajectory import text_to_ids


class CalibratedRouter(Gate):
    def __init__(self, head: DecisionHead, text_fn=None):
        self.head = head
        # state text at decision time: task + pending call (deviation
        # recorded in decision_data.py; production = transcript encoding)
        self.text_fn = text_fn or (
            lambda call, ctx: f"{ctx.get('task', '')} [{call.name}]")

    def _ids(self, call, context):
        return torch.tensor(text_to_ids(self.text_fn(call, context)))

    def decide(self, call, context):
        ans, conf = self.head.decide("route", self._ids(call, context))
        return Route(ans)

    def escalate(self, call, context):
        raise NotImplementedError(
            "CalibratedRouter never fabricates oracle observations")

    def ask_batch(self, call, context, questions):
        """One forward pass for every registered question. Questions the
        head was not trained for raise — no silent defaults."""
        out = self.head.forward(self._ids(call, context))
        answers = {}
        for name, kind, text in questions:
            if name not in out:
                raise KeyError(
                    f"CalibratedRouter: question {name!r} not registered "
                    "in the DecisionHead")
            ans, conf = out[name]
            if kind == "choice":
                options = self.head.qmeta[name][1]
                answers[name] = Choice(text, tuple(options), ans, conf)
            elif kind == "noul":
                answers[name] = Noul(text, ans)   # v5.36: ans IS P(yes)
            elif kind == "score":
                answers[name] = Score(text, ans, conf)
            else:
                raise ValueError(f"unknown kind {kind!r}")
        return answers

    def uncertainty(self, call, context, question):
        """v5.36: continuous-Noul uncertainty readout. P(yes) near 0.5
        is uncertainty; distance from 0.5 is how decided it is. The
        old `certainty` Score workaround is retired by this."""
        if question not in self.head.qmeta or \
                self.head.qmeta[question][0] != "noul":
            raise KeyError(f"register a continuous-noul question {question!r}")
        p, _ = self.head.forward(self._ids(call, context))[question]
        return p
