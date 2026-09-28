from enum import Enum

try:
    from .schema import ToolCallError
except ImportError:
    from schema import ToolCallError


class Route(Enum):
    DIRECT = "DIRECT"
    ESCALATE = "ESCALATE"


class GateViolation(Exception):
    pass


class Gate:
    def decide(self, call, context: dict) -> Route:
        raise NotImplementedError

    def escalate(self, call, context: dict) -> str:
        raise NotImplementedError

    # --- v5.32 typed decision interface (System One / Jev direction) ------
    # Additive: decide/escalate contract above is unchanged (T11-T19).
    # ask() turns the route + confidence into a typed, schema-checked
    # decision; ask_batch() lets a non-autoregressive head answer several
    # questions in one pass (see decision_head.py). The default ask_batch
    # is serial — correctness first, speed is an override.

    def ask(self, call, context: dict, kind: str, question: str):
        from decision import Choice, Noul, Score
        if kind == "choice":
            # route is our only Choice question; the descriptive text is
            # the question label, not the dispatch key
            route = self.decide(call, context)
            conf = context.get("confidence")
            return Choice(question, tuple(r.value for r in Route),
                          route.value, float(conf if conf is not None else 0.5))
        if kind == "noul":
            # conservative default: any ESCALATE decision means "yes, this
            # call needs scrutiny" for scrutiny-type questions
            route = self.decide(call, context)
            return Noul(question, "no" if route == Route.DIRECT else "yes",
                        float(context.get("confidence") or 0.5))
        if kind == "score":
            return Score(question, 0.5, 0.5)  # unopinionated default
        raise NotImplementedError(f"unknown decision kind {kind!r}")

    def ask_batch(self, call, context: dict, questions: tuple) -> dict:
        return {name: self.ask(call, context, kind, text)
                for name, kind, text in questions}


class FixedGate(Gate):
    def __init__(self, route: Route):
        self.route = route

    def decide(self, call, context) -> Route:
        return self.route

    def escalate(self, call, context) -> str:
        raise NotImplementedError("FixedGate has no oracle")


class OracleGate(Gate):
    def __init__(self, oracle):
        self.oracle = oracle

    def decide(self, call, context) -> Route:
        return Route.ESCALATE

    def escalate(self, call, context) -> str:
        return str(self.oracle(call, context))


class ThresholdGate(Gate):
    def __init__(self, tau: float, oracle):
        if not 0.0 <= tau <= 1.0:
            raise ValueError("tau must be in [0, 1]")
        self.tau = float(tau)
        self._oracle = oracle

    def decide(self, call, context) -> Route:
        conf = context.get("confidence")
        if conf is None:
            raise ToolCallError("ThresholdGate: context lacks 'confidence'")
        return Route.DIRECT if conf >= self.tau else Route.ESCALATE

    def escalate(self, call, context) -> str:
        return str(self._oracle(call, context))


def routing_gate(tau_correctness: list) -> None:
    for (t0, c0), (t1, c1) in zip(tau_correctness, tau_correctness[1:]):
        if t1 < t0:
            raise GateViolation("input must be sorted by ascending tau")
        if c1 < c0:
            raise GateViolation(
                f"monotonicity violated: tau {t0}->{t1} dropped correctness "
                f"{c0:.4f}->{c1:.4f} — this is a bug, not a trade-off")


def eval_tau_grid(taus, make_gate, run_eval) -> list:
    return [(tau, run_eval(make_gate(tau))) for tau in sorted(taus)]
