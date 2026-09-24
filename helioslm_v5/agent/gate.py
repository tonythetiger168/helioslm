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
