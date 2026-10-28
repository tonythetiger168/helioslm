"""HeliosLM v5.53 — TrustGate CLARIFY: three-state action routing.

M3 adds the missing middle state between "act" and "escalate". A tool call
that is syntactically a call but lacks required information is not a trust
failure and must not become a guess:

    DIRECT   — complete schema + trusted enough to execute
    CLARIFY  — required parameter missing; ask a concrete question
    ESCALATE — complete but distrusted, or schema-invalid beyond clarification

The wrapper is deliberately thin: schema-gap detection lives here, while the
trust decision remains owned by the wrapped gate (TrustGateV2 in production,
a FixedGate in oracle tests). Missing arguments are checked BEFORE trust so a
high-confidence partial call cannot execute by accident.
"""
try:
    from .gate import Gate, Route
    from .schema import ToolCallError, ToolRegistry
except ImportError:
    from gate import Gate, Route
    from schema import ToolCallError, ToolRegistry

SCHEMA_ID = "trust-gate-clarify/1"


class ClarifySchemaError(ValueError):
    """Raised for malformed clarify-gate inputs or vacuous questions."""


def inspect_call(call, registry: ToolRegistry) -> dict:
    """Classify a parsed call without executing it.

    Returns a deterministic record with status:
    - "complete": exact arg keys and exact builtin types match the ToolSpec;
    - "missing": one or more required params are absent and the present
      ones are exact (CLARIFY candidate);
    - "invalid": unknown tool, non-object args, extras, or wrong types.

    The distinction matters: missing information is a user question; invalid
    structure is an escalation/oracle problem. Both are loud, neither is
    executed.
    """
    try:
        spec = registry.spec(call.name)
    except ToolCallError as e:
        return {"status": "invalid", "missing": [], "extra": [],
                "wrong_types": [], "detail": str(e)}
    if not isinstance(call.args, dict):
        return {"status": "invalid", "missing": [], "extra": [],
                "wrong_types": [],
                "detail": f"{call.name}: args must be an object"}

    got, want = set(call.args), set(spec.params)
    missing = sorted(want - got)
    extra = sorted(got - want)
    wrong_types = []
    for key in sorted(got & want):
        expected = spec.params[key]
        value = call.args[key]
        if type(value) is not expected:
            wrong_types.append({
                "param": key,
                "expected": expected.__name__,
                "got": type(value).__name__,
            })
    # Invalid structure wins over missing: a call that is both incomplete
    # AND has extras/wrong types is malformed, not merely underspecified —
    # asking for the missing field would still leave a broken call.
    if missing and not (extra or wrong_types):
        status = "missing"
        detail = "missing required parameter(s): " + ", ".join(missing)
    elif extra or wrong_types:
        status = "invalid"
        bits = []
        if missing:
            bits.append("missing parameter(s): " + ", ".join(missing))
        if extra:
            bits.append("extra parameter(s): " + ", ".join(extra))
        if wrong_types:
            bits.append("wrong type(s): " + ", ".join(
                f"{r['param']}({r['got']} != {r['expected']})"
                for r in wrong_types))
        detail = "; ".join(bits)
    else:
        status = "complete"
        detail = ""
    return {"status": status, "missing": missing, "extra": extra,
            "wrong_types": wrong_types, "detail": detail}


class TrustGateClarify(Gate):
    """Three-state wrapper around any Gate.

    Args:
        inner: gate consulted only for schema-complete calls. Use
            TrustGateV2 for production trust math; FixedGate is enough for
            constructive oracle tests.
        registry: ToolRegistry whose ToolSpec params define required fields.

    The wrapper owns ONLY the missing/invalid split. It does not weaken the
    inner gate: a complete call still routes exactly as the inner gate says.
    """

    def __init__(self, inner: Gate, registry: ToolRegistry):
        if inner is None:
            raise ClarifySchemaError("TrustGateClarify requires an inner gate")
        if not isinstance(registry, ToolRegistry):
            raise ClarifySchemaError("registry must be a ToolRegistry")
        self.inner = inner
        self.registry = registry
        self.log: list = []

    def decide_explain(self, call, context: dict) -> dict:
        info = inspect_call(call, self.registry)
        if info["status"] == "missing":
            route = Route.CLARIFY
        elif info["status"] == "invalid":
            route = Route.ESCALATE
        else:
            route = self.inner.decide(call, context)
            if not isinstance(route, Route):
                raise ClarifySchemaError(
                    f"inner gate returned non-Route {route!r}")
        rec = {
            "schema": SCHEMA_ID,
            "call": call.name,
            "status": info["status"],
            "missing": info["missing"],
            "extra": info["extra"],
            "wrong_types": info["wrong_types"],
            "detail": info["detail"],
            "route": route.value,
        }
        self.log.append(rec)
        return rec

    def decide(self, call, context: dict) -> Route:
        return Route[self.decide_explain(call, context)["route"]]

    def clarify(self, call, context: dict) -> str:
        info = inspect_call(call, self.registry)
        if info["status"] != "missing":
            raise ClarifySchemaError(
                f"CLARIFY requested for non-missing call: {info['status']}")
        spec = self.registry.spec(call.name)
        wants = ", ".join(
            f"{name}: {spec.params[name].__name__}" for name in info["missing"])
        question = (f"CLARIFY: {call.name} is missing required parameter(s) "
                    f"{wants}. Please provide "
                    f"{', '.join(info['missing'])}.")
        if len(question) < 16:
            raise ClarifySchemaError("vacuous clarification question")
        return question

    def escalate(self, call, context: dict) -> str:
        return self.inner.escalate(call, context)

    def ask(self, call, context: dict, kind: str, question: str):
        return self.inner.ask(call, context, kind, question)
