"""hsh H3: the scoped tool registry and its guarded execution pipeline.

After dsh's ``core/tools`` (docs/tool-execution-pipeline.md,
docs/subsystems/tools.md). One call goes through, in order:

  parse + validate arguments
  -> ``tools/pre-execute``   waterfall: allow (call next), deny, or ask
  -> ``ctx.approval``         only for ask; absent or unanswered = deny
  -> guards                   monotonic: any guard may deny, none may allow
  -> ``tools/execute``        waterfall around the tool body (timeout,
                              retry, metrics); the body returns a JSON value
  -> ``project_content``      the tool's own content projection
  -> ``tools/post-execute``   waterfall: accept, replace or block the result
  -> ``finalize_content``     the tool's last content-only transform; runs
                              for every outcome, failures included
  -> ``tools/result``         emit: the frozen final outcome

Any exception inside the pipeline becomes an error result (code
``pipeline-error``, or ``tool-error`` from the body); it never escapes to
the agent loop. A tool hidden from a scope (restricted away, or never
registered) refuses with the same ``unknown-tool`` result, so a scope
cannot tell the two apart.

Scopes (dsh glossary): a tool registered through a scoped context belongs
to that scope only and shadows a global tool of the same name there.
``restrict(allow=..., deny=...)`` filters the *global* tools one scope
inherits; several restrictions intersect; the scope's own tools are
exempt. The pipeline events are dispatched scoped, so listeners
registered through an agent's context see only that agent's calls.
"""
from __future__ import annotations

import copy
import json

from .kernel import Service
from .schema import SchemaError, check_schema, validate

_EVENTS = {"tools/change": "emit", "tools/pre-execute": "waterfall",
           "tools/execute": "waterfall", "tools/post-execute": "waterfall",
           "tools/result": "emit"}


class ToolDefinition:
    """A tool: model-facing schema plus host-only behaviour.

    Only ``name``, ``description`` and ``parameters`` ever reach a model
    request (see :meth:`schema`).
    """

    def __init__(self, name, description, parameters, execute,
                 output=None, render=None, project_content=None,
                 finalize_content=None, concurrency_safe=None, meta=None):
        if not name or not isinstance(name, str):
            raise ValueError("a tool needs a name")
        self.name = name
        self.description = description
        self.parameters = check_schema(parameters)
        if parameters.get("type", "object") != "object":
            raise SchemaError(f"{name}: parameters must be an object schema")
        self.execute = execute                  # (args, run) -> JSON value
        self.output = check_schema(output) if output is not None else None
        self.render = render                    # (args, value) -> str
        self.project_content = project_content  # (exec, result) -> str | None
        self.finalize_content = finalize_content
        self.concurrency_safe = concurrency_safe
        self.meta = meta

    def schema(self):
        return {"name": self.name, "description": self.description,
                "parameters": copy.deepcopy(self.parameters)}


def tool(name, description, parameters, **kw):
    """Decorator form: ``@tool("calc", "...", {...})`` over the body."""
    def wrap(fn):
        return ToolDefinition(name, description, parameters, fn, **kw)
    return wrap


class Execution:
    """One call moving through the pipeline. Identity fields are fixed."""

    __slots__ = ("call_id", "name", "arguments", "scope", "parent", "tool",
                 "contexts", "body_ran")

    def __init__(self, call_id, name, arguments, scope, parent=None):
        self.call_id = call_id
        self.name = name
        self.arguments = arguments
        self.scope = scope
        self.parent = parent
        self.tool = None
        self.contexts = []      # deferred contexts, appended after tool/result
        self.body_ran = False

    def defer_context(self, text, source="tool"):
        self.contexts.append({"content": text, "source": source})


class ToolResult:
    """A normalized outcome. ``content`` is what the model reads."""

    def __init__(self, content, is_error=False, value=None, error=None,
                 meta=None, contexts=None):
        self.content = content
        self.is_error = is_error
        self.value = value
        self.error = error        # {"name", "code", "reason"?}, errors only
        self.meta = meta
        self.contexts = list(contexts or [])

    def replace(self, **kw):
        fields = dict(content=self.content, is_error=self.is_error,
                      value=self.value, error=self.error, meta=self.meta,
                      contexts=self.contexts)
        fields.update(kw)
        return ToolResult(**fields)

    def to_json(self):
        out = {"content": self.content, "is_error": self.is_error}
        for k in ("value", "error", "meta"):
            if getattr(self, k) is not None:
                out[k] = getattr(self, k)
        if self.contexts:
            out["contexts"] = self.contexts
        return json.loads(json.dumps(out, allow_nan=False))

    def session_data(self, turn, step, call_id):
        """The ``tool/result`` event payload for the session log."""
        msg = {"role": "tool", "call_id": call_id, "content": self.content}
        if self.is_error:
            msg["is_error"] = True
        data = {"turn": turn, "step": step, "message": msg}
        if self.is_error and self.error:
            data["error"] = self.error
        if self.meta is not None:
            data["meta"] = self.meta
        return data


def _fail(name, code, reason):
    return ToolResult(f"Error: {reason}", is_error=True,
                      error={"name": name, "code": code, "reason": reason})


class Tools(Service):
    """``ctx.tools``: register, restrict, guard, list and execute tools."""

    key = "tools"

    def start(self):
        for ev, mode in _EVENTS.items():
            self.ctx.declare(ev, mode)
        self._tools = {}          # scope (None = global) -> {name: def}
        self._restrictions = {}   # scope -> [ {"allow": set|None, "deny": set} ]
        self._guards = []         # [(scope, fn)]

    # ----- registration -----------------------------------------------
    def register(self, definition, ctx=None):
        """Register through ``ctx`` (default: the service's own context).

        The registration is an effect of ``ctx``'s plugin and is scoped to
        ``ctx.scope``. Returns the disposer.
        """
        ctx = ctx or self.ctx
        table = self._tools.setdefault(ctx.scope, {})
        if definition.name in table:
            raise ValueError(f"tool {definition.name!r} already registered here")
        table[definition.name] = definition
        self.ctx.emit("tools/change")

        def dispose():
            if table.get(definition.name) is definition:
                del table[definition.name]
                self.ctx.emit("tools/change")

        return ctx.effect(dispose)

    def restrict(self, ctx, allow=None, deny=None):
        """Filter the global tools ``ctx.scope`` inherits."""
        rule = {"allow": set(allow) if allow is not None else None,
                "deny": set(deny or ())}
        rules = self._restrictions.setdefault(ctx.scope, [])
        rules.append(rule)
        self.ctx.emit("tools/change")

        def dispose():
            if rule in rules:
                rules.remove(rule)
                self.ctx.emit("tools/change")

        return ctx.effect(dispose)

    def guard(self, fn, ctx=None):
        """``fn(execution) -> reason | None``; a reason denies the call."""
        ctx = ctx or self.ctx
        entry = (ctx.scope, fn)
        self._guards.append(entry)

        def dispose():
            if entry in self._guards:
                self._guards.remove(entry)

        return ctx.effect(dispose)

    # ----- visibility -------------------------------------------------
    def visible(self, scope=None):
        out = {}
        for name, d in self._tools.get(None, {}).items():
            if all((r["allow"] is None or name in r["allow"]) and name not in r["deny"]
                   for r in self._restrictions.get(scope, ())):
                out[name] = d
        if scope is not None:
            out.update(self._tools.get(scope, {}))  # own tools shadow globals
        return out

    def schemas(self, scope=None):
        """Model-facing schemas, sorted by name (stable across runs)."""
        return [d.schema() for _, d in sorted(self.visible(scope).items())]

    # ----- execution --------------------------------------------------
    def execute(self, call_id, name, arguments, scope=None, parent=None):
        """Run one call through the whole pipeline; always returns a result."""
        parse_error = "arguments must be a JSON object"
        if isinstance(arguments, str):
            try:
                args = json.loads(arguments) if arguments.strip() else {}
            except json.JSONDecodeError as exc:
                args = None
                parse_error = f"arguments are not valid JSON: {exc.msg}"
        else:
            args = copy.deepcopy(arguments)
        exe = Execution(call_id, name, args, scope, parent)
        sctx = self.ctx.scoped(scope)
        defn = self.visible(scope).get(name)
        exe.tool = defn
        try:
            if defn is None:
                result = _fail(name, "unknown-tool", f"unknown tool {name!r}")
            elif args is None:
                result = _fail(name, "invalid-arguments", parse_error)
            elif errs := validate(args, defn.parameters):
                result = _fail(name, "invalid-arguments", "; ".join(errs))
            else:
                result = self._admitted(sctx, exe, defn)
        except Exception as exc:
            result = _fail(name, "pipeline-error", f"{type(exc).__name__}: {exc}")
        if defn is not None and defn.finalize_content is not None:
            try:
                new = defn.finalize_content(exe, result)
                if new is not None:
                    result = result.replace(content=new)
            except Exception as exc:  # finalize must be total; contain it
                result = _fail(name, "pipeline-error", f"finalize_content: {exc}")
        result.contexts = result.contexts + exe.contexts
        frozen = ToolResult(**_unpack(result.to_json()))
        sctx.emit("tools/result", exe, frozen)
        return frozen

    def _admitted(self, sctx, exe, defn):
        decision = sctx.waterfall("tools/pre-execute", exe,
                                  final=lambda e: {"action": "allow"})
        action = (decision or {}).get("action", "allow")
        if action == "ask":
            approve = self.ctx.get("approval")
            ok = False
            if approve is not None:
                ok = approve(exe, decision.get("question", f"Run {exe.name}?")) is True
            if not ok:
                return _fail(exe.name, "denied", "approval refused or unavailable")
        elif action == "deny":
            return _fail(exe.name, "denied", decision.get("reason", "denied by policy"))
        elif action != "allow":
            raise ValueError(f"unknown pre-execute action {action!r}")
        for scope, fn in list(self._guards):
            if scope is None or scope is exe.scope:
                reason = fn(exe)
                if reason:
                    return _fail(exe.name, "denied", str(reason))

        result = sctx.waterfall("tools/execute", exe,
                                final=lambda e: self._run_body(e, defn))
        if defn.project_content is not None:
            new = defn.project_content(exe, result)
            if new is not None:
                result = result.replace(content=new)
        return sctx.waterfall("tools/post-execute", exe, result,
                              final=lambda e, r: r)

    def _run_body(self, exe, defn):
        exe.body_ran = True
        try:
            value = defn.execute(copy.deepcopy(exe.arguments), exe)
        except Exception as exc:
            return _fail(exe.name, "tool-error", f"{type(exc).__name__}: {exc}")
        try:
            json.dumps(value, allow_nan=False)
        except (TypeError, ValueError):
            return _fail(exe.name, "invalid-output", "tool returned a non-JSON value")
        if defn.output is not None:
            errs = validate(value, defn.output)
            if errs:
                return _fail(exe.name, "invalid-output", "; ".join(errs))
        if defn.render is not None:
            content = defn.render(exe.arguments, value)
        elif isinstance(value, str):
            content = value
        else:
            content = json.dumps(value, ensure_ascii=False, sort_keys=True)
        return ToolResult(content, value=value)


def _unpack(d):
    return {"content": d["content"], "is_error": d["is_error"],
            "value": d.get("value"), "error": d.get("error"),
            "meta": d.get("meta"), "contexts": d.get("contexts")}
