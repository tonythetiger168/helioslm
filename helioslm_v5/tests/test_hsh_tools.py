"""hsh H3: the tool registry and execution pipeline (hsh/tools.py, schema.py).

Pins:
1. the schema subset: unsupported keywords refused at registration; the
   validator's type, required, additionalProperties, enum, bounds, items
   and oneOf rules (bool is not an integer);
2. schemas() sends only name/description/parameters, sorted;
3. pipeline order: pre-execute -> guards -> execute -> project ->
   post-execute -> finalize -> tools/result, observed by listeners;
4. deny / ask (approval yes, no, absent) / guard deny skip the body;
   no guard can allow what another denies;
5. invalid JSON, schema violations, unknown tools, body exceptions,
   non-JSON and schema-violating output all become error results, and
   finalize_content still runs on them;
6. scopes: own tools shadow globals, restrictions intersect and hide a
   tool exactly like a missing one, scoped listeners see only their
   scope's calls; registrations unwind with their plugin;
7. a result converts to the session log's tool/result payload.
"""
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from helioslm_v5.hsh import (Context, SchemaError, ToolDefinition, Tools,  # noqa: E402
                             check_schema, tool, validate)

ADD = {"type": "object", "properties": {"a": {"type": "integer"}, "b": {"type": "integer"}},
       "required": ["a", "b"], "additionalProperties": False}


@tool("add", "Add two integers.", ADD)
def add_tool(args, run):
    return args["a"] + args["b"]


def setup(*defs):
    ctx = Context()
    ctx.plugin(Tools)
    for d in defs:
        ctx.tools.register(d)
    return ctx


# ----- schema ------------------------------------------------------------

def test_schema_subset_enforced():
    check_schema({"type": "object", "properties": {"x": {"type": ["string", "null"]}}})
    for bad in ({"type": "object", "patternProperties": {}},
                {"type": "strange"},
                {"oneOf": [{"type": "string"}]},
                {"properties": {"x": {"$ref": "#/a"}}},
                {"required": "x"}):
        with pytest.raises(SchemaError):
            check_schema(bad)
    with pytest.raises(SchemaError):
        ToolDefinition("t", "", {"type": "string"}, lambda a, r: 1)


def test_validator_rules():
    assert validate({"a": 1, "b": 2}, ADD) == []
    assert validate({"a": True, "b": 2}, ADD)          # bool is not integer
    assert validate({"a": 1}, ADD)                     # missing b
    assert validate({"a": 1, "b": 2, "c": 3}, ADD)     # extra property
    assert validate(2.0, {"type": "integer"}) == []
    s = {"type": "array", "items": {"enum": ["x", "y"]}, "maxItems": 2}
    assert validate(["x", "y"], s) == [] and validate(["z"], s) and validate(["x"] * 3, s)
    assert validate(5, {"type": "number", "minimum": 0, "maximum": 4})
    one = {"oneOf": [{"type": "string"}, {"type": "string", "maxLength": 1}]}
    assert validate("ab", one) == [] and validate("a", one)  # "a" matches both
    assert validate({"k": 1}, {"additionalProperties": {"type": "string"}})
    assert validate(None, {}) == []


def test_schemas_only_model_fields_sorted():
    b = ToolDefinition("b", "B", {"type": "object"}, lambda a, r: 0, meta={"secret": 1})
    ctx = setup(add_tool, b)
    out = ctx.tools.schemas()
    assert [s["name"] for s in out] == ["add", "b"]
    assert all(set(s) == {"name", "description", "parameters"} for s in out)


# ----- pipeline ----------------------------------------------------------

def test_happy_path_and_order():
    order = []

    def project(exe, res):
        order.append("project")
        return None

    def finalize(exe, res):
        order.append("finalize")
        return res.content + " (final)"

    t = ToolDefinition("add", "", ADD, lambda a, r: order.append("body") or a["a"] + a["b"],
                       project_content=project, finalize_content=finalize)
    ctx = setup(t)
    ctx.on("tools/pre-execute", lambda e, n: order.append("pre") or n())
    ctx.tools.guard(lambda e: order.append("guard"))
    ctx.on("tools/execute", lambda e, n: order.append("exec-in") or n())
    ctx.on("tools/post-execute", lambda e, r, n: order.append("post") or n())
    ctx.on("tools/result", lambda e, r: order.append(("result", r.content)))
    res = ctx.tools.execute("c1", "add", '{"a": 2, "b": 3}')
    assert not res.is_error and res.value == 5 and res.content == "5 (final)"
    assert order == ["pre", "guard", "exec-in", "body", "project", "post",
                     "finalize", ("result", "5 (final)")]


@pytest.mark.parametrize("decision,approval,expect_ran", [
    ({"action": "deny", "reason": "no"}, None, False),
    ({"action": "ask"}, None, False),               # no approval service
    ({"action": "ask"}, lambda e, q: False, False),
    ({"action": "ask"}, lambda e, q: "yes", False),  # only True approves
    ({"action": "ask"}, lambda e, q: True, True),
])
def test_deny_and_ask(decision, approval, expect_ran):
    ran = []
    t = ToolDefinition("add", "", ADD, lambda a, r: ran.append(1) or 0)
    ctx = setup(t)
    if approval is not None:
        ctx.provide("approval", approval)
    ctx.on("tools/pre-execute", lambda e, n: decision)
    res = ctx.tools.execute("c", "add", {"a": 1, "b": 1})
    assert bool(ran) == expect_ran
    assert res.is_error == (not expect_ran)
    if not expect_ran:
        assert res.error["code"] == "denied"


def test_guards_are_monotonic():
    ran = []
    ctx = setup(ToolDefinition("add", "", ADD, lambda a, r: ran.append(1) or 0))
    ctx.tools.guard(lambda e: None)                       # abstain
    ctx.tools.guard(lambda e: "rm -rf is not allowed")    # deny
    ctx.on("tools/pre-execute", lambda e, n: {"action": "allow"})  # explicit allow
    res = ctx.tools.execute("c", "add", {"a": 1, "b": 1})
    assert res.is_error and "rm -rf" in res.content and not ran


def test_post_execute_replaces():
    ctx = setup(add_tool)
    ctx.on("tools/post-execute", lambda e, r, n: r.replace(content="redacted"))
    assert ctx.tools.execute("c", "add", {"a": 1, "b": 1}).content == "redacted"


@pytest.mark.parametrize("name,args,code", [
    ("add", "{bad json", "invalid-arguments"),
    ("add", '{"a": 1}', "invalid-arguments"),
    ("add", None, "invalid-arguments"),
    ("nope", "{}", "unknown-tool"),
])
def test_input_failures_still_finalize(name, args, code):
    seen = []
    t = ToolDefinition("add", "", ADD, lambda a, r: 0,
                       finalize_content=lambda e, r: seen.append(r.error["code"]) or None)
    ctx = setup(t)
    res = ctx.tools.execute("c", name, args)
    assert res.is_error and res.error["code"] == code
    if name == "add":
        assert seen == [code]


def test_body_and_output_failures():
    boom = ToolDefinition("boom", "", {"type": "object"}, lambda a, r: 1 / 0)
    obj = ToolDefinition("obj", "", {"type": "object"}, lambda a, r: object())
    typed = ToolDefinition("typed", "", {"type": "object"}, lambda a, r: "x",
                           output={"type": "integer"})
    ctx = setup(boom, obj, typed)
    assert ctx.tools.execute("1", "boom", {}).error["code"] == "tool-error"
    assert ctx.tools.execute("2", "obj", {}).error["code"] == "invalid-output"
    assert ctx.tools.execute("3", "typed", {}).error["code"] == "invalid-output"


def test_listener_exception_is_contained():
    ctx = setup(add_tool)
    ctx.on("tools/pre-execute", lambda e, n: 1 / 0)
    res = ctx.tools.execute("c", "add", {"a": 1, "b": 1})
    assert res.is_error and res.error["code"] == "pipeline-error"


def test_deferred_context_and_args_not_mutable():
    def body(args, run):
        args["a"] = 999
        run.defer_context("file changed: x.py")
        return "ok"

    t = ToolDefinition("t", "", {"type": "object"}, body)
    ctx = setup(t)
    seen = []
    ctx.on("tools/result", lambda e, r: seen.append(e.arguments))
    res = ctx.tools.execute("c", "t", {"a": 1})
    assert res.contexts == [{"content": "file changed: x.py", "source": "tool"}]
    assert seen == [{"a": 1}]


# ----- scopes ------------------------------------------------------------

def test_scoped_shadow_restrict_and_dispatch():
    ctx = setup(add_tool, ToolDefinition("bash", "", {"type": "object"}, lambda a, r: "sh"))
    agent = object()
    actx = ctx.scoped(agent)
    ctx.tools.register(ToolDefinition("add", "agent add", ADD, lambda a, r: "mine"), ctx=actx)
    assert ctx.tools.execute("1", "add", {"a": 1, "b": 1}).content == "2"
    assert ctx.tools.execute("2", "add", {"a": 1, "b": 1}, scope=agent).content == "mine"

    ctx.tools.restrict(actx, deny=["bash"])
    ctx.tools.restrict(actx, allow=["bash", "add"])
    names = [s["name"] for s in ctx.tools.schemas(agent)]
    assert names == ["add"]                      # own add survives the filters
    hidden = ctx.tools.execute("3", "bash", {}, scope=agent)
    missing = ctx.tools.execute("4", "zzz", {}, scope=agent)
    assert hidden.error["code"] == missing.error["code"] == "unknown-tool"
    assert ctx.tools.execute("5", "bash", {}).content == "sh"   # global scope unaffected

    seen = []
    actx.on("tools/result", lambda e, r: seen.append(e.call_id))
    ctx.tools.execute("6", "add", {"a": 1, "b": 1})
    ctx.tools.execute("7", "add", {"a": 1, "b": 1}, scope=agent)
    assert seen == ["7"]


def test_registrations_unwind_with_plugin():
    ctx = setup()
    changes = []
    ctx.on("tools/change", lambda: changes.append(1))

    def plugin(c, cfg):
        c.tools.register(add_tool, ctx=c)
        c.tools.guard(lambda e: "blocked", ctx=c)

    plugin.inject = ["tools"]
    fork = ctx.plugin(plugin)
    assert ctx.tools.execute("1", "add", {"a": 1, "b": 1}).content == "Error: blocked"
    fork.dispose()
    assert ctx.tools.schemas() == [] and ctx.tools._guards == []
    assert len(changes) == 2
    with pytest.raises(ValueError):
        ctx.tools.register(add_tool)
        ctx.tools.register(add_tool)


def test_session_payload():
    ctx = setup(add_tool)
    ok = ctx.tools.execute("c1", "add", {"a": 1, "b": 1})
    bad = ctx.tools.execute("c2", "add", "{")
    assert ok.session_data(1, 2, "c1") == {
        "turn": 1, "step": 2, "message": {"role": "tool", "call_id": "c1", "content": "2"}}
    d = bad.session_data(1, 2, "c2")
    assert d["message"]["is_error"] is True and d["error"]["code"] == "invalid-arguments"
