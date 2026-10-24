"""hsh H1: the plugin kernel (evastar/hsh/kernel.py).

Pins:
1. services: provide / lookup / duplicate refused / withdrawn on dispose;
2. inject: a plugin waits for its services, activates when they appear,
   unwinds when one disappears and comes back when it returns;
3. effects unwind last-in first-out, exactly once, including nested
   plugins and the services they provided;
4. a failing apply rolls back what it registered and leaves the fork
   'failed';
5. each dispatch mode behaves as declared, and dispatching with another
   mode (or an undeclared event) raises;
6. waterfall: next() delegates, argument replacement, short-circuit;
7. scopes: a scoped dispatch reaches untagged listeners plus its own.
"""
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from evastar.hsh import (Context, EventModeError, KernelError,  # noqa: E402
                             ListenerErrors, Service)


def test_provide_lookup_and_duplicate():
    ctx = Context()
    ctx.provide("llm", "adapter")
    assert ctx.llm == "adapter" and ctx.has("llm")
    with pytest.raises(KernelError):
        ctx.provide("llm", "other")
    with pytest.raises(AttributeError):
        ctx.missing
    with pytest.raises(KernelError):
        ctx.tools = 1


def test_service_withdrawn_when_plugin_disposed():
    ctx = Context()
    fork = ctx.plugin(lambda c, cfg: c.provide("tools", {"n": cfg}), 3)
    assert ctx.tools == {"n": 3}
    fork.dispose()
    assert not ctx.has("tools") and fork.state == "disposed"


def test_inject_waits_activates_and_reactivates():
    ctx = Context()
    seen = []

    def consumer(c, cfg):
        seen.append(("up", c.tools))
        c.effect(lambda: seen.append(("down",)))

    consumer.inject = ["tools"]
    fork = ctx.plugin(consumer)
    assert fork.state == "pending" and seen == []

    provider = ctx.plugin(lambda c, cfg: c.provide("tools", "T1"))
    assert fork.state == "active" and seen == [("up", "T1")]

    provider.dispose()
    assert fork.state == "pending" and seen[-1] == ("down",)

    ctx.plugin(lambda c, cfg: c.provide("tools", "T2"))
    assert fork.state == "active" and seen[-1] == ("up", "T2")


def test_effects_unwind_lifo_once_and_nested():
    ctx = Context()
    log = []

    def child(c, cfg):
        c.effect(lambda: log.append("child-effect"))

    def parent(c, cfg):
        c.effect(lambda: log.append("first"))
        c.plugin(child)
        c.effect(lambda: log.append("last"))

    fork = ctx.plugin(parent)
    fork.dispose()
    fork.dispose()
    assert log == ["last", "child-effect", "first"]


def test_disposer_runs_once_and_leaves_stack():
    ctx = Context()
    log = []

    def p(c, cfg):
        d = c.effect(lambda: log.append("x"))
        d()
        d()

    fork = ctx.plugin(p)
    assert log == ["x"] and fork._effects == []
    fork.dispose()
    assert log == ["x"]


def test_failed_apply_rolls_back():
    ctx = Context()
    ctx.declare("tick", "emit")

    def bad(c, cfg):
        c.provide("half", 1)
        c.on("tick", lambda: None)
        raise ValueError("boom")

    with pytest.raises(ValueError):
        ctx.plugin(bad)
    assert not ctx.has("half")
    assert ctx._root.listeners["tick"] == []
    assert [f.state for f in ctx.forks] == ["failed"]


def test_failure_during_settle_is_recorded_not_raised():
    ctx = Context()

    def bad(c, cfg):
        raise RuntimeError("late")

    bad.inject = ["x"]
    fork = ctx.plugin(bad)
    ctx.provide("x", 1)
    assert fork.state == "failed" and isinstance(fork.error, RuntimeError)
    assert ctx.errors and ctx.errors[-1][0] == "plugin/bad"


class Counter(Service):
    key = "counter"

    def start(self):
        self.n = 0
        self.ctx.declare("count", "emit")
        self.ctx.on("count", self.bump)
        self.started = True

    def bump(self):
        self.n += 1

    def stop(self):
        self.started = False


def test_service_subclass_lifecycle():
    ctx = Context()
    fork = ctx.plugin(Counter)
    svc = ctx.counter
    ctx.emit("count")
    ctx.emit("count")
    assert svc.n == 2 and svc.started
    fork.dispose()
    assert not ctx.has("counter") and not svc.started
    ctx.emit("count")
    assert svc.n == 2  # its listener went with it


def test_dependents_see_started_service():
    ctx = Context()
    seen = []

    def user(c, cfg):
        seen.append(c.counter.started)

    user.inject = ["counter"]
    ctx.plugin(user)
    ctx.plugin(Counter)
    assert seen == [True]


def test_modes_are_contracts():
    ctx = Context()
    with pytest.raises(EventModeError):
        ctx.emit("never-declared")
    ctx.declare("e", "emit")
    with pytest.raises(EventModeError):
        ctx.serial("e")
    with pytest.raises(EventModeError):
        ctx.declare("e", "bail")
    with pytest.raises(KernelError):
        ctx.declare("x", "broadcast")


def test_emit_swallows_and_records():
    ctx = Context()
    ctx.declare("e", "emit")
    log = []
    ctx.on("e", lambda v: 1 / 0)
    ctx.on("e", lambda v: log.append(v))
    ctx.emit("e", 7)
    assert log == [7] and isinstance(ctx.errors[0][1], ZeroDivisionError)


def test_serial_parallel_bail():
    ctx = Context()
    ctx.declare("s", "serial")
    ctx.declare("p", "parallel")
    ctx.declare("b", "bail")
    ctx.on("s", lambda x: x + 1)
    ctx.on("s", lambda x: x * 2)
    assert ctx.serial("s", 5) == [6, 10]

    ran = []
    ctx.on("p", lambda: ran.append(1) or 1 / 0)
    ctx.on("p", lambda: ran.append(2))
    ctx.on("p", lambda: (_ for _ in ()).throw(KeyError("k")))
    with pytest.raises(ListenerErrors) as info:
        ctx.parallel("p")
    assert ran == [1, 2] and len(info.value.errors) == 2

    ctx.on("b", lambda: None)
    ctx.on("b", lambda: "second")
    ctx.on("b", lambda: "third")
    assert ctx.bail("b") == "second"


def test_waterfall_delegate_replace_shortcircuit():
    ctx = Context()
    ctx.declare("w", "waterfall")
    order = []

    def outer(req, nxt):
        order.append("outer-in")
        out = nxt({**req, "outer": True})
        order.append("outer-out")
        return out + "!"

    def inner(req, nxt):
        order.append("inner")
        return nxt()

    ctx.on("w", outer)
    ctx.on("w", inner)
    out = ctx.waterfall("w", {"x": 1}, final=lambda r: f"final{sorted(r)}")
    assert out == "final['outer', 'x']!"
    assert order == ["outer-in", "inner", "outer-out"]

    ctx.on("w", lambda req, nxt: "denied", prepend=True)
    assert ctx.waterfall("w", {}, final=lambda r: "ran") == "denied"


def test_waterfall_without_listeners_runs_final():
    ctx = Context()
    ctx.declare("w", "waterfall")
    assert ctx.waterfall("w", 2, final=lambda v: v * 10) == 20
    assert ctx.waterfall("w", 2) == 2


def test_scoped_dispatch_filter_and_unwind():
    ctx = Context()
    ctx.declare("step", "serial")
    a, b = object(), object()
    ctx.on("step", lambda: "global")
    ctx.scoped(a).on("step", lambda: "a")
    ctx.scoped(b).on("step", lambda: "b")
    assert ctx.serial("step") == ["global"]
    assert ctx.scoped(a).serial("step") == ["global", "a"]
    assert ctx.scoped(b).serial("step") == ["global", "b"]

    def agent(c, cfg):
        c.on("step", lambda: "agent-only")

    key = object()
    fork = ctx.scoped(key).plugin(agent)
    assert ctx.scoped(key).serial("step") == ["global", "agent-only"]
    fork.dispose()
    assert ctx.scoped(key).serial("step") == ["global"]


def test_chain_of_services_cascades():
    # llm <- loop <- ui: removing llm unwinds loop, which withdraws its
    # service, which unwinds ui; providing llm again brings both back.
    ctx = Context()
    log = []

    def loop(c, cfg):
        c.provide("loop", "L")
        c.effect(lambda: log.append("loop-down"))

    loop.inject = ["llm"]

    def ui(c, cfg):
        log.append("ui-up")
        c.effect(lambda: log.append("ui-down"))

    ui.inject = ["loop"]
    ctx.plugin(ui)
    ctx.plugin(loop)
    llm = ctx.plugin(lambda c, cfg: c.provide("llm", "M"))
    assert log == ["ui-up"]
    llm.dispose()
    assert log == ["ui-up", "loop-down", "ui-down"]
    ctx.plugin(lambda c, cfg: c.provide("llm", "M2"))
    assert log[-1] == "ui-up" and ctx.loop == "L"
