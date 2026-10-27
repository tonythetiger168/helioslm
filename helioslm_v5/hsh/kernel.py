"""hsh kernel: plugins, services, reversible effects and typed events.

Python counterpart of the Cordis layer under DeepSeek Harness (dsh, MIT,
github.com/deepseek-ai/deepseek-harness, docs/cordis-primer.md). Written
from the design documents, not ported from the TypeScript source.

Five ideas, as in Cordis:

1. A plugin is a callable or a ``Service`` subclass, with an optional
   ``inject`` list of service keys it needs.
2. A ``Context`` is a repository of services: ``ctx.provide("tools", t)``
   and then ``ctx.tools`` anywhere in the tree.
3. ``inject`` expresses load order: a plugin whose services are missing
   waits, activates when they appear, and is unwound (then waits again)
   when one of them disappears.
4. Events have a declared dispatch mode (emit, waterfall, parallel, serial,
   bail); dispatching with a different method raises, so the mode is part
   of the event's contract.
5. Every registration is an effect with a disposer, kept on the plugin
   that made it. Disposing the plugin unwinds its effects last-in
   first-out, including the services it provided, which in turn unwinds
   the plugins that injected them.

Scopes (dsh glossary, "agent-scope"): ``ctx.scoped(key)`` returns a
context whose registrations are tagged with ``key``. A scoped dispatch
reaches untagged listeners plus the listeners of its own scope only.

Unlike Cordis this kernel is synchronous: the harness runs on CPU in one
process and determinism matters more than concurrency for the oracle
tests. ``parallel`` calls every listener and raises the collected errors
together instead of running them concurrently.
"""
from __future__ import annotations

import itertools

MODES = ("emit", "waterfall", "parallel", "serial", "bail")

_ids = itertools.count(1)


class KernelError(RuntimeError):
    pass


class EventModeError(KernelError):
    """An event was dispatched with a method other than its declared mode."""


class ListenerErrors(KernelError):
    """Errors raised by listeners of a ``parallel`` dispatch, collected."""

    def __init__(self, event, errors):
        super().__init__(f"{event}: {len(errors)} listener(s) failed")
        self.event = event
        self.errors = errors


class _Listener:
    __slots__ = ("id", "fn", "scope")

    def __init__(self, fn, scope):
        self.id = next(_ids)
        self.fn = fn
        self.scope = scope


class Fork:
    """A plugin mounted on a context: its state and its effect stack."""

    def __init__(self, root, plugin, config, parent_fork, scope):
        self.id = next(_ids)
        self.root = root
        self.plugin = plugin
        self.config = config
        self.parent = parent_fork
        self.scope = scope
        self.inject = tuple(getattr(plugin, "inject", ()) or ())
        self.state = "pending"     # pending | active | failed | disposed
        self.error = None
        self.instance = None       # the Service object, for Service plugins
        self._effects = []         # [(id, disposer)] in registration order
        self.ctx = Context(root, self, scope)

    @property
    def name(self):
        return getattr(self.plugin, "name", None) or getattr(
            self.plugin, "__name__", type(self.plugin).__name__)

    def ready(self):
        return all(self.root._has(k) for k in self.inject)

    def _activate(self):
        if self.state != "pending" or not self.ready():
            return
        self.state = "active"
        try:
            if isinstance(self.plugin, type) and issubclass(self.plugin, Service):
                # start before provide, so dependents never see a service
                # that has not started; unwinding withdraws it, then stops it
                self.instance = self.plugin(self.ctx, self.config)
                self.instance.start()
                self.ctx.effect(self.instance.stop)
                self.ctx.provide(self.plugin.key, self.instance)
            elif hasattr(self.plugin, "apply"):
                self.plugin.apply(self.ctx, self.config)
            else:
                self.plugin(self.ctx, self.config)
        except Exception as exc:  # roll back what apply registered
            self._unwind()
            self.state = "failed"
            self.error = exc
            raise

    def _unwind(self):
        while self._effects:
            _, dispose = self._effects.pop()
            dispose()

    def _deactivate(self):
        """A required service vanished: unwind and wait for it again."""
        if self.state == "active":
            self.state = "pending"
            self._unwind()

    def dispose(self):
        if self.state == "disposed":
            return
        self._unwind()
        self.state = "disposed"
        self.root._forks.discard(self)
        if self.parent is not None:
            self.parent._drop_effect_for(self)

    def _drop_effect_for(self, child):
        self._effects = [(i, d) for i, d in self._effects
                         if getattr(d, "_fork", None) is not child]


class Service:
    """Base for a plugin that owns one ``ctx.<key>``.

    Subclasses set ``key`` (and optionally ``inject``) and may override
    ``start`` and ``stop``. ``self.ctx`` is the plugin's own context, so
    registrations made through it unwind with the service.
    """

    key = None
    inject = ()

    def __init__(self, ctx, config=None):
        if not self.key:
            raise KernelError(f"{type(self).__name__} has no key")
        self.ctx = ctx
        self.config = config

    def start(self):
        pass

    def stop(self):
        pass


class _Root:
    """State shared by every context of one tree."""

    def __init__(self):
        self.services = {}       # key -> (value, providing fork or None)
        self.modes = {}          # event -> mode
        self.listeners = {}      # event -> [_Listener]
        self._forks = set()
        self.errors = []         # errors swallowed by emit

    def _has(self, key):
        return key in self.services

    def _pending(self):
        return sorted((f for f in self._forks if f.state == "pending"),
                      key=lambda f: f.id)

    def _settle(self):
        # Activate every pending fork whose services now exist. Activation
        # can provide further services, so repeat until nothing changes.
        progress = True
        while progress:
            progress = False
            for fork in self._pending():
                if fork.ready():
                    try:
                        fork._activate()
                    except Exception as exc:  # recorded on the fork too
                        self.errors.append((f"plugin/{fork.name}", exc))
                    progress = True

    def _service_removed(self, key):
        for fork in sorted(self._forks, key=lambda f: -f.id):
            if fork.state == "active" and key in fork.inject:
                fork._deactivate()


class Context:
    """A view on a plugin tree, owned by one fork (None for the root)."""

    def __init__(self, root=None, fork=None, scope=None):
        object.__setattr__(self, "_root", root or _Root())
        object.__setattr__(self, "_fork", fork)
        object.__setattr__(self, "_scope", scope)

    # ----- services -------------------------------------------------
    def provide(self, key, value):
        root = self._root
        if key in root.services:
            raise KernelError(f"service '{key}' is already provided")
        root.services[key] = (value, self._fork)

        def dispose():
            if root.services.get(key, (None, None))[0] is value:
                del root.services[key]
                root._service_removed(key)

        self.effect(dispose)
        root._settle()
        return value

    def has(self, key):
        return key in self._root.services

    def get(self, key, default=None):
        return self._root.services.get(key, (default, None))[0]

    def __getattr__(self, name):
        if name.startswith("_"):
            raise AttributeError(name)
        try:
            return self._root.services[name][0]
        except KeyError:
            raise AttributeError(f"no service '{name}'") from None

    def __setattr__(self, name, value):
        raise KernelError("use ctx.provide(key, value) to add a service")

    # ----- effects --------------------------------------------------
    def effect(self, dispose):
        """Record ``dispose`` on this context's plugin; returns a disposer.

        At the root (no plugin) the effect lives until it is disposed by
        hand.
        """
        fork = self._fork
        eid = next(_ids)
        done = []

        def disposer():
            if done:
                return
            done.append(True)
            if fork is not None:
                fork._effects = [(i, d) for i, d in fork._effects if i != eid]
            dispose()

        if fork is not None:
            fork._effects.append((eid, disposer))
        return disposer

    # ----- plugins --------------------------------------------------
    def plugin(self, plugin, config=None):
        fork = Fork(self._root, plugin, config, self._fork, self._scope)
        self._root._forks.add(fork)
        d = self.effect(fork.dispose)
        d._fork = fork
        if fork.ready():
            fork._activate()   # a failing apply raises to the caller
            self._root._settle()
        return fork

    @property
    def forks(self):
        return sorted(self._root._forks, key=lambda f: f.id)

    # ----- scopes ---------------------------------------------------
    def scoped(self, key):
        """A context whose registrations and dispatches carry ``key``."""
        return Context(self._root, self._fork, key)

    @property
    def scope(self):
        return self._scope

    # ----- events ---------------------------------------------------
    def declare(self, event, mode):
        if mode not in MODES:
            raise KernelError(f"unknown mode '{mode}'")
        old = self._root.modes.get(event)
        if old is not None and old != mode:
            raise EventModeError(f"{event} is declared '{old}', not '{mode}'")
        self._root.modes[event] = mode

    def on(self, event, fn, prepend=False):
        lst = self._root.listeners.setdefault(event, [])
        entry = _Listener(fn, self._scope)
        if prepend:
            lst.insert(0, entry)
        else:
            lst.append(entry)

        def dispose():
            if entry in lst:
                lst.remove(entry)

        return self.effect(dispose)

    def _targets(self, event, mode):
        declared = self._root.modes.get(event)
        if declared is None:
            raise EventModeError(f"{event} was never declared")
        if declared != mode:
            raise EventModeError(
                f"{event} is declared '{declared}', dispatched as '{mode}'")
        scope = self._scope
        return [entry.fn for entry in list(self._root.listeners.get(event, ()))
                if entry.scope is None or entry.scope is scope]

    def emit(self, event, *args):
        """Observe: every listener runs; errors are recorded, not raised."""
        for fn in self._targets(event, "emit"):
            try:
                fn(*args)
            except Exception as exc:
                self._root.errors.append((event, exc))

    def serial(self, event, *args):
        """Run listeners in order; return their results as a list."""
        return [fn(*args) for fn in self._targets(event, "serial")]

    def parallel(self, event, *args):
        """Run every listener even if some fail; raise the failures together."""
        errors = []
        for fn in self._targets(event, "parallel"):
            try:
                fn(*args)
            except Exception as exc:
                errors.append(exc)
        if errors:
            raise ListenerErrors(event, errors)

    def bail(self, event, *args):
        """Return the first non-None listener result (None if none bails)."""
        for fn in self._targets(event, "bail"):
            out = fn(*args)
            if out is not None:
                return out
        return None

    def waterfall(self, event, *args, final=None):
        """Around-middleware: ``listener(*args, next)``.

        Calling ``next()`` delegates to the next listener (``next(*new)``
        replaces the arguments downstream); returning without calling it
        short-circuits. ``final(*args)`` is the innermost step.
        """
        chain = self._targets(event, "waterfall")

        def step(i, cur):
            if i == len(chain):
                return final(*cur) if final is not None else (
                    cur[0] if len(cur) == 1 else cur)

            def nxt(*new):
                return step(i + 1, new if new else cur)

            return chain[i](*cur, nxt)

        return step(0, args)

    @property
    def errors(self):
        return self._root.errors
