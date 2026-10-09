"""helios-harness core - Context (Cordis-inspired temporal/spatial composability)."""
import uuid

class _Namespace:
    def __init__(self, mapping):
        self._m = mapping
    def __getattr__(self, name):
        if name in self._m:
            return self._m[name]
        raise AttributeError(name)

class Effect:
    def __init__(self, kind, payload, replay_fn=None):
        self.id = str(uuid.uuid4())[:8]
        self.kind = kind
        self.payload = payload
        self.replay_fn = replay_fn
    def verify(self):
        if self.replay_fn is None:
            return True
        return self.replay_fn(self.payload)

class Context:
    def __init__(self):
        self._services = {}
        self._plugins = []
        self.effects = []
    def register(self, name, service):
        self._services[name] = service
        return service
    def __getattr__(self, name):
        if name in self._services:
            return self._services[name]
        dotted = {k[len(name)+1:]: v for k, v in self._services.items()
                  if k.startswith(name + ".")}
        if dotted:
            return _Namespace(dotted)
        raise AttributeError(f"no service '{name}'")
    def effect(self, kind, payload, replay_fn=None):
        eff = Effect(kind, payload, replay_fn)
        self.effects.append(eff)
        return eff
    def verify_all_effects(self):
        bad = [e for e in self.effects if not e.verify()]
        return {"total": len(self.effects), "failed": len(bad),
                "ok": len(self.effects) - len(bad)}
    def use(self, plugin):
        plugin.apply(self)
        self._plugins.append(plugin)
        return plugin
