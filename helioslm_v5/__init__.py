"""Backward-compatible alias for the ``evastar`` package (renamed in v5.134).

``import helioslm_v5.x.y`` loads ``evastar.x.y`` and registers the very same
module object under both names, so classes, isinstance checks and module
state are shared. New code should import ``evastar``.
"""
import importlib
import importlib.abc
import importlib.util
import os
import sys

_OLD, _NEW = "helioslm_v5", "evastar"


class _AliasLoader(importlib.abc.Loader):
    def __init__(self, new_name):
        self._new_name = new_name

    def _real_origin(self, fullname):
        """Resolve the aliased module's real source file without executing
        it (find_spec imports parent packages only). Loud error: an alias
        whose target has no file origin cannot serve code."""
        spec = importlib.util.find_spec(self._new_name)
        if spec is None or not spec.origin or not os.path.isfile(spec.origin):
            raise ImportError(
                f"alias {fullname!r} -> {self._new_name!r} has no readable "
                "source origin")
        return spec.origin

    def get_code(self, fullname):
        # v5.153: runpy (`python -m helioslm_v5.x.y` — the daily-mandated
        # suite entry point) requires loader.get_code; without it the call
        # died with AttributeError. Compiling the real evastar file keeps
        # tracebacks pointed at the canonical path, and runpy re-executes
        # the code as __main__ exactly as `python -m evastar.x.y` would.
        origin = self._real_origin(fullname)
        with open(origin, "rb") as f:
            return compile(f.read(), origin, "exec")

    def get_source(self, fullname):
        # inspect/pdb and runpy's get_code fallback path use get_source.
        origin = self._real_origin(fullname)
        with open(origin, "r", encoding="utf-8") as f:
            return f.read()

    def create_module(self, spec):
        module = importlib.import_module(self._new_name)
        self._real_spec = module.__spec__
        return module

    def exec_module(self, module):
        # The import system stamps the alias spec onto the shared module;
        # put the real one back so the module still reports its own name.
        module.__spec__ = self._real_spec


class _AliasFinder(importlib.abc.MetaPathFinder):
    def find_spec(self, fullname, path=None, target=None):
        if not fullname.startswith(_OLD + "."):
            return None
        new_name = _NEW + fullname[len(_OLD):]
        real_spec = importlib.util.find_spec(new_name)
        if real_spec is None:
            return None
        # v5.153: carry the real spec's origin (and search locations for
        # packages). spec_from_loader defaults origin to None, and runpy
        # stamps spec.origin into __file__ — an alias spec without an
        # origin runs the module with __file__ = None, breaking any
        # Path(__file__) usage at import time.
        is_pkg = real_spec.submodule_search_locations is not None
        spec = importlib.util.spec_from_loader(
            fullname, _AliasLoader(new_name),
            origin=real_spec.origin, is_package=is_pkg)
        if is_pkg:
            spec.submodule_search_locations = list(
                real_spec.submodule_search_locations)
        return spec


if not any(isinstance(f, _AliasFinder) for f in sys.meta_path):
    sys.meta_path.insert(0, _AliasFinder())
