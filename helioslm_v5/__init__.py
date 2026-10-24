"""Backward-compatible alias for the ``evastar`` package (renamed in v5.134).

``import helioslm_v5.x.y`` loads ``evastar.x.y`` and registers the very same
module object under both names, so classes, isinstance checks and module
state are shared. New code should import ``evastar``.
"""
import importlib
import importlib.abc
import importlib.util
import sys

_OLD, _NEW = "helioslm_v5", "evastar"


class _AliasLoader(importlib.abc.Loader):
    def __init__(self, new_name):
        self._new_name = new_name

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
        if importlib.util.find_spec(new_name) is None:
            return None
        return importlib.util.spec_from_loader(
            fullname, _AliasLoader(new_name),
            is_package=hasattr(importlib.import_module(new_name), "__path__"))


if not any(isinstance(f, _AliasFinder) for f in sys.meta_path):
    sys.meta_path.insert(0, _AliasFinder())
