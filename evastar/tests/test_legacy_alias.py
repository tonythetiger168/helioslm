"""v5.134: the pre-rename names keep working after helioslm_v5 -> evastar."""
import importlib
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))


def test_old_package_is_same_module():
    new = importlib.import_module("evastar.src.model_v5")
    old = importlib.import_module("helioslm_v5.src.model_v5")
    assert old is new
    assert new.__spec__.name == "evastar.src.model_v5"


def test_old_from_import():
    from helioslm_v5.configs.config_v5 import HeliosLMv5Config
    from evastar.configs.config_v5 import EvastarV5Config
    assert HeliosLMv5Config is EvastarV5Config


def test_class_alias():
    from evastar.src.model_v5 import EvastarV5, HeliosLMv5
    assert HeliosLMv5 is EvastarV5


def test_missing_old_module_still_errors():
    try:
        importlib.import_module("helioslm_v5.no_such_module")
    except ModuleNotFoundError:
        return
    raise AssertionError("expected ModuleNotFoundError")
