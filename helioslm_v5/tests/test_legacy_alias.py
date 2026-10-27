"""v5.164: the Evastar-era names keep working after evastar -> helioslm_v5."""
import importlib
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))


def test_old_package_is_same_module():
    new = importlib.import_module("helioslm_v5.src.model_v5")
    old = importlib.import_module("evastar.src.model_v5")
    assert old is new
    assert new.__spec__.name == "helioslm_v5.src.model_v5"


def test_old_from_import():
    from evastar.configs.config_v5 import EvastarV5Config
    from helioslm_v5.configs.config_v5 import HeliosLMv5Config
    assert EvastarV5Config is HeliosLMv5Config


def test_class_alias():
    from helioslm_v5.src.model_v5 import HeliosLMv5, EvastarV5
    assert EvastarV5 is HeliosLMv5


def test_missing_old_module_still_errors():
    try:
        importlib.import_module("evastar.no_such_module")
    except ModuleNotFoundError:
        return
    raise AssertionError("expected ModuleNotFoundError")
