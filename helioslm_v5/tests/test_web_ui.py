"""T71 - v1.29: backend priority + mix SFT script."""
import sys, os
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from harness.web_ui import _resolve_backend


def test_api_backend_priority(monkeypatch=None):
    # set env, resolve, assert api first
    os.environ["OPENAI_BASE"] = "https://api.deepseek.com/v1"
    os.environ["OPENAI_KEY"] = "fake-key-for-test"
    name, fn = _resolve_backend(None)
    assert name == "api"
    del os.environ["OPENAI_BASE"], os.environ["OPENAI_KEY"]
    print("PASS test_api_backend_priority")


def test_qwen_fallback_when_no_api():
    old_base = os.environ.pop("OPENAI_BASE", None)
    old_key = os.environ.pop("OPENAI_KEY", None)
    old_ak = os.environ.pop("OPENAI_API_KEY", None)
    name, fn = _resolve_backend(None)
    # should be qwen if qwen/ exists, else smoke
    assert name in ("qwen-local", "smoke")
    if old_base: os.environ["OPENAI_BASE"] = old_base
    if old_key: os.environ["OPENAI_KEY"] = old_key
    if old_ak: os.environ["OPENAI_API_KEY"] = old_ak
    print(f"PASS test_qwen_fallback_when_no_api ({name})")


if __name__ == "__main__":
    test_api_backend_priority()
    test_qwen_fallback_when_no_api()
    print("backend priority tests done")
