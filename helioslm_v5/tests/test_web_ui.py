"""T68 - v1.26: chat backend resolution."""
import sys, os
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from harness.web_ui import _resolve_backend


def test_backend_priority(monkeypatch=None):
    # smoke fallback (no qwen/, no API env)
    old_qwen = os.path.exists("qwen/config.json")
    name, fn = _resolve_backend(None)
    assert name == "smoke"
    assert "heard you say" in fn("##user## hello", 0, 0)
    print(f"PASS test_backend_priority (smoke fallback)")


def test_echo_response():
    name, fn = _resolve_backend(None)
    r = fn("##user## what is AI?", 0, 0)
    assert "what is AI?" in r
    print("PASS test_echo_response")


if __name__ == "__main__":
    test_backend_priority()
    test_echo_response()
    print("backend resolution tests done")
