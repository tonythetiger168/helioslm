"""T69 - v1.27: direct chat (no tool loop)."""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from harness.web_ui import _resolve_backend, HarnessHandler, _sessions


def test_direct_chat_no_parse_error():
    name, fn = _resolve_backend(None)
    # transcript style (no ##user## markers)
    r = fn("User: hello\nAssistant:", 0, 0)
    assert "hello" in r.lower()
    print("PASS test_direct_chat_no_parse_error")


def test_chat_handler():
    from harness.web_ui import _boot_context
    ctx = _boot_context("full")
    HarnessHandler.ctx = ctx
    r = HarnessHandler._chat(None, "test_session", "what is AI?")
    assert "response" in r
    assert "AI" in r["response"] or "heard" in r["response"]
    assert r["backend"] == "smoke"
    # multi-turn
    r2 = HarnessHandler._chat(None, "test_session", "tell me more")
    assert "heard" in r2["response"]
    print("PASS test_chat_handler")


if __name__ == "__main__":
    test_direct_chat_no_parse_error()
    test_chat_handler()
    print("v1.27 chat tests done")
