"""T70 - v1.28: reply cleaning + repetition control."""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from harness.web_ui import _clean_reply


def test_clean_reply_strips_hallucinations():
    raw = "##assistant## ## I am HeliosLM. Hello!\nHello!"
    cleaned = _clean_reply(raw)
    assert "HeliosLM" not in cleaned
    assert "##assistant##" not in cleaned
    assert "Hello!" in cleaned
    print("PASS test_clean_reply_strips_hallucinations")


def test_clean_reply_dedups():
    raw = "line one\nline one\nline one\nline two"
    cleaned = _clean_reply(raw)
    assert cleaned.count("line one") == 1
    assert "line two" in cleaned
    print("PASS test_clean_reply_dedups")


if __name__ == "__main__":
    test_clean_reply_strips_hallucinations()
    test_clean_reply_dedups()
    print("v1.28 reply cleaning tests done")
