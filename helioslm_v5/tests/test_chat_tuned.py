"""T21 - v5.30: chat-tuned checkpoint artifact oracles.

Checks the seeded training summary (checkpoints/chat_tuned_v5.30.json):
- structure & reproducibility metadata (seed, init_from hot-start)
- mode-choice metric presence: the v5.30 capability is CHOOSING between
  text reply and tool block; the summary must record it
- recorded, not hidden: if exact-match is still toy-model-limited, the
  summary says so — this test asserts the metric EXISTS and is parsed,
  not that the toy model is good (same discipline as T19)

Run from repo root: python3 helioslm_v5/tests/test_chat_tuned.py
"""
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))


def test_t21_chat_tuned_summary():
    out = Path(__file__).resolve().parent.parent.parent / "checkpoints" / \
        "chat_tuned_v5.30.json"
    if not out.exists():
        print("SKIP: run examples/train_chat_tuned.py first")
        return
    meta = json.loads(out.read_text())
    assert meta["arch"] == "helioslm_v5_lite"
    assert meta["version"].startswith("v5.30")
    assert meta["seed"] == 6
    # hot-start from the tool-tuned ep2 weights (tool ability preserved)
    assert meta["init_from"] == "tool_tuned_v5.27.pt (hot-start)"
    # the v5.30 capability metric must be recorded
    assert "eval_mode_choice" in meta, \
        "mode-choice metric missing — chat capability unmeasured"
    num, den = meta["eval_mode_choice"].split("/")
    assert int(den) > 0
    # v5.30.2+: eval must be stratified — both types measured (the v5.30
    # filter bias left the eval with zero text targets; never again)
    if meta["version"] >= "v5.30.2":
        for k in ("eval_mode_tool", "eval_mode_text"):
            assert k in meta, f"{k} missing — eval not stratified"
            t, d = meta[k].split("/")
            assert int(d) > 0, f"{k} has no targets — stratification failed"
    print(f"PASS test_t21_chat_tuned_summary "
          f"(exact={meta['eval_exact']}, mode_choice={meta['eval_mode_choice']}"
          + (f", tool={meta.get('eval_mode_tool')}, "
             f"text={meta.get('eval_mode_text')})"
             if "eval_mode_tool" in meta else ")"))


if __name__ == "__main__":
    test_t21_chat_tuned_summary()
    print("\n1/1 v5.30 artifact tests passed")
