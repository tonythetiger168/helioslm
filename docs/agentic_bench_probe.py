"""docs/AGENTIC_BENCH.md probe: measured toy-scale agentic eval numbers.

Reproduces every number in the doc's measured section:
1. tool-call recovery battery — 7 repairable corruption classes x 40
   samples (strict parser must reject, recovery must restore exactly),
   plus 6 never-repaired classes that must stay loud failures.
2. term_bench surface counts (tasks per difficulty tier) read from the
   module itself.
3. The Agentic Index aggregation math on the committed schema fixtures
   (is_model_score: false — a contract check, not a model score).

Prints measured JSON only; asserts nothing about the values. Seeds are
fixed so runs are reproducible on CPU.
"""
import json
import random
import sys
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parents[1]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))


def tool_recovery_battery():
    from helioslm_v5.agent.schema import (ToolCallError, parse_tool_call)
    from helioslm_v5.agent.tool_recovery import recover_tool_call
    from helioslm_v5.agent.tools import ToolRegistry, ToolSpec
    from helioslm_v5.tests import test_tool_recovery as t

    reg = ToolRegistry([
        ToolSpec("read_file", "read a file", {"path": str}),
        ToolSpec("run_tests", "run tests",
                 {"target": str, "verbose": bool}),
        ToolSpec("edit", "replace text",
                 {"path": str, "old": str, "new": str, "count": int}),
    ])
    rng = random.Random(1)
    repaired = {}
    for name, (fn, expect) in t.REPAIRABLE.items():
        ok = 0
        for _ in range(40):
            calls = t._sample_calls(rng)
            text = fn(calls)
            try:
                parse_tool_call(text, reg)
                continue  # strict parser accepted: not a corruption
            except ToolCallError:
                pass
            got, repairs = recover_tool_call(text, reg)
            if got == calls and expect in repairs:
                ok += 1
        repaired[name] = {"recovered": ok, "of": 40, "repair": expect}
    refused = {}
    for name, fn in t.NOT_REPAIRED.items():
        try:
            recover_tool_call(fn(), reg)
            refused[name] = False
        except ToolCallError:
            refused[name] = True
    return {"repairable_classes": repaired, "never_repaired": refused,
            "repair_pipeline": list(
                __import__("helioslm_v5.agent.tool_recovery",
                           fromlist=["REPAIRS"]).REPAIRS)}


def term_bench_surface():
    from helioslm_v5.eval import term_bench as tb
    tiers = {}
    for t in tb.TASKS:
        tiers[t.get("tier", "?")] = tiers.get(t.get("tier", "?"), 0) + 1
    return {"tasks": len(tb.TASKS), "tiers": tiers}


def agentic_index_math():
    import tempfile
    from helioslm_v5.eval.agentic_index import (WEIGHTS, build_index,
                                                load_components)
    from helioslm_v5.tests.case_agentic_index import _fixture
    with tempfile.TemporaryDirectory() as td:
        root = _fixture(Path(td) / "results")
        idx = build_index(load_components(root), root)
    return {"weights": WEIGHTS, "fixture_index": idx["agentic_index"],
            "evidence_type": idx["evidence_type"],
            "is_model_score": idx["is_model_score"]}


def main():
    print(json.dumps({
        "tool_recovery": tool_recovery_battery(),
        "term_bench_surface": term_bench_surface(),
        "agentic_index_math": agentic_index_math(),
    }, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
