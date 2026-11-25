"""Synthetic oracles for the harness-protocol SFT dataset builder (T0).

Non-executing with respect to any model: the builder replays scripted
oracles in sandboxes and re-parses every generated document. These tests
pin the dataset contract:

1. every row carries a known track and a D1 chat record;
2. term_bench rows re-parse with the harness parser (```bash blocks and
   DONE lines), and the replayed solutions pass their own verifiers;
3. long_horizon rows re-execute under the real AgentLoop to the scripted
   answer (the builder raises before a bad trajectory can be recorded);
4. tool_recovery targets round-trip through recover_tool_call unchanged;
5. HDL rows use the pinned prompt envelope, contain no upstream
   VerilogEval text, and (with Icarus) the generator suite compiles and
   simulates clean;
6. the manifest SHA-256 matches the written dataset byte-for-byte.

Run from repo root: PYTHONPATH=. python3 helioslm_v5/tests/case_harness_sft_data.py
"""
import json
import sys
import tempfile
from pathlib import Path

_REPO = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(_REPO))

from helioslm_v5.agent.tool_recovery import recover_tool_call  # noqa: E402
from helioslm_v5.eval import harness_sft_data as gen  # noqa: E402
from helioslm_v5.eval import term_bench  # noqa: E402
from helioslm_v5.tests.case_common import _pass  # noqa: E402


def test_dataset_shape_and_tracks():
    rows, report = gen.build_dataset(n_variants=3, n_recovery=8,
                                     n_generated=5)
    assert report["total_rows"] == len(rows)
    assert set(report) >= set(gen.TRACKS) | {"total_rows"}
    assert report["term_bench"]["rows"] > 0
    assert report["long_horizon"]["rows"] > 0
    assert report["tool_recovery"]["rows"] == 8
    assert report["hdl"]["rows"] == len(gen._HDL_SUITE) + 5
    assert report["grounded_chat"]["rows"] == len(gen._GROUNDED)
    for row in rows:
        assert row["track"] in gen.TRACKS
        rec = row["record"]
        assert rec["id"] and rec["messages"] and rec["content"]
        assert all(m["role"] in ("system", "user", "assistant")
                   for m in rec["messages"])
    print("PASS test_dataset_shape_and_tracks")


def test_term_bench_rows_follow_protocol():
    rows = gen.gen_term_bench()
    assert len(rows) == sum(len(t["solution"]) + 1
                            for t in term_bench.load_tasks())
    for row in rows:
        target = row["record"]["content"]
        kind, _ = term_bench.parse_reply(target)
        assert kind in ("cmd", "done"), (row["record"]["id"], kind)
        prompt = row["record"]["messages"][0]["content"]
        assert prompt.startswith("You are operating a bash shell.")
        assert "```bash" in prompt
    n_done = sum(1 for row in rows
                 if row["record"]["content"].startswith("DONE"))
    assert n_done == len(term_bench.load_tasks())
    print("PASS test_term_bench_rows_follow_protocol")


def test_long_horizon_rows_match_agentloop_shape():
    rows = gen.gen_long_horizon(n_variants=2)
    assert rows
    for row in rows:
        prompt = row["record"]["messages"][0]["content"]
        assert prompt.startswith("You are an agent.")
        assert "@@tool@@" in prompt
        target = row["record"]["content"]
        assert target.startswith("@@tool@@") and target.endswith("@@end@@")
    # The scripted scenarios must be represented.
    ids = {row["record"]["id"] for row in rows}
    for sc_id in ("recover_missing_file", "prompt_injection_in_file",
                  "accumulate_ten_steps", "answer_key_beside_task",
                  "write_then_report"):
        assert any(i.startswith(f"long_horizon/{sc_id}/") for i in ids)
    print("PASS test_long_horizon_rows_match_agentloop_shape")


def test_tool_recovery_rows_roundtrip():
    rows = gen.gen_tool_recovery(n=6)
    assert len(rows) == 6
    for row in rows:
        target = row["record"]["content"]
        calls, repairs = recover_tool_call(target, gen._RECOVERY_REGISTRY)
        assert repairs == ()  # the canonical target needs no repair
        assert calls
        user = row["record"]["messages"][0]["content"]
        assert "Broken block:" in user
        broken = user.split("Broken block:\n", 1)[1]
        recovered, _ = recover_tool_call(broken, gen._RECOVERY_REGISTRY)
        from helioslm_v5.agent.schema import render_tool_call
        assert render_tool_call(recovered) == target
    print("PASS test_tool_recovery_rows_roundtrip")


def test_hdl_rows_envelope_and_oracle():
    rows, checked = gen.gen_hdl(n_generated=6)
    assert len(rows) == len(gen._HDL_SUITE) + 6
    for row in rows:
        rec = row["record"]
        assert rec["messages"][0]["role"] == "system"
        user = rec["messages"][1]["content"]
        assert user.startswith("\nQuestion:\n")
        assert "Enclose your code with [BEGIN] and [DONE]." in user
        assert user.endswith("\nAnswer:\n")
        content = rec["content"]
        assert content.startswith("[BEGIN]\n") and content.endswith("[DONE]")
        # No upstream VerilogEval text or code is used: the suite is
        # hand-written or parameterised by this generator and references no
        # ProbNNN ids.
        assert "Prob" not in rec["id"]
    # Icarus gating: when the simulator is installed the whole suite must
    # verify; the builder never silently emits an unverified row.
    import shutil
    if shutil.which("iverilog") and shutil.which("vvp"):
        assert checked == len(rows)
    print("PASS test_hdl_rows_envelope_and_oracle")


def test_manifest_matches_written_bytes():
    rows, _ = gen.build_dataset(n_variants=2, n_recovery=4, n_generated=2)
    with tempfile.TemporaryDirectory() as td:
        manifest = gen.write_dataset(rows, td)
        data = Path(td) / "harness_sft.jsonl"
        import hashlib
        assert manifest["sha256"] == hashlib.sha256(
            data.read_bytes()).hexdigest()
        assert manifest["schema"] == "harness-sft/1"
        assert manifest["n_rows"] == len(rows)
        lines = data.read_text(encoding="utf-8").splitlines()
        assert len(lines) == len(rows)
        for line in lines:
            parsed = json.loads(line)
            assert set(parsed) == {"track", "record"}
        reloaded = json.loads(
            (Path(td) / "MANIFEST.json").read_text(encoding="utf-8"))
        assert reloaded["sha256"] == manifest["sha256"]
    print("PASS test_manifest_matches_written_bytes")


def test_harness_sft_data_suite():
    test_dataset_shape_and_tracks()
    test_term_bench_rows_follow_protocol()
    test_long_horizon_rows_match_agentloop_shape()
    test_tool_recovery_rows_roundtrip()
    test_hdl_rows_envelope_and_oracle()
    test_manifest_matches_written_bytes()
    _pass("test_harness_sft_data_suite",
          "Harness-protocol SFT dataset (T0): five tracks machine-generated "
          "from in-repo oracles; term_bench replays verify in a sandbox, "
          "long-horizon trajectories re-execute under the real AgentLoop, "
          "tool-recovery targets round-trip exactly, the HDL generator "
          "suite is Icarus-gated with no upstream text, and the manifest "
          "SHA-256 matches the written bytes; no model, network, upstream "
          "benchmark text or teacher API involved")


if __name__ == "__main__":
    test_harness_sft_data_suite()
