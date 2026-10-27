"""T159 — HDL track: UVM testbench benchmark, CVDP language split and
gate-count ratio, v5.182.

Pins:

1. ``parse_ports`` reads RefModule's ANSI port list (shared direction
   and range, ``reg``/``logic`` dropped, comments ignored);
2. mutants are single swaps after the port list, in a fixed order, and
   the wrapper muxes every output by ``+UVMB_VARIANT``;
3. the UVM report summary is parsed, and a run without one is None;
4. the prompt carries the design renamed to TopModule and the rules;
5. with Icarus: ``prepare`` keeps only mutants that VerilogEval's own
   testbench tells apart from the reference;
6. with Verilator >= 5.052 and a UVM cache in ``HELIOS_UVM_CACHE``: the
   reference testbench builds, is clean on the golden design and kills
   every mutant (and ``calibrate`` keeps them all); a testbench that
   never checks kills none; the reference testbench asserts reset rarely;
7. CVDP ``language`` and ``pass@k_by_language``; ``area_ratio``;
8. ``default_workers`` follows the cgroup CPU quota and memory limit.

Run from repo root: python3 -m pytest helioslm_v5/tests/test_uvm_bench.py
"""
import os
import shutil
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from helioslm_v5.eval import cvdp_bench as cb  # noqa: E402
from helioslm_v5.eval import hdl_bench as hb  # noqa: E402
from helioslm_v5.eval import uvm_bench as ub  # noqa: E402

FIX = Path(__file__).parent / "fixtures" / "verilog_eval"
UVM_CACHE = os.environ.get("HELIOS_UVM_CACHE")
needs_iverilog = pytest.mark.skipif(not shutil.which("iverilog"),
                                    reason="Icarus Verilog not installed")
needs_uvm = pytest.mark.skipif(
    not (shutil.which("verilator") and UVM_CACHE),
    reason="needs verilator and HELIOS_UVM_CACHE (uvm_bench fetch)")


def _prob(name):
    return {"name": name, **{k: (FIX / f"{name}_{s}").read_text()
                             for k, s in (("prompt", "prompt.txt"),
                                          ("ref", "ref.sv"), ("test", "test.sv"))}}


def test_parse_ports():
    ref = ("module RefModule (\n  input clk, reset, // two inputs\n"
           "  input signed [7:0] a,\n  output reg [3:0] q,\n"
           "  output logic done\n);\nendmodule\n")
    assert ub.parse_ports(ref) == [
        ("input", "", "clk"), ("input", "", "reset"),
        ("input", "signed [7:0]", "a"), ("output", "[3:0]", "q"),
        ("output", "", "done")]
    with pytest.raises(ub.UvmBenchError):
        ub.parse_ports("module RefModule; endmodule")


def test_mutants_and_wrapper():
    p = _prob("Prob120_fsm3s")
    ms = ub.mutants(p["ref"])
    assert ms == ub.mutants(p["ref"])                     # deterministic
    head_end = p["ref"].index(");") + 2
    for _, text in ms:
        assert text[:head_end] == p["ref"][:head_end]     # ports untouched
        assert text != p["ref"]
    assert any(label.startswith("==->!=") for label, _ in ms)
    assert any(label.startswith("? B : A->? A : B") for label, _ in ms)
    w = ub.wrapper(ub.parse_ports(p["ref"]), 3)
    assert w.startswith("module TopModule (\n  input clk,")
    assert "TopModule_v2 u_v2 (.clk(clk), .in(in), .reset(reset), .out(out_v2));" in w
    assert ("assign out = (uvmb_variant == 1) ? out_v1 : (uvmb_variant == 2) ? out_v2 "
            ": out_v0;") in w
    task = {"ref": p["ref"], "ports": ub.parse_ports(p["ref"]),
            "mutants": [{"text": t} for _, t in ms[:2]]}
    src = ub.design_sources(task)
    assert "module TopModule_v0" in src and "module TopModule_v2" in src
    assert "RefModule" not in src


def test_summary_counts():
    log = ("UVM_INFO @ 0: reporter [RNTST] Running test t...\n"
           "--- UVM Report Summary ---\n\n** Report counts by severity\n"
           "UVM_INFO :    3\nUVM_WARNING :    0\nUVM_ERROR :    2\n"
           "UVM_FATAL :    0\n")
    assert ub.summary_counts(log) == {"ERROR": 2, "FATAL": 0}
    assert ub.summary_counts("UVM_ERROR : 1\n") is None


def test_prompt():
    p = _prob("Prob120_fsm3s")
    system, user = ub.build_prompt({**p, "ports": [], "mutants": []})
    assert "UVM" in system
    assert "module TopModule (" in user and "RefModule" not in user
    assert "named `tb`" in user and "uvm_error" in user
    assert user.rstrip().endswith("do NOT output anything else.")


@needs_iverilog
def test_prepare_keeps_only_killable_mutants():
    data = ub.prepare([_prob("Prob120_fsm3s"), _prob("Prob031_dff")], workers=1)
    assert [t["name"] for t in data["tasks"]] == ["Prob120_fsm3s"]
    assert data["skipped"] == {"Prob031_dff": "no killable mutant"}
    t = data["tasks"][0]
    assert 1 <= len(t["mutants"]) <= ub.N_MUTANTS
    for m in t["mutants"]:
        r = hb.simulate(hb.reference_sample({"ref": m["text"]}), _prob("Prob120_fsm3s"))
        assert r["code"] in ("R", "r")


@needs_iverilog
@needs_uvm
def test_reference_tb_kills_mutants():
    task = ub.prepare([_prob("Prob120_fsm3s")], workers=1)["tasks"][0]
    r = ub.run_tb(ub.reference_tb(task), task, ub.uvm_dir(UVM_CACHE))
    assert r["build"] and r["golden"] and r["passed"]
    assert r["killed"] == len(task["mutants"])
    lazy = ub.reference_tb(task).replace("`uvm_error", "`uvm_info")
    lazy = lazy.replace('$sformatf("out=%0h exp %0h", tb.out, tb.out_exp))',
                        '$sformatf("out=%0h exp %0h", tb.out, tb.out_exp), UVM_LOW)')
    r = ub.run_tb(lazy, task, ub.uvm_dir(UVM_CACHE))
    assert r["build"] and r["golden"] and r["killed"] == 0 and not r["passed"]
    assert r["kills"] == [False] * len(task["mutants"])
    kept, dropped = ub.calibrate([task], ub.uvm_dir(UVM_CACHE), workers=1)
    assert kept == [task] and dropped == {}


def test_cvdp_language():
    row = {"output": {"context": {"rtl/a.sv": ""}}}
    assert cb.language(row) == "sv"
    assert cb.language({"output": {"context": {"rtl/a.v": ""}}}) == "v"
    assert cb.language({"output": {"context": {"docs/a.md": ""}}}) == "other"


def test_area_ratio():
    assert hb.area_ratio({"ok": True, "cells": 10}, {"ok": True, "cells": 5}) == 2.0
    assert hb.area_ratio({"ok": True, "cells": 10}, {"ok": False, "cells": 5}) is None
    assert hb.area_ratio(None, {"ok": True, "cells": 5}) is None
    assert hb.area_ratio({"ok": True, "cells": 10}, {"ok": True, "cells": None}) is None


def test_random_expr_covers_wide_ports():
    assert ub._random_expr("") == "$urandom"
    assert ub._random_expr("[31:0]") == "$urandom"
    assert ub._random_expr("[99:0]") == "{$urandom, $urandom, $urandom, $urandom}"


def test_default_workers_reads_cgroup(tmp_path):
    (tmp_path / "cpu.max").write_text("1600000 100000\n")        # 16 CPUs
    (tmp_path / "memory.max").write_text(str(62 << 30) + "\n")
    assert ub.cpu_limit(tmp_path) == min(16, os.cpu_count())
    assert ub.default_workers(root=tmp_path) == min(16, os.cpu_count()) // 2
    (tmp_path / "memory.max").write_text(str(12 << 30) + "\n")   # memory bound
    assert ub.default_workers(root=tmp_path) == min(3, min(16, os.cpu_count()) // 2)
    (tmp_path / "cpu.max").write_text("max 100000\n")
    assert ub.cpu_limit(tmp_path) == os.cpu_count()


def test_reference_tb_resets_rarely():
    p = _prob("Prob120_fsm3s")
    tb = ub.reference_tb({"ports": ub.parse_ports(p["ref"])}, cycles=50)
    assert "tb.reset = (first || $urandom_range(15) == 0) ? 1'b1 : 1'b0;" in tb
    assert "tb.in = $urandom;" in tb and "repeat (50)" in tb
    assert "TopModule_v0 gold (" in tb and "run_test(\"t\")" in tb
