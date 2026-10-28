"""T162 — HDL track H5: project-level data with a lint + synth gate,
v5.186.

Pins:

1. ``parse`` / ``dep_order``: a file comes after the files it needs;
   unordered files go packages, interfaces, modules, then UVM classes by
   role; macro-only headers are compiled first, but not the variants a
   header picks by ``ifdef``;
2. ``top_candidates`` prefers the module named like the unit directory
   and never a testbench;
3. ``project_docs`` keeps a unit only when its gate passes, packs it as
   one document in dependency order, keeps shared design files only when
   a passing synthesis read them, drops near-duplicate projects and
   benchmark-contaminated files;
4. ``synth_gate`` runs Verilator and Yosys (skipped without them);
5. ``fetch_mixture``: permissive-licence row filter, SHL-0.51 LICENSE,
   ``per-repository`` sources, and the H5 mixture.

Run from repo root: python3 -m pytest helioslm_v5/tests/test_hdl_projects.py
"""
import json
import shutil
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(ROOT))

from examples import fetch_mixture as fm  # noqa: E402
from helioslm_v5.src.training import hdl_projects as hp  # noqa: E402
from helioslm_v5.src.training.data_mixture import (  # noqa: E402
    load_mixture, validate_mixture)

FILES = {
    "u/rtl/top.sv": "`include \"defs.svh\"\nmodule top(input clk, output y);\n"
                    "  import u_pkg::*;\n  sub u0(.clk(clk), .y(y));\nendmodule\n",
    "u/rtl/sub.sv": "module sub(input clk, output y);\n  assign y = u_pkg::ONE;\n"
                    "endmodule\n",
    "u/rtl/u_pkg.sv": "package u_pkg;\n  localparam bit ONE = 1'b1;\nendpackage\n",
    "u/rtl/u_if.sv": "interface u_if(input clk);\n  logic y;\nendinterface\n",
    "u/rtl/defs.svh": "`define W 4\n",
    "u/dv/u_env.sv": "class u_env extends uvm_env;\n  u_agent a;\nendclass\n",
    "u/dv/u_agent.sv": "class u_agent extends uvm_agent;\nendclass\n",
    "u/dv/tb.sv": "module tb;\n  top dut(.clk(1'b0), .y());\nendmodule\n",
}


def test_parse_and_dep_order():
    info, defined = hp.index(FILES)
    assert info["u/rtl/top.sv"]["defs"] == [("module", "top")]
    assert {"u_pkg", "sub"} <= info["u/rtl/top.sv"]["refs"]
    assert info["u/rtl/top.sv"]["includes"] == ["defs.svh"]
    assert "top" in info["u/dv/tb.sv"]["refs"]
    order = hp.dep_order(list(FILES), info, defined)
    pos = {p: i for i, p in enumerate(order)}
    assert pos["u/rtl/u_pkg.sv"] < pos["u/rtl/sub.sv"] < pos["u/rtl/top.sv"]
    assert pos["u/rtl/defs.svh"] < pos["u/rtl/top.sv"]
    assert pos["u/rtl/top.sv"] < pos["u/dv/tb.sv"]
    assert pos["u/rtl/u_pkg.sv"] < pos["u/rtl/u_if.sv"] < pos["u/rtl/sub.sv"]
    assert pos["u/dv/u_agent.sv"] < pos["u/dv/u_env.sv"]


def test_comments_ignored():
    info = hp.parse("// module fake(\n/* package p; */\nmodule real_m;\nendmodule\n")
    assert info["defs"] == [("module", "real_m")]


def test_top_candidates():
    info, defined = hp.index(FILES)
    assert hp.top_candidates(list(FILES), info, defined)[0] == (
        "top", "u/rtl/top.sv")
    f = dict(FILES, **{"u/rtl/u.sv": "module u(input a);\nendmodule\n"})
    info, defined = hp.index(f)
    assert hp.top_candidates(list(f), info, defined, prefer="u")[0][0] == "u"
    names = [n for n, _ in hp.top_candidates(list(f), info, defined)]
    assert "tb" not in names


def test_closure_skips_testbench_and_classes():
    info, defined = hp.index(FILES)
    c = hp.closure(["u/rtl/top.sv"], info, defined)
    assert c == {"u/rtl/top.sv", "u/rtl/sub.sv", "u/rtl/u_pkg.sv",
                 "u/rtl/defs.svh"}


def _fake_gate(passing):
    def gate(root, paths, info, defined, prefer=None, tool=None, tries=1,
             timeout=600, all_text=None):
        tops = hp.top_candidates(paths, info, defined, prefer)
        ok = bool(tops) and tops[0][0] in passing
        return {"top": tops[0][0] if tops else None, "files": 1,
                "lint": ok, "synth": ok, "cells": 5 if ok else None,
                "pass": ok, "error": None if ok else "synth: x",
                "paths": sorted(hp.closure([tops[0][1]], info, defined))
                if tops else []}
    return gate


REPO = {
    "ip/a/rtl/a.sv": "module a(input i, output o);\n  lib l(.i(i), .o(o));\nendmodule\n",
    "ip/a/dv/a_env.sv": "class a_env extends uvm_env;\nendclass\n",
    "ip/b/rtl/b.sv": "module b(input i, output o);\n  assign o = ~i;\nendmodule\n",
    "ip/prim/rtl/lib.sv": "module lib(input i, output o);\n  assign o = i;\nendmodule\n",
    "ip/prim/rtl/unused.sv": "module unused(input i);\nendmodule\n",
    "dv/common.sv": "class common_seq;\nendclass\n",
}


def test_project_docs_gate(monkeypatch, tmp_path):
    monkeypatch.setattr(hp, "gate_project", _fake_gate({"a"}))
    report = []
    docs = hp.project_docs("r", REPO, REPO, tmp_path, units=["ip/*"],
                           keep=["ip/prim/*", "dv/*"], report=report)
    assert len(docs) == 2
    assert docs[0].startswith("// project: r/ip/a\n")
    assert "// file: ip/a/rtl/a.sv" in docs[0] and "a_env" in docs[0]
    assert "module b" not in "".join(docs)
    shared = docs[1]
    assert shared.startswith("// project: r/shared")
    assert "ip/prim/rtl/lib.sv" in shared and "dv/common.sv" in shared
    assert "unused.sv" not in shared
    rows = {r["project"]: r for r in report}
    assert rows["r/ip/a"]["pass"] and not rows["r/ip/b"]["pass"]
    assert "paths" not in rows["r/ip/a"]


def test_project_docs_nothing_passes(monkeypatch, tmp_path):
    monkeypatch.setattr(hp, "gate_project", _fake_gate(set()))
    assert hp.project_docs("r", REPO, REPO, tmp_path, units=["ip/*"],
                           keep=["ip/prim/*", "dv/*"]) == []


def test_near_dup_and_decontam(monkeypatch, tmp_path):
    monkeypatch.setattr(hp, "gate_project", _fake_gate({"top"}))
    body = "".join(f"  assign w{i} = a{i} ^ b{i} & c{i};\n" for i in range(60))
    rows = [("o1/r", "rtl/top.v", f"module top;\n{body}endmodule\n"),
            ("o2/fork", "rtl/top.v", f"module top;\n{body}  // edit\nendmodule\n"),
            ("o3/other", "top.v", "module top;\n" + body.replace("^", "|")
             .replace("&", "+") + "endmodule\n")]
    dedup = hp.MinHashDedup()
    docs = list(hp.repo_docs(rows, dedup=dedup, workdir=tmp_path))
    assert len(docs) == 2
    assert "github/o1/r" in docs[0] and "github/o3/other" in docs[1]
    bench = hp.Decontam([f"module top;\n{body}endmodule\n"])
    docs = list(hp.repo_docs(rows, decontam=bench, workdir=tmp_path))
    assert docs == [d for d in docs if "o3/other" in d]


def test_minhash_and_decontam_units():
    text = " ".join(f"tok{i}" for i in range(300))
    d = hp.MinHashDedup()
    assert not d.seen(text) and d.seen(text)
    assert not d.seen(" ".join(f"x{i}" for i in range(300)))
    c = hp.Decontam(["assign y = a & b | c ^ d ; assign z = y ;"])
    assert c.hit("// x\nassign y = a & b | c ^ d ; assign z = y ; endmodule")
    assert not c.hit("assign y = a & b ;")
    assert not hp.Decontam().hit("anything at all here")


def test_bench_texts(tmp_path):
    (tmp_path / "p.sv").write_text("module m; endmodule")
    (tmp_path / "x.jsonl").write_text(json.dumps({"q": "q" * 80, "n": 1}) + "\n")
    texts = list(hp.bench_texts([tmp_path]))
    assert "module m; endmodule" in texts and "q" * 80 in texts


TOOLS = hp.tools(verilator=shutil.which("verilator"))
needs_tools = pytest.mark.skipif(
    not (TOOLS["verilator"] and TOOLS["yosys"]), reason="needs verilator + yosys")


@needs_tools
def test_synth_gate_real(tmp_path):
    files = {"rtl/top.v": "`include \"defs.vh\"\nmodule top(input clk, input [3:0] a,"
                          " output [3:0] y);\n  wire [3:0] t;\n  add1 u0(.a(a), .y(t));\n"
                          "  reg [3:0] q;\n  always @(posedge clk) q <= t;\n"
                          "  assign y = q;\nendmodule\n",
             "rtl/add1.v": "module add1(input [3:0] a, output [3:0] y);\n"
                           "  assign y = a + `ONE;\nendmodule\n",
             "inc/defs.vh": "`define ONE 4'd1\n"}
    for p, t in files.items():
        (tmp_path / p).parent.mkdir(parents=True, exist_ok=True)
        (tmp_path / p).write_text(t)
    info, defined = hp.index(files)
    r = hp.synth_gate("top", "rtl/top.v", tmp_path, info, defined, TOOLS,
                      all_text=files)
    assert r["pass"] and r["lint"] and r["cells"] > 0, r
    (tmp_path / "rtl/add1.v").write_text("module add1(input a output y);\nendmodule\n")
    files["rtl/add1.v"] = (tmp_path / "rtl/add1.v").read_text()
    info, defined = hp.index(files)
    r = hp.synth_gate("top", "rtl/top.v", tmp_path, info, defined, TOOLS,
                      all_text=files)
    assert not r["pass"] and r["error"].startswith("lint")


def test_gate_without_tools(tmp_path):
    info, defined = hp.index(FILES)
    r = hp.synth_gate("top", "u/rtl/top.sv", tmp_path, info, defined,
                      {"verilator": None, "yosys": None, "sv2v": None})
    assert not r["pass"] and "not found" in r["error"]


def test_licence_filter():
    assert fm.licence_ok(["MIT", "Apache-2.0"])
    assert not fm.licence_ok(["MIT", "GPL-3.0"])
    assert not fm.licence_ok([])
    rows = [{"max_stars_repo_name": "o/a", "max_stars_repo_path": "x.sv",
             "max_stars_repo_licenses": ["MIT"], "content": "m"},
            {"max_stars_repo_name": "o/b", "max_stars_repo_path": "y.sv",
             "max_stars_repo_licenses": ["GPL-2.0"], "content": "n"}]
    assert list(fm.stack_rows(rows, {}, "content")) == [("o/a", "x.sv", "m")]


def test_shl_licence(tmp_path):
    (tmp_path / "LICENSE").write_text("SOLDERPAD HARDWARE LICENSE version 0.51\n")
    assert fm.git_licence(tmp_path) == "SHL-0.51"


def test_per_repository_sources():
    src = {"name": "s", "source": "hf:o/stack", "license": "per-repository",
           "weight": 1.0}
    with pytest.raises(ValueError, match="per-repository"):
        validate_mixture({"name": "m", "sources": [src]})
    src["projects"] = {}
    validate_mixture({"name": "m", "sources": [src]})
    rows = fm.plan({"name": "m", "sources": [src]}, 1e6, 0.2, {"s": "other"})
    assert rows["s"]["usable"]


def test_h5_mixture():
    mix = load_mixture(ROOT / "data/mixtures/hdl_h5_sv_uvm_v1.json")
    src = {s["name"]: s for s in mix["sources"]}
    names = [s["name"] for s in mix["sources"]]
    assert abs(sum(s["weight"] for s in mix["sources"]) - 1) < 1e-9
    for n in ("opentitan-sv", "ibex-sv", "cv32e40p-sv", "stack-systemverilog",
              "verilog-github"):
        assert isinstance(src[n].get("projects"), dict), n
    assert src["cv32e40p-sv"]["license"] == "SHL-0.51"
    assert len(src["cv32e40p-sv"]["source"].rpartition("@")[2]) == 40
    assert names.index("cv32e40p-sv") < names.index("core-v-verif-sv")
    assert src["core-v-verif-sv"]["requires"] == ["cv32e40p-sv"]
    curated = sum(src[n]["weight"] for n in ("opentitan-sv", "ibex-sv",
                                             "cv32e40p-sv", "core-v-verif-sv"))
    github = src["stack-systemverilog"]["weight"] + src["verilog-github"]["weight"]
    assert curated > github


def test_gate_fixes_from_real_cores():
    """v5.187: what Ibex / CV32E40P / OpenTitan needed."""
    files = {"rtl/core.sv": '`include "dv_fcov_macros.svh"\n'
                            '`include "cc/registers.svh"\nmodule core;\nendmodule\n',
             "rtl/core_tracing.sv": "module core_tracing;\n  core u();\nendmodule\n",
             "vendor/dv/sv/dv_fcov_macros.svh": "`define FCOV\n",
             "vendor/cc/include/cc/registers.svh": "`define FF\n"}
    info, defined = hp.index(files)
    c = hp.closure(["rtl/core.sv"], info, defined)
    assert "vendor/dv/sv/dv_fcov_macros.svh" in c       # header on a dv path
    assert "vendor/cc/include" in hp.include_dirs(sorted(c), info)
    assert hp.top_candidates(list(files), info, defined)[0][0] == "core_tracing"
    assert hp.top_candidates(list(files), info, defined,
                             prefer="core")[0][0] == "core"
    text = "a;\n// pragma translate_off\nassert property (x);\n" \
           "// pragma translate_on\nb;\n"
    out = hp.drop_translate_off(text)
    assert "assert" not in out and out.count("\n") == text.count("\n")
