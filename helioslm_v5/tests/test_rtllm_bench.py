"""T143 — HDL track H0: RTLLM v2.1 harness (Icarus), v5.175.

Pins:

0. code extraction keeps helper modules (from the first module to the
   last endmodule when the answer has no [BEGIN]/fenced block);
1. build_cache finds designs by their makefile, keeps data files, writes a
   sha256 manifest; load rejects a tampered cache;
2. the requested module name comes from the description;
3. reference_sample renames the reference's top module, including the
   irregular names (verified_booth4_mul.v, verified_multi_pipe);
4. with Icarus Verilog installed: the references pass (multi_booth_8bit
   reads test_data.dat), a wrong design fails function, a syntax error
   fails syntax, and score reports syntax@k and pass@k.

Fixtures: three upstream designs (MIT) in tests/fixtures/rtllm/.
Run from repo root: python3 -m pytest helioslm_v5/tests/test_rtllm_bench.py
"""
import re
import shutil
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from helioslm_v5.eval import rtllm_bench as rb  # noqa: E402

FIX = Path(__file__).parent / "fixtures" / "rtllm"
NAMES = ("adder_8bit", "multi_booth_8bit", "multi_pipe_4bit")
needs_iverilog = pytest.mark.skipif(
    not (shutil.which("iverilog") and shutil.which("vvp")),
    reason="Icarus Verilog not installed")


@pytest.fixture
def designs(tmp_path):
    rb.build_cache(FIX, tmp_path, designs=NAMES)
    return rb.load(tmp_path)


def test_cache_and_manifest(tmp_path):
    m = rb.build_cache(FIX, tmp_path, designs=NAMES)
    assert m["commit"] == rb.COMMIT and m["designs"] == list(NAMES)
    assert "multi_booth_8bit/test_data.dat" in m["sha256"]
    assert not any(k.endswith("makefile") for k in m["sha256"])
    ds = rb.load(tmp_path)
    assert ds[1]["aux"].keys() == {"test_data.dat"}
    (tmp_path / "adder_8bit" / "testbench.v").write_text("module tb; endmodule\n")
    with pytest.raises(rb.RtllmError, match="sha256"):
        rb.load(tmp_path)


def test_missing_design(tmp_path):
    with pytest.raises(rb.RtllmError, match="not in checkout"):
        rb.build_cache(FIX, tmp_path, designs=("adder_8bit", "nope"))


def test_top_name():
    assert rb.top_name("Module name:  \n    adder_8bit   \nInput") == "adder_8bit"
    with pytest.raises(rb.RtllmError):
        rb.top_name("no name here")


def test_reference_renamed(designs):
    for d in designs:
        s = rb.reference_sample(d)
        assert d["top"] in re.findall(r"^\s*module\s+(\w+)", s, re.M)
    # helper modules keep their names
    assert "module full_adder" in rb.reference_sample(designs[0])


def test_prompt_is_description(designs):
    assert rb.build_prompt(designs[0]["prompt"]).startswith(
        "Please act as a professional verilog designer.")


def test_extract_keeps_helper_modules():
    two = "module a(); b u(); endmodule\nmodule b(); endmodule"
    assert rb.extract_code("Here:\n" + two + "\nDone.") == two + "\n"
    # a closing fence on the endmodule line is one fence: fallback span
    assert rb.extract_code("```verilog\n" + two + "```") == two + "\n"
    assert rb.extract_code("```\n" + two + "\n```\n") == two + "\n"
    assert rb.extract_code("[BEGIN]\n" + two + "\n[DONE]") == two + "\n"


@needs_iverilog
def test_self_check(designs):
    assert rb.self_check(designs) == (3, [])


@needs_iverilog
def test_failures(designs):
    add = designs[0]
    wrong = ("module adder_8bit(input [7:0] a, b, input cin, "
             "output [7:0] sum, output cout);\n"
             "  assign {cout, sum} = a - b;\nendmodule\n")
    r = rb.simulate(wrong, add)
    assert r["syntax"] and not r["passed"]
    r = rb.simulate("module adder_8bit(\n  oops\nendmodule\n", add)
    assert not r["syntax"] and not r["passed"]


@needs_iverilog
def test_score(designs):
    def gen(system, user, i):
        return [f"```verilog\n{rb.reference_sample(designs[i])}\n```", "no code"]
    out = rb.score(designs, gen, n=2, ks=(1, 2), exclude=("multi_pipe_4bit",))
    assert out["n_designs"] == 2 and out["excluded"] == ["multi_pipe_4bit"]
    assert out["pass@1"] == pytest.approx(0.5)
    assert out["pass@2"] == pytest.approx(1.0)
    assert out["syntax@1"] == pytest.approx(0.5)
