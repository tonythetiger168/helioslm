"""T141 — HDL track H0: VerilogEval v2 spec-to-rtl harness, v5.147.

Pins:

1. the prompt matches upstream sv-generate (spec-to-rtl, 0-shot, no rules);
2. code extraction follows upstream: [BEGIN]/[DONE], else backticks;
3. log classification follows sv-iv-analyze's failure codes;
4. fetch writes a sha256 manifest and load rejects a tampered cache;
5. unbiased pass@k;
6. with Icarus Verilog installed: references renamed to TopModule pass,
   a wrong design fails as R, a syntax error as S, an endless loop as T.

Fixtures: two upstream problems (MIT) in tests/fixtures/verilog_eval/.
Run from repo root: python3 -m pytest helioslm_v5/tests/test_hdl_bench.py
"""
import hashlib
import json
import shutil
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from helioslm_v5.eval import hdl_bench as hb  # noqa: E402

FIX = Path(__file__).parent / "fixtures" / "verilog_eval"
NAMES = ["Prob001_zero", "Prob031_dff"]
needs_iverilog = pytest.mark.skipif(
    not (shutil.which("iverilog") and shutil.which("vvp")),
    reason="Icarus Verilog not installed")


def _fake_get(url):
    rel = url.split(hb.COMMIT + "/", 1)[1]
    if rel.endswith("problems.txt"):
        return ("\n".join(NAMES) + "\n").encode()
    if rel.startswith("scripts/"):
        return b"// one-shot prefix\n"
    return (FIX / rel.split("/", 1)[1]).read_bytes()


@pytest.fixture
def probs(tmp_path):
    hb.fetch(tmp_path, limit=2, get=_fake_get)
    return hb.load(tmp_path)


def test_prompt_matches_upstream():
    system, user = hb.build_prompt("  spec line\n\n")
    assert system == ("You are a Verilog RTL designer that only writes code "
                      "using correct Verilog syntax.")
    assert user == ("\nQuestion:\nspec line\n"
                    "Enclose your code with [BEGIN] and [DONE]. Only output "
                    "the code snippet\nand do NOT output anything else.\n"
                    "\nAnswer:\n")


def test_extract_begin_done():
    t = "sure\n[BEGIN]\nmodule TopModule();\nendmodule\n[DONE]\nbye"
    assert hb.extract_code(t) == "module TopModule();\nendmodule\n"
    t = "[BEGIN]module TopModule();\nendmodule[DONE]"
    assert hb.extract_code(t) == "module TopModule();\nendmodule\n"


def test_extract_backticks_fallback():
    t = "here:\n```verilog\nmodule TopModule();\nendmodule\n```\ntext"
    assert hb.extract_code(t) == "module TopModule();\nendmodule\n"
    # no fences: everything up to and including endmodule
    t = "module TopModule();\nendmodule\ntrailing words"
    assert hb.extract_code(t) == "module TopModule();\nendmodule\n"


@pytest.mark.parametrize("log,sample,code", [
    ("Mismatches: 0 in 40 samples\n", "", "."),
    ("Mismatches: 3 in 40 samples\n", "", "R"),
    ("Mismatches: 3 in 40 samples\n", "always @(posedge clk, posedge reset)", "r"),
    ("sample.sv:3: syntax error\nI give up.\n", "", "S"),
    ("sample.sv:3: error: Unknown module type: foo\n", "", "m"),
    ("sample.sv:3: error: Unable to bind wire/reg/memory `x'\n", "", "p"),
    ("sample.sv:3: error: something else\n", "", "C"),
    ("Hint: ...\nTIMEOUT\n", "", "T"),
])
def test_classify_log(log, sample, code):
    assert hb.classify_log(log, sample) == code


def test_fetch_manifest_and_tamper(tmp_path):
    m = hb.fetch(tmp_path, limit=2, get=_fake_get)
    assert m["commit"] == hb.COMMIT and m["problems"] == NAMES
    rel = f"{hb.TASK_DIR}/Prob001_zero_ref.sv"
    data = (tmp_path / rel).read_bytes()
    assert m["sha256"][rel] == hashlib.sha256(data).hexdigest()
    disk = json.loads((tmp_path / "MANIFEST.json").read_text())
    assert disk == m
    probs = hb.load(tmp_path)
    assert [p["name"] for p in probs] == NAMES
    assert hb.shot_prefix(tmp_path, 1) == "// one-shot prefix\n"
    (tmp_path / rel).write_text("module RefModule(); endmodule\n")
    with pytest.raises(hb.HdlBenchError, match="sha256"):
        hb.load(tmp_path)


def test_fetch_requires_full_set(tmp_path):
    with pytest.raises(hb.HdlBenchError, match="expected 156"):
        hb.fetch(tmp_path, get=_fake_get)


def test_pass_at_k():
    assert hb.pass_at_k(10, 0, 1) == 0.0
    assert hb.pass_at_k(10, 10, 5) == 1.0
    assert hb.pass_at_k(10, 3, 1) == pytest.approx(0.3)
    assert hb.pass_at_k(5, 1, 5) == 1.0


def test_reference_sample_renames(probs):
    s = hb.reference_sample(probs[0])
    assert "module TopModule" in s and "RefModule" not in s


@needs_iverilog
def test_self_check_passes(probs):
    ok, fails = hb.self_check(probs)
    assert (ok, fails) == (2, [])


@needs_iverilog
def test_failures_classified(probs):
    zero, dff = probs
    wrong = "module TopModule(output zero);\n  assign zero = 1'b1;\nendmodule\n"
    assert hb.simulate(wrong, zero)["code"] == "R"
    bad = "module TopModule(output zero)\n  assign zero = ;\nendmodule\n"
    assert hb.simulate(bad, zero)["code"] == "S"
    hang = ("module TopModule(input clk, input d, output reg q);\n"
            "  initial while (1) q = d;\nendmodule\n")
    assert hb.simulate(hang, dff, timeout=3)["code"] == "T"


@needs_iverilog
def test_score_end_to_end(probs):
    def gen(system, user, i):
        ref = hb.reference_sample(probs[i])
        return [f"[BEGIN]\n{ref}[DONE]", "nonsense"]
    out = hb.score(probs, gen, n=2, ks=(1, 2))
    assert out["pass@1"] == pytest.approx(0.5)
    assert out["pass@2"] == pytest.approx(1.0)
    assert out["failure_codes"]["."] == 2
