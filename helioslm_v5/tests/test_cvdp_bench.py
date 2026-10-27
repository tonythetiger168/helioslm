"""T144 — HDL track H0: CVDP v1.1.0 non-agentic code generation, v5.177.

Pins:

1. load keeps open-source datapoints and lists the rest with a reason
   (a commercial EDA image here);
2. docker-compose and .env parsing for the CVDP harness shape, and the
   /code, /src, /rundir rewrite;
3. prompts follow upstream: plain text for one file, the JSON schema for
   several; response parsing for both;
4. model-chosen file names cannot leave the repo dir;
5. with Icarus Verilog and cocotb installed: the golden output passes,
   a wrong one fails, and score reports pass@k per category;
6. env_file as a list is read (v5.179).

Fixtures: two upstream example datapoints (CC BY 4.0 / Apache-2.0) in
tests/fixtures/cvdp/.
Run from repo root: python3 -m pytest helioslm_v5/tests/test_cvdp_bench.py
"""
import importlib.util
import json
import shutil
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from helioslm_v5.eval import cvdp_bench as cb  # noqa: E402

FIX = Path(__file__).parent / "fixtures" / "cvdp"
OSS = FIX / "cvdp_v1.1.0_example_nonagentic_code_generation_no_commercial_with_solutions.jsonl"
COM = FIX / "cvdp_v1.1.0_example_nonagentic_code_generation_commercial_with_solutions.jsonl"
needs_sim = pytest.mark.skipif(
    not (shutil.which("iverilog") and shutil.which("pytest")
         and importlib.util.find_spec("cocotb")),
    reason="Icarus Verilog / cocotb not installed")


def test_load_and_skip():
    rows, skipped = cb.load(OSS)
    assert [r["id"] for r in rows] == ["cvdp_copilot_lfsr_0001"] and not skipped
    rows, skipped = cb.load(COM)
    assert not rows
    assert skipped == {"cvdp_copilot_gf_multiplier_0039":
                       "service xrun uses image __VERIF_EDA_IMAGE__"}


def test_compose_env_rewrite():
    row = cb.load(OSS)[0][0]
    svc = cb.parse_compose(row["harness"]["files"]["docker-compose.yml"])
    assert list(svc) == ["direct"]
    assert svc["direct"]["env_file"] == "./src/.env"
    assert svc["direct"]["command"].startswith("pytest -s")
    env = cb.parse_env(row["harness"]["files"]["src/.env"])
    assert env["TOPLEVEL"] == "lfsr_8bit" and env["SIM"] == "icarus"
    assert cb._rewrite("A=/code/rtl/x.sv P=/src q=rtl/src/y", "/T") == \
        "A=/T/code/rtl/x.sv P=/T/src q=rtl/src/y"


@pytest.mark.skipif(not shutil.which("iverilog"), reason="needs iverilog")
def test_env_file_list():
    """v5.179: env_file given as a YAML list (seen on the full dataset)."""
    row = json.loads(json.dumps(cb.load(OSS)[0][0]))
    files = row["harness"]["files"]
    files["docker-compose.yml"] = (
        "services:\n  direct:\n    image: __OSS_SIM_IMAGE__\n"
        "    env_file:\n      - ./src/.env\n"
        "    command: test \"$TOPLEVEL\" = lfsr_8bit\n")
    svc = cb.parse_compose(files["docker-compose.yml"])
    assert svc["direct"]["env_file"] == ["./src/.env"]
    assert cb.run(row, row["output"]["context"])["passed"]


def test_prompt_single_file():
    row = cb.load(OSS)[0][0]
    system, user = cb.build_prompt(row)
    assert system.startswith("You are a helpful assistance.")
    assert "'Specification to RTL Translation'" in system
    assert "JSON" not in system
    assert user.startswith("\nProvide me one answer for this request: Design the RTL")
    assert user.endswith("saved directly to: rtl/lfsr_8bit.sv.\n")


def test_prompt_multi_file():
    row = json.loads(OSS.read_text().splitlines()[0])
    row["input"]["context"] = {"docs/spec.md": "spec"}
    row["output"]["context"] = {"rtl/a.sv": "", "rtl/b.sv": ""}
    system, user = cb.build_prompt(row)
    assert '{ "code": [{ "<name>" : "<code>"}] }' in system
    assert user.startswith("\nConsider the following content for the file docs/spec.md:")
    assert user.endswith("Name the files as: ['rtl/a.sv', 'rtl/b.sv'].\n")


def test_parse_response():
    one = ["rtl/a.sv"]
    assert cb.parse_response("x\n```verilog\nmodule a; endmodule\n```\ny", one) == \
        {"rtl/a.sv": "module a; endmodule\n"}
    assert cb.parse_response("  module a; endmodule  ", one) == \
        {"rtl/a.sv": "module a; endmodule"}
    two = ["rtl/a.sv", "rtl/b.sv"]
    text = 'Sure: {"code": [{"rtl/a.sv": "module a;\\nendmodule"}, {"rtl/b.sv": "module b; endmodule"}]}'
    assert cb.parse_response(text, two) == {"rtl/a.sv": "module a;\nendmodule",
                                            "rtl/b.sv": "module b; endmodule"}
    assert cb.parse_response("no json here", two) == {}


def test_safe_name():
    assert cb.safe_name("rtl/a.sv")
    assert not cb.safe_name("../a.sv")
    assert not cb.safe_name("/etc/x")
    assert not cb.safe_name("rtl/../../x")


@needs_sim
def test_self_check():
    rows, _ = cb.load(OSS)
    assert cb.self_check(rows, workers=1) == (1, [])


@needs_sim
def test_score():
    rows, _ = cb.load(OSS)
    gold = rows[0]["output"]["context"]["rtl/lfsr_8bit.sv"]
    # a constant 0 would satisfy the upstream LFSR test (0 maps to 0);
    # a constant 8'hFF does not
    wrong = ("module lfsr_8bit(input clock, reset, input [7:0] lfsr_seed, "
             "output reg [7:0] lfsr_out);\n  always @(posedge clock) "
             "lfsr_out <= 8'hFF;\nendmodule\n")
    broken = "module lfsr_8bit(input clock\nendmodule\n"
    texts = {rows[0]["id"]: [f"```verilog\n{gold}\n```", wrong, broken, "x"]}
    out = cb.score(rows, texts, ks=(1, 4), workers=2)
    assert out["pass@1"] == pytest.approx(0.25)
    assert out["pass@4"] == pytest.approx(1.0)
    assert out["pass@1_by_category"] == {"cid003": pytest.approx(0.25)}
