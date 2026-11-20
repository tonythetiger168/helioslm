"""T161 — HDL track H4: simulation-equivalence reward and the RL helpers,
v5.185.

Pins:

1. ``top_ports`` reads ANSI and non-ANSI port lists, folds parameter
   defaults into widths and refuses inout ports;
2. ``top_module`` picks the module nothing instantiates;
3. ``extract_answer`` takes the last fenced block with a module (fence
   mid-line), else module..endmodule;
4. with Icarus: an equivalent answer gets 1, a wrong one 0, a missing
   module 0; a golden design with submodules works; X outputs of the
   golden design are not compared;
5. with yosys: ``ref_cells`` is set and a smaller correct answer gets
   the gate bonus;
6. ``row_prompt_golden`` reads verl-style rows; ``group_advantages``
   drops groups with one reward value; ``pg_loss`` raises the log-prob
   of a positive-advantage answer (tiny random model, CPU);
7. ``mutants`` is deterministic and never edits comments or the module
   header; ``audit_task`` classifies killed, false-accept, equivalent
   and unknown mutants with injectable simulators / equivalence checkers;
   ``formal_equiv`` maps Yosys proof, counterexample and unproven logs.

Run from repo root: python3 -m pytest helioslm_v5/tests/test_hdl_reward.py
"""
import json
import shutil
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "examples"))

from helioslm_v5.eval import hdl_reward as hr  # noqa: E402

needs_iverilog = pytest.mark.skipif(not shutil.which("iverilog"), reason="Icarus not installed")
needs_yosys = pytest.mark.skipif(not shutil.which("yosys"), reason="yosys not installed")

COUNTER = """module counter #(parameter W = 4) (input clk, input rst_n, input en,
    output reg [W-1:0] q);
  always @(posedge clk or negedge rst_n) if (!rst_n) q <= 0; else if (en) q <= q + 1'b1;
endmodule
"""
MUX = """module mux(a, b, s, y);
  input [7:0] a, b; input s; output [7:0] y;
  sub u (.a(a), .b(b), .s(s), .y(y));
endmodule
module sub(input [7:0] a, input [7:0] b, input s, output [7:0] y);
  assign y = s ? b : a;
endmodule
"""
AUDIT_SRC = """module m(input [1:0] a, input [1:0] b, input s, output [1:0] y);
  // keep this comment: a & b -> 0
  wire [1:0] t = a & b;
  assign y = s ? t : a;
endmodule
"""


def _audit_task():
    return {"golden": AUDIT_SRC, "top": "m", "ports": hr.top_ports(AUDIT_SRC, "m")}


def test_mutants_deterministic_and_interface_preserving():
    first = hr.mutants(AUDIT_SRC, top="m", max_mutants=4)
    assert first == hr.mutants(AUDIT_SRC, top="m", max_mutants=4)
    assert len(first) == 2
    assert len({label for label, _ in first}) == len(first)
    header = AUDIT_SRC.split(");", 1)[0] + ");"
    for _, text in first:
        assert text.startswith(header)
        assert "// keep this comment: a & b -> 0" in text
        assert hr.top_module(text) == "m"
        assert hr.top_ports(text, "m") == hr.top_ports(AUDIT_SRC, "m")


def test_audit_killed_mutants_and_seed_forwarding():
    calls = []

    def sim(answer, golden, name, ports, seed=1):
        calls.append((answer, seed))
        bad = answer != golden
        return {"ok": not bad, "mismatches": int(bad), "checks": 3, "log": ""}

    out = hr.audit_task(_audit_task(), seeds=(7, 8, 9), max_mutants=2,
                        simulate_fn=sim, equiv_fn=lambda *args: False)
    assert out["golden_ok"] and out["ok"]
    assert out["counts"]["killed"] == 2
    assert [seed for _, seed in calls[:3]] == [7, 8, 9]
    assert [seed for _, seed in calls[3:]] == [7, 7]


def test_audit_false_accept_and_equivalent_mutants():
    def sim(*args, seed=1):
        return {"ok": True, "mismatches": 0, "checks": 3, "log": ""}

    false = hr.audit_task(_audit_task(), seeds=(5,), max_mutants=2,
                          simulate_fn=sim, equiv_fn=lambda *args: (False, "diff"))
    assert not false["ok"]
    assert false["counts"]["false_accept"] == 2
    assert false["false_accept"] == [row["label"] for row in false["results"]]

    equal = hr.audit_task(_audit_task(), seeds=(5,), max_mutants=2,
                          simulate_fn=sim, equiv_fn=lambda *args: True)
    assert equal["ok"] and equal["counts"]["equivalent"] == 2


def test_audit_survivor_without_formal_verdict_is_unknown():
    def sim(*args, seed=1):
        return {"ok": True, "mismatches": 0, "checks": 3, "log": ""}

    out = hr.audit_task(_audit_task(), seeds=(5, 6), max_mutants=2,
                        simulate_fn=sim)
    assert out["ok"]                       # no proven false accept
    assert out["counts"]["unknown"] == 2   # but simulation did not prove safety
    assert len(out["unknown"]) == 2


def test_formal_equiv_verdicts(monkeypatch):
    monkeypatch.setattr(hr.shutil, "which", lambda tool: "/bin/yosys")
    logs = iter([
        (0, "SAT proof finished - no model found: SUCCESS"),
        (1, "SAT proof finished - model found: FAIL"),
        (1, "ERROR: unproven"),
    ])

    def run(cmd, cwd, capture_output, text, timeout):
        assert "miter -equiv" in cmd[3] and "-set-init-undef" in cmd[3]
        code, log = next(logs)
        return SimpleNamespace(returncode=code, stdout=log, stderr="")

    monkeypatch.setattr(hr.subprocess, "run", run)
    assert hr.formal_equiv(AUDIT_SRC, AUDIT_SRC, "m")[0] is True
    assert hr.formal_equiv(AUDIT_SRC, AUDIT_SRC, "m")[0] is False
    assert hr.formal_equiv(AUDIT_SRC, AUDIT_SRC, "m")[0] is None


def test_audit_cli_writes_summary(tmp_path, monkeypatch):
    import hdl_rl_hf as rl
    task_path = tmp_path / "tasks.jsonl"
    out_path = tmp_path / "audit" / "result.json"
    task_path.write_text(json.dumps(_audit_task()) + "\n", encoding="utf-8")

    def fake(task, seeds, max_mutants, simulate_fn=None, equiv_fn=None):
        assert seeds == (3, 4) and max_mutants == 2 and equiv_fn is None
        return {"ok": True, "golden_ok": True, "seeds": list(seeds),
                "n_mutants": 2, "counts": {"killed": 1, "equivalent": 0,
                                           "false_accept": 0, "unknown": 1,
                                           "invalid": 0},
                "false_accept": [], "unknown": ["x"], "invalid": [],
                "results": []}

    monkeypatch.setattr(rl.hr, "audit_task", fake)
    out = rl.cmd_audit(SimpleNamespace(tasks=str(task_path), out=str(out_path),
                                       seeds=[3, 4], mutants=2, formal=False,
                                       workers=1))
    assert out["summary"] == {"tasks": 1, "golden_ok": 1,
                              "false_accept_tasks": 0, "unknown_tasks": 1,
                              "counts": {"killed": 1, "equivalent": 0,
                                         "false_accept": 0, "unknown": 1,
                                         "invalid": 0},
                              "ok": True}
    assert json.loads(out_path.read_text(encoding="utf-8"))["schema"] == \
        "hdl-reward-audit/1"


def test_ports_and_top():
    assert hr.top_ports(COUNTER) == [("input", 1, "clk"), ("input", 1, "rst_n"),
                                     ("input", 1, "en"), ("output", 4, "q")]
    assert hr.top_module(MUX) == "mux"
    assert hr.top_ports(MUX) == [("input", 8, "a"), ("input", 8, "b"),
                                 ("input", 1, "s"), ("output", 8, "y")]
    assert hr.top_ports("module m #(parameter N=3)(input [2*N-1:0] x, output y); endmodule") \
        == [("input", 6, "x"), ("output", 1, "y")]
    with pytest.raises(hr.RewardError):
        hr.top_ports("module m(inout a, output b); endmodule")


def test_extract_answer():
    t = "think\n<answer> ```verilog\nmodule a; endmodule\n``` </answer>"
    assert hr.extract_answer(t) == "module a; endmodule\n"
    t = "```verilog\nmodule a; endmodule\n```\nfix:\n```verilog\nmodule b; endmodule\n```"
    assert hr.extract_answer(t) == "module b; endmodule\n"
    assert hr.extract_answer("so module c; endmodule ok") == "module c; endmodule\n"
    assert hr.extract_answer("nothing") == ""


@needs_iverilog
def test_reward_equivalence():
    task = hr.make_task("count", COUNTER, synth=None)
    good = ("```verilog\nmodule counter(input clk, input rst_n, input en, output reg [3:0] q);\n"
            "always @(posedge clk or negedge rst_n) if (!rst_n) q <= 4'd0; else q <= q + en;\n"
            "endmodule\n```")
    assert hr.reward(good, task, gate_weight=0)[0] == 1.0
    assert hr.reward(good.replace("q + en", "q + 1"), task, gate_weight=0)[0] == 0.0
    assert hr.reward("module other; endmodule", task)[0] == 0.0
    mux = hr.make_task("mux", MUX, synth=None)
    flat = ("```verilog\nmodule mux(input [7:0] a, input [7:0] b, input s, output [7:0] y);\n"
            "assign y = s ? b : a;\nendmodule\n```")
    assert hr.reward(flat, mux, gate_weight=0)[0] == 1.0
    assert hr.reward(flat.replace("s ? b : a", "s ? a : b"), mux, gate_weight=0)[0] == 0.0


@needs_iverilog
def test_x_outputs_not_compared():
    gold = ("module r(input clk, input d, output reg q);\n"
            "always @(posedge clk) q <= q;\nendmodule\n")   # q stays X
    task = hr.make_task("x", gold, synth=None)
    ans = "module r(input clk, input d, output reg q = 0);\nendmodule\n"
    assert hr.reward(ans, task, gate_weight=0)[0] == 1.0


@needs_iverilog
@needs_yosys
def test_gate_bonus():
    gold = ("module add(input [3:0] a, input [3:0] b, output [3:0] y);\n"
            "wire [3:0] t1 = a + b; wire [3:0] t2 = a + b;\n"
            "assign y = (a[0] ? t1 : t2) ^ (a & ~a);\nendmodule\n")
    task = hr.make_task("add", gold)
    assert task["ref_cells"] > 0
    small = "module add(input [3:0] a, input [3:0] b, output [3:0] y); assign y = a + b; endmodule"
    r, info = hr.reward(small, task)
    assert 1.0 <= r <= 1.0 + hr.GATE_WEIGHT and info["cells"] <= task["ref_cells"]


def test_row_prompt_golden():
    import hdl_rl_hf as rl
    row = {"prompt": [{"role": "system", "content": "sys"},
                      {"role": "user", "content": "Write a mux."}],
           "reward_model": {"ground_truth": json.dumps({"answer": "```verilog\n" + MUX + "```"})},
           "data_source": "codev"}
    p, g = rl.row_prompt_golden(row)
    assert p == "Write a mux." and g.startswith("module mux(a, b, s, y);")
    with pytest.raises(hr.RewardError):
        rl.row_prompt_golden({"prompt": "x", "other": "no code here"})
    raw = "// this module adds\n`timescale 1ns/1ps\nmodule m(input a, output y);\nassign y = a;\nendmodule\n"
    assert rl.row_prompt_golden({"prompt": "x", "reward_model": {"ground_truth": raw}})[1] == raw
    assert hr.extract_answer("The module below:\nmodule q(output y); endmodule") \
        == "module q(output y); endmodule\n"


def test_group_advantages():
    import hdl_rl_hf as rl
    a = rl.group_advantages([1, 0, 1, 0, 1, 1, 1, 1], 4)
    assert a[4:] == [None] * 4
    assert a[0] > 0 > a[1] and abs(sum(a[:4])) < 1e-9


def test_pg_loss_raises_logprob_of_positive_answer():
    torch = pytest.importorskip("torch")
    transformers = pytest.importorskip("transformers")
    import hdl_rl_hf as rl
    torch.manual_seed(0)
    cfg = transformers.GPT2Config(vocab_size=32, n_positions=32, n_embd=16, n_layer=1, n_head=2)
    model = transformers.GPT2LMHeadModel(cfg)
    prompt, resp = [1, 2, 3], [4, 5]

    def logp():
        with torch.no_grad():
            ids = torch.tensor([prompt + resp])
            lg = model(input_ids=ids[:, :-1]).logits[0, -2:]
            return -torch.nn.functional.cross_entropy(lg, ids[0, -2:], reduction="sum").item()

    before = logp()
    opt = torch.optim.SGD(model.parameters(), lr=0.5)
    rl.pg_loss(model, prompt, resp, 1.0, len(resp), "cpu").backward()
    opt.step()
    assert logp() > before
