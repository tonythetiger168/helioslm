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
   of a positive-advantage answer (tiny random model, CPU).

Run from repo root: python3 -m pytest helioslm_v5/tests/test_hdl_reward.py
"""
import json
import shutil
import sys
from pathlib import Path

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
