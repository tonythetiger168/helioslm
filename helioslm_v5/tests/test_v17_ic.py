"""T49 - v1.7: IC design + verification plugin (RTL/UVM)."""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from harness.harness_core import Context
from harness.harness_plugins import (ICPlugin, ICWorkflowPlugin,
                                     ModelPlugin, PresetPlugin, StatePlugin)


def _fake_model(prompt, seed, step):
    # UVM first: the gen_uvm prompt embeds the RTL (which contains
    # "module"), so checking "module" first misroutes UVM requests to RTL
    if "UVM" in prompt:
        return """class adder_test extends uvm_test;
  function void build_phase(uvm_phase phase);
  endfunction
endclass"""
    if "Verilog" in prompt or "module" in prompt:
        return """module adder(input [7:0] a, b, output [8:0] sum);
  assign sum = a + b;
endmodule"""
    return """class adder_test extends uvm_test;
  function void build_phase(uvm_phase phase);
  endfunction
endclass"""


def test_ic_gen_and_lint():
    ctx = Context()
    ctx.use(PresetPlugin("full", model_fn=None))
    ctx.use(ICPlugin())
    rtl = ctx.ic.gen_rtl("an 8-bit adder", _fake_model)
    assert "module adder" in rtl
    lint = ctx.ic.lint(rtl)
    assert lint["ok"], lint["errors"]
    print("PASS test_ic_gen_and_lint")


def test_ic_lint_catches_errors():
    ctx = Context()
    ctx.use(ICPlugin())
    bad = "module foo(input a; output b; assign b = a;"
    lint = ctx.ic.lint(bad)
    assert not lint["ok"]
    assert any("endmodule" in e for e in lint["errors"])
    print("PASS test_ic_lint_catches_errors")


def test_ic_workflow_end_to_end():
    ctx = Context()
    ctx.use(PresetPlugin("full", model_fn=None))
    ctx.use(ICPlugin())
    ctx.use(ICWorkflowPlugin())
    ctx.use(ModelPlugin("ic", _fake_model))
    r = ctx.ic.run("an 8-bit adder with carry out", ctx.model.ic)
    assert r["lint"]["ok"]
    assert "uvm_test" in r["uvm"]
    rep = ctx.verify_all_effects()
    assert rep["failed"] == 0
    print("PASS test_ic_workflow_end_to_end (effects all verified)")


def test_coverage():
    ctx = Context()
    ctx.use(ICPlugin())
    c = ctx.ic.coverage(goal=100, hit=87)
    assert c["pct"] == 87.0
    print("PASS test_coverage")


if __name__ == "__main__":
    test_ic_gen_and_lint()
    test_ic_lint_catches_errors()
    test_ic_workflow_end_to_end()
    test_coverage()
    print("\n4/4 IC plugin tests passed")
