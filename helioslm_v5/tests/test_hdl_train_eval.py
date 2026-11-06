"""T157 — HDL track: SFT on prompt / response records and the
three-benchmark eval runner (v5.178, v5.179).

Pins:

1. sft_hf.render / pack: chat-template prompt tokens get mask 0, response
   + EOS mask 1; a record whose prompt fills --seq is dropped, a long
   response is cut; windows() cuts [n, seq + 1] id / mask rows;
2. sft_hf.masked_loss scores mask-1 targets only (prompt tokens do not
   change it); train() on a tiny torch LM lowers the masked loss and
   reports a validation loss;
3. cpt_hf.cast_frozen (v5.179): frozen parameters in bf16 stay fixed
   while the rest trains under autocast;
4. hdl_eval_all.collect / score_all: every VerilogEval and RTLLM fixture
   gets one job with its harness's own prompt, RTLLM exclusions are
   skipped, and generated references score pass@1 = 1 (needs iverilog);
5. v5.229 repair loop: only failing VerilogEval / RTLLM samples are sent
   back, with the whole conversation and the tool messages; chat_prompt
   handles multi-turn and templates without a system role;
   hb.tool_feedback compiles the answer alone (needs iverilog);
6. v5.233 strip_reasoning: thinking-mode answers are scored without
   their reasoning;
7. v5.234 cvdp_check: single-file CVDP answers are compiled with their
   RTL context only, and the repair loop resends the failing ones.

Run from repo root: python3 -m pytest helioslm_v5/tests/test_hdl_train_eval.py
"""
import shutil
import sys
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "examples"))

import sft_hf  # noqa: E402

FIX = ROOT / "helioslm_v5" / "tests" / "fixtures"


class Tok:
    """Characters as ids; the chat template is '<u>' + text + '<a>'."""
    eos_token_id = 1
    chat_template = "x"

    def apply_chat_template(self, msgs, tokenize, add_generation_prompt):
        if msgs[0]["role"] == "system":
            raise ValueError("no system role")
        return "<u>" + msgs[0]["content"] + "<a>"

    def __call__(self, text, add_special_tokens=False):
        return {"input_ids": [2 + (ord(c) % 50) for c in text]}


def test_render_and_pack():
    tok = Tok()
    ids, mask = sft_hf.render(tok, "sys", "hi", "ok")
    assert len(ids) == len(mask) == len("<u>sys\n\nhi<a>") + 3
    assert mask[-3:] == [1, 1, 1] and ids[-1] == 1 and sum(mask) == 3
    recs = [{"prompt": "p", "response": "r"},
            {"prompt": "x" * 50, "response": "r"},           # prompt too long
            {"prompt": "p", "response": "y" * 40}]           # cut response
    ids, mask, st = sft_hf.pack(recs, tok, 20, ("system", "prompt", "response"))
    assert st["kept"] == 2 and st["dropped_long_prompt"] == 1
    assert st["cut_response"] == 1 and len(ids) == len("<u>p<a>") + 2 + 20
    assert ids.dtype == np.uint32 and mask.dtype == np.uint8
    wi, wm = sft_hf.windows(ids, mask, 8)
    assert wi.shape == wm.shape == ((len(ids) - 1) // 8, 9)
    assert (wi[1, 0] == ids[8]) and (wi[0, -1] == ids[8])



def test_no_think_and_drop_long():
    tok = Tok()
    assert sft_hf.strip_think("<think>a\nb</think>\n\nmodule m;") == "module m;"
    assert sft_hf.strip_think("module m;") == "module m;"
    recs = [{"prompt": "p", "response": "<think>" + "z" * 30 + "</think>ok"},
            {"prompt": "p", "response": "y" * 40}]
    ids, mask, st = sft_hf.pack(recs, tok, 20, ("system", "prompt", "response"),
                                think=False, drop_long=True)
    assert st["kept"] == 1 and st["dropped_long"] == 1 and st["cut_response"] == 0
    assert int(mask.sum()) == 3                            # "ok" + EOS

torch = pytest.importorskip("torch")


class TinyLM(torch.nn.Module):
    def __init__(self, vocab=64, dim=32):
        super().__init__()
        self.emb = torch.nn.Embedding(vocab, dim)
        self.rnn = torch.nn.GRU(dim, dim, batch_first=True)
        self.head = torch.nn.Linear(dim, vocab)

    def forward(self, input_ids):
        h, _ = self.rnn(self.emb(input_ids))
        return (self.head(h),)


def test_masked_loss_ignores_prompt():
    torch.manual_seed(0)
    m = TinyLM()
    x = torch.randint(0, 64, (1, 10))
    y = torch.randint(0, 64, (1, 10))
    mk = torch.zeros(1, 10, dtype=torch.long)
    mk[0, 6:] = 1
    a = sft_hf.masked_loss(m, x, y, mk)
    y2 = y.clone()
    y2[0, :6] = (y2[0, :6] + 1) % 64                   # change prompt targets
    assert torch.allclose(a, sft_hf.masked_loss(m, x, y2, mk))


def test_train_lowers_masked_loss():
    torch.manual_seed(0)
    rng = np.random.default_rng(0)
    seq, n = 16, 64
    ids = np.zeros((n, seq + 1), dtype=np.uint32)
    mask = np.zeros((n, seq + 1), dtype=np.uint8)
    for i in range(n):
        p = rng.integers(2, 60, size=8)
        ids[i, :8], ids[i, 8:] = p, 7                  # response: always 7
        mask[i, 8:] = 1
    m = TinyLM()
    rows = []
    res = sft_hf.train(m, ids[:56], mask[:56], epochs=8, micro_batch=4,
                       grad_accum=1, lr=1e-2, warmup=2, val_ids=ids[56:],
                       val_mask=mask[56:], autocast=False, log=rows.append,
                       eval_every=0)
    assert res["steps"] == 8 * 56 // 4 and res["stopped"] is None
    assert rows[-1]["loss"] < rows[0]["loss"] / 4
    assert res["val_loss"] < 0.5


def test_frozen_bf16_trains():
    """v5.179: frozen parameters stored in bf16 stay fixed; training of the
    rest still runs under autocast."""
    from cpt_hf import cast_frozen, freeze
    torch.manual_seed(0)
    m = TinyLM()
    m.norm = torch.nn.LayerNorm(32)
    tr, fr = freeze(m, ["emb"])
    saved = cast_frozen(m, torch.bfloat16)
    assert fr == 64 * 32 and saved == fr * 2
    assert m.emb.weight.dtype == torch.bfloat16
    assert m.head.weight.dtype == torch.float32
    before = m.emb.weight.detach().clone()
    ids = np.full((8, 9), 7, dtype=np.uint32)
    mask = np.ones((8, 9), dtype=np.uint8)
    res = sft_hf.train(m, ids, mask, epochs=2, micro_batch=4, grad_accum=1,
                       lr=1e-2, warmup=1, autocast=True, eval_every=0)
    assert res["steps"] == 4 and res["stopped"] is None
    assert torch.equal(m.emb.weight, before)


@pytest.mark.skipif(shutil.which("iverilog") is None, reason="needs iverilog")
def test_eval_all_references(tmp_path):
    import json
    import hdl_eval_all as ea
    from helioslm_v5.eval import hdl_bench as hb
    from helioslm_v5.eval import rtllm_bench as rb
    names = ("adder_8bit", "multi_booth_8bit", "multi_pipe_4bit")
    rb.build_cache(FIX / "rtllm", tmp_path / "rt", designs=names)
    jobs, data = ea.collect(("rtllm",), None, tmp_path / "rt", None,
                            exclude=("multi_pipe_4bit",))
    designs = data["rtllm"]
    assert [designs[k]["name"] for (_, k, _, _) in jobs] == list(names[:2])
    assert jobs[0][2] == hb.SYSTEM_MSG
    assert jobs[0][3] == rb.build_prompt(designs[0]["prompt"])
    outs = [["```verilog\n" + rb.reference_sample(designs[k]) + "\n```"]
            for (_, k, _, _) in jobs]
    res = ea.score_all(jobs, outs, data, 1, (1,), exclude=("multi_pipe_4bit",))
    assert res["rtllm"]["pass@1"] == 1.0 and res["rtllm"]["n_designs"] == 2
    assert ea.summary(res)["rtllm"]["pass@1"] == 1.0
    json.dumps(res)
    # --samples rescores saved generations without a GPU (v5.194)
    samples = tmp_path / "s.samples.json"
    samples.write_text(json.dumps(
        [{"bench": b, "key": k, "outputs": o} for (b, k, _, _), o in zip(jobs, outs)]
        + [{"bench": "cvdp", "key": "unused", "outputs": ["x"]}]))
    out = tmp_path / "r.json"
    assert ea.main(["--model", "m", "--samples", str(samples), "--out", str(out),
                    "--benches", "rtllm", "--rtllm-cache", str(tmp_path / "rt"),
                    "--rtllm-exclude", "multi_pipe_4bit", "--workers", "1"]) == 0
    assert json.loads(out.read_text())["summary"]["rtllm"]["pass@1"] == 1.0


# ---- v5.229: agent repair loop with open-tool feedback (no testbench) ----

def test_repair_loop_resends_only_failing_samples():
    import hdl_eval_all as ea
    jobs = [("verilog-eval", 0, "sys", "write A"),
            ("rtllm", 1, "sys", "write B"),
            ("cvdp", "c", "sys", "write C")]           # not repaired
    outputs = [["BAD a", "module ok a"], ["BAD b"], ["BAD c"]]
    calls = []

    def gen(rjobs, n, t, p, new):
        calls.append(rjobs)
        # round 1 fixes job 0 only; round 2 fixes job 1
        return [["module fixed"] if (len(calls) == 1 and k == 0) or len(calls) == 2
                else ["BAD again"] for _, k, _, _ in rjobs]

    def check(code):
        ok = code.lstrip().startswith("module")
        return {"ok": ok, "feedback": "" if ok else "line 1: syntax error"}

    out, stats = ea.repair(gen, jobs, outputs, 3, check, workers=2)
    assert out[0] == ["module fixed", "module ok a"]
    assert out[1] == ["module fixed"] and out[2] == ["BAD c"]
    assert stats == [{"round": 1, "failing": 2, "fixed": 1},
                     {"round": 2, "failing": 1, "fixed": 1},
                     {"round": 3, "failing": 0, "fixed": 0}]
    # round 2 carries the whole conversation: prompt, answer, feedback x2
    msgs = calls[1][0][3]
    assert [m["role"] for m in msgs] == ["user", "assistant", "user",
                                        "assistant", "user"]
    assert msgs[0]["content"] == "write B" and "syntax error" in msgs[2]["content"]
    assert msgs[3]["content"] == "BAD again"


def test_chat_prompt_multi_turn_and_no_system_role():
    import hdl_eval_all as ea

    class Tok:
        chat_template = "x"

        def __init__(self, system_ok):
            self.system_ok = system_ok

        def apply_chat_template(self, msgs, tokenize, add_generation_prompt,
                                **kw):
            if not self.system_ok and msgs[0]["role"] == "system":
                raise ValueError("no system role")
            head = f"think={kw['enable_thinking']}|" if kw else ""
            return head + "|".join(f"{m['role']}:{m['content']}" for m in msgs)

    turns = [{"role": "user", "content": "u1"},
             {"role": "assistant", "content": "a1"},
             {"role": "user", "content": "u2"}]
    assert ea.chat_prompt(Tok(True), "S", "u") == "system:S|user:u"
    assert ea.chat_prompt(Tok(False), "S", turns) == "user:S\n\nu1|assistant:a1|user:u2"
    assert ea.chat_prompt(Tok(True), "S", "u", thinking=False) == "think=False|system:S|user:u"


@pytest.mark.skipif(shutil.which("iverilog") is None, reason="needs iverilog")
def test_tool_feedback_compile_only():
    from helioslm_v5.eval import hdl_bench as hb
    good = "module TopModule(input a, output b);\n  assign b = ~a;\nendmodule\n"
    bad = "module TopModule(input a, output b);\n  assign b = ~a\nendmodule\n"
    assert hb.tool_feedback(good)["ok"]
    r = hb.tool_feedback(bad)
    assert not r["ok"] and "line" in r["feedback"]
    assert not hb.tool_feedback("no code")["ok"]


def test_strip_reasoning_keeps_only_the_answer():
    """v5.233: thinking-mode output is cut after the reasoning marker."""
    import hdl_eval_all as ea
    gemma = ("<|channel>thought\nadd 8 bits, ripple carry [BEGIN] not this"
             "<channel|>```verilog\nmodule a(input x, output y);\n"
             "  assign y = x < 1'b1;\nendmodule\n```<turn|>")
    out = ea.strip_reasoning(gemma)
    assert out.startswith("```verilog") and "thought" not in out
    assert "x < 1'b1" in out and "<turn|>" not in out
    assert ea.strip_reasoning("<think>a</think>\n[BEGIN]\nm\n[DONE]") == "[BEGIN]\nm\n[DONE]"
    assert ea.strip_reasoning("module m; endmodule") == "module m; endmodule"
    oss = ("<|channel|>analysis<|message|>use assign<|end|><|start|>assistant"
           "<|channel|>final<|message|>module m; endmodule<|return|>")
    assert ea.strip_reasoning(oss) == "module m; endmodule"


def test_cvdp_check_and_repair():
    """v5.234: CVDP answers in the repair loop (compile stubbed)."""
    import hdl_eval_all as ea
    rows = {"one": {"input": {"context": {"rtl/pkg.sv": "package p; endpackage",
                                          "verif/tb.sv": "module tb; endmodule",
                                          "rtl/top.sv": "old"}},
                    "output": {"context": {"rtl/top.sv": ""}}},
            "two": {"input": {"context": {}},
                    "output": {"context": {"rtl/a.sv": "", "rtl/b.sv": ""}}}}
    seen = []

    def fake_compile(td, target, others):
        seen.append((target, others))
        code = Path(td, target).read_text()
        return None if code.startswith("module") else f"{target}:1: syntax error\nI give up."
    check = ea.cvdp_check(rows, compile_fn=fake_compile)
    bad = check("one", "rtl/top.sv\nmodule top; endmodule")
    assert not bad["ok"] and bad["feedback"] == "line 1: syntax error"
    assert seen[-1] == ("rtl/top.sv", ["rtl/pkg.sv"])          # no tb, no old copy
    assert check("one", "```verilog\nmodule top; endmodule\n```")["ok"]
    assert check("two", "{}")["ok"]                            # JSON multi-file: skipped
    assert not check("one", "sorry")["ok"]
    jobs = [("cvdp", "one", "sys", "write top")]

    def gen(rjobs, n, t, p, new):
        return [["module top; endmodule"] for _ in rjobs]
    out, stats = ea.repair(gen, jobs, [["rtl/top.sv\nmodule top; endmodule"]], 2,
                           None, workers=1, bench_checks={"cvdp": check})
    assert out == [["module top; endmodule"]]
    assert stats[0] == {"round": 1, "failing": 1, "fixed": 1}


def test_consensus_pick_and_select():
    import hdl_eval_all as ea
    same = lambda a, b: a.strip() == b.strip()        # noqa: E731
    # the majority behaviour wins; a non-compiling candidate never votes
    assert ea.consensus_pick(["A", "B", "B", "C"], [True, True, True, True], same) == 1
    assert ea.consensus_pick(["B", "A", "A", "A"], [True, False, True, True], same) == 2
    assert ea.consensus_pick(["x", "y"], [False, False], same) == 0
    assert ea.consensus_pick(["x", "y"], [False, True], same) == 1
    assert ea.consensus_pick(["A", "B"], [True, True], same) == 0     # tie -> first
    jobs = [("cvdp", "c1", "", "p"), ("uvm", "u1", "", "p")]
    outs = [["bad", "good", "good2"], ["u0", "u1"]]
    got, picks = ea.select_consensus(
        jobs, outs, check=None,
        bench_checks={"cvdp": lambda k, t: {"ok": t.startswith("good")}})
    assert picks == [1, 0] and got == [["good"], ["u0"]]


def test_tb_pick_prefers_candidates_passing_model_testbenches():
    import hdl_eval_all as ea
    table = {("A", "t1"): (3, 10), ("A", "t2"): (0, 10),
             ("B", "t1"): (0, 10), ("B", "t2"): (0, 10),
             ("C", "t1"): (0, 10), ("C", "t2"): (0, 10),
             ("D", "t1"): None, ("D", "t2"): None}
    run = lambda c, t: table.get((c, t))                              # noqa: E731
    same = lambda a, b: {a, b} == {"B", "C"} or a == b               # noqa: E731
    pick, st = ea.tb_pick(["A", "B", "C", "D"], [True] * 4, ["t1", "t2", "t3"], run, same)
    assert pick == 1 and st == {"valid_tbs": 2, "best_pass": 2}     # t3 runs nowhere
    # no compiling candidate -> 0; no valid testbench -> falls back to votes
    assert ea.tb_pick(["A"], [False], ["t1"], run, same)[0] == 0
    pick, st = ea.tb_pick(["A", "B", "C"], [True] * 3, ["t3"], run, same)
    assert pick == 1 and st["valid_tbs"] == 0
    gen = lambda jobs, n, *a: [["```verilog\nmodule tb; endmodule\n```"] * n for _ in jobs]  # noqa: E731
    jobs = [("verilog-eval", "p1", "", "spec"), ("cvdp", "c1", "", "p")]
    outs = [["x", "y"], ["bad", "good"]]
    got, picks, stats, tbs = ea.select_tb(
        gen, jobs, outs, check=lambda c: {"ok": False}, n_tb=2,
        bench_checks={"cvdp": lambda k, t: {"ok": t == "good"}})
    assert picks == [0, 1] and got == [["x"], ["good"]] and stats["tb_jobs"] == 1
    assert tbs["p1"][0].startswith("module tb")


def test_select_consensus_votes_cvdp_when_asked(monkeypatch):
    import hdl_eval_all as ea
    seen = {}

    def fake_pick(codes, ok, equiv=None):
        seen["ok"] = ok
        return 2
    monkeypatch.setattr(ea, "consensus_pick", fake_pick)
    jobs = [("cvdp", "c1", "", "p")]
    outs = [["bad", "good", "good2"]]
    chk = {"cvdp": lambda k, t: {"ok": t.startswith("good")}}
    got, picks = ea.select_consensus(jobs, outs, None, bench_checks=chk, vote_checked=("cvdp",))
    assert picks == [2] and got == [["good2"]] and seen["ok"] == [False, True, True]
    got, picks = ea.select_consensus(jobs, outs, None, bench_checks=chk)
    assert picks == [1]
