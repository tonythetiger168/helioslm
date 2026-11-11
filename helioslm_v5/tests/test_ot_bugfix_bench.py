"""T169 — OpenTitan real-IP bug-fix / completion benchmark, v5.228.

Pins:

1. ``code_mask`` / ``module_body_span``: comments, strings and the port
   header are never mutated;
2. ``mutate``: one ``bugfix`` site (operator / inverted if / off-by-one
   sized constant / flipped reset value) or one masked ``assign`` /
   ``always`` block, the same for the same seed, with the line count
   kept for ``complete``;
3. ``prompt`` asks for the whole file between [BEGIN] and [DONE];
4. ``candidates`` keeps single-module ``hw/ip/*/rtl`` files only;
   ``seen_ips`` reads passing OpenTitan units from an H5 manifest;
5. ``summarize`` groups by kind, split, kind x split and operator;
6. with yosys + sv2v installed: ``build`` on a toy IP keeps tasks whose
   mutant is not equivalent, the original file scores as a pass and the
   mutant as a fail (skipped without the tools);
7. v5.230 ``splice``: a ``complete`` answer holding only the missing
   block is put in place of the TODO comment (never for ``bugfix``);
8. v5.231 ``edit_pairs``: SFT records (mutated file -> whole original)
   skip every evaluation IP and file, are seeded, and also come from
   packed project documents (``split_packed``).

Plain functions, no pytest fixtures (folded into the daily gate).
Run from repo root: python3 -m pytest helioslm_v5/tests/test_ot_bugfix_bench.py
"""
import json
import random
import sys
import tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(ROOT))

from helioslm_v5.eval import ot_bugfix_bench as ob  # noqa: E402
from helioslm_v5.src.training import hdl_projects as hp  # noqa: E402

TOY = """// toy counter, the comment has == and if (x) in it
module toy_cnt #(
  parameter int W = 4
) (
  input  logic         clk_i,
  input  logic         rst_ni,
  input  logic         en_i,
  output logic [W-1:0] cnt_o,
  output logic         wrap_o
);
  logic [W-1:0] cnt_q;
  always_ff @(posedge clk_i or negedge rst_ni) begin
    if (!rst_ni) begin
      cnt_q <= '0;
    end else if (en_i) begin
      cnt_q <= cnt_q + 4'd1;
    end
  end
  assign cnt_o = cnt_q;
  assign wrap_o = en_i && (cnt_q == 4'd15);
endmodule
"""


def test_mask_and_body_span():
    mask = ob.code_mask(TOY)
    assert not mask[TOY.index("==")]           # in the comment
    lo, hi = ob.module_body_span(TOY)
    assert TOY[lo:hi].lstrip().startswith("logic [W-1:0] cnt_q;")
    assert "input" not in TOY[lo:hi] and "endmodule" not in TOY[lo:hi]


def test_bugfix_mutation_sites():
    span = ob.module_body_span(TOY)
    sites = ob._sites(TOY, span, ob.code_mask(TOY))
    kinds = {s[0] for s in sites}
    assert kinds == {"op", "cond", "const", "reset"}
    for _, s, _, _ in sites:
        assert s >= span[0]                     # never the header/comment
    reps = {(k, r) for k, _, _, r in sites}
    assert ("reset", "<= '1;") in reps
    assert ("const", "4'd2") in reps and ("const", "4'd14") in reps
    assert ("op", "||") in reps and ("op", "-") in reps


def test_mutate_is_seeded_and_single_site():
    a = ob.mutate(TOY, "bugfix", random.Random(3))
    b = ob.mutate(TOY, "bugfix", random.Random(3))
    assert a == b and a["text"] != TOY
    assert a["op"] in ob.SYMPTOM
    diff = [i for i, (x, y) in enumerate(zip(TOY.splitlines(),
                                             a["text"].splitlines())) if x != y]
    assert len(diff) == 1 and diff[0] + 1 == a["line"]
    ops = {ob.mutate(TOY, "bugfix", random.Random(s))["op"] for s in range(40)}
    assert ops == set(ob.SYMPTOM)               # every operator kind is reachable


def test_complete_masks_one_block():
    seen = set()
    for s in range(30):
        m = ob.mutate(TOY, "complete", random.Random(s))
        assert "// TODO: one" in m["text"]
        assert m["text"].count("\n") == TOY.count("\n")
        seen.add(m["op"])
        if m["op"] == "always":
            assert "always_ff" not in m["text"]
            assert "assign cnt_o" in m["text"]
    assert seen == {"assign", "always"}


def test_always_end_follows_else():
    text = ("module m(input logic c, input logic a, output logic q);\n"
            "  always_ff @(posedge c) begin\n"
            "    if (a) begin q <= 1'b1; end\n"
            "  end\n"
            "  always_comb if (a) begin q = 1; end else begin q = 0; end\n"
            "endmodule\n")
    span = ob.module_body_span(text)
    blocks = ob._blocks(text, span, ob.code_mask(text))
    always = [text[s:e] for s, e, k in blocks if k == "always"]
    assert always[0].endswith("end\n  end")
    assert always[1].endswith("else begin q = 0; end")


def test_prompt_asks_for_whole_file():
    m = ob.mutate(TOY, "bugfix", random.Random(0))
    task = {"kind": "bugfix", "op": m["op"], "path": "hw/ip/toy/rtl/toy_cnt.sv",
            "ip": "toy", "prompt_text": m["text"]}
    p = ob.prompt(task)
    assert "[BEGIN]" in p and "[DONE]" in p and ob.SYMPTOM[m["op"]] in p
    assert m["text"] in p
    task.update(kind="complete")
    assert "TODO" in ob.prompt(task)


def test_candidates_and_seen_ips():
    texts = {"hw/ip/toy/rtl/toy_cnt.sv": TOY,
             "hw/ip/toy/rtl/two.sv": TOY + TOY.replace("toy_cnt", "toy2"),
             "hw/ip/toy/dv/toy_tb.sv": TOY.replace("toy_cnt", "toy_tb"),
             "hw/ip/toy/rtl/toy_pkg.sv": "package toy_pkg; endpackage\n",
             "hw/top/rtl/x.sv": TOY.replace("toy_cnt", "x")}
    info, _ = hp.index(texts)
    assert ob.candidates(texts, info) == [("toy", "hw/ip/toy/rtl/toy_cnt.sv",
                                           "toy_cnt")]
    man = {"sources": {"opentitan-sv": {"projects": [
        {"project": "opentitan-sv/hw/ip/uart", "pass": True},
        {"project": "opentitan-sv/hw/ip/spi_host", "pass": False},
        {"project": "opentitan-sv/hw/ip/gpio", "pass": True, "near_dup": True}]},
        "ibex-sv": {"projects": [{"project": "ibex-sv/hw/ip/x", "pass": True}]}}}
    with tempfile.TemporaryDirectory() as tmp:
        f = Path(tmp, "manifest.json")
        f.write_text(json.dumps(man))
        assert ob.seen_ips(f) == {"uart"}
    assert ob.seen_ips(None) == set()


def test_summarize_groups():
    tasks = [{"id": "a", "kind": "bugfix", "split": "seen", "op": "op"},
             {"id": "b", "kind": "complete", "split": "unseen", "op": "assign"},
             {"id": "c", "kind": "bugfix", "split": "unseen", "op": "cond"}]
    res = {"a": [{"pass": True, "synth": True}],
           "b": [{"pass": False, "synth": True}, {"pass": True, "synth": True}],
           "c": [{"pass": False, "synth": False}]}
    s = ob.summarize(tasks, res)
    assert s["all"] == {"n": 3, "pass": 0.5, "synth": 0.6667}
    assert s["bugfix"]["pass"] == 0.5 and s["complete/unseen"]["pass"] == 0.5
    assert s["op:cond"] == {"n": 1, "pass": 0.0, "synth": 0.0}


def test_build_and_score_with_tools():
    """End to end on a one-file toy IP (needs yosys + sv2v + git)."""
    if not ob.have_tools():
        return
    with tempfile.TemporaryDirectory() as tmp:
        f = Path(tmp, "hw/ip/toy/rtl/toy_cnt.sv")
        f.parent.mkdir(parents=True)
        f.write_text(TOY)
        tasks = ob.build(tmp, seed=1, per_ip=4, log=lambda *_: None)
        assert tasks and {t["split"] for t in tasks} == {"unseen"}
        scorer = ob.Scorer(tmp)
        for t in tasks:
            good = scorer.score(t, f"[BEGIN]\n{TOY}[DONE]\n")
            assert good["pass"], (t["id"], good)
            if t["kind"] == "bugfix":
                bad = scorer.score(t, f"```systemverilog\n{t['prompt_text']}```")
                assert bad["synth"] and not bad["pass"], (t["id"], bad)
        assert not scorer.score(tasks[0], "no code here")["pass"]


def test_splice_snippet_answers():
    m = ob.mutate(TOY, "complete", random.Random(0))
    task = {"kind": "complete", "prompt_text": m["text"]}
    hole = next(ln for ln in m["text"].splitlines() if "// TODO: one" in ln)
    snippet = "  assign cnt_o = cnt_q;"
    out = ob.splice(task, "// TODO: one assign block was removed here; restore it\n"
                    + snippet + "\n")
    assert out is not None and snippet in out and hole not in out
    assert out.count("module") == m["text"].count("module")
    assert ob.splice(dict(task, kind="bugfix"), snippet) is None
    assert ob.splice(task, "  \n") is None


def test_edit_pairs_exclude_eval_ips():
    texts = {"hw/ip/toy/rtl/toy_cnt.sv": TOY,
             "hw/ip/other/rtl/other_cnt.sv": TOY.replace("toy_cnt", "other_cnt"),
             "hw/ip/prim/rtl/prim_cnt.sv": TOY.replace("toy_cnt", "prim_cnt")}
    tasks = [{"ip": "toy", "path": "hw/ip/toy/rtl/toy_cnt.sv"}]
    ips, paths = ob.eval_exclusions(tasks)
    recs = ob.edit_pairs(texts, exclude_ips=ips, exclude_paths=paths,
                         per_file=2, seed=0)
    assert {r["path"] for r in recs} == {"hw/ip/other/rtl/other_cnt.sv",
                                         "hw/ip/prim/rtl/prim_cnt.sv"}
    assert len(recs) == 4 and {r["kind"] for r in recs} == set(ob.KINDS)
    for r in recs:
        assert r["response"] == f"[BEGIN]\n{texts[r['path']]}[DONE]"
        assert "[BEGIN]" in r["prompt"] and r["system"] == ob.SYSTEM
        assert texts[r["path"]] not in r["prompt"]       # the prompt is mutated
    assert recs == ob.edit_pairs(texts, exclude_ips=ips, exclude_paths=paths,
                                 per_file=2, seed=0)
    doc = hp.pack("p", ["a/x.v", "a/y.v"], {"a/x.v": TOY, "a/y.v": "// y\n"})
    files = ob.split_packed(doc)
    assert files["a/x.v"].strip() == TOY.strip() and files["a/y.v"] == "// y\n"
    other = ob.edit_pairs(files, source="stack")
    assert other and "open-source `a` IP" in other[0]["prompt"]

    # Exercise the CLI path that renames packed files before creating pairs.
    from argparse import Namespace
    from unittest.mock import patch
    from examples import ot_bugfix_run

    with tempfile.TemporaryDirectory() as directory:
        root = Path(directory)
        packed = root / "packed.jsonl"
        output = root / "pairs.jsonl"
        packed_files = {
            "fork/copied.sv": TOY,
            "other/clean.sv": TOY.replace("toy_cnt", "clean_cnt"),
        }
        packed.write_text(json.dumps({"text": hp.pack(
            "fork", list(packed_files), packed_files)}), encoding="utf-8")
        args = Namespace(tasks=None, ot=None, docs=[packed], out=output,
                         per_file=2, seed=0, max_records=100)
        with patch.object(ot_bugfix_run, "read_tasks", return_value=tasks), \
             patch.object(ob, "checkout"), \
             patch.object(ob, "repo_texts", return_value=texts):
            ot_bugfix_run.cmd_pairs(args)
        written = [json.loads(line) for line in output.read_text().splitlines()]
        assert any(row["path"].endswith("other/clean.sv") for row in written)
        assert not any(row["path"].endswith("fork/copied.sv") for row in written)


def test_fail_type_v5265():
    """v5.265: failure stage from new ``fail`` keys and from old score files."""
    assert ob.fail_type({"pass": True, "synth": True, "error": None}) is None
    assert ob.fail_type({"pass": False, "fail": "synth"}) == "synth"
    old = [({"pass": False, "synth": False, "error": "no module in answer"}, "no_module"),
           ({"pass": False, "synth": False, "error": "sv2v: error, called at src/Convert/Scoper.hs"}, "convert"),
           ({"pass": False, "synth": False, "error": "lint: ERROR: syntax"}, "synth"),
           ({"pass": False, "synth": True, "error": "ERROR: Found 3 unproven $equiv cells"}, "not_equivalent"),
           ({"pass": False, "synth": False, "error": "gold conversion failed"}, "gold")]
    for res, want in old:
        assert ob.fail_type(res) == want, (res, want)
    assert set(w for _, w in old) == set(ob.FAIL_TYPES)


def test_report_csv_v5265():
    """v5.265 ``report``: one CSV row per task, per-model pass/fail/seconds."""
    sys.path.insert(0, str(ROOT / "examples"))
    import ot_bugfix_run as run
    with tempfile.TemporaryDirectory() as d:
        d = Path(d)
        tasks = [{"id": "a", "ip": "x", "kind": "bugfix", "split": "seen", "op": "reset"},
                 {"id": "b", "ip": "y", "kind": "complete", "split": "unseen", "op": "assign"}]
        (d / "t.jsonl").write_text("\n".join(json.dumps(t) for t in tasks))
        (d / "m.json").write_text(json.dumps({"results": {
            "a": [{"pass": True, "synth": True, "error": None, "seconds": 1.5}],
            "b": [{"pass": False, "synth": False, "error": "sv2v: error"}]}}))
        run.main(["report", "--tasks", str(d / "t.jsonl"), "--scores", str(d / "m.json"),
                  "--out", str(d / "r.csv")])
        rows = (d / "r.csv").read_text().splitlines()
        assert rows[0].startswith("id,ip,kind,split,op,m:pass,m:fail,m:seconds")
        assert rows[1].startswith("a,x,bugfix,seen,reset,1,,1.5")
        assert rows[2].startswith("b,y,complete,unseen,assign,0,convert,")
        assert run.fail_counts(tasks, {"b": [{"pass": False, "error": "sv2v: x"}]}) == \
            {"all": {"convert": 1}, "complete": {"convert": 1}}
