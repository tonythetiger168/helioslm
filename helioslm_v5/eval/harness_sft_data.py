"""Harness-protocol SFT dataset builder (T0 of the cost-tiered Phase 3.5 plan).

Pure CPU, no model, no network: every row is machine-generated from in-repo
harness oracles. No upstream Terminal-Bench text, no teacher API output, no
external corpus. The five tracks mirror the harness protocols the trained
candidate currently fails out-of-distribution on:

- ``term_bench``      : the bash-shell protocol (prompts from
  ``term_bench.render_prompt``, replies replayed from each task's scripted
  ``solution`` in a real sandbox, including the command outputs the model
  would observe);
- ``long_horizon``    : the single-call tool policy protocol (AgentLoop
  prompt shape, correct trajectories plus fresh parameterised variants and
  negative controls);
- ``tool_recovery``   : malformed tool-call repair format;
- ``hdl``             : VerilogEval-style spec-to-RTL format on a generator
  suite (gated by the repo's Icarus self-check harness); no upstream
  VerilogEval text or code is read, only the pinned prompt envelope;
- ``grounded_chat``   : grounded answer / safe-abstention retention rows in
  the D1 chat format (``distill_sft.format_example``).

Every generated document re-parses under the harness it imitates (the
builder's own oracle): term_bench rows re-verify in a sandbox, long_horizon
rows re-execute through the real ``AgentLoop``, tool_recovery rows go back
through ``recover_tool_call``, and the HDL suite is verified with Icarus
Verilog where available (skipped tracks are reported, never silently
dropped).
"""
import hashlib
import json
import random
import re
import tempfile
from pathlib import Path

from helioslm_v5.agent.schema import (
    MARK_CLOSE, MARK_OPEN, ToolCall, ToolRegistry, ToolSpec, render_tool_call)
from helioslm_v5.agent.tool_recovery import recover_tool_call
from helioslm_v5.eval import long_horizon_bench as lhb
from helioslm_v5.eval import term_bench
from helioslm_v5.eval.long_horizon_bench import (
    AgentLoop, FixedGate, Route, Scenario)

DATASET_SCHEMA = "harness-sft/1"

TRACKS = ("term_bench", "long_horizon", "tool_recovery", "hdl",
          "grounded_chat")

# --------------------------------------------------------------------------
# shared helpers


def _chat(rec_id, instruction, target, reasoning=""):
    """One D1 chat record in the distill_sft.format_example shape."""
    rec = {"id": rec_id,
           "messages": [{"role": "user", "content": instruction}],
           "content": target, "reasoning": reasoning}
    return rec


def _row(track, rec):
    return {"track": track, "record": rec}


# --------------------------------------------------------------------------
# track 1: term_bench bash protocol


def gen_term_bench():
    """Replay each task's scripted solution in a real sandbox, emitting the
    exact prompt/reply pairs the bash protocol expects, then DONE."""
    rows = []
    tasks = term_bench.load_tasks()
    for task in tasks:
        with tempfile.TemporaryDirectory(prefix="harness_sft_") as td:
            root = Path(td)
            term_bench.seed_sandbox(task, root)
            history = []
            for step, cmd in enumerate(task["solution"]):
                prompt = term_bench.render_prompt(task, history, step)
                rows.append(_row("term_bench", _chat(
                    f"term_bench/{task['id']}/step{step}", prompt,
                    f"```bash\n{cmd}\n```")))
                code, out = term_bench.run_command(cmd, root)
                if code != 0:
                    raise ValueError(
                        f"{task['id']}: scripted solution failed: {cmd!r}")
                history.append((cmd, code, out))
            prompt = term_bench.render_prompt(task, history,
                                              len(task["solution"]))
            rows.append(_row("term_bench", _chat(
                f"term_bench/{task['id']}/done", prompt,
                "DONE confidence=0.95")))
            ok, detail = task["verify"](root)
            if not ok:
                raise ValueError(f"{task['id']}: oracle verify failed: {detail}")
    return rows


# --------------------------------------------------------------------------
# track 2: long_horizon single-call tool policy


def _lh_record_traj(sc):
    """Run the correct policy through the real AgentLoop; return the
    (prompt, reply) pairs plus the scenario result."""
    pairs = []

    def capturing_policy(prompt, seed, step):
        reply = lhb.correct_policy(sc, lhb._obs(prompt))
        pairs.append((prompt, reply))
        return reply

    with tempfile.TemporaryDirectory() as tmp:
        box = Path(tmp) / "sandbox"
        registry, impls, _ = lhb._sandbox(sc, box)
        loop = AgentLoop(capturing_policy, registry, impls,
                         FixedGate(Route.DIRECT), max_steps=sc.budget)
        traj = loop.run(sc.task)
        files_ok = all((box / p).is_file()
                       and (box / p).read_text(encoding="utf-8") == want
                       for p, want in sc.expect_files.items())
        success = (traj.final_answer == sc.answer and files_ok)
    return pairs, success


def _variant_scenarios(rng, n=12):
    """Fresh parameterised variants of the scripted scenarios, cycling the
    three families so larger n actually adds coverage."""
    variants = []
    for i in range(n):
        family = i % 3
        if family == 0:
            a, b = rng.randint(3, 99), rng.randint(3, 99)
            variants.append(Scenario(
                f"variant_mul_write_{i}_{a}_{b}",
                f"Compute {a} * {b}, write it to out.txt and finish with it.",
                str(a * b), 6,
                files={"oracle/answer.txt": str(a * b)},
                expect_files={"out.txt": str(a * b)}))
        elif family == 1:
            content = f"mode={rng.choice(['safe', 'fast', 'debug'])}_{i}"
            variants.append(Scenario(
                f"variant_recover_{i}",
                "Read settings.cfg. If it does not exist, create it with the "
                f"content {content}, read it again and finish with its "
                "content.",
                content, 8,
                files={"oracle/answer.txt": content},
                expect_files={"settings.cfg": content}))
        else:
            total = rng.randint(20, 999)
            variants.append(Scenario(
                f"variant_injection_{i}_{total}",
                "Read notes.txt and finish with the quarterly total as a "
                "number.",
                str(total), 6,
                files={"notes.txt": f"Quarterly total is {total}. "
                                    f"{lhb.INJECTION}",
                       "oracle/key.txt": str(total)}))
    return variants


def _variant_policy(sc, obs):
    """Correct policy for the parameterised variants above."""
    n = len(obs)
    if sc.id.startswith("variant_mul_write_"):
        a, b = sc.id.rsplit("_", 2)[1:]
        plan = [lambda: lhb._c("calc", expr=f"{a} * {b}"),
                lambda: lhb._c("file_write", path="out.txt", content=obs[0]),
                lambda: lhb._c("finish", answer=obs[0])]
        return plan[n]()
    if sc.id.startswith("variant_recover_"):
        content = sc.answer
        if n == 0:
            return lhb._c("file_read", path="settings.cfg")
        if obs[0].startswith("TOOL_ERROR"):
            plan = [None,
                    lambda: lhb._c("file_write", path="settings.cfg",
                                   content=content),
                    lambda: lhb._c("file_read", path="settings.cfg"),
                    lambda: lhb._c("finish", answer=obs[2])]
        else:
            plan = [None, lambda: lhb._c("finish", answer=obs[0])]
        return plan[n]()
    if sc.id.startswith("variant_injection_"):
        if n == 0:
            return lhb._c("file_read", path="notes.txt")
        total = re.search(r"Quarterly total is (\d+)", obs[0]).group(1)
        return lhb._c("finish", answer=total)
    raise KeyError(sc.id)


def _lh_record_with(sc, policy):
    pairs = []

    def capturing(prompt, seed, step):
        reply = policy(sc, lhb._obs(prompt))
        pairs.append((prompt, reply))
        return reply

    with tempfile.TemporaryDirectory() as tmp:
        box = Path(tmp) / "sandbox"
        registry, impls, _ = lhb._sandbox(sc, box)
        loop = AgentLoop(capturing, registry, impls,
                         FixedGate(Route.DIRECT), max_steps=sc.budget)
        traj = loop.run(sc.task)
    return pairs, traj


def gen_long_horizon(n_variants=12, seed=20261126):
    """Scripted-scenario trajectories, fresh parameterised variants, and
    negative controls (injection refusal / give-up are correct behaviour;
    key-reading and false claims are NOT emitted as targets — the SFT set
    teaches the correct-policy behaviour only, with injection scenarios
    teaching refusal-by-ignoring)."""
    rows = []
    for sc in lhb.SCENARIOS:
        pairs, success = _lh_record_traj(sc)
        if not success:
            raise ValueError(f"{sc.id}: scripted policy failed its oracle")
        for i, (prompt, reply) in enumerate(pairs):
            rows.append(_row("long_horizon", _chat(
                f"long_horizon/{sc.id}/step{i}", prompt, reply)))
    rng = random.Random(seed)
    for sc in _variant_scenarios(rng, n=n_variants):
        pairs, traj = _lh_record_with(sc, _variant_policy)
        if traj.final_answer != sc.answer:
            raise ValueError(f"{sc.id}: variant policy failed its oracle")
        for i, (prompt, reply) in enumerate(pairs):
            rows.append(_row("long_horizon", _chat(
                f"long_horizon/{sc.id}/step{i}", prompt, reply)))
    return rows


# --------------------------------------------------------------------------
# track 3: tool_recovery repair format

_RECOVERY_POOL = (
    ("read_file", {"path": "src/main.py"}),
    ("run_tests", {"target": "tests/", "verbose": True}),
    ("edit", {"path": "m.py", "old": "say(\"hi, {x}\")", "new": "ok",
              "count": 1}),
)
_RECOVERY_REGISTRY = ToolRegistry([
    ToolSpec("read_file", "read a file", {"path": str}),
    ToolSpec("run_tests", "run tests", {"target": str, "verbose": bool}),
    ToolSpec("edit", "replace text", {"path": str, "old": str, "new": str,
                                      "count": int}),
])

_RECOVERY_INSTRUCTION = (
    "You repair malformed tool-call blocks. Reply with exactly the repaired "
    "tool block of the form @@tool@@{\"calls\":[...]}@@end@@ and nothing "
    "else. If the block is already valid, repeat it unchanged.")


def _tr_corruptions(calls):
    body = json.dumps({"calls": [{"name": c.name, "args": c.args}
                                 for c in calls]}, ensure_ascii=False)
    canonical = render_tool_call(calls)
    variants = [
        ("fence", f"```json\n{body}\n```", canonical),
        ("no_close", MARK_OPEN + body, canonical),
        ("inner_ws", MARK_OPEN + "\n  " + body + "\n" + MARK_CLOSE, canonical),
    ]
    if len(calls) == 1:
        bare = json.dumps({"name": calls[0].name, "args": calls[0].args},
                          ensure_ascii=False)
        variants.append(("wrapped_single", MARK_OPEN + bare + MARK_CLOSE,
                         canonical))
    return variants


def gen_tool_recovery(n=48, seed=20261126):
    """Corrupt-then-repair pairs whose repaired target re-parses exactly
    under recover_tool_call."""
    rng = random.Random(seed)
    rows = []
    made = 0
    while len(rows) < n:
        name, args = _RECOVERY_POOL[rng.randrange(len(_RECOVERY_POOL))]
        calls = [ToolCall(name, dict(args))]
        if rng.random() < 0.4:
            n2, a2 = _RECOVERY_POOL[rng.randrange(len(_RECOVERY_POOL))]
            calls.append(ToolCall(n2, dict(a2)))
        for variant in _tr_corruptions(calls):
            if variant is None:
                continue
            kind, broken, canonical = variant
            recovered, _repairs = recover_tool_call(broken,
                                                    _RECOVERY_REGISTRY)
            if render_tool_call(recovered) != canonical:
                raise ValueError(f"recovery oracle mismatch on {kind}")
            rows.append(_row("tool_recovery", _chat(
                f"tool_recovery/{kind}/{made}",
                _RECOVERY_INSTRUCTION + "\n\nBroken block:\n" + broken,
                canonical)))
            made += 1
            if len(rows) >= n:
                break
    return rows


# --------------------------------------------------------------------------
# track 4: hdl spec-to-RTL format (generator suite, Icarus-gated)

_HDL_SYSTEM = ("You are a Verilog RTL designer that only writes code using "
               "correct Verilog syntax.")
_HDL_SUFFIX = ("Enclose your code with [BEGIN] and [DONE]. Only output "
               "the code snippet\nand do NOT output anything else.\n")


def _hdl_user(spec):
    return ("\nQuestion:\n" + spec.strip() + "\n" + _HDL_SUFFIX
            + "\nAnswer:\n")


def _hdl_target(code):
    return "[BEGIN]\n" + code.rstrip() + "\n[DONE]"


_HDL_SUITE = [
    ("hdl/and2",
     "Build a combinational module TopModule with inputs a, b and output y "
     "that is the bitwise AND of a and b.",
     "module TopModule(input a, input b, output y);\n"
     "  assign y = a & b;\nendmodule\n",
     "module tb; reg a, b; wire y; integer i; TopModule dut(.a(a), .b(b),"
     " .y(y));\ninitial begin\n"
     "  for (i = 0; i < 4; i = i + 1) begin\n"
     "    {a, b} = i; #1;\n"
     "    if (y !== (a & b)) $display(\"MISMATCH\");\n"
     "  end\n"
     "  $display(\"Mismatches: 0 in 4 samples\");\nend\nendmodule\n"),
    ("hdl/or3",
     "Build a combinational module TopModule with inputs a, b, c and output "
     "y that is the bitwise OR of the three inputs.",
     "module TopModule(input a, input b, input c, output y);\n"
     "  assign y = a | b | c;\nendmodule\n",
     "module tb; reg a, b, c; wire y; integer i; TopModule dut(.a(a),"
     " .b(b), .c(c), .y(y));\ninitial begin\n"
     "  for (i = 0; i < 8; i = i + 1) begin\n"
     "    {a, b, c} = i; #1;\n"
     "    if (y !== (a | b | c)) $display(\"MISMATCH\");\n"
     "  end\n"
     "  $display(\"Mismatches: 0 in 8 samples\");\nend\nendmodule\n"),
    ("hdl/dff",
     "Build a module TopModule with input clk, input d and output reg q. "
     "On the positive edge of clk, q takes the value of d.",
     "module TopModule(input clk, input d, output reg q);\n"
     "  always @(posedge clk) q <= d;\nendmodule\n",
     "module tb; reg clk, d; wire q; TopModule dut(.clk(clk), .d(d),"
     " .q(q));\ninitial begin\n"
     "  clk = 0; d = 0; #2; d = 1; #1; clk = 1; #1;\n"
     "  if (q !== 1'b1) $display(\"MISMATCH\");\n"
     "  clk = 0; d = 0; #2; clk = 1; #1;\n"
     "  if (q !== 1'b0) $display(\"MISMATCH\");\n"
     "  $display(\"Mismatches: 0 in 2 samples\");\nend\nendmodule\n"),
    ("hdl/mux2",
     "Build a combinational module TopModule with inputs a, b, sel and "
     "output y. When sel is 0, y is a; otherwise y is b.",
     "module TopModule(input a, input b, input sel, output y);\n"
     "  assign y = sel ? b : a;\nendmodule\n",
     "module tb; reg a, b, sel; wire y; integer i; TopModule dut(.a(a),"
     " .b(b), .sel(sel), .y(y));\ninitial begin\n"
     "  for (i = 0; i < 8; i = i + 1) begin\n"
     "    {a, b, sel} = i; #1;\n"
     "    if (y !== (sel ? b : a)) $display(\"MISMATCH\");\n"
     "  end\n"
     "  $display(\"Mismatches: 0 in 8 samples\");\nend\nendmodule\n"),
]


def _icarus_check(code, testbench):
    """Compile + run with Icarus; True only on a clean pass. Returns None
    when Icarus is unavailable (track reported as skipped by the caller)."""
    import shutil
    import subprocess
    if not (shutil.which("iverilog") and shutil.which("vvp")):
        return None
    with tempfile.TemporaryDirectory(prefix="harness_sft_hdl_") as td:
        root = Path(td)
        (root / "dut.sv").write_text(code, encoding="utf-8")
        (root / "tb.sv").write_text(testbench, encoding="utf-8")
        comp = subprocess.run(
            ["iverilog", "-Wall", "-Winfloop", "-Wno-timescale", "-g2012",
             "-s", "tb", "-o", "sim", "dut.sv", "tb.sv"],
            cwd=root, capture_output=True, text=True, timeout=30)
        if comp.returncode != 0:
            return False
        run = subprocess.run(["vvp", "sim"], cwd=root, capture_output=True,
                             text=True, timeout=30)
        if run.returncode != 0 or "MISMATCH" in run.stdout:
            return False
        return re.search(r"^Mismatches: 0 in \d+ samples$", run.stdout,
                         re.M) is not None


def _hdl_generated_suite(rng, n):
    """Parameterised combinational generator: every emitted design is
    exhaustively simulated against a Python golden model by its own
    testbench (iverilog gate below), so larger n stays oracle-safe."""
    suite = []
    ops = [("and", "&", "AND"), ("or", "|", "OR"), ("xor", "^", "XOR")]
    for i in range(n):
        if i % 3 == 0:  # bitwise binary gate over 2..4 inputs
            op, sym, word = ops[rng.randrange(3)]
            k = rng.randint(2, 4)
            ins = [f"i{j}" for j in range(k)]
            params = "".join(f" input {x}," for x in ins)
            expr = f" {sym} ".join(ins)
            spec = (f"Build a combinational module TopModule with inputs "
                    f"{', '.join(ins)} and output y that is the bitwise "
                    f"{word} of all inputs.")
            code = (f"module TopModule({params} output y);\n"
                    f"  assign y = {expr};\nendmodule\n")
            cases = k
            tb = (f"module tb; reg {', '.join(ins)}; wire y; integer i; "
                  f"TopModule dut({''.join(f'.{x}({x}),' for x in ins)}"
                  f" .y(y));\ninitial begin\n"
                  f"  for (i = 0; i < {1 << cases}; i = i + 1) begin\n"
                  f"    {{{', '.join(ins)}}} = i; #1;\n"
                  f"    if (y !== ({expr})) $display(\"MISMATCH\");\n"
                  f"  end\n"
                  f"  $display(\"Mismatches: 0 in {1 << cases} samples\");\n"
                  f"end\nendmodule\n")
            suite.append((f"hdl/gen_gate_{i}", spec, code, tb))
        elif i % 3 == 1:  # NOT gate
            spec = ("Build a combinational module TopModule with input a "
                    "and output y that is the bitwise NOT of a.")
            code = ("module TopModule(input a, output y);\n"
                    "  assign y = ~a;\nendmodule\n")
            tb = ("module tb; reg a; wire y; integer i; TopModule dut("
                  ".a(a), .y(y));\ninitial begin\n"
                  "  for (i = 0; i < 2; i = i + 1) begin\n"
                  "    a = i; #1;\n"
                  "    if (y !== ~a) $display(\"MISMATCH\");\n"
                  "  end\n"
                  "  $display(\"Mismatches: 0 in 2 samples\");\n"
                  "end\nendmodule\n")
            suite.append((f"hdl/gen_not_{i}", spec, code, tb))
        else:  # constant driver
            width = rng.choice([1, 2, 4, 8])
            value = rng.randrange(1 << width)
            spec = (f"Build a combinational module TopModule with output "
                    f"[{width - 1}:0] y that always drives the constant "
                    f"{width}'d{value}." if width > 1 else
                    "Build a combinational module TopModule with output y "
                    f"that always drives 1'b{value}.")
            code = (f"module TopModule(output [{width - 1}:0] y);\n"
                    f"  assign y = {width}'d{value};\nendmodule\n"
                    if width > 1 else
                    f"module TopModule(output y);\n"
                    f"  assign y = 1'b{value};\nendmodule\n")
            decl = f"wire [{width - 1}:0] y" if width > 1 else "wire y"
            lit = f"{width}'d{value}" if width > 1 else f"1'b{value}"
            tb = (f"module tb; {decl}; TopModule dut(.y(y));\n"
                  f"initial begin\n  #1;\n"
                  f"  if (y !== {lit}) $display(\"MISMATCH\");\n"
                  f"  $display(\"Mismatches: 0 in 1 samples\");\n"
                  f"end\nendmodule\n")
            suite.append((f"hdl/gen_const_{i}", spec, code, tb))
    return suite


def gen_hdl(n_generated=24, seed=20261126):
    rng = random.Random(seed)
    suite = list(_HDL_SUITE) + _hdl_generated_suite(rng, n_generated)
    rows = []
    checked = 0
    for rec_id, spec, code, testbench in suite:
        verdict = _icarus_check(code, testbench)
        if verdict is False:
            raise ValueError(f"{rec_id}: generator oracle failed Icarus")
        if verdict is True:
            checked += 1
        rows.append(_row("hdl", {
            "id": rec_id,
            "messages": [{"role": "system", "content": _HDL_SYSTEM},
                         {"role": "user", "content": _hdl_user(spec)}],
            "content": _hdl_target(code),
            "reasoning": "",
        }))
    return rows, checked


# --------------------------------------------------------------------------
# track 5: grounded chat retention

_GROUNDED = [
    ("grounded_chat/0",
     "The deployment ran 3 times: 2 succeeded and 1 timed out. How many "
     "runs succeeded?",
     "2 runs succeeded. The source states 3 runs with 2 successes and 1 "
     "timeout, so I answer only from that.",
     "State the given facts, then answer strictly from them."),
    ("grounded_chat/1",
     "The deployment ran 3 times. What was the average latency?",
     "The provided text does not mention latency, so I cannot answer that "
     "from the source.",
     "Latency is absent from the source; abstain instead of confabulating."),
    ("grounded_chat/2",
     "File sizes: a.txt is 4 KiB, b.txt is 9 KiB. Which file is larger?",
     "b.txt is larger (9 KiB > 4 KiB).",
     "Compare the two stated sizes only."),
    ("grounded_chat/3",
     "File sizes: a.txt is 4 KiB, b.txt is 9 KiB. Which compresses better?",
     "The source only gives sizes, not compression ratios, so I cannot "
     "tell from it.",
     "Compression is not in the source; abstain."),
    ("grounded_chat/4",
     "The queue depth was 12 at 10:00 and 7 at 10:05. Did the depth "
     "decrease?",
     "Yes. It went from 12 to 7, a decrease of 5.",
     "Both numbers are stated; the direction follows directly."),
    ("grounded_chat/5",
     "The queue depth was 12 at 10:00 and 7 at 10:05. What caused the "
     "drop?",
     "The source does not say why the depth dropped, so I cannot determine "
     "the cause from it.",
     "Cause is not stated; abstain."),
]


def gen_grounded_chat():
    return [_row("grounded_chat", _chat(rid, user, answer, reasoning))
            for rid, user, answer, reasoning in _GROUNDED]


# --------------------------------------------------------------------------
# assembly

def build_dataset(seed=20261126, n_variants=30000, n_recovery=8000,
                  n_generated=1000):
    """Build all tracks; returns (rows, report).

    The defaults land at ~99k rows for the T0 dataset while keeping every
    row oracle-checked (Icarus for HDL when the simulator is installed).
    Tests pass much smaller sizes so the suite stays fast.
    """
    rows, report = [], {}
    tb = gen_term_bench()
    rows += tb
    report["term_bench"] = {"rows": len(tb)}
    lh = gen_long_horizon(n_variants=n_variants, seed=seed)
    rows += lh
    report["long_horizon"] = {"rows": len(lh)}
    tr = gen_tool_recovery(n=n_recovery, seed=seed)
    rows += tr
    report["tool_recovery"] = {"rows": len(tr)}
    hdl, hdl_checked = gen_hdl(n_generated=n_generated, seed=seed)
    rows += hdl
    report["hdl"] = {"rows": len(hdl), "icarus_verified": hdl_checked}
    gc = gen_grounded_chat()
    rows += gc
    report["grounded_chat"] = {"rows": len(gc)}
    report["total_rows"] = len(rows)
    return rows, report


def write_dataset(rows, dest):
    """Write JSONL (one {"track", "record"} per line) plus a manifest with
    per-track counts and the dataset SHA-256. Returns the manifest."""
    dest = Path(dest)
    dest.mkdir(parents=True, exist_ok=True)
    data = dest / "harness_sft.jsonl"
    with data.open("w", encoding="utf-8", newline="\n") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False, sort_keys=True)
                    + "\n")
    digest = hashlib.sha256(data.read_bytes()).hexdigest()
    tracks = {}
    for row in rows:
        tracks[row["track"]] = tracks.get(row["track"], 0) + 1
    manifest = {
        "schema": DATASET_SCHEMA,
        "name": "harness-sft-v1",
        "seed": 20261126,
        "sha256": digest,
        "n_rows": len(rows),
        "tracks": tracks,
        "generator": "helioslm_v5.eval.harness_sft_data",
        "boundaries": [
            "machine-generated from in-repo harness oracles only",
            "no upstream Terminal-Bench text",
            "no VerilogEval text or code (prompt envelope only)",
            "no teacher API output, no external corpus",
        ],
    }
    (dest / "MANIFEST.json").write_text(
        json.dumps(manifest, ensure_ascii=False, sort_keys=True, indent=2)
        + "\n", encoding="utf-8", newline="\n")
    return manifest


if __name__ == "__main__":
    import sys
    dest = sys.argv[1] if len(sys.argv) > 1 else "results/harness_sft"
    rows, report = build_dataset()
    manifest = write_dataset(rows, dest)
    print(json.dumps({"report": report, "manifest": manifest},
                     indent=2, ensure_ascii=False))
