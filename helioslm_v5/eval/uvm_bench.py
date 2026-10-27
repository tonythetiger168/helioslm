"""UVM testbench benchmark on VerilogEval v2 designs (v5.182, HDL track).

The model gets a VerilogEval spec and the working RTL of ``TopModule``
and must write a self-checking UVM testbench for it (SystemVerilog,
UVM 1.2 / IEEE 1800.2-2017 class library). A testbench is scored on
three gates, all run with Verilator and upstream UVM 2017-1.0:

1. ``build``  : it compiles with the design (``verilator --binary
   --timing``, top module ``tb``);
2. ``golden`` : on the correct design it ends with a UVM report summary
   showing ``UVM_ERROR : 0`` and ``UVM_FATAL : 0``;
3. ``kill``   : on each mutant of the design (one operator or constant
   changed) it reports an error. A sample ``passes`` when it builds, is
   clean on the golden design and kills every mutant.

Mutants are made by ``prepare`` from the reference RTL and kept only if
VerilogEval's own testbench (Icarus) tells them apart from the
reference, so no mutant is equivalent to the design. All variants are
compiled into one binary: a wrapper ``TopModule`` with the design's
ports instantiates the golden design and every mutant and drives its
outputs from the one chosen by ``+UVMB_VARIANT=<k>``, so a testbench is
built once and run ``1 + mutants`` times.

UVM comes from Verilator's own tree at a pinned tag
(``test_regress/t/uvm/uvm_pkg_all_v2017_1_0_nodpi.svh``, Apache-2.0,
UVM without DPI); ``fetch`` stores it with a sha256 manifest. Verilator
must be at least that tag (5.052). Testbenches are untrusted code: run
them on the pod.
"""
import hashlib
import json
import re
import shutil
import subprocess
import tempfile
import urllib.request
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

from helioslm_v5.eval import hdl_bench as hb

SCHEMA_ID = "uvm-bench/1"
VERILATOR_TAG = "v5.052"
UVM_REL = "test_regress/t/uvm/uvm_pkg_all_v2017_1_0_nodpi.svh"
UVM_URL = f"https://raw.githubusercontent.com/verilator/verilator/{VERILATOR_TAG}/{UVM_REL}"
UVM_FILE = "uvm_pkg_all_v2017_1_0_nodpi.svh"
N_MUTANTS = 3
BUILD_TIMEOUT_S = 900
RUN_TIMEOUT_S = 60
VARIANT_ARG = "UVMB_VARIANT"
VERILATOR_FLAGS = ("--binary", "--timing", "-CFLAGS", "-O0", "-Wno-fatal", "-Wno-lint",
                   "-Wno-style", "--top-module", "tb")

SYSTEM_MSG = ("You are a design verification engineer. You write UVM "
              "testbenches in SystemVerilog (IEEE 1800-2017, UVM 1.2 class "
              "library).")
RULES = """Write a complete, self-checking UVM testbench for the module TopModule below.
Rules:
- The top-level module must be named `tb`. It instantiates `TopModule dut (...)`, \
generates any clock and reset, and calls `run_test("<your test class>")` from an initial block.
- `import uvm_pkg::*;` and `include "uvm_macros.svh"` are available.
- Use UVM components (sequence item, sequence, driver, monitor, scoreboard, env, test) \
and a virtual interface or equivalent to reach the DUT ports.
- The scoreboard must predict the expected outputs from the specification with its own \
reference model and report every mismatch with `uvm_error`. Do not read signals inside \
the DUT; use only its ports.
- Drive enough stimulus (directed and random) to expose any bug in TopModule, then end \
the test (raise/drop an objection) so the simulation finishes.
Enclose your code with [BEGIN] and [DONE]. Only output the code snippet and do NOT \
output anything else."""

# operator / constant swaps tried in order on the module body; a match is
# replaced one occurrence at a time (a callable gets the match)
_SWAPS = (
    (r"(?<![&])&(?![&=])", "|"),
    (r"(?<![|])\|(?![|=])", "&"),
    (r"(?<![\^~])\^(?![~=])", "~^"),
    (r"(?<![+])\+(?![+:=])", "-"),
    (r"==", "!="),
    (r"!=", "=="),
    (r"(?<![<])<(?![<=])", ">="),
    (r"(?<![>-])>(?![>=])", "<="),
    (r"1'b0", "1'b1"),
    (r"1'b1", "1'b0"),
    (r"&&", "||"),
    (r"\|\|", "&&"),
    (r"(?<![\w'])~(?=[\w(])", ""),
    (r"\?\s*(\w+)\s*:\s*(\w+)", lambda m: f"? {m.group(2)} : {m.group(1)}"),
    (r"(?<![\w'\[:$.])(\d+)(?![\w'\]:.])", lambda m: str(int(m.group(1)) + 1)),
)
_PORT_HEAD = re.compile(r"\bmodule\s+RefModule\s*(#\s*\(.*?\)\s*)?\((.*?)\)\s*;", re.S)
_SUMMARY = re.compile(r"UVM_(ERROR|FATAL)\s*:\s*(\d+)")


class UvmBenchError(ValueError):
    """Raised for a missing tool, a bad cache or a malformed design."""


# ------------------------------------------------------------------ data

def fetch(dest, get=hb._get):
    """Download the UVM package file into ``dest`` (with a manifest)."""
    dest = Path(dest)
    dest.mkdir(parents=True, exist_ok=True)
    data = get(UVM_URL)
    if b"package uvm_pkg" not in data:
        raise UvmBenchError("downloaded file is not the UVM package")
    (dest / UVM_FILE).write_bytes(data)
    (dest / "uvm_macros.svh").write_text(
        "// macros come from the concatenated UVM package file\n", encoding="utf-8")
    man = {"schema": SCHEMA_ID, "verilator_tag": VERILATOR_TAG, "uvm": UVM_REL,
           "licence": "Apache-2.0", "sha256": {UVM_FILE: hashlib.sha256(data).hexdigest()}}
    (dest / "MANIFEST.json").write_text(json.dumps(man, indent=1) + "\n", encoding="utf-8")
    return man


def uvm_dir(cache):
    """The verified UVM directory of a ``fetch`` cache."""
    cache = Path(cache)
    mpath = cache / "MANIFEST.json"
    if not mpath.is_file():
        raise UvmBenchError(f"no MANIFEST.json in {cache}; run fetch first")
    man = json.loads(mpath.read_text(encoding="utf-8"))
    for rel, digest in man["sha256"].items():
        p = cache / rel
        if not p.is_file() or hashlib.sha256(p.read_bytes()).hexdigest() != digest:
            raise UvmBenchError(f"sha256 mismatch or missing: {rel}")
    return cache


# ------------------------------------------------------------------ design

def _strip_comments(text):
    text = re.sub(r"/\*.*?\*/", lambda m: re.sub(r"[^\n]", " ", m.group(0)), text, flags=re.S)
    return re.sub(r"//[^\n]*", lambda m: " " * len(m.group(0)), text)


def parse_ports(ref):
    """``[(direction, range, name)]`` of RefModule's ANSI port list;
    ``range`` keeps ``signed`` and packed dimensions ("" for one bit)."""
    m = _PORT_HEAD.search(_strip_comments(ref))
    if not m:
        raise UvmBenchError("no ANSI port list on RefModule")
    ports, cur_dir, cur_rng = [], None, ""
    for item in m.group(2).split(","):
        toks = item.split()
        if not toks:
            continue
        if toks[0] in ("input", "output", "inout"):
            cur_dir, toks = toks[0], toks[1:]
            toks = [t for t in toks if t not in ("wire", "reg", "logic", "var")]
            cur_rng = " ".join(toks[:-1])
            toks = toks[-1:]
        if cur_dir is None or len(toks) != 1 or not re.fullmatch(r"\w+", toks[0]):
            raise UvmBenchError(f"cannot parse port {item.strip()!r}")
        ports.append((cur_dir, cur_rng, toks[0]))
    if not ports:
        raise UvmBenchError("RefModule has no ports")
    return ports


def mutants(ref):
    """Every single-occurrence swap of ``_SWAPS`` in the body after the
    port list, in a fixed order, as ``(label, text)``."""
    clean = _strip_comments(ref)
    m = _PORT_HEAD.search(clean)
    if not m:
        raise UvmBenchError("no ANSI port list on RefModule")
    start = m.end()
    out, seen = [], set()
    for pat, rep in _SWAPS:
        for hit in re.finditer(pat, clean[start:]):
            a, b = start + hit.start(), start + hit.end()
            new = rep(hit) if callable(rep) else rep
            text = ref[:a] + new + ref[b:]
            if text not in seen:
                seen.add(text)
                out.append((f"{hit.group(0)}->{new or 'del'}@{a}", text))
    return out


def rename(text, name):
    return re.sub(r"\bRefModule\b", name, text)


def _decl(*parts):
    return " ".join(x for x in parts if x)


def wrapper(ports, n_variants):
    """``TopModule`` with RefModule's ports: instantiates
    ``TopModule_v0..`` and drives every output from the variant chosen
    by ``+UVMB_VARIANT=<k>`` (0 = golden)."""
    decl = ",\n".join("  " + _decl(d, r, n) for d, r, n in ports)
    lines = [f"module TopModule (\n{decl}\n);",
             "  int sel = 0;",
             f'  initial if (!$value$plusargs("{VARIANT_ARG}=%d", sel)) sel = 0;']
    outs = [(r, n) for d, r, n in ports if d == "output"]
    for v in range(n_variants):
        for r, n in outs:
            lines.append("  " + _decl("wire", r, f"{n}_v{v};"))
        conns = ", ".join(f".{n}({n}_v{v})" if d == "output" else f".{n}({n})"
                          for d, r, n in ports)
        lines.append(f"  TopModule_v{v} u_v{v} ({conns});")
    for r, n in outs:
        chain = " : ".join([f"(sel == {v}) ? {n}_v{v}" for v in range(1, n_variants)]
                           + [f"{n}_v0"])
        lines.append(f"  assign {n} = {chain};")
    lines.append("endmodule")
    return "\n".join(lines) + "\n"


def design_sources(task):
    """Golden + mutants renamed ``TopModule_v<k>`` plus the wrapper."""
    variants = [task["ref"]] + [m["text"] for m in task["mutants"]]
    parts = [rename(t, f"TopModule_v{i}") for i, t in enumerate(variants)]
    parts.append(wrapper([tuple(p) for p in task["ports"]], len(variants)))
    return "\n".join(parts)


def prepare(probs, n_mutants=N_MUTANTS, workers=4):
    """Tasks for ``probs`` (``hdl_bench.load``): for each design, its
    ports and the first ``n_mutants`` mutants that compile and fail
    VerilogEval's own testbench. Designs with no such mutant or an
    unparsable port list are left out (listed under ``skipped``)."""
    def one(p):
        try:
            ports = parse_ports(p["ref"])
        except UvmBenchError as e:
            return p["name"], None, str(e)
        kept = []
        for label, text in mutants(p["ref"]):
            r = hb.simulate(hb.reference_sample({"ref": text}), p)
            if r["code"] in ("R", "r"):          # compiles, behaves differently
                kept.append({"label": label, "text": text})
                if len(kept) == n_mutants:
                    break
        if not kept:
            return p["name"], None, "no killable mutant"
        return p["name"], {"name": p["name"], "prompt": p["prompt"], "ref": p["ref"],
                           "ports": ports, "mutants": kept}, None
    with ThreadPoolExecutor(max_workers=workers) as ex:
        res = list(ex.map(one, probs))
    tasks = [t for _, t, _ in res if t]
    skipped = {n: why for n, t, why in res if not t}
    return {"schema": SCHEMA_ID, "source": "verilog-eval-v2/spec-to-rtl",
            "commit": hb.COMMIT, "n_mutants": n_mutants, "tasks": tasks,
            "skipped": skipped}


def load(path):
    data = json.loads(Path(path).read_text(encoding="utf-8"))
    if data.get("schema") != SCHEMA_ID:
        raise UvmBenchError(f"{path}: not a {SCHEMA_ID} task file")
    return data["tasks"]


# ------------------------------------------------------------------ prompt

def build_prompt(task):
    """``(system, user)``: spec, the design and the testbench rules."""
    dut = rename(task["ref"], "TopModule").strip()
    user = ("Specification:\n" + task["prompt"].strip() + "\n\nDesign under test:\n"
            "```systemverilog\n" + dut + "\n```\n\n" + RULES + "\n")
    return SYSTEM_MSG, user


# ------------------------------------------------------------------ scoring

def need_verilator():
    path = shutil.which("verilator")
    if not path:
        raise UvmBenchError("verilator not found on PATH (build "
                            f"{VERILATOR_TAG} or newer on the pod)")
    return path


def summary_counts(log):
    """``{"ERROR": n, "FATAL": n}`` from the last UVM report summary, or
    None when the run printed none."""
    i = log.rfind("Report counts by severity")
    if i < 0:
        return None
    found = dict((k, int(v)) for k, v in _SUMMARY.findall(log[i:]))
    return found if {"ERROR", "FATAL"} <= found.keys() else None


def _run(cmd, cwd, timeout):
    try:
        r = subprocess.run(cmd, cwd=cwd, capture_output=True, text=True, timeout=timeout)
        return r.returncode, r.stdout + r.stderr
    except subprocess.TimeoutExpired as e:
        return None, hb._text(e.stdout) + "\nTIMEOUT\n"


def run_tb(tb, task, uvm, build_timeout=BUILD_TIMEOUT_S, run_timeout=RUN_TIMEOUT_S,
           jobs=2):
    """Build ``tb`` against the task's design and run every variant.

    Returns ``{"build", "golden", "killed", "mutants", "passed", "log"}``.
    """
    verilator = need_verilator()
    uvm = Path(uvm).resolve()
    with tempfile.TemporaryDirectory(prefix="uvmb_") as td:
        td = Path(td)
        (td / "dut.sv").write_text(design_sources(task), encoding="utf-8")
        (td / "tb.sv").write_text(tb, encoding="utf-8")
        rc, log = _run([verilator, *VERILATOR_FLAGS, "-j", str(jobs),
                        f"+incdir+{uvm}", str(uvm / UVM_FILE), "dut.sv", "tb.sv"],
                       td, build_timeout)
        n_mut = len(task["mutants"])
        res = {"build": rc == 0 and (td / "obj_dir" / "Vtb").is_file(),
               "golden": False, "killed": 0, "mutants": n_mut, "passed": False}
        if not res["build"]:
            res["log"] = log[-4000:]
            return res
        logs = []
        for v in range(n_mut + 1):
            rc, out = _run([str(td / "obj_dir" / "Vtb"), "+UVM_NO_RELNOTES",
                            f"+{VARIANT_ARG}={v}"], td, run_timeout)
            counts = summary_counts(out)
            if v == 0:
                res["golden"] = (rc == 0 and counts is not None
                                 and counts["ERROR"] == 0 and counts["FATAL"] == 0)
                logs.append(out[-3000:])
                if not res["golden"]:
                    break
            elif rc is not None and (counts is None or counts["ERROR"] or counts["FATAL"]
                                     or rc != 0):
                res["killed"] += 1          # a timeout is not counted as a kill
        res["passed"] = res["golden"] and res["killed"] == n_mut
        res["log"] = "\n".join(logs)[-4000:]
    return res


def score(tasks, texts, uvm, ks=(1,), workers=4):
    """Score ``texts[name] -> [response, ...]`` (same n for every task)."""
    ns = {len(texts[t["name"]]) for t in tasks}
    if len(ns) != 1:
        raise UvmBenchError(f"uneven sample counts: {sorted(ns)}")
    n = ns.pop()
    jobs = [(t, x) for t in tasks for x in texts[t["name"]]]

    def one(job):
        t, x = job
        r = run_tb(hb.extract_code(x), t, uvm)
        return t["name"], {k: r[k] for k in ("build", "golden", "killed", "mutants", "passed")}

    with ThreadPoolExecutor(max_workers=workers) as ex:
        results = list(ex.map(one, jobs))
    per = {t["name"]: [] for t in tasks}
    for name, r in results:
        per[name].append(r)
    m = max(len(tasks), 1)
    out = {"schema": SCHEMA_ID, "bench": "uvm-tb/verilog-eval-v2",
           "verilator_tag": VERILATOR_TAG, "n_problems": len(tasks), "n": n,
           "problems": [{"name": k, "samples": v} for k, v in per.items()]}
    for k in ks:
        if k <= n:
            for key in ("build", "golden", "passed"):
                label = "pass" if key == "passed" else key
                out[f"{label}@{k}"] = sum(
                    hb.pass_at_k(n, sum(s[key] for s in v), k) for v in per.values()) / m
    out["kill_rate"] = sum(s["killed"] / s["mutants"] for v in per.values() for s in v) \
        / max(sum(len(v) for v in per.values()), 1)
    return out


def reference_tb(task):
    """A small hand-written UVM testbench used by ``self-check``: it drives
    random inputs and compares every output with the golden design
    instantiated inside the testbench, so it is clean on the golden
    variant and catches every mutant that VerilogEval's testbench does."""
    ports = [tuple(p) for p in task["ports"]]
    ins = [(r, n) for d, r, n in ports if d == "input"]
    outs = [(r, n) for d, r, n in ports if d == "output"]
    clk = any(n == "clk" for _, n in ins)
    decls = "\n".join("  " + _decl("logic", r, n + ";") for r, n in ins + outs)
    exp = "\n".join("  " + _decl("logic", r, n + "_exp;") for r, n in outs)
    conn = ", ".join(f".{n}({n})" for _, _, n in ports)
    conn_e = ", ".join(f".{n}({n}_exp)" if d == "output" else f".{n}({n})"
                       for d, _, n in ports)
    drive = "\n".join(f"        tb.{n} = $urandom;" for r, n in ins if n != "clk")
    check = "\n".join(
        f"        if (tb.{n} !== tb.{n}_exp) `uvm_error(\"MISMATCH\", "
        f"$sformatf(\"{n}=%0h exp %0h\", tb.{n}, tb.{n}_exp))" for _, n in outs)
    wait = "@(negedge tb.clk);" if clk else "#5;"
    settle = "" if clk else "#5;"
    clock = "  initial clk = 0;\n  always #5 clk = ~clk;\n" if clk else ""
    return f"""`include "uvm_macros.svh"
import uvm_pkg::*;
module tb;
{decls}
{exp}
{clock}  TopModule dut ({conn});
  TopModule_v0 gold ({conn_e});
  class t extends uvm_test;
    `uvm_component_utils(t)
    function new(string name, uvm_component parent = null);
      super.new(name, parent);
    endfunction
    task run_phase(uvm_phase phase);
      phase.raise_objection(this);
      repeat (400) begin
{drive}
        {wait}
        {settle}
{check}
      end
      phase.drop_objection(this);
    endtask
  endclass
  initial run_test("t");
endmodule
"""


def self_check(tasks, uvm, workers=4):
    """Run ``reference_tb`` on every task: must build, be clean on the
    golden design and kill every mutant. Returns ``(passed, failures)``."""
    def one(t):
        r = run_tb(reference_tb(t), t, uvm)
        return t["name"], r
    with ThreadPoolExecutor(max_workers=workers) as ex:
        res = list(ex.map(one, tasks))
    fails = [(n, {k: r[k] for k in ("build", "golden", "killed", "mutants")})
             for n, r in res if not r["passed"]]
    return len(tasks) - len(fails), fails


def main(argv=None):
    import argparse
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("cmd", choices=("fetch", "prepare", "self-check"))
    ap.add_argument("--cache", default="data/hdl_bench/uvm")
    ap.add_argument("--ve-cache", default="data/hdl_bench/verilog_eval_v2")
    ap.add_argument("--limit", type=int)
    ap.add_argument("--workers", type=int, default=4)
    a = ap.parse_args(argv)
    cache = Path(a.cache)
    if a.cmd == "fetch":
        print(fetch(cache))
        return 0
    if a.cmd == "prepare":
        probs = hb.load(a.ve_cache)[:a.limit]
        data = prepare(probs, workers=a.workers)
        cache.mkdir(parents=True, exist_ok=True)
        (cache / "tasks.json").write_text(json.dumps(data, indent=1) + "\n", encoding="utf-8")
        print(f"{len(data['tasks'])} tasks, {len(data['skipped'])} skipped -> {cache / 'tasks.json'}")
        return 0
    tasks = load(cache / "tasks.json")[:a.limit]
    ok, fails = self_check(tasks, uvm_dir(cache), a.workers)
    print(f"self-check: {ok}/{len(tasks)} reference testbenches pass")
    for name, r in fails:
        print(f"  FAIL {name}: {r}")
    return 0 if not fails else 1


if __name__ == "__main__":
    raise SystemExit(main())
