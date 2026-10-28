"""Simulation-equivalence reward for Verilog RL (v5.185, HDL track H4).

A task is a natural-language spec plus a golden design (CodeV-R1 RL
data: GitHub code whose behaviour DeepSeek-R1 reproduced). A sampled
answer is rewarded when Icarus simulation cannot tell its top module
from the golden one:

1. ``top_ports`` reads the golden top module's ports, ANSI or not, with
   ``parameter`` defaults folded into the widths;
2. ``equiv_tb`` instantiates the golden design (every module renamed
   ``gold_<name>``) and the answer's module of the same name, drives
   random inputs (a clock-named input toggles, a reset-named input is
   asserted for the first cycles and then rarely) and counts the cycles
   where an output differs; a golden output that is X is not compared,
   so an answer may pick any value for state the golden leaves
   uninitialised;
3. ``reward`` is 1 for an equivalent answer, else 0. With ``ref_cells``
   (yosys cells of the golden design) a correct answer gains
   ``gate_weight * clamp(log2(ref_cells / cells), -1, 1)``: smaller than
   the golden design is better, by at most ``gate_weight``.

Random simulation is a test, not a proof: a rare-path difference can be
missed. ``prepare`` keeps only tasks whose golden design compiles,
simulates against itself and has a port list that can be driven.
Answers are untrusted code: run this on the pod.
"""
import math
import re
import shutil
import subprocess
import tempfile
from pathlib import Path

from helioslm_v5.eval import hdl_bench as hb

CYCLES = 400
SIM_TIMEOUT_S = 20
COMPILE_TIMEOUT_S = 30
GATE_WEIGHT = 0.1

_RESET = re.compile(r"^a?(reset|rst)(_?n|_?b)?$|^(rst|reset)_?\w*$", re.I)
_CLOCK = re.compile(r"^(clk|clock)\w*$|^\w*_(clk|clock)$", re.I)
_MODULE = re.compile(r"\bmodule\s+(\w+)")


class RewardError(Exception):
    """A golden design this reward cannot drive."""


def strip_comments(text):
    text = re.sub(r"/\*.*?\*/", " ", text, flags=re.S)
    return re.sub(r"//[^\n]*", "", text)


def module_names(text):
    return _MODULE.findall(strip_comments(text))


def top_module(text):
    """The module no other module of ``text`` instantiates (the last
    such one when several are)."""
    clean = strip_comments(text)
    names = module_names(clean)
    if not names:
        raise RewardError("no module")
    used = set()
    for n in names:
        for other in names:
            if other != n and re.search(rf"^\s*{other}\s*(#\s*\(.*?\)\s*)?\w+\s*\(",
                                        _body(clean, n), re.M | re.S):
                used.add(other)
    tops = [n for n in names if n not in used]
    return (tops or names)[-1]


def _body(text, name):
    m = re.search(rf"\bmodule\s+{name}\b(.*?)\bendmodule\b", text, re.S)
    return m.group(1) if m else ""


def _params(header_and_body):
    """``{name: value}`` of ``parameter`` / ``localparam`` integer
    defaults that evaluate to an int (others are ignored)."""
    out = {}
    for m in re.finditer(r"\b(?:parameter|localparam)\b\s*(?:integer|int|\[[^\]]*\])?\s*"
                         r"((?:\w+\s*=\s*[^,;)]+\s*,?\s*)+)", header_and_body):
        for k, v in re.findall(r"(\w+)\s*=\s*([^,;)]+)", m.group(1)):
            try:
                out[k] = int(_eval(v, out))
            except (ValueError, SyntaxError, ZeroDivisionError, TypeError, NameError):
                pass
    return out


_NUM = re.compile(r"(\d+)?'[sS]?([dDhHbBoO])([0-9a-fA-F_xXzZ]+)")


def _eval(expr, params):
    """Integer value of a constant expression (Verilog literals, + - * /
    % << >> and parameters); raises on anything else."""
    def lit(m):
        base = {"d": 10, "h": 16, "b": 2, "o": 8}[m.group(2).lower()]
        return str(int(m.group(3).replace("_", ""), base))
    e = _NUM.sub(lit, expr.strip())
    e = re.sub(r"\$clog2\s*\(", "_clog2(", e)
    for k in sorted(params, key=len, reverse=True):
        e = re.sub(rf"\b{k}\b", str(params[k]), e)
    if not re.fullmatch(r"[\d\s+\-*/%()<>_clog2]*", e):
        raise ValueError(f"not constant: {expr!r}")
    e = e.replace("/", "//")
    return int(eval(e, {"__builtins__": {}},                         # noqa: S307
                    {"_clog2": lambda x: max(0, math.ceil(math.log2(x))) if x > 0 else 0}))


def _width(rng, params):
    """Bit width of a packed range text ("" is 1)."""
    rng = re.sub(r"\b(signed|unsigned|wire|reg|logic|var)\b", "", rng).strip()
    if not rng:
        return 1
    dims = re.findall(r"\[([^:\]]+):([^\]]+)\]", rng)
    if not dims or re.sub(r"\[[^\]]*\]", "", rng).strip():
        raise RewardError(f"cannot size {rng!r}")
    w = 1
    for hi, lo in dims:
        w *= abs(_eval(hi, params) - _eval(lo, params)) + 1
    return w


def top_ports(text, name=None):
    """``[(direction, width, port)]`` of module ``name`` (default: the
    top module), ANSI or non-ANSI."""
    clean = strip_comments(text)
    name = name or top_module(clean)
    m = re.search(rf"\bmodule\s+{name}\s*(#\s*\((.*?)\)\s*)?\((.*?)\)\s*;(.*?)\bendmodule\b",
                  clean, re.S)
    if not m:
        raise RewardError(f"no port list on {name}")
    params = _params((m.group(2) or "") + ";" + m.group(4))
    head, body = m.group(3), m.group(4)
    ports, cur = [], None
    if re.search(r"\b(input|output|inout)\b", head):
        for item in head.split(","):
            toks = item.strip()
            mm = re.match(r"(input|output|inout)\b(.*)", toks, re.S)
            if mm:
                cur = mm.group(1)
                rest = mm.group(2)
            else:
                rest = toks
            if cur is None:
                raise RewardError(f"cannot parse port {item.strip()!r}")
            nm = re.search(r"(\w+)\s*$", rest)
            if not nm:
                raise RewardError(f"cannot parse port {item.strip()!r}")
            rng = rest[:nm.start()]
            if mm is None and ports:            # "input [3:0] a, b": b shares a's range
                rng = ports[-1][3]
            ports.append((cur, _width(rng, params), nm.group(1), rng))
    else:
        names = [p.strip() for p in head.split(",") if p.strip()]
        decl = {}
        for dm in re.finditer(r"\b(input|output|inout)\b([^;]*);", body):
            rest = dm.group(2)
            rng = "".join(re.findall(r"\[[^\]]*\]", rest.split(",")[0]))
            for n in re.findall(r"(\w+)\s*(?:,|$)", re.sub(r"\[[^\]]*\]", " ", rest)):
                if n not in ("wire", "reg", "logic", "signed", "unsigned"):
                    decl[n] = (dm.group(1), rng)
        for n in names:
            if n not in decl:
                raise RewardError(f"no direction for port {n}")
            ports.append((decl[n][0], _width(decl[n][1], params), n, decl[n][1]))
    if any(d == "inout" for d, *_ in ports):
        raise RewardError("inout ports are not driven")
    if not any(d == "output" for d, *_ in ports):
        raise RewardError("no outputs")
    return [(d, w, n) for d, w, n, _ in ports]


def rename_all(text, prefix):
    """Every module of ``text`` (definitions and instantiations) renamed
    ``<prefix><name>``."""
    for n in sorted(set(module_names(text)), key=len, reverse=True):
        text = re.sub(rf"\b{n}\b", prefix + n, text)
    return text


def _rand(width):
    calls = (width + 31) // 32
    return "$urandom" if calls == 1 else "{" + ", ".join(["$urandom"] * calls) + "}"


def equiv_tb(name, ports, cycles=CYCLES, seed=1):
    """Testbench ``tb`` comparing ``gold_<name>`` with ``<name>``; prints
    ``MISMATCHES <n> CHECKS <m>``."""
    ins = [(w, n) for d, w, n in ports if d == "input"]
    outs = [(w, n) for d, w, n in ports if d == "output"]
    clocks = [n for w, n in ins if w == 1 and _CLOCK.match(n)]
    resets = [n for w, n in ins if w == 1 and _RESET.match(n) and n not in clocks]
    vec = lambda w: f"[{w - 1}:0] " if w > 1 else ""            # noqa: E731
    lines = ["`timescale 1ns/1ps", "module tb;"]
    lines += [f"  reg {vec(w)}{n};" for w, n in ins]
    lines += [f"  wire {vec(w)}g_{n}, s_{n};" for w, n in outs]
    lines.append("  integer mism = 0, checks = 0, i, seed;")
    conn = lambda pre: ", ".join(                               # noqa: E731
        f".{n}({n})" if d == "input" else f".{n}({pre}{n})" for d, _, n in ports)
    lines.append(f"  gold_{name} gold ({conn('g_')});")
    lines.append(f"  {name} dut ({conn('s_')});")
    check = "\n".join(
        f"      if (!(^g_{n} === 1'bx) && g_{n} !== s_{n}) mism = mism + 1;"
        for _, n in outs)
    def drive(first):
        out = []
        for w, n in ins:
            if n in clocks:
                continue
            if n in resets:
                act = "0" if re.search(r"(_?n|_?b)$", n, re.I) and not n.lower().endswith("en") \
                    else "1"
                cond = "1" if first else "($urandom % 32 == 0)"
                out.append(f"      {n} = {cond} ? 1'b{act} : 1'b{1 - int(act)};")
            else:
                out.append(f"      {n} = {_rand(w)};")
        return "\n".join(out)
    lines.append("  initial begin")
    lines.append(f"    seed = {seed}; i = $urandom(seed);")
    for c in clocks:
        lines.append(f"    {c} = 0;")
    if clocks:
        lines.append("    for (i = 0; i < %d; i = i + 1) begin" % cycles)
        lines.append("      if (i < 3) begin")
        lines.append(drive(True))
        lines.append("      end else begin")
        lines.append(drive(False))
        lines.append("      end")
        lines.append("      #5; " + " ".join(f"{c} = 1;" for c in clocks))
        lines.append("      #4;")
        lines.append("      checks = checks + 1;")
        lines.append(check)
        lines.append("      #1; " + " ".join(f"{c} = 0;" for c in clocks))
        lines.append("    end")
    else:
        lines.append("    for (i = 0; i < %d; i = i + 1) begin" % cycles)
        lines.append(drive(False))
        lines.append("      #5;")
        lines.append("      checks = checks + 1;")
        lines.append(check)
        lines.append("    end")
    lines.append('    $display("MISMATCHES %0d CHECKS %0d", mism, checks);')
    lines.append("    $finish;")
    lines.append("  end")
    lines.append("endmodule")
    return "\n".join(lines) + "\n"


_RESULT = re.compile(r"MISMATCHES (\d+) CHECKS (\d+)")


def need(tool):
    path = shutil.which(tool)
    if not path:
        raise RewardError(f"{tool} not found on PATH")
    return path


def simulate(answer, golden, name, ports, cycles=CYCLES):
    """``{"ok", "mismatches", "checks", "log"}`` for one answer."""
    iverilog, vvp = need("iverilog"), need("vvp")
    with tempfile.TemporaryDirectory(prefix="hdlr_") as td:
        td = Path(td)
        (td / "gold.v").write_text(rename_all(golden, "gold_"), encoding="utf-8")
        (td / "dut.v").write_text(answer, encoding="utf-8")
        (td / "tb.v").write_text(equiv_tb(name, ports, cycles), encoding="utf-8")
        try:
            c = subprocess.run([iverilog, "-g2012", "-s", "tb", "-o", "sim",
                                "gold.v", "dut.v", "tb.v"], cwd=td,
                               capture_output=True, text=True, timeout=COMPILE_TIMEOUT_S)
        except subprocess.TimeoutExpired:
            return {"ok": False, "mismatches": None, "checks": 0, "log": "compile timeout"}
        if c.returncode:
            return {"ok": False, "mismatches": None, "checks": 0,
                    "log": (c.stdout + c.stderr)[-2000:]}
        try:
            r = subprocess.run([vvp, "-n", "sim"], cwd=td, capture_output=True,
                               text=True, timeout=SIM_TIMEOUT_S)
            out = r.stdout + r.stderr
        except subprocess.TimeoutExpired:
            return {"ok": False, "mismatches": None, "checks": 0, "log": "sim timeout"}
    m = _RESULT.search(out)
    if not m:
        return {"ok": False, "mismatches": None, "checks": 0, "log": out[-2000:]}
    mism, checks = int(m.group(1)), int(m.group(2))
    return {"ok": mism == 0 and checks > 0, "mismatches": mism, "checks": checks,
            "log": out[-500:]}


def synth_cells(code, top, timeout=120):
    """yosys ``synth -flatten`` of module ``top``: ``{"ok", "cells"}``,
    None without yosys."""
    yosys = shutil.which("yosys")
    if not yosys:
        return None
    with tempfile.TemporaryDirectory(prefix="hdlr_") as td:
        (Path(td) / "d.v").write_text(code, encoding="utf-8")
        try:
            r = subprocess.run([yosys, "-q", "-p", f"read_verilog -sv d.v; synth -flatten -top {top}; "
                                "tee -q -o stat.txt stat"], cwd=td, capture_output=True,
                               text=True, timeout=timeout)
        except subprocess.TimeoutExpired:
            return {"ok": False, "cells": None}
        st = Path(td) / "stat.txt"
        report = st.read_text(encoding="utf-8", errors="replace") if st.is_file() else ""
    return {"ok": r.returncode == 0, "cells": hb.stat_cells(report)}


_FENCE = re.compile(r"```[ \t]*(?:verilog|systemverilog|sv|v)?[ \t]*\n(.*?)```", re.S | re.I)


def extract_answer(text):
    """The Verilog of an answer: the last fenced block that holds a
    module (fences may start mid-line, as in ``<answer> ```verilog``),
    else ``[BEGIN]``/``[DONE]``, else from the first ``module`` to the
    last ``endmodule``."""
    text = re.sub(r"<think>.*?</think>", "", text, flags=re.S)
    blocks = [b for b in _FENCE.findall(text) if re.search(r"\bmodule\b", b)]
    if blocks:
        return blocks[-1].strip() + "\n"
    code = hb.extract_code(text)
    if re.search(r"\bmodule\b", code) and "[BEGIN]" in text:
        return code
    a, b = text.find("module"), text.rfind("endmodule")
    return text[a:b + len("endmodule")] + "\n" if a >= 0 and b > a else ""


def reward(text, task, gate_weight=GATE_WEIGHT, synth=synth_cells):
    """``(reward, info)`` for one sampled answer to ``task`` (``golden``,
    ``top``, ``ports`` and optionally ``ref_cells``)."""
    code = extract_answer(text)
    if not code or not re.search(rf"\bmodule\s+{task['top']}\b", code):
        return 0.0, {"ok": False, "why": "no module " + task["top"]}
    res = simulate(code, task["golden"], task["top"],
                   [tuple(p) for p in task["ports"]])
    info = {k: res[k] for k in ("ok", "mismatches", "checks")}
    if not res["ok"]:
        return 0.0, info
    r = 1.0
    if gate_weight and task.get("ref_cells"):
        s = synth(code, task["top"])
        if s and s.get("ok") and s.get("cells"):
            info["cells"] = s["cells"]
            r += gate_weight * max(-1.0, min(1.0, math.log2(task["ref_cells"] / s["cells"])))
    return r, info


def make_task(prompt, golden, synth=synth_cells):
    """A task dict, or raises RewardError when the golden design cannot
    be used (no drivable ports, or it is not equivalent to itself)."""
    top = top_module(golden)
    ports = top_ports(golden, top)
    self_check = simulate(golden, golden, top, ports)
    if not self_check["ok"]:
        kind = ("timeout" if "timeout" in self_check["log"] else
                "compile" if self_check["mismatches"] is None else "mismatch")
        raise RewardError(f"golden self-check {kind}: " + self_check["log"][-300:])
    task = {"prompt": prompt, "golden": golden, "top": top,
            "ports": [list(p) for p in ports]}
    if synth is not None:
        s = synth(golden, top)
        if s and s.get("ok") and s.get("cells"):
            task["ref_cells"] = s["cells"]
    return task
