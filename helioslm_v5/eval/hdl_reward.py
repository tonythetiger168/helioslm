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
``audit_task`` measures that risk before RL: it replays the golden design
on several seeds, then tries deterministic one-token mutants and calls an
optional formal-equivalence callback for every mutant the simulator fails
to kill. Answers are untrusted code: run this on the pod.
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
FORMAL_TIMEOUT_S = 120
FORMAL_SEQ = 8

_RESET = re.compile(r"^a?(reset|rst)(_?n|_?b)?$|^(rst|reset)_?\w*$", re.I)
_CLOCK = re.compile(r"^(clk|clock)\w*$|^\w*_(clk|clock)$", re.I)
_MODULE = re.compile(r"\bmodule\s+(\w+)")
# One-token mutations tried in source order. Comments are blanked with
# spaces first, so an offset always addresses the original text.
_MUTANT_SWAPS = (
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
AUDIT_STATES = ("killed", "equivalent", "false_accept", "unknown", "invalid")


class RewardError(Exception):
    """A golden design this reward cannot drive."""


def strip_comments(text):
    text = re.sub(r"/\*.*?\*/", " ", text, flags=re.S)
    return re.sub(r"//[^\n]*", "", text)


def _mask_comments(text):
    """Comments replaced by spaces, preserving every offset and newline."""
    text = re.sub(r"/\*.*?\*/", lambda m: re.sub(r"[^\n]", " ", m.group(0)),
                  text, flags=re.S)
    return re.sub(r"//[^\n]*", lambda m: " " * len(m.group(0)), text)


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


def _module_body_span(clean, name):
    """``(start, end)`` of module ``name``'s body in comment-masked text."""
    m = re.search(rf"\bmodule\s+{re.escape(name)}\b", clean)
    if not m:
        raise RewardError(f"no module {name}")
    endm = re.search(r"\bendmodule\b", clean[m.end():])
    if not endm:
        raise RewardError(f"module {name} has no endmodule")
    body_end = m.end() + endm.start()
    depth = 0
    for pos in range(m.end(), body_end):
        ch = clean[pos]
        if ch in "([{":
            depth += 1
        elif ch in ")]}":
            depth = max(0, depth - 1)
        elif ch == ";" and depth == 0:
            return pos + 1, body_end
    raise RewardError(f"module {name} has no header terminator")


def mutants(golden, top=None, max_mutants=None):
    """Deterministic one-token mutants of ``golden``'s top module body.

    Returns ``(label, text)`` pairs in source order. Comments are masked
    without changing offsets, so mutation never edits a comment. Interface
    drift is rejected by ``audit_task`` before a mutant is simulated.
    """
    if max_mutants is not None and max_mutants < 0:
        raise ValueError("max_mutants must be >= 0")
    if max_mutants == 0:
        return []
    name = top or top_module(golden)
    clean = _mask_comments(golden)
    start, end = _module_body_span(clean, name)
    out, seen = [], {golden}
    for pat, rep in _MUTANT_SWAPS:
        for hit in re.finditer(pat, clean[start:end]):
            a, b = start + hit.start(), start + hit.end()
            new = rep(hit) if callable(rep) else rep
            text = golden[:a] + new + golden[b:]
            if text not in seen:
                seen.add(text)
                out.append((f"{hit.group(0)}->{new or 'del'}@{a}", text))
                if max_mutants is not None and len(out) >= max_mutants:
                    return out
    return out


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


def simulate(answer, golden, name, ports, cycles=CYCLES, seed=1):
    """``{"ok", "mismatches", "checks", "log"}`` for one answer."""
    iverilog, vvp = need("iverilog"), need("vvp")
    with tempfile.TemporaryDirectory(prefix="hdlr_") as td:
        td = Path(td)
        (td / "gold.v").write_text(rename_all(golden, "gold_"), encoding="utf-8")
        (td / "dut.v").write_text(answer, encoding="utf-8")
        (td / "tb.v").write_text(equiv_tb(name, ports, cycles, seed=seed),
                                  encoding="utf-8")
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


def formal_equiv(answer, golden, top, timeout=FORMAL_TIMEOUT_S):
    """Yosys miter/SAT verdict for ``answer`` versus ``golden``.

    Returns ``True`` when the bounded temporal-induction proof succeeds,
    ``False`` when SAT finds a difference, and ``None`` for no tool,
    timeout, parse failure or an unproven result. The callback contract of
    ``audit_task`` also accepts this value as ``(verdict, log)``.
    """
    yosys = shutil.which("yosys")
    if not yosys:
        return None, "yosys not found on PATH"
    if not re.fullmatch(r"[A-Za-z_]\w*", top):
        raise RewardError(f"unsafe module name {top!r}")
    script = "; ".join((
        "read_verilog -sv gold.v",
        f"prep -top gold_{top} -flatten",
        "design -stash gold",
        "read_verilog -sv dut.v",
        f"prep -top {top} -flatten",
        "design -stash gate",
        f"design -copy-from gold -as gold gold_{top}",
        f"design -copy-from gate -as gate {top}",
        "miter -equiv -flatten -ignore_gold_x gold gate miter",
        "hierarchy -top miter",
        f"sat -verify -tempinduct -prove trigger 0 -set-init-undef "
        f"-set-def-inputs -seq {FORMAL_SEQ} miter",
    ))
    with tempfile.TemporaryDirectory(prefix="hdlr_eq_") as td:
        td = Path(td)
        (td / "gold.v").write_text(rename_all(golden, "gold_"), encoding="utf-8")
        (td / "dut.v").write_text(answer, encoding="utf-8")
        try:
            r = subprocess.run([yosys, "-q", "-p", script], cwd=td,
                               capture_output=True, text=True, timeout=timeout)
        except subprocess.TimeoutExpired:
            return None, f"formal timeout after {timeout}s"
    log = (r.stdout + r.stderr)[-2000:]
    if r.returncode == 0:
        return True, log
    if re.search(r"model found|counterexample|FAIL", log, re.I):
        return False, log
    return None, log


def _equiv_verdict(result):
    """Normalize ``equiv_fn`` output: bool, None, or ``(bool|None, log)``."""
    detail = None
    if isinstance(result, tuple):
        verdict = result[0] if result else None
        detail = result[1] if len(result) > 1 else None
    else:
        verdict = result
    return (None if verdict is None else bool(verdict)), detail


def _invalid_result(label, why):
    return {"label": label, "status": "invalid", "why": str(why)[-500:]}


def audit_task(task, seeds=(1, 2, 3), max_mutants=8, simulate_fn=None,
               equiv_fn=None):
    """Audit how often this task's random-simulation reward can false-accept.

    The golden design must be self-equivalent on every seed. Each mutant is
    first rejected when it changes the module interface, then simulated on
    every seed. A surviving mutant is classified by ``equiv_fn(mutant,
    golden, top)`` (bool, None for no verdict, or ``(verdict, log)``):

    - ``killed``: simulation found a mismatch;
    - ``equivalent``: formal proof says the mutant is genuinely equivalent;
    - ``false_accept``: formal proof says it differs but simulation missed it;
    - ``unknown``: simulation missed it and no formal verdict is available;
    - ``invalid``: the mutant does not compile, times out or changes ports.

    ``ok`` means the golden self-check passed and no proven false accept was
    found. ``unknown`` mutants are reported separately and are not a proof.
    """
    sim = simulate_fn or simulate
    seeds = tuple(seeds)
    if not seeds:
        raise ValueError("seeds must not be empty")
    if max_mutants < 1:
        raise ValueError("max_mutants must be >= 1")
    top, golden = task["top"], task["golden"]
    ports = [tuple(p) for p in task["ports"]]
    counts = {state: 0 for state in AUDIT_STATES}
    base = {"ok": False, "golden_ok": False, "seeds": list(seeds),
            "n_mutants": 0, "counts": counts, "false_accept": [],
            "unknown": [], "invalid": [], "results": []}
    try:
        candidates = mutants(golden, top=top, max_mutants=max_mutants)
    except RewardError as e:
        return {**base, "why": str(e)}

    for seed in seeds:
        try:
            res = sim(golden, golden, top, ports, seed=seed)
        except Exception as e:                         # noqa: BLE001
            return {**base, "why": f"golden seed {seed} failed: {e}"}
        if not res.get("ok"):
            why = res.get("log", "")[-300:]
            return {**base, "why": f"golden seed {seed} failed: {why}"}
    base["golden_ok"] = True
    base["n_mutants"] = len(candidates)

    for label, mutant in candidates:
        row = {"label": label}
        try:
            if top_module(mutant) != top or top_ports(mutant, top) != ports:
                row = _invalid_result(label, "module interface changed")
                counts["invalid"] += 1
                base["invalid"].append(label)
                base["results"].append(row)
                continue
        except Exception as e:                         # noqa: BLE001
            row = _invalid_result(label, e)
            counts["invalid"] += 1
            base["invalid"].append(label)
            base["results"].append(row)
            continue

        killed = invalid = False
        for seed in seeds:
            try:
                res = sim(mutant, golden, top, ports, seed=seed)
            except Exception as e:                     # noqa: BLE001
                row = _invalid_result(label, e)
                invalid = True
                break
            row.update({"seed": seed, "mismatches": res.get("mismatches"),
                        "checks": res.get("checks")})
            if not res.get("ok"):
                if res.get("mismatches"):
                    row["status"] = "killed"
                    killed = True
                else:
                    row = _invalid_result(label, res.get("log", "simulation failed"))
                    invalid = True
                break
        if killed:
            counts["killed"] += 1
        elif invalid:
            counts["invalid"] += 1
            base["invalid"].append(label)
        else:
            verdict, detail = (None, None)
            if equiv_fn is not None:
                try:
                    verdict, detail = _equiv_verdict(equiv_fn(mutant, golden, top))
                except Exception as e:                 # noqa: BLE001
                    detail = str(e)
            if verdict is True:
                row["status"] = "equivalent"
                counts["equivalent"] += 1
            elif verdict is False:
                row["status"] = "false_accept"
                counts["false_accept"] += 1
                base["false_accept"].append(label)
            else:
                row["status"] = "unknown"
                counts["unknown"] += 1
                base["unknown"].append(label)
            if detail:
                row["formal_log"] = str(detail)[-500:]
        base["results"].append(row)

    base["ok"] = base["golden_ok"] and counts["false_accept"] == 0
    return base


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
    m = re.search(r"^[ \t]*module\s+\w+", text, re.M) or re.search(r"\bmodule\s+\w+", text)
    b = text.rfind("endmodule")
    return text[m.start():b + len("endmodule")].strip() + "\n" if m and b > m.start() else ""


def reward(text, task, gate_weight=GATE_WEIGHT, synth=synth_cells, seed=1):
    """``(reward, info)`` for one sampled answer to ``task`` (``golden``,
    ``top``, ``ports`` and optionally ``ref_cells``)."""
    code = extract_answer(text)
    if not code or not re.search(rf"\bmodule\s+{task['top']}\b", code):
        return 0.0, {"ok": False, "why": "no module " + task["top"]}
    res = simulate(code, task["golden"], task["top"],
                   [tuple(p) for p in task["ports"]], seed=seed)
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
