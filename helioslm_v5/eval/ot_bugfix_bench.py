"""ot_bugfix_bench.py — OpenTitan real-IP bug-fix / completion benchmark
(v5.228, HDL track).

VerilogEval / RTLLM / CVDP are small, self-contained problems. This
benchmark asks the model to repair real RTL: single-module files of
OpenTitan IPs (Apache-2.0, pinned commit ``OT_COMMIT``), checked by a
formal equivalence proof against the original module.

1. ``candidates`` lists the ``hw/ip/<ip>/rtl/*.sv`` files that define
   exactly one module, are not too long for a prompt, and whose module
   passes the H5 synth gate (``hdl_projects.synth_gate``: Verilator lint,
   sv2v, Yosys synth) as its own top;
2. ``mutate`` makes one single-site bug (``bugfix``: a swapped operator,
   an inverted ``if`` condition, an off-by-one sized constant, a flipped
   ``'0``/``'1`` reset or default value) or masks one ``assign`` /
   ``always`` block (``complete``); the seed fixes every choice;
3. ``build`` keeps a task only when the original module is proven
   equivalent to itself (the scorer works on it) and, for ``bugfix``,
   the mutant still lints and synthesises but is NOT proven equivalent
   (the bug is observable). Tasks are tagged ``seen`` when their IP was
   a passing unit of the H5 training data (``seen_ips`` reads that
   manifest), else ``unseen``;
4. ``score`` puts the answer in place of the file, converts the closure
   with sv2v, and passes when Yosys synthesises it and ``equiv_make`` /
   ``equiv_simple -seq`` / ``equiv_induct -seq`` / ``equiv_status
   -assert`` prove it equivalent to the original. A timeout or an
   unproven equivalence is a fail, so a correct answer that renames
   state registers may be scored as wrong: the numbers are a lower bound.

Answers are untrusted code: score on the pod. ``examples/ot_bugfix_run.py``
is the CLI (build / generate / score).
"""
import json
import random
import re
import shutil
import subprocess
import tempfile
from collections import Counter
from pathlib import Path

from helioslm_v5.eval import hdl_bench as hb
from helioslm_v5.src.training import hdl_projects as hp

OT_URL = "https://github.com/lowRISC/opentitan"
OT_COMMIT = "ecedaa5a3fe66a373fbd7bdb6f4aac608b82537d"   # = hdl_h5_sv_uvm_v1
MAX_CHARS = 12000          # file text in the prompt
MIN_CELLS, MAX_CELLS = 10, 20000
EQUIV_SEQ = 8
EQUIV_TIMEOUT_S = 120
SYSTEM = ("You are an expert SystemVerilog hardware engineer. You fix and "
          "complete RTL without changing anything else.")
KINDS = ("bugfix", "complete")

SYMPTOM = {
    "op": "an operator is wrong (comparison, logic or arithmetic)",
    "cond": "an if condition has the wrong polarity",
    "const": "a sized constant is off by one",
    "reset": "a reset or default value is wrong ('0 versus '1)",
}


# ------------------------------------------------------------ text helpers

def code_mask(text):
    """A bool per character: True where it is code, False inside comments
    and string literals (mutations never touch those)."""
    mask = [True] * len(text)
    for rx in (hp._COMMENT, hp._STRING):
        for m in rx.finditer(text):
            for i in range(m.start(), m.end()):
                mask[i] = False
    return mask


def _in_code(mask, start, end):
    return all(mask[start:end])


def module_body_span(text):
    """(start, end) of the text between the first ``module`` header's
    ``);`` and ``endmodule`` (ports and parameters are never mutated)."""
    mask = code_mask(text)
    m = next((m for m in re.finditer(r"\bmodule\b", text)
              if mask[m.start()]), None)
    e = next((m for m in re.finditer(r"\bendmodule\b", text)
              if mask[m.start()]), None)
    if not m or not e:
        return None
    depth, i = 0, m.end()
    while i < e.start():
        c = text[i]
        if mask[i]:
            if c == "(":
                depth += 1
            elif c == ")":
                depth -= 1
            elif c == ";" and depth == 0:
                return i + 1, e.start()
        i += 1
    return None


_OPS = [(r"(?<=\s)==(?=\s)", "!="), (r"(?<=\s)!=(?=\s)", "=="),
        (r"(?<=\s)&&(?=\s)", "||"), (r"(?<=\s)\|\|(?=\s)", "&&"),
        (r"(?<=\s)\+(?=\s)", "-"), (r"(?<=\s)-(?=\s)", "+"),
        (r"(?<=\s)&(?=\s)", "|"), (r"(?<=\s)\|(?=\s)", "&"),
        (r"(?<=\s)>(?=\s)", "<"), (r"(?<=\s)<(?=\s)", ">")]
_IF = re.compile(r"\bif\s*\(")
_SIZED = re.compile(r"\b(\d+)'d(\d+)\b")
_RESET_VAL = re.compile(r"(<=|=)(\s*)'([01])(\s*);")
_ASSIGN = re.compile(r"^[ \t]*assign\b", re.M)
_ALWAYS = re.compile(r"^[ \t]*always(_ff|_comb|_latch)?\b", re.M)


def _sites(text, span, mask):
    """Every mutation site in the body: (kind, start, end, replacement)."""
    lo, hi = span
    sites = []
    for rx, rep in _OPS:
        for m in re.finditer(rx, text[lo:hi]):
            s, e = lo + m.start(), lo + m.end()
            if _in_code(mask, s, e):
                sites.append(("op", s, e, rep))
    for m in _IF.finditer(text, lo, hi):
        if not mask[m.start()]:
            continue
        close = _match_paren(text, m.end() - 1, mask)
        if close is None or close > hi:
            continue
        inner = text[m.end():close]
        sites.append(("cond", m.end(), close, f"!({inner})"))
    for m in _SIZED.finditer(text, lo, hi):
        if not _in_code(mask, m.start(), m.end()):
            continue
        width, val = int(m.group(1)), int(m.group(2))
        if width > 64:
            continue
        new = val + 1 if val + 1 < (1 << width) else val - 1
        if new < 0:
            continue
        sites.append(("const", m.start(), m.end(), f"{width}'d{new}"))
    for m in _RESET_VAL.finditer(text, lo, hi):
        if not _in_code(mask, m.start(), m.end()):
            continue
        flip = "1" if m.group(3) == "0" else "0"
        sites.append(("reset", m.start(), m.end(),
                      f"{m.group(1)}{m.group(2)}'{flip}{m.group(4)};"))
    return sites


def _match_paren(text, open_at, mask):
    depth = 0
    for i in range(open_at, len(text)):
        if not mask[i]:
            continue
        if text[i] == "(":
            depth += 1
        elif text[i] == ")":
            depth -= 1
            if depth == 0:
                return i
    return None


def _blocks(text, span, mask):
    """Maskable statements of the body: (start, end, kind) of each
    top-level ``assign ...;`` and ``always ... begin ... end`` (or a
    single-statement always)."""
    lo, hi = span
    out = []
    for m in _ASSIGN.finditer(text, lo, hi):
        s = m.start() + len(m.group(0)) - len("assign")
        if not mask[s]:
            continue
        semi = next((i for i in range(s, hi) if mask[i] and text[i] == ";"), None)
        if semi is not None:
            out.append((s, semi + 1, "assign"))
    for m in _ALWAYS.finditer(text, lo, hi):
        s = m.start() + len(m.group(0)) - len(m.group(0).lstrip())
        if not mask[s]:
            continue
        end = _always_end(text, s, hi, mask)
        if end:
            out.append((s, end, "always"))
    return out


_KW = re.compile(r"\b(begin|end|case|casez|casex|endcase)\b|;")


def _always_end(text, start, hi, mask):
    """End offset of the always block at ``start``: the ``end`` closing
    its first ``begin``, or the first ``;`` when the statement has no
    ``begin`` before it."""
    depth = 0
    for m in _KW.finditer(text, start, hi):
        if not mask[m.start()]:
            continue
        w = m.group(0)
        if w in ("begin", "case", "casez", "casex"):
            depth += 1
        elif w in ("end", "endcase"):
            depth -= 1
            if depth == 0 and not re.match(r"\s*else\b", text[m.end():]):
                return m.end()
        elif w == ";" and depth == 0:
            return m.end()
    return None


def mutate(text, kind, rng):
    """One task body from ``text``: returns {"kind", "op", "text",
    "line"} or None when the file has no site of that kind.
    ``bugfix`` changes one site; ``complete`` replaces one block by a
    TODO comment (kept line count so the line number is meaningful)."""
    span = module_body_span(text)
    if not span:
        return None
    mask = code_mask(text)
    if kind == "bugfix":
        sites = _sites(text, span, mask)
        if not sites:
            return None
        ops = sorted({s[0] for s in sites})
        op = rng.choice(ops)                  # balance operator kinds
        _, s, e, rep = rng.choice([x for x in sites if x[0] == op])
        new = text[:s] + rep + text[e:]
        return {"kind": kind, "op": op, "text": new,
                "line": text.count("\n", 0, s) + 1}
    if kind == "complete":
        blocks = _blocks(text, span, mask)
        blocks = [b for b in blocks if 1 <= text.count("\n", b[0], b[1]) <= 40
                  or b[2] == "assign"]
        if not blocks:
            return None
        s, e, what = rng.choice(blocks)
        lines = text.count("\n", s, e)
        hole = (f"// TODO: one {what} block was removed here; restore it"
                + "\n" * lines)
        return {"kind": kind, "op": what, "text": text[:s] + hole + text[e:],
                "line": text.count("\n", 0, s) + 1}
    raise ValueError(kind)


def prompt(task):
    if task["kind"] == "bugfix":
        ask = (f"This file has exactly one bug: {SYMPTOM[task['op']]}. "
               "Fix it.")
    else:
        ask = ("One block of this file was removed and replaced by a "
               "`// TODO` comment. Write it back so the module works.")
    return (f"The file below is `{task['path']}` from the OpenTitan "
            f"`{task['ip']}` IP. {ask} Return the complete corrected file "
            "between [BEGIN] and [DONE], with no other changes.\n\n"
            f"```systemverilog\n{task['prompt_text']}```\n")


# ----------------------------------------------------------------- tools

def tools(verilator=None, yosys=None, sv2v=None):
    return hp.tools(verilator, yosys, sv2v)


def _ordered(start, info, defined, all_text):
    """The closure of ``start`` in compile order, as ``synth_gate``
    builds it (macro headers first)."""
    files = sorted(hp.closure([start], info, defined))
    order = [p for p in hp.dep_order(files, info, defined)
             if not p.endswith((".svh", ".vh")) and info[p]["defs"]]
    heads = [p for p in files if p not in order and not info[p]["defs"]
             and "`define" in all_text.get(p, "")]
    inner = {Path(i).name for p in heads for i in info[p]["includes"]}
    order = [p for p in heads if Path(p).name not in inner] + order
    return files, order


def to_verilog(top, start, files, order, info, texts, tool,
               replace=None, timeout=300):
    """sv2v the closure with ``start``'s text replaced by ``replace``.
    Returns (verilog text, None) or (None, error)."""
    if not tool.get("sv2v"):
        return None, "sv2v not found"
    with tempfile.TemporaryDirectory() as tmp:
        src = Path(tmp, "src")
        for p in files:
            f = src / p
            f.parent.mkdir(parents=True, exist_ok=True)
            t = replace if (replace is not None and p == start) else texts[p]
            f.write_text(hp.drop_translate_off(t), encoding="utf-8")
        inc = [f"-I{d}" for d in hp.include_dirs(files, info)]
        out = Path(tmp, "out.v")
        cmd = [tool["sv2v"], "-DSYNTHESIS", *inc, f"--top={top}",
               f"--write={out}", *order]
        code, log = hp._run(cmd, src, timeout)
        if code and "Resource temporarily unavailable" in log:
            code, log = hp._run(cmd, src, timeout)   # GHC thread limit: once more
        if code or not out.exists():
            return None, "sv2v: " + hp._first_error(log)
        return out.read_text(encoding="utf-8", errors="replace"), None


def lint_synth(verilog, top, tool, timeout=300):
    """Yosys synth of a converted design (plus Verilator lint when
    installed). Returns (pass, error)."""
    with tempfile.TemporaryDirectory() as tmp:
        f = Path(tmp, "d.v")
        f.write_text(verilog, encoding="utf-8")
        if tool.get("verilator"):
            code, log = hp._run([tool["verilator"], "--lint-only",
                                 "-Wno-fatal", "-Wno-lint", "-Wno-style",
                                 "--top-module", top, str(f)], tmp, timeout)
            if code:
                return False, "lint: " + hp._first_error(log)
        stat = Path(tmp, "stat.txt")
        code, log = hp._run([tool["yosys"], "-q", "-p",
                             f"read_verilog {f}; hierarchy -check -top {top}; "
                             f"synth -top {top}; tee -q -o {stat} stat"],
                            tmp, timeout)
        cells = hp._stat_cells(stat.read_text()) if stat.exists() else None
        if code or not cells:
            return False, "synth: " + (hp._first_error(log) if code else "no cells")
        return True, cells


def equiv_script(gold, gate, top, seq=EQUIV_SEQ):
    """The Yosys script proving ``gate`` (file) equivalent to ``gold``."""
    prep = f"prep -flatten -top {top}; memory_map; opt_clean"
    return (f"read_verilog {gold}; {prep}; rename {top} gold; "
            f"design -stash gold; "
            f"read_verilog {gate}; {prep}; rename {top} gate; "
            f"design -stash gate; "
            f"design -copy-from gold -as gold gold; "
            f"design -copy-from gate -as gate gate; "
            f"async2sync; equiv_make gold gate equiv; hierarchy -top equiv; "
            f"equiv_simple -seq {seq}; equiv_induct -seq {seq}; "
            f"equiv_status -assert")


def equivalent(gold_v, gate_v, top, tool, timeout=EQUIV_TIMEOUT_S):
    """True when Yosys proves the two converted designs equivalent."""
    with tempfile.TemporaryDirectory() as tmp:
        g, c = Path(tmp, "gold.v"), Path(tmp, "gate.v")
        g.write_text(gold_v, encoding="utf-8")
        c.write_text(gate_v, encoding="utf-8")
        code, log = hp._run([tool["yosys"], "-q", "-p",
                             equiv_script(g, c, top)], tmp, timeout)
        return code == 0, None if code == 0 else hp._first_error(log)


# ----------------------------------------------------------------- build

def repo_texts(root):
    root = Path(root)
    out = {}
    for p in sorted(root.rglob("*")):
        if p.suffix in (".sv", ".svh", ".v", ".vh") and p.is_file() \
                and ".git" not in p.parts and "vendor" not in p.parts:
            out[str(p.relative_to(root))] = p.read_text(encoding="utf-8",
                                                       errors="replace")
    return out


_RTL = re.compile(r"^hw/ip/([^/]+)/rtl/[^/]+\.sv$")


def candidates(texts, info):
    """[(ip, path, module)] of single-module RTL files short enough for
    a prompt (no gate run yet)."""
    out = []
    for p, t in texts.items():
        m = _RTL.match(p)
        if not m or len(t) > MAX_CHARS or hp.TB_PATH.search(p):
            continue
        mods = [n for k, n in info[p]["defs"] if k in ("module", "macromodule")]
        if len(mods) == 1 and len(info[p]["defs"]) == 1:
            out.append((m.group(1), p, mods[0]))
    return sorted(out)


def seen_ips(manifest):
    """IPs whose ``hw/ip/<ip>`` unit passed the gate in the H5 data
    manifest (``fetch_mixture`` output), i.e. were trained on."""
    if not manifest:
        return set()
    m = json.loads(Path(manifest).read_text())
    seen = set()
    for name, s in m.get("sources", {}).items():
        if "opentitan" not in name:
            continue
        for r in s.get("projects") or []:
            mm = re.search(r"hw/ip/([^/]+)$", r.get("project", ""))
            if mm and r.get("pass") and not r.get("near_dup"):
                seen.add(mm.group(1))
    return seen


def _build_ip(ip, files_ip, texts, info, defined, seen, seed, per_ip, tool):
    """Tasks of one IP (its own seeded RNG, so IPs can run in parallel)."""
    rng = random.Random(f"{seed}:{ip}")
    files_ip = sorted(files_ip)
    rng.shuffle(files_ip)
    tasks, stats = [], Counter()
    for p, mod in files_ip:
        if len(tasks) >= per_ip:
            break
        files, order = _ordered(p, info, defined, texts)
        gold, err = to_verilog(mod, p, files, order, info, texts, tool)
        ok, cells = (False, err) if gold is None else lint_synth(gold, mod, tool)
        if not ok or not (MIN_CELLS <= cells <= MAX_CELLS):
            stats["gold_gate_fail" if not ok else "size"] += 1
            continue
        if not equivalent(gold, gold, mod, tool)[0]:
            stats["self_equiv_fail"] += 1
            continue
        for kind in KINDS:
            if len(tasks) >= per_ip:
                break
            mut = mutate(texts[p], kind, rng)
            if not mut:
                stats[f"{kind}_no_site"] += 1
                continue
            if kind == "bugfix":
                bad, _ = to_verilog(mod, p, files, order, info, texts,
                                    tool, replace=mut["text"])
                if bad is None or not lint_synth(bad, mod, tool)[0]:
                    stats["mutant_broken"] += 1
                    continue
                if equivalent(gold, bad, mod, tool)[0]:
                    stats["mutant_equivalent"] += 1
                    continue
            task = {"id": f"ot_{ip}_{Path(p).stem}_{kind}",
                    "ip": ip, "path": p, "top": mod, "kind": kind,
                    "op": mut["op"], "line": mut["line"],
                    "split": "seen" if ip in seen else "unseen",
                    "cells": cells, "prompt_text": mut["text"],
                    "commit": OT_COMMIT}
            task["prompt"] = prompt(task)
            tasks.append(task)
            stats["kept"] += 1
    return tasks, stats


def build(root, *, seed=0, per_ip=10, max_tasks=200, manifest=None,
          tool=None, workers=8, log=print):
    """Tasks (list of dicts) from an OpenTitan checkout at ``root``.
    IPs are built in parallel; the result is in IP order and capped at
    ``max_tasks`` (the same tasks for the same seed and checkout)."""
    from concurrent.futures import ThreadPoolExecutor
    tool = tool or tools()
    if not tool.get("yosys") or not tool.get("sv2v"):
        raise RuntimeError("build needs yosys and sv2v")
    texts = repo_texts(root)
    info, defined = hp.index(texts)
    seen = seen_ips(manifest)
    by_ip = {}
    for ip, p, mod in candidates(texts, info):
        by_ip.setdefault(ip, []).append((p, mod))
    stats = Counter()
    with ThreadPoolExecutor(max(1, workers)) as ex:
        futs = {ip: ex.submit(_build_ip, ip, by_ip[ip], texts, info, defined,
                              seen, seed, per_ip, tool) for ip in sorted(by_ip)}
        per = {}
        for ip, f in futs.items():
            per[ip], st = f.result()
            stats.update(st)
            log(f"{ip}: {len(per[ip])} tasks")
    tasks = [t for ip in sorted(per) for t in per[ip]][:max_tasks]
    log("build stats: " + json.dumps(dict(stats)) + f" -> {len(tasks)} tasks")
    return tasks


# ----------------------------------------------------------------- score

_HOLE = re.compile(r"^([ \t]*)// TODO: one \w+ block was removed here; restore it\n", re.M)


def splice(task, code):
    """A ``complete`` answer that holds only the missing block (no
    ``module``): the task file with the TODO comment replaced by it.
    None for a ``bugfix`` task or an empty answer."""
    if task["kind"] != "complete" or not code.strip():
        return None
    m = _HOLE.search(task["prompt_text"])
    if not m:
        return None
    body = code.strip("\n")
    if "// TODO: one" in body:          # the model echoed the comment
        body = "\n".join(ln for ln in body.splitlines() if "// TODO: one" not in ln)
    t = task["prompt_text"]
    return t[:m.start()] + body + "\n" + t[m.end():]


class Scorer:
    """Scores answers against one checkout; caches each task's gold."""

    def __init__(self, root, tool=None):
        self.tool = tool or tools()
        self.texts = repo_texts(root)
        self.info, self.defined = hp.index(self.texts)
        self._gold = {}

    def _closure(self, task):
        return _ordered(task["path"], self.info, self.defined, self.texts)

    def gold(self, task):
        key = task["path"]
        if key not in self._gold:
            files, order = self._closure(task)
            self._gold[key] = to_verilog(task["top"], task["path"], files,
                                         order, self.info, self.texts,
                                         self.tool)[0]
        return self._gold[key]

    def score(self, task, answer_text):
        """{"synth": bool, "pass": bool, "error": str|None} for one raw
        model answer."""
        code = hb.extract_code(answer_text)
        res = {"synth": False, "pass": False, "error": None}
        res["spliced"] = False
        if "module" not in code:
            spliced = splice(task, code)
            if spliced is None:
                res["error"] = "no module in answer"
                return res
            code, res["spliced"] = spliced, True
        files, order = self._closure(task)
        gate, err = to_verilog(task["top"], task["path"], files, order,
                               self.info, self.texts, self.tool, replace=code)
        if gate is None:
            res["error"] = err
            return res
        ok, info = lint_synth(gate, task["top"], self.tool)
        res["synth"] = ok
        if not ok:
            res["error"] = info
            return res
        gold = self.gold(task)
        if gold is None:
            res["error"] = "gold conversion failed"
            return res
        res["pass"], res["error"] = equivalent(gold, gate, task["top"], self.tool)
        return res


def summarize(tasks, results):
    """Pass / synth rates overall, by kind, by split and by kind x split.
    ``results``: {task id: [score dicts, one per sample]}."""
    groups = {}
    for t in tasks:
        rs = results.get(t["id"]) or []
        if not rs:
            continue
        p = sum(r["pass"] for r in rs) / len(rs)
        s = sum(r["synth"] for r in rs) / len(rs)
        for key in ("all", t["kind"], t["split"], f"{t['kind']}/{t['split']}",
                    f"op:{t['op']}"):
            g = groups.setdefault(key, [0, 0.0, 0.0])
            g[0] += 1
            g[1] += p
            g[2] += s
    return {k: {"n": n, "pass": round(p / n, 4), "synth": round(s / n, 4)}
            for k, (n, p, s) in sorted(groups.items())}


def checkout(dest, url=OT_URL, commit=OT_COMMIT):
    """Shallow checkout of the pinned commit."""
    dest = Path(dest)
    if (dest / ".git").is_dir():
        return dest
    dest.mkdir(parents=True, exist_ok=True)
    for cmd in (["git", "init", "-q"], ["git", "remote", "add", "origin", url],
                ["git", "fetch", "-q", "--depth", "1", "origin", commit],
                ["git", "checkout", "-q", "FETCH_HEAD"]):
        subprocess.run(cmd, cwd=dest, check=True)
    return dest


def have_tools(tool=None):
    tool = tool or tools()
    return bool(tool.get("yosys") and tool.get("sv2v")
                and shutil.which("git"))
