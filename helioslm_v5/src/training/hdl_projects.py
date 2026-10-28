"""hdl_projects.py — project-level HDL pretraining data (HDL track H5).

A *project* is a group of HDL files that belong together: one IP of a
large repository (``hw/ip/uart``), one core, or one GitHub repository of
The Stack. This module turns a project into one training document and
decides whether it may be used at all:

1. ``index`` parses each file's definitions (package, interface, module,
   class, program) and references (``import p::*``, ``p::x``, module
   instantiations, ``extends``, ``\\`include``);
2. ``dep_order`` sorts the files so that a definition comes before its
   uses (packages, then interfaces, then modules, then classes / UVM
   environments, agents and sequences among files with no order between
   them);
3. ``synth_gate`` finds the project's top module, collects the files it
   needs from the whole repository (vendored dependencies included, even
   though they are not trained on), runs ``verilator --lint-only`` and a
   Yosys ``synth`` of that top, and passes only when both succeed and the
   netlist has cells. SystemVerilog goes through ``sv2v`` first when it is
   installed (Yosys' own reader handles only part of SystemVerilog);
4. ``pack`` writes the project as one document: ``// project:`` header,
   then each file as ``// file: <path>`` + its text, in dependency order;
5. ``MinHashDedup`` drops near-duplicate projects (forks) and
   ``Decontam`` drops files sharing a 13-gram with the evaluation
   benchmarks (VerilogEval, RTLLM, CVDP).

No third-party Python packages; the gate needs ``verilator`` and
``yosys`` on PATH (or passed in) and uses ``sv2v`` if present.
"""
import hashlib
import re
import shutil
import subprocess
import tempfile
import zlib
from collections import defaultdict
from pathlib import Path

import numpy as np

HDL_SUFFIXES = (".v", ".sv", ".svh", ".vh")
_COMMENT = re.compile(r"//[^\n]*|/\*.*?\*/", re.S)
_STRING = re.compile(r'"(?:\\.|[^"\\\n])*"')
_DEF = re.compile(
    r"^[ \t]*(?:(?:virtual|static|automatic|extern)\s+)*"
    r"(module|macromodule|package|interface|program|class)\s+"
    r"(?:(?:static|automatic)\s+)?([A-Za-z_]\w*)", re.M)
_IMPORT = re.compile(r"\b([A-Za-z_]\w*)\s*::")
_EXTENDS = re.compile(r"\bextends\s+([A-Za-z_]\w*)")
_INST = re.compile(r"(?:^|;)[ \t]*([A-Za-z_]\w*)\s*(?:#\s*\(|[A-Za-z_]\w*\s*"
                   r"(?:\[[^\]]*\]\s*)?\()", re.M)
_INCLUDE = re.compile(r'^[ \t]*`include\s+"([^"]+)"', re.M)
KIND_RANK = {"package": 0, "interface": 1, "module": 2, "macromodule": 2,
             "program": 3, "class": 3}
_UVM_ORDER = ("_if", "_pkg", "cfg", "item", "seq_item", "driver", "monitor",
              "sequencer", "agent", "scoreboard", "cov", "env", "vseq", "seq",
              "test", "tb")
TB_PATH = re.compile(r"(^|/)((pre_)?dv|tb|test|tests|testbench|sim|verif|bench|formal|fpv)(/|$)|"
                     r"(_tb|_test|tb_)[^/]*$", re.I)
_TB_NAME = re.compile(r"(^tb_|_tb$|^tb$|test)", re.I)


def strip_comments(text):
    """Text without comments; string literals kept (they hold includes)."""
    keep = {}

    def save(m):
        keep[len(keep)] = m.group(0)
        return f"\x00{len(keep) - 1}\x00"
    text = _STRING.sub(save, text)
    text = _COMMENT.sub(lambda m: "\n" * m.group(0).count("\n"), text)
    return re.sub(r"\x00(\d+)\x00", lambda m: keep[int(m.group(1))], text)


def parse(text):
    """{"defs": [(kind, name)], "refs": set(names), "includes": [paths]}"""
    code = strip_comments(text)
    defs = [(k, n) for k, n in _DEF.findall(code)]
    refs = set(_IMPORT.findall(code)) | set(_EXTENDS.findall(code))
    refs |= set(_INST.findall(code))
    refs -= {n for _, n in defs}
    return {"defs": defs, "refs": refs, "includes": _INCLUDE.findall(code)}


def index(files):
    """Parse ``files`` ({path: text}); returns (info {path: parse()},
    defined {name: [paths]})."""
    info = {p: parse(t) for p, t in files.items()}
    defined = defaultdict(list)
    for p, i in info.items():
        for _, n in i["defs"]:
            defined[n].append(p)
    return info, defined


def _pick(paths, near):
    """The definition to use among several (variants such as
    prim_generic / prim_xilinx): same directory tree as ``near`` first,
    then a ``generic`` path, then the shortest path."""
    def key(p):
        shared = 0
        for a, b in zip(Path(near).parts, Path(p).parts):
            if a != b:
                break
            shared += 1
        return (-shared, "generic" not in p, len(p), p)
    return min(paths, key=key)


def _include_target(inc, paths_by_name, near):
    cands = paths_by_name.get(Path(inc).name, [])
    full = [p for p in cands if p.endswith(inc)]
    return _pick(full or cands, near) if (full or cands) else None


def deps(path, info, defined, paths_by_name):
    """Files ``path`` needs: definitions of its references and its
    includes (one file per name)."""
    out = set()
    for r in info[path]["refs"]:
        if r in defined:
            out.add(_pick(defined[r], path))
    for inc in info[path]["includes"]:
        t = _include_target(inc, paths_by_name, path)
        if t:
            out.add(t)
    out.discard(path)
    return out


def _by_name(paths):
    d = defaultdict(list)
    for p in paths:
        d[Path(p).name].append(p)
    return d


def closure(start, info, defined, design_only=True):
    """``start`` plus everything it needs, transitively. ``design_only``
    leaves out testbench paths and files that define only classes /
    programs."""
    names = _by_name(info)
    seen, todo = set(), list(start)
    while todo:
        p = todo.pop()
        if p in seen:
            continue
        seen.add(p)
        for d in deps(p, info, defined, names):
            kinds = {k for k, _ in info[d]["defs"]}
            if design_only and (TB_PATH.search(d) or (
                    kinds and kinds <= {"class", "program"})):
                continue
            todo.append(d)
    return seen


def _rank(path, info):
    kinds = [KIND_RANK[k] for k, _ in info[path]["defs"]]
    kind = min(kinds) if kinds else (0 if path.endswith((".svh", ".vh")) else 2)
    stem = Path(path).stem.lower()
    uvm = next((i for i, s in enumerate(_UVM_ORDER) if stem.endswith(s)),
               len(_UVM_ORDER))
    return (kind, uvm, path)


def dep_order(paths, info, defined):
    """``paths`` sorted so that a file comes after the files it needs
    (Kahn's algorithm; ties by kind, then UVM role, then path). Files in a
    cycle keep that tie order."""
    paths = list(paths)
    pset, names = set(paths), _by_name(paths)
    need = {p: deps(p, info, defined, names) & pset for p in paths}
    users = defaultdict(set)
    for p, ds in need.items():
        for d in ds:
            users[d].add(p)
    left = {p: len(ds) for p, ds in need.items()}
    out, ready = [], sorted((p for p in paths if not left[p]),
                            key=lambda p: _rank(p, info))
    while ready:
        p = ready.pop(0)
        out.append(p)
        for u in users[p]:
            left[u] -= 1
            if not left[u]:
                ready.append(u)
        ready.sort(key=lambda q: _rank(q, info))
    rest = sorted((p for p in paths if p not in set(out)),
                  key=lambda p: _rank(p, info))
    return out + rest


def pack(name, paths, files):
    """One document: a ``// project:`` line, then every file in order."""
    parts = [f"// project: {name}\n"]
    for p in paths:
        parts.append(f"// file: {p}\n{files[p].rstrip()}\n")
    return "\n".join(parts)


def top_candidates(paths, info, defined, prefer=None):
    """Modules of ``paths`` (non-testbench files) that no other module of
    ``paths`` instantiates; ``prefer`` (a name) first, then the ones with
    the largest design closure."""
    design = [p for p in paths if not TB_PATH.search(p)]
    mods = {n: p for p in design for k, n in info[p]["defs"]
            if k in ("module", "macromodule") and not _TB_NAME.search(n)}
    used = set()
    for p in design:
        used |= info[p]["refs"]
    tops = [n for n in mods if n not in used]

    def size(n):
        return len(closure([mods[n]], info, defined))
    tops.sort(key=lambda n: (n != prefer, -size(n), n))
    return [(n, mods[n]) for n in tops]


def _run(cmd, cwd, timeout, stdin=None):
    try:
        r = subprocess.run(cmd, cwd=cwd, capture_output=True, text=True,
                           timeout=timeout, stdin=stdin)
        return r.returncode, (r.stdout + r.stderr)[-4000:]
    except subprocess.TimeoutExpired:
        return -1, "timeout"
    except OSError as e:
        return -1, f"{type(e).__name__}: {e}"


def _stat_cells(report):
    found = re.findall(r"Number of cells:\s+(\d+)|^\s*(\d+)\s+cells\s*$",
                       report, re.M)
    return int(next(g for g in found[-1] if g)) if found else None


def tools(verilator=None, yosys=None, sv2v=None):
    """Tool paths: arguments, else PATH. Missing verilator / yosys -> None."""
    return {"verilator": verilator or shutil.which("verilator"),
            "yosys": yosys or shutil.which("yosys"),
            "sv2v": sv2v or shutil.which("sv2v")}


def synth_gate(top, start, root, info, defined, tool=None, timeout=600,
               defines=("SYNTHESIS",), all_text=None):
    """Lint and synthesise module ``top`` (defined in ``start``) of the
    checkout / directory ``root``. Returns {"top", "files", "lint",
    "synth", "cells", "pass", "error"}. ``all_text``: {path: text}, to
    find macro-only headers."""
    tool = tool or tools()
    all_text = all_text or {}
    files = sorted(closure([start], info, defined))
    order = [p for p in dep_order(files, info, defined)
             if not p.endswith((".svh", ".vh")) and info[p]["defs"]]
    # macro headers first: a `define used by a file compiled before the
    # file that includes it (filelist order) must already be known. Only
    # headers no other header includes: those are chosen by `ifdef there
    heads = [p for p in files if p not in order and not info[p]["defs"]
             and "`define" in all_text.get(p, "")]
    inner = {Path(i).name for p in heads for i in info[p]["includes"]}
    order = [p for p in heads if Path(p).name not in inner] + order
    res = {"top": top, "files": len(order), "lint": False, "synth": False,
           "cells": None, "pass": False, "error": None, "paths": files}
    if not tool.get("verilator") or not tool.get("yosys"):
        res["error"] = "verilator / yosys not found"
        return res
    incdirs = sorted({str(Path(p).parent) for p in files} | {"."})
    inc = [f"-I{d}" for d in incdirs]
    dfs = [f"-D{d}" for d in defines]
    code, log = _run([tool["verilator"], "--lint-only", "-Wno-fatal",
                      "-Wno-lint", "-Wno-style", "--top-module", top,
                      *dfs, *inc, *order], root, timeout)
    res["lint"] = code == 0
    if not res["lint"]:
        res["error"] = "lint: " + _first_error(log)
        return res
    with tempfile.TemporaryDirectory() as tmp:
        stat = Path(tmp, "stat.txt")
        if tool.get("sv2v") and any(p.endswith(".sv") for p in order):
            out = Path(tmp, "sv2v.v")
            code, log = _run([tool["sv2v"], *dfs, *inc, f"--top={top}",
                              f"--write={out}", *order], root, timeout)
            if code:
                res["error"] = "sv2v: " + _first_error(log)
                return res
            read = f"read_verilog {out}"
        else:
            read = "read_verilog -sv " + " ".join(
                [*dfs, *inc, *order])
        script = (f"{read}; hierarchy -check -top {top}; synth -top {top}; "
                  f"tee -q -o {stat} stat")
        code, log = _run([tool["yosys"], "-q", "-p", script], root, timeout)
        cells = _stat_cells(stat.read_text()) if stat.exists() else None
    res.update(synth=code == 0, cells=cells)
    res["pass"] = res["synth"] and bool(cells)
    if not res["pass"]:
        res["error"] = "synth: " + (_first_error(log) if code else "no cells")
    return res


def _first_error(log):
    for line in log.splitlines():
        if "rror" in line or "ERROR" in line:
            return line.strip()[:300]
    return log.strip().splitlines()[-1][:300] if log.strip() else "failed"


def gate_project(root, unit_paths, info, defined, prefer=None, tool=None,
                 tries=1, timeout=600, all_text=None):
    """Gate the unit's top module: the one named like the unit directory,
    else the uninstantiated module with the largest design. ``tries`` > 1
    falls back to the next candidates, but a small helper module passing
    says little about the project, so the default is 1. Returns the
    passing result, or the first failure, or a ``no top module`` result."""
    first = None
    for top, start in top_candidates(unit_paths, info, defined,
                                     prefer)[:tries]:
        r = synth_gate(top, start, root, info, defined, tool, timeout,
                       all_text=all_text)
        if r["pass"]:
            return r
        first = first or r
    return first or {"top": None, "files": 0, "lint": False, "synth": False,
                     "cells": None, "pass": False, "error": "no top module",
                     "paths": []}


# ---------------------------------------------------------------- dedup ---

_TOKEN = re.compile(r"\w+|[^\s\w]")
_P = (1 << 61) - 1


def tokens(text):
    return _TOKEN.findall(strip_comments(text))


def shingles(text, n=5):
    t = tokens(text)
    return {zlib.crc32(" ".join(t[i:i + n]).encode())
            for i in range(max(1, len(t) - n + 1))}


class MinHashDedup:
    """Near-duplicate filter: MinHash (``perms`` permutations) with LSH
    (``bands`` bands); ``seen(text)`` is True when an earlier text shares
    a band, i.e. their Jaccard similarity is likely above ~(1/bands) **
    (1/rows)  (0.8 for 128 / 16)."""

    def __init__(self, perms=128, bands=16, seed=0):
        rng = np.random.default_rng(seed)
        self.a = rng.integers(1, _P, perms, dtype=np.uint64)
        self.b = rng.integers(0, _P, perms, dtype=np.uint64)
        self.bands, self.rows = bands, perms // bands
        self.buckets = [dict() for _ in range(bands)]

    def signature(self, text):
        x = np.fromiter(shingles(text), dtype=np.uint64)
        if not len(x):
            x = np.zeros(1, dtype=np.uint64)
        h = (np.outer(x, self.a) + self.b) % np.uint64(_P)
        return h.min(axis=0)

    def seen(self, text, key=None):
        sig, dup = self.signature(text), False
        for i, bucket in enumerate(self.buckets):
            band = sig[i * self.rows:(i + 1) * self.rows].tobytes()
            if band in bucket:
                dup = True
            else:
                bucket[band] = key
        return dup


class Decontam:
    """13-gram overlap with benchmark texts (prompts and reference
    solutions). ``hit(text)`` is True when ``text`` shares any 13-token
    window with them."""

    def __init__(self, texts=(), n=13):
        self.n, self.grams = n, set()
        for t in texts:
            self.add(t)

    def _grams(self, text):
        t = tokens(text)
        return {hashlib.blake2b(" ".join(t[i:i + self.n]).encode(),
                                digest_size=8).digest()
                for i in range(len(t) - self.n + 1)}

    def add(self, text):
        self.grams |= self._grams(text)

    def hit(self, text):
        return bool(self.grams) and not self.grams.isdisjoint(self._grams(text))


def bench_texts(paths):
    """Texts of benchmark files under ``paths`` (files or directories):
    HDL sources, prompts (.txt), and every string value of .json /
    .jsonl rows."""
    import json
    for root in paths:
        root = Path(root)
        for p in ([root] if root.is_file() else sorted(root.rglob("*"))):
            if not p.is_file() or p.stat().st_size > 50 << 20:
                continue
            s = p.suffix.lower()
            if s in HDL_SUFFIXES + (".txt",):
                yield p.read_text(encoding="utf-8", errors="replace")
            elif s in (".json", ".jsonl"):
                raw = p.read_text(encoding="utf-8", errors="replace")
                rows = (raw.splitlines() if s == ".jsonl" else [raw])
                for line in rows:
                    try:
                        yield from _strings(json.loads(line))
                    except ValueError:
                        continue


def _strings(obj):
    if isinstance(obj, str):
        if len(obj) > 60:
            yield obj
    elif isinstance(obj, dict):
        for v in obj.values():
            yield from _strings(v)
    elif isinstance(obj, list):
        for v in obj:
            yield from _strings(v)


# ------------------------------------------------------------- projects ---

def unit_of(path, patterns):
    """The unit directory of ``path``: the prefix matched by the first
    glob of ``patterns`` (e.g. ``hw/ip/*`` -> ``hw/ip/uart``), or None."""
    import fnmatch
    parts = path.split("/")
    for pat in patterns:
        n = pat.count("/") + 1
        if len(parts) > n and fnmatch.fnmatch("/".join(parts[:n]), pat):
            return "/".join(parts[:n])
    return None


def project_docs(name, files, all_files, root, *, units=None, keep=(),
                 gate=True, isolate=False, tool=None, dedup=None,
                 decontam=None, report=None, timeout=600, workers=1):
    """Documents of one repository (``files``: the trainable {path: text},
    ``all_files``: every HDL file of the checkout, for dependencies).

    ``units``: globs naming the project directories (``hw/ip/*``); None
    makes the whole repository one project. A unit is used only when its
    synth gate passes (``gate``); its files (design and testbench, in
    dependency order) become one document. ``isolate``: resolve a unit's
    dependencies inside the unit only (one GitHub repository per unit).
    Files matching ``keep`` (shared libraries such as OpenTitan's
    ``hw/ip/prim`` and ``hw/dv``) and files outside every unit form one
    more document: once a unit passed, its testbench-path files, and
    design files only when a passing unit's synthesis read them.
    ``report`` (a list) receives one row per unit; ``workers`` units are
    gated at once."""
    import fnmatch
    from concurrent.futures import ThreadPoolExecutor
    groups = defaultdict(list)
    for p in files:
        lib = any(fnmatch.fnmatch(p, g) for g in keep)
        groups[None if lib else (unit_of(p, units) if units else "")].append(p)
    shared = None if isolate else index(all_files)

    def job(u):
        if shared:
            info, defined = shared
        else:
            info, defined = index({p: t for p, t in all_files.items()
                                   if (unit_of(p, units) if units else "") == u})
        g = None
        if gate:
            prefer = Path(u).name if u else None
            g = gate_project(root, groups[u], info, defined, prefer, tool,
                             timeout=timeout, all_text=all_files)
        return u, info, defined, g

    with ThreadPoolExecutor(max(1, workers)) as ex:
        results = list(ex.map(job, sorted(k for k in groups if k is not None)))
    out, passed, synthesised = [], 0, set()
    for u, info, defined, g in results:
        paths = groups[u]
        row = {"project": f"{name}/{u}".rstrip("/"), "files": len(paths)}
        if g is not None:
            g = dict(g)
            if g["pass"]:
                synthesised |= set(g["paths"])
            del g["paths"]
            row.update(g, gate_files=g["files"], files=len(paths))
            if not g["pass"]:
                if report is not None:
                    report.append(row)
                continue
        kept = [p for p in paths
                if not (decontam and decontam.hit(all_files[p]))]
        row["decontam_dropped"] = len(paths) - len(kept)
        doc = pack(row["project"], dep_order(kept, info, defined), all_files)
        if kept and dedup is not None and dedup.seen(doc, row["project"]):
            row["near_dup"] = True
        elif kept:
            out.append(doc)
            passed += 1
        if report is not None:
            report.append(row)
    # shared libraries: verification code once a unit passed, design
    # code only when a passing unit's synthesis used it
    rest = [p for p in groups.get(None, [])
            if not gate or (passed and (p in synthesised or TB_PATH.search(p)))]
    if rest:
        info, defined = shared or index({p: all_files[p] for p in rest})
        rest = [p for p in rest
                if not (decontam and decontam.hit(all_files[p]))]
        out.append(pack(f"{name}/shared", dep_order(rest, info, defined),
                        all_files))
        if report is not None:
            report.append({"project": f"{name}/shared", "files": len(rest),
                           "pass": None})
    return out


def read_tree(root, max_bytes=1 << 20):
    """{repo-relative path: text} of every HDL file under ``root``."""
    root = Path(root)
    return {p.relative_to(root).as_posix(): p.read_text(encoding="utf-8",
                                                        errors="replace")
            for p in sorted(root.rglob("*"))
            if p.is_file() and p.suffix.lower() in HDL_SUFFIXES
            and ".git" not in p.parts and p.stat().st_size <= max_bytes}


def repo_docs(rows, *, batch=256, workdir=None, **kw):
    """Documents of GitHub repositories given as rows ``(repo, path,
    text)`` grouped by repository (as The Stack orders them, or not): each
    repository is one isolated project (``project_docs`` with
    ``units=["*/*"]`` over ``owner/name/path``). Repositories are written
    to a scratch directory ``batch`` at a time; yields documents."""
    by_repo = defaultdict(dict)
    for repo, path, text in rows:
        if repo and path and "/" in repo and ".." not in path:
            by_repo[repo][path.lstrip("/")] = text
    repos = sorted(by_repo)
    for i in range(0, len(repos), batch):
        with tempfile.TemporaryDirectory(dir=workdir) as tmp:
            files = {}
            for r in repos[i:i + batch]:
                for p, t in by_repo[r].items():
                    f = Path(tmp, r, p)
                    f.parent.mkdir(parents=True, exist_ok=True)
                    f.write_text(t, encoding="utf-8")
                    files[f"{r}/{p}"] = t
            yield from project_docs("github", files, files, tmp,
                                    units=["*/*"], isolate=True, **kw)
