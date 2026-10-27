"""HDL benchmark harness: VerilogEval v2 spec-to-rtl (v5.147, HDL track H0).

VerilogEval v2 (NVlabs/verilog-eval, MIT) has 156 problems. Each one is a
natural-language spec for a module named ``TopModule``, a reference
``RefModule`` and a self-checking testbench. A sample passes when Icarus
Verilog compiles it with the testbench and reference and the simulation
prints ``Mismatches: 0 in N samples``.

This module reproduces the upstream pipeline so our numbers compare with
published ones:

- prompt   : upstream ``sv-generate`` spec-to-rtl prompt (system message,
             ``Question:`` / ``Answer:``, the ``[BEGIN]``/``[DONE]`` suffix,
             optional 1-shot prefix, no rules suffix);
- extract  : ``[BEGIN]``/``[DONE]`` first, else the first backtick block;
- simulate : ``iverilog -Wall -Winfloop -Wno-timescale -g2012 -s tb``, then
             ``vvp`` under a 30 s timeout (upstream Makefile);
- classify : the ``sv-iv-analyze`` failure codes (S syntax, C compile, T
             timeout, R runtime mismatch, ... and "." for a pass).

Two optional extra checks, reported but never part of the pass rate:
``pyslang`` lint (SystemVerilog front end) and ``yosys`` synthesis, which
catches designs that simulate but are not synthesizable. With synthesis,
each passing sample also gets ``area_ratio`` = reference cells / sample
cells, averaged as ``area_ratio_mean`` (v5.182; gate count, higher is
smaller).

Data is fetched from a pinned upstream commit into a cache directory with
a sha256 manifest (``fetch``); nothing is vendored. ``self_check`` renames
each reference to ``TopModule`` and must score 100%: it proves the local
toolchain before a model run spends GPU time.

Execution safety: samples are untrusted code run in a simulator. Each one
runs in its own temporary directory under a timeout. Run untrusted models'
samples inside a container (the RunPod pod is one).
"""
import hashlib
import json
import re
import shutil
import subprocess
import tempfile
import urllib.request
from pathlib import Path

SCHEMA_ID = "hdl-bench/1"
UPSTREAM = "NVlabs/verilog-eval"
COMMIT = "c498220d0a52248f8e3fdffe279075215bde2da6"
RAW = f"https://raw.githubusercontent.com/{UPSTREAM}/{COMMIT}"
TASK_DIR = "dataset_spec-to-rtl"
N_PROBLEMS = 156
SIM_TIMEOUT_S = 30
COMPILE_TIMEOUT_S = 60
IVERILOG_FLAGS = ("-Wall", "-Winfloop", "-Wno-timescale", "-g2012", "-s", "tb")

SYSTEM_MSG = ("You are a Verilog RTL designer that only writes code using "
              "correct Verilog syntax.")
NO_EXPLAIN_SUFFIX = ("Enclose your code with [BEGIN] and [DONE]. Only output "
                     "the code snippet\nand do NOT output anything else.\n")
SHOT_FILE = "scripts/verilog-example-prefix_spec-to-rtl_{n}-shot.txt"

_MISMATCH = re.compile(r"^Mismatches: (\d+) in \d+ samples$")
# sv-iv-analyze: compile/run log substrings checked line by line, in order;
# the first hit decides the code.
_LOG_CODES = (
    ("syntax error", "S"),
    ("error: This assignment requires an explicit cast", "e"),
    ("error: Sized numeric constant must have a size greater than zero", "0"),
    ("warning: always_comb process has no sensitivities", "n"),
    ("found no sensitivities so it will never trigger", "n"),
    ("is declared here as wire", "w"),
    ("Unknown module type", "m"),
    ("Unable to bind wire/reg/memory `clk'", "c"),
    ("TIMEOUT", "T"),
)
CODES = {
    ".": "pass", "S": "syntax error", "e": "explicit cast needed",
    "0": "zero-width constant", "n": "no sensitivities", "w": "reg declared as wire",
    "m": "unknown module", "c": "clk not bound", "T": "timeout",
    "p": "unbound wire/reg", "C": "other compile error",
    "r": "reset sensitivity (runtime mismatch)", "R": "runtime mismatch",
}


class HdlBenchError(ValueError):
    """Raised for a missing tool, a bad cache or a malformed problem."""


# ------------------------------------------------------------------ data

def _get(url):
    with urllib.request.urlopen(url, timeout=60) as r:
        return r.read()


def fetch(dest, limit=None, shots=(1,), get=_get):
    """Download the spec-to-rtl problems at ``COMMIT`` into ``dest``.

    Writes ``MANIFEST.json`` (commit, file sha256s). ``get(url) -> bytes``
    is injectable for tests. Returns the manifest dict.
    """
    dest = Path(dest)
    (dest / TASK_DIR).mkdir(parents=True, exist_ok=True)
    names = _get_text(get, f"{RAW}/{TASK_DIR}/problems.txt").split()
    if limit is None and len(names) != N_PROBLEMS:
        raise HdlBenchError(f"expected {N_PROBLEMS} problems, got {len(names)}")
    names = names[:limit] if limit else names
    files = {}
    rels = [f"{TASK_DIR}/{n}_{s}" for n in names
            for s in ("prompt.txt", "ref.sv", "test.sv")]
    rels += [SHOT_FILE.format(n=k) for k in shots]
    for rel in rels:
        data = get(f"{RAW}/{rel}")
        out = dest / rel
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_bytes(data)
        files[rel] = hashlib.sha256(data).hexdigest()
    manifest = {"schema": SCHEMA_ID, "upstream": UPSTREAM, "commit": COMMIT,
                "licence": "MIT", "problems": names, "sha256": files}
    (dest / "MANIFEST.json").write_text(json.dumps(manifest, indent=1) + "\n",
                                        encoding="utf-8")
    return manifest


def _get_text(get, url):
    return get(url).decode("utf-8")


def load(cache):
    """Load problems from a ``fetch`` cache, verifying every sha256."""
    cache = Path(cache)
    mpath = cache / "MANIFEST.json"
    if not mpath.is_file():
        raise HdlBenchError(f"no MANIFEST.json in {cache}; run fetch first")
    manifest = json.loads(mpath.read_text(encoding="utf-8"))
    if manifest.get("commit") != COMMIT:
        raise HdlBenchError(f"cache commit {manifest.get('commit')} != {COMMIT}")
    for rel, digest in manifest["sha256"].items():
        p = cache / rel
        if not p.is_file() or hashlib.sha256(p.read_bytes()).hexdigest() != digest:
            raise HdlBenchError(f"sha256 mismatch or missing: {rel}")
    probs = []
    for name in manifest["problems"]:
        base = cache / TASK_DIR / name
        rec = {"name": name}
        for key, suf in (("prompt", "prompt.txt"), ("ref", "ref.sv"),
                         ("test", "test.sv")):
            rec[key] = Path(f"{base}_{suf}").read_text(encoding="utf-8")
        if "module RefModule" not in rec["ref"]:
            raise HdlBenchError(f"{name}: reference has no RefModule")
        probs.append(rec)
    return probs


def shot_prefix(cache, shots):
    if not shots:
        return ""
    p = Path(cache) / SHOT_FILE.format(n=shots)
    if not p.is_file():
        raise HdlBenchError(f"{shots}-shot prefix not in cache: {p.name}")
    return p.read_text(encoding="utf-8")


# ------------------------------------------------------------------ prompt

def build_prompt(spec, prefix=""):
    """Return ``(system, user)`` exactly as upstream spec-to-rtl builds it."""
    user = prefix + "\nQuestion:\n" + spec.strip() + "\n"
    user = user.rstrip() + "\n" + NO_EXPLAIN_SUFFIX
    user += "\nAnswer:\n"
    return SYSTEM_MSG, user


def extract_code(text):
    """Pull the Verilog out of a response (upstream sv-generate rules)."""
    lines = text.splitlines()
    found, start, end = [], False, False
    for line in lines:
        if not start:
            if line.strip() == "[BEGIN]":
                start = True
            elif line.lstrip().startswith("[BEGIN]"):
                rest = line.lstrip().replace("[BEGIN]", "")
                if rest.rstrip().endswith("[DONE]"):
                    found.append(rest.rstrip().replace("[DONE]", ""))
                    start = end = True
                    break
                found.append(rest)
                start = True
        elif not end:
            if line.strip() == "[DONE]":
                end = True
            elif line.rstrip().endswith("[DONE]"):
                found.append(line.rstrip().replace("[DONE]", ""))
                end = True
            else:
                found.append(line)
    if start and end:
        return "\n".join(found) + "\n"
    # Fallback: inside the first backtick pair if there are two fences,
    # else everything up to the first endmodule.
    fences = sum(1 for ln in lines if ln.startswith("```"))
    out, first, second, endmod = [], False, False, False
    for line in lines:
        echo = True
        if fences >= 2:
            if not first or second:
                echo = False
        elif endmod:
            echo = False
        if line.startswith("```"):
            if not first:
                first = True
            else:
                second = True
            echo = False
        elif line.strip().startswith("endmodule"):
            endmod = True
        if echo:
            out.append(line)
    return "\n".join(out) + "\n"


# ------------------------------------------------------------------ scoring

def classify_log(log, sample=""):
    """Map a compile+run log (and the sample source) to an upstream code."""
    code, err_c, err_p, ok = "?", False, False, False
    for line in log.splitlines():
        hit = next((c for s, c in _LOG_CODES if s in line), None)
        if "error" in line:
            err_c = True
        if "Unable to bind wire/reg" in line:
            err_p = True
        if hit:
            code = hit
            break
        m = _MISMATCH.match(line.rstrip())
        if m and int(m.group(1)) == 0:
            ok = True
    if code == "?" and err_p:
        code = "p"
    if code == "?" and err_c:
        code = "C"
    if code == "?" and ok:
        code = "."
    if code == "?":
        resets = ("posedge reset", "negedge reset", "posedge r)")
        code = "r" if any(r in ln for ln in sample.splitlines()
                          for r in resets) else "R"
    return code


def need_tool(name):
    path = shutil.which(name)
    if not path:
        raise HdlBenchError(f"{name} not found on PATH (install it on the pod)")
    return path


def simulate(sample, prob, timeout=SIM_TIMEOUT_S):
    """Compile and run one sample against ``prob``'s testbench and reference.

    Returns ``{"code", "passed", "log"}``.
    """
    iverilog, vvp = need_tool("iverilog"), need_tool("vvp")
    with tempfile.TemporaryDirectory(prefix="hdlb_") as td:
        td = Path(td)
        for fname, text in (("sample.sv", sample), ("test.sv", prob["test"]),
                            ("ref.sv", prob["ref"])):
            (td / fname).write_text(text, encoding="utf-8")
        log = []
        try:
            c = subprocess.run([iverilog, *IVERILOG_FLAGS, "-o", "sim",
                                "sample.sv", "test.sv", "ref.sv"], cwd=td,
                               capture_output=True, text=True,
                               timeout=COMPILE_TIMEOUT_S)
            log.append(c.stdout + c.stderr)
            if c.returncode == 0:
                try:
                    r = subprocess.run([vvp, "-n", "sim"], cwd=td,
                                       capture_output=True, text=True,
                                       timeout=timeout)
                    log.append(r.stdout + r.stderr)
                except subprocess.TimeoutExpired as e:
                    log.append(_text(e.stdout) + "\nTIMEOUT\n")
        except subprocess.TimeoutExpired:
            log.append("compile error: iverilog timed out\n")
        text = "".join(log)
    code = classify_log(text, sample)
    return {"code": code, "passed": code == ".", "log": text[-4000:]}


def _text(b):
    if b is None:
        return ""
    return b.decode("utf-8", "replace") if isinstance(b, bytes) else b


def lint(sample):
    """pyslang lint: ``{"ok", "errors"}``, or None if pyslang is absent."""
    try:
        import pyslang
    except ImportError:
        return None
    tree = pyslang.SyntaxTree.fromText(sample)
    comp = pyslang.Compilation()
    comp.addSyntaxTree(tree)
    diags = [d for d in comp.getAllDiagnostics() if d.isError()]
    return {"ok": not diags, "errors": len(diags)}


def synth(sample, timeout=120):
    """yosys synthesis of TopModule: ``{"ok", "cells"}``, None if absent."""
    yosys = shutil.which("yosys")
    if not yosys:
        return None
    with tempfile.TemporaryDirectory(prefix="hdlb_") as td:
        p = Path(td) / "sample.sv"
        p.write_text(sample, encoding="utf-8")
        try:
            # -q also silences stat, so its report goes to a file (v5.184)
            r = subprocess.run([yosys, "-q", "-p",
                                "read_verilog -sv sample.sv; "
                                "synth -top TopModule; tee -q -o stat.txt stat"],
                               cwd=td, capture_output=True, text=True,
                               timeout=timeout)
        except subprocess.TimeoutExpired:
            return {"ok": False, "cells": None}
        st = Path(td) / "stat.txt"
        report = st.read_text(encoding="utf-8", errors="replace") if st.is_file() else ""
    return {"ok": r.returncode == 0, "cells": stat_cells(report)}


def stat_cells(report):
    """Cell count from a yosys ``stat`` report: ``Number of cells: N``
    (yosys <= 0.4x) or ``N cells`` (newer); the last one is the top."""
    found = re.findall(r"Number of cells:\s+(\d+)|^\s*(\d+)\s+cells\s*$", report, re.M)
    return int(next(g for g in found[-1] if g)) if found else None


def reference_sample(prob):
    """The reference renamed to TopModule: must always pass."""
    return re.sub(r"\bRefModule\b", "TopModule", prob["ref"])


def pass_at_k(n, c, k):
    """Unbiased pass@k (Chen et al. 2021) for n samples with c correct."""
    if n - c < k:
        return 1.0
    p = 1.0
    for i in range(n - c + 1, n + 1):
        p *= 1.0 - k / i
    return 1.0 - p


def score(probs, generate_fn, n=1, ks=(1,), prefix="", extra=False):
    """Run ``generate_fn(system, user, i) -> [text]*n`` on every problem.

    Returns the result dict (schema ``hdl-bench/1``).
    """
    rows, codes, ratios = [], {}, []
    for i, prob in enumerate(probs):
        system, user = build_prompt(prob["prompt"], prefix)
        texts = generate_fn(system, user, i)
        if len(texts) != n:
            raise HdlBenchError(f"{prob['name']}: got {len(texts)} samples, want {n}")
        samples = []
        ref = synth(reference_sample(prob)) if extra else None
        for t in texts:
            code = extract_code(t)
            sim = simulate(code, prob)
            codes[sim["code"]] = codes.get(sim["code"], 0) + 1
            s = {"code": sim["code"], "passed": sim["passed"]}
            if extra:
                s["lint"], s["synth"] = lint(code), synth(code)
                r = area_ratio(ref, s["synth"]) if s["passed"] else None
                if r is not None:
                    s["area_ratio"] = r
                    ratios.append(r)
            samples.append(s)
        c = sum(s["passed"] for s in samples)
        row = {"name": prob["name"], "n": n, "correct": c, "samples": samples}
        if ref is not None:
            row["ref_cells"] = ref["cells"]
        rows.append(row)
    out = {"schema": SCHEMA_ID, "bench": "verilog-eval-v2/spec-to-rtl",
           "commit": COMMIT, "n_problems": len(rows), "n": n,
           "failure_codes": dict(sorted(codes.items())), "problems": rows}
    for k in ks:
        if k <= n:
            out[f"pass@{k}"] = (sum(pass_at_k(n, r["correct"], k) for r in rows)
                                / max(len(rows), 1))
    if ratios:
        out["area_ratio_mean"] = sum(ratios) / len(ratios)
        out["area_ratio_n"] = len(ratios)
    return out


def area_ratio(ref, sample):
    """Reference cells / sample cells after yosys ``synth`` (v5.182): above
    1 the passing sample is smaller than the reference design. None when
    either side did not synthesize or has no cells."""
    if not ref or not sample or not ref.get("cells") or not sample.get("cells"):
        return None
    if not (ref["ok"] and sample["ok"]):
        return None
    return ref["cells"] / sample["cells"]


def self_check(probs):
    """Score the references themselves; returns ``(passed, failures)``."""
    fails = []
    for prob in probs:
        r = simulate(reference_sample(prob), prob)
        if not r["passed"]:
            fails.append((prob["name"], r["code"]))
    return len(probs) - len(fails), fails


def main(argv=None):
    import argparse
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("cmd", choices=("fetch", "self-check"))
    ap.add_argument("--cache", default="data/hdl_bench/verilog_eval_v2")
    ap.add_argument("--limit", type=int)
    a = ap.parse_args(argv)
    if a.cmd == "fetch":
        m = fetch(a.cache, a.limit)
        print(f"fetched {len(m['problems'])} problems @ {COMMIT[:12]} -> {a.cache}")
        return 0
    probs = load(a.cache)
    ok, fails = self_check(probs)
    print(f"self-check: {ok}/{len(probs)} references pass")
    for name, code in fails:
        print(f"  FAIL {name}: {code} ({CODES.get(code, '?')})")
    return 0 if not fails else 1


if __name__ == "__main__":
    raise SystemExit(main())
