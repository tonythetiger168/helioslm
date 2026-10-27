"""HDL benchmark harness: RTLLM v2.1 (v5.175, HDL track H0).

RTLLM (hkust-zhiyao/RTLLM, MIT) has 50 designs (adders to a RISC-V ALU
and a calendar). Each one has a natural-language ``design_description.txt``
that names the module and its ports, a ``testbench.v`` that prints
"Passed" on success, a ``verified_<name>.v`` reference and, for a few
designs, data files the testbench reads.

Upstream scores with Synopsys VCS: a sample counts for *syntax* when it
compiles with the testbench, and for *function* when the simulation output
contains "Pass" or "pass" (``auto_run.py``). VCS is commercial, so this
harness uses Icarus Verilog (``iverilog -g2012`` + ``vvp``) with the same
two rules. Numbers are therefore "RTLLM v2.1 (Icarus)" and are only
compared with our own runs; ``self_check`` lists the references Icarus
cannot run so they can be excluded by name rather than silently scored 0.

Upstream's pass@k divides by 51 for 50 designs (it appends a 0); this
harness reports the plain mean over designs.

Data is fetched at a pinned commit with git (``fetch``) and a sha256
manifest of every file used; nothing is vendored. Samples are untrusted
code: run them only inside a container (the RunPod pod is one).
"""
import hashlib
import json
import re
import shutil
import subprocess
import tempfile
from pathlib import Path

from helioslm_v5.eval import hdl_bench as hb

SCHEMA_ID = "rtllm-bench/1"
UPSTREAM = "https://github.com/hkust-zhiyao/RTLLM"
COMMIT = "51ed553d0ffd32797a1a0a13e051656bf302c81f"
N_DESIGNS = 50
SIM_TIMEOUT_S = 30
COMPILE_TIMEOUT_S = 60
IVERILOG_FLAGS = ("-g2012", "-Wno-timescale")
# upstream auto_run.py order
DESIGNS = (
    "adder_8bit", "adder_16bit", "adder_32bit", "adder_pipe_64bit", "adder_bcd",
    "sub_64bit", "multi_8bit", "multi_16bit", "multi_booth_8bit",
    "multi_pipe_4bit", "multi_pipe_8bit", "div_16bit", "radix2_div",
    "comparator_3bit", "comparator_4bit", "accu", "fixed_point_adder",
    "fixed_point_substractor", "float_multi", "asyn_fifo", "LIFObuffer",
    "right_shifter", "LFSR", "barrel_shifter", "fsm", "sequence_detector",
    "counter_12", "JC_counter", "ring_counter", "up_down_counter",
    "signal_generator", "square_wave", "clkgenerator", "instr_reg", "ROM", "RAM",
    "alu", "pe", "freq_div", "freq_divbyeven", "freq_divbyodd", "freq_divbyfrac",
    "calendar", "traffic_light", "width_8to16", "synchronizer", "edge_detect",
    "pulse_detect", "parallel2serial", "serial2parallel")
_SKIP = ("makefile", "design_description.txt")


class RtllmError(ValueError):
    """Raised for a missing tool, a bad cache or a malformed design."""


# ------------------------------------------------------------------ data

def _git(args, cwd):
    subprocess.run(["git", *args], cwd=cwd, check=True, capture_output=True,
                   text=True, timeout=600)


def clone(dest):
    """Shallow-fetch ``COMMIT`` of the upstream repo into ``dest/repo``."""
    repo = Path(dest) / "repo"
    repo.mkdir(parents=True, exist_ok=True)
    _git(["init", "-q"], repo)
    _git(["fetch", "-q", "--depth", "1", UPSTREAM, COMMIT], repo)
    _git(["checkout", "-q", "FETCH_HEAD"], repo)
    return repo


def build_cache(repo, dest, designs=DESIGNS):
    """Copy each design's files from a checkout into ``dest/<design>/``.

    Writes ``MANIFEST.json`` with a sha256 per file. Returns the manifest.
    """
    repo, dest = Path(repo), Path(dest)
    found = {}
    for mk in repo.rglob("makefile"):
        if any(p.startswith("_") for p in mk.relative_to(repo).parts):
            continue
        found[mk.parent.name] = mk.parent
    missing = [d for d in designs if d not in found]
    if missing:
        raise RtllmError(f"designs not in checkout: {missing}")
    files = {}
    for d in designs:
        src = found[d]
        out = dest / d
        out.mkdir(parents=True, exist_ok=True)
        for f in sorted(src.iterdir()):
            if f.is_file() and f.name != "makefile":
                data = f.read_bytes()
                (out / f.name).write_bytes(data)
                files[f"{d}/{f.name}"] = hashlib.sha256(data).hexdigest()
        for need in ("design_description.txt", "testbench.v"):
            if f"{d}/{need}" not in files:
                raise RtllmError(f"{d}: missing {need}")
        refs = [f for f in files if f.startswith(f"{d}/verified_")]
        if len(refs) != 1:
            raise RtllmError(f"{d}: expected one verified_*.v, got {refs}")
    manifest = {"schema": SCHEMA_ID, "upstream": UPSTREAM, "commit": COMMIT,
                "licence": "MIT", "designs": list(designs), "sha256": files}
    (dest / "MANIFEST.json").write_text(json.dumps(manifest, indent=1) + "\n",
                                        encoding="utf-8")
    return manifest


def fetch(dest):
    """Clone at ``COMMIT`` and build the cache in ``dest``."""
    with tempfile.TemporaryDirectory(prefix="rtllm_") as td:
        m = build_cache(clone(td), dest)
    if len(m["designs"]) != N_DESIGNS:
        raise RtllmError(f"expected {N_DESIGNS} designs, got {len(m['designs'])}")
    return m


def load(cache):
    """Load designs from a cache, verifying every sha256."""
    cache = Path(cache)
    mpath = cache / "MANIFEST.json"
    if not mpath.is_file():
        raise RtllmError(f"no MANIFEST.json in {cache}; run fetch first")
    manifest = json.loads(mpath.read_text(encoding="utf-8"))
    if manifest.get("commit") != COMMIT:
        raise RtllmError(f"cache commit {manifest.get('commit')} != {COMMIT}")
    for rel, digest in manifest["sha256"].items():
        p = cache / rel
        if not p.is_file() or hashlib.sha256(p.read_bytes()).hexdigest() != digest:
            raise RtllmError(f"sha256 mismatch or missing: {rel}")
    out = []
    for d in manifest["designs"]:
        base = cache / d
        aux = {f.name: f.read_bytes() for f in sorted(base.iterdir())
               if f.name not in _SKIP and f.name != "testbench.v"
               and not f.name.startswith("verified_")}
        ref = next(base.glob("verified_*.v"))
        out.append({"name": d,
                    "prompt": (base / "design_description.txt").read_text(encoding="utf-8"),
                    "test": (base / "testbench.v").read_text(encoding="utf-8"),
                    "ref": ref.read_text(encoding="utf-8"),
                    "top": top_name((base / "design_description.txt")
                                    .read_text(encoding="utf-8")),
                    "aux": aux})
    return out


# ------------------------------------------------------------------ scoring

def top_name(description):
    """The module name the description asks for (and the testbench uses)."""
    m = re.search(r"Module name:\s*(\w+)", description)
    if not m:
        raise RtllmError("description has no 'Module name:'")
    return m.group(1)


def reference_sample(design):
    """The reference with its top module renamed to the requested name.

    References name the top ``verified_<x>`` or something else again
    (``verified_multi_pipe``, ``verified_adder_64bit``); the top is the
    module no other module in the file instantiates.
    """
    ref, top = design["ref"], design["top"]
    defs = re.findall(r"^\s*module\s+(\w+)", ref, re.M)
    used = {d for d in defs if len(re.findall(rf"\b{d}\b", ref)) > 1}
    roots = [d for d in defs if d not in used] or defs
    return re.sub(rf"\b{roots[-1]}\b", top, ref)


def simulate(sample, design, timeout=SIM_TIMEOUT_S):
    """Compile ``sample`` with the design's testbench and run it.

    Returns ``{"syntax", "passed", "log"}`` (upstream's two rules).
    """
    iverilog, vvp = hb.need_tool("iverilog"), hb.need_tool("vvp")
    with tempfile.TemporaryDirectory(prefix="rtllm_") as td:
        td = Path(td)
        (td / "design.v").write_text(sample, encoding="utf-8")
        (td / "testbench.v").write_text(design["test"], encoding="utf-8")
        for name, data in design["aux"].items():
            (td / name).write_bytes(data)
        log, syntax, out = [], False, ""
        try:
            c = subprocess.run([iverilog, *IVERILOG_FLAGS, "-o", "sim",
                                "design.v", "testbench.v"], cwd=td,
                               capture_output=True, text=True,
                               timeout=COMPILE_TIMEOUT_S)
            log.append(c.stdout + c.stderr)
            syntax = c.returncode == 0 and (td / "sim").is_file()
            if syntax:
                try:
                    r = subprocess.run([vvp, "-n", "sim"], cwd=td,
                                       capture_output=True, text=True,
                                       timeout=timeout)
                    out = r.stdout
                    log.append(r.stdout + r.stderr)
                except subprocess.TimeoutExpired as e:
                    log.append(hb._text(e.stdout) + "\nTIMEOUT\n")
        except subprocess.TimeoutExpired:
            log.append("compile timed out\n")
    passed = syntax and ("Pass" in out or "pass" in out)
    return {"syntax": syntax, "passed": passed, "log": "".join(log)[-4000:]}


def extract_code(text):
    """Code from a response: a [BEGIN]/[DONE] or fenced block, else the span
    from the first ``module`` to the last ``endmodule``.

    VerilogEval's fallback stops at the first ``endmodule``; RTLLM answers
    often carry helper modules, so the fallback keeps all of them.
    """
    fences = [i for i, ln in enumerate(text.splitlines()) if ln.startswith("```")]
    if "[BEGIN]" in text or len(fences) >= 2:
        return hb.extract_code(text)
    m = re.search(r"\bmodule\b", text)
    end = text.rfind("endmodule")
    if not m or end < m.start():
        return text
    return text[m.start():end + len("endmodule")] + "\n"


def build_prompt(description):
    """Upstream sends the description as the whole user message."""
    return description.strip() + "\n"


def score(designs, generate_fn, n=1, ks=(1,), exclude=()):
    """Run ``generate_fn(system, user, i) -> [text]*n`` on every design."""
    rows = []
    for i, d in enumerate(designs):
        if d["name"] in exclude:
            continue
        texts = generate_fn(hb.SYSTEM_MSG, build_prompt(d["prompt"]), i)
        if len(texts) != n:
            raise RtllmError(f"{d['name']}: got {len(texts)} samples, want {n}")
        sims = [simulate(extract_code(t), d) for t in texts]
        rows.append({"name": d["name"], "n": n,
                     "syntax": sum(s["syntax"] for s in sims),
                     "correct": sum(s["passed"] for s in sims)})
    out = {"schema": SCHEMA_ID, "bench": "rtllm-v2.1 (icarus)", "commit": COMMIT,
           "n_designs": len(rows), "n": n, "excluded": sorted(exclude),
           "designs": rows}
    m = max(len(rows), 1)
    for k in ks:
        if k <= n:
            out[f"syntax@{k}"] = sum(hb.pass_at_k(n, r["syntax"], k) for r in rows) / m
            out[f"pass@{k}"] = sum(hb.pass_at_k(n, r["correct"], k) for r in rows) / m
    return out


def self_check(designs):
    """Run each reference; returns ``(passed, failures)``."""
    fails = []
    for d in designs:
        r = simulate(reference_sample(d), d)
        if not r["passed"]:
            fails.append((d["name"], "syntax" if not r["syntax"] else "function"))
    return len(designs) - len(fails), fails


def main(argv=None):
    import argparse
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("cmd", choices=("fetch", "self-check"))
    ap.add_argument("--cache", default="data/hdl_bench/rtllm_v2")
    a = ap.parse_args(argv)
    if a.cmd == "fetch":
        if not shutil.which("git"):
            raise RtllmError("git not found")
        m = fetch(a.cache)
        print(f"fetched {len(m['designs'])} designs @ {COMMIT[:12]} -> {a.cache}")
        return 0
    designs = load(a.cache)
    ok, fails = self_check(designs)
    print(f"self-check: {ok}/{len(designs)} references pass under Icarus")
    for name, why in fails:
        print(f"  FAIL {name}: {why}")
    if fails:
        print("exclude with: --exclude " + ",".join(n for n, _ in fails))
    return 0 if not fails else 1


if __name__ == "__main__":
    raise SystemExit(main())
