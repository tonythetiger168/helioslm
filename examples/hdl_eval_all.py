"""hdl_eval_all.py — HDL track: score one model on VerilogEval v2,
RTLLM v2.1 and CVDP in one vLLM batch (v5.178), optionally the UVM
testbench benchmark (``--benches ...,uvm``, v5.182).

  # on a GPU pod with vLLM, iverilog, cocotb==2.0.1, pytest==8.3.2
  python -m helioslm_v5.eval.hdl_bench fetch
  python -m helioslm_v5.eval.rtllm_bench fetch
  python -m helioslm_v5.eval.cvdp_bench fetch
  python examples/hdl_eval_all.py --model google/gemma-4-E4B-it \\
      --out results/hdl/e4b_it.json

All prompts of the three benchmarks go to vLLM at once (greedy, or ``--n``
samples at ``--temperature``); simulation then runs on the CPU cores.
The prompts are built exactly as each benchmark's own harness builds them
(``hdl_bench_run.py`` is the transformers path for one benchmark). The
output holds one ``hdl-bench/1``-style result per benchmark plus a
summary line. ``--model`` may also be a local checkpoint directory.
``uvm`` needs Verilator >= 5.052 and ``uvm_bench fetch`` + ``prepare``;
its prompts get at least ``--uvm-new-tokens`` (a UVM testbench is long).
``--extra`` adds yosys gate counts (``area_ratio_mean``) to VerilogEval.
``--repair-rounds N`` (v5.229) is the agent loop: a VerilogEval / RTLLM
answer that fails an iverilog compile of the answer alone (no testbench,
so no expected outputs leak) goes back to the model with the tool
messages, up to N times; the first-round generations are saved to
``<out>.round0.samples.json`` and the per-round stats under ``repair``.
``--select consensus --n 8`` (v5.241) submits one answer per problem
without the answer key: the compiling sample that is random-simulation
equivalent to the most other samples (CVDP: the first that passes its
compile check); all samples go to ``<out>.all.samples.json``.
``--select tb`` (v5.242) ranks the samples by ``--tb-n`` self-checking
testbenches the model writes from the problem text alone (CodeT-style),
then by the same simulation votes; the testbenches go to ``<out>.tbs.json``.
The raw generations are written to ``<out>.samples.json`` before any
scoring, so a scoring failure never costs the GPU time (v5.179).
"""
import argparse
import json
import re
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from helioslm_v5.eval import cvdp_bench as cb  # noqa: E402
from helioslm_v5.eval import hdl_bench as hb  # noqa: E402
from helioslm_v5.eval import rtllm_bench as rb  # noqa: E402
from helioslm_v5.eval import uvm_bench as ub  # noqa: E402

BENCHES = ("verilog-eval", "rtllm", "cvdp", "uvm")
DEFAULT_BENCHES = ("verilog-eval", "rtllm", "cvdp")


def collect(benches, ve_cache, rtllm_cache, cvdp_file, exclude=(),
            uvm_cache="data/hdl_bench/uvm"):
    """[(bench, key, system, user)] for every problem, and the loaded data."""
    jobs, data = [], {}
    if "verilog-eval" in benches:
        data["verilog-eval"] = probs = hb.load(ve_cache)
        for i, p in enumerate(probs):
            jobs.append(("verilog-eval", i, *hb.build_prompt(p["prompt"])))
    if "rtllm" in benches:
        data["rtllm"] = designs = rb.load(rtllm_cache)
        for i, d in enumerate(designs):
            if d["name"] not in exclude:
                jobs.append(("rtllm", i, hb.SYSTEM_MSG, rb.build_prompt(d["prompt"])))
    if "cvdp" in benches:
        rows, skipped = cb.load(cvdp_file)
        data["cvdp"] = rows
        data["cvdp_skipped"] = len(skipped)
        for r in rows:
            jobs.append(("cvdp", r["id"], *cb.build_prompt(r)))
    if "uvm" in benches:
        data["uvm"] = tasks = ub.load(Path(uvm_cache) / "tasks.json")
        data["uvm_dir"] = ub.uvm_dir(uvm_cache)
        for t in tasks:
            jobs.append(("uvm", t["name"], *ub.build_prompt(t)))
    return jobs, data


def score_all(jobs, outputs, data, n, ks, exclude=(), workers=8, extra=False):
    """Score generated ``outputs`` (aligned with ``jobs``) per benchmark.
    A benchmark whose scoring raises gets ``{"error": ...}`` and the
    others are still scored (v5.179)."""
    got = {(b, k): o for (b, k, _, _), o in zip(jobs, outputs)}
    runs = {
        "verilog-eval": lambda: hb.score(
            data["verilog-eval"], lambda s, u, i: got[("verilog-eval", i)],
            n=n, ks=ks, extra=extra),
        "rtllm": lambda: rb.score(
            data["rtllm"], lambda s, u, i: got[("rtllm", i)], n=n, ks=ks,
            exclude=tuple(exclude)),
        "cvdp": lambda: dict(cb.score(
            data["cvdp"], {r["id"]: got[("cvdp", r["id"])] for r in data["cvdp"]},
            ks=ks, workers=workers), skipped_unsupported=data["cvdp_skipped"]),
        "uvm": lambda: ub.score(
            data["uvm"], {t["name"]: got[("uvm", t["name"])] for t in data["uvm"]},
            data.get("uvm_dir"), ks=ks, workers=min(workers, ub.default_workers())),
    }
    res = {}
    for b, fn in runs.items():
        if b in data:
            try:
                res[b] = fn()
            except Exception as e:          # keep the other benchmarks
                res[b] = {"error": f"{type(e).__name__}: {e}"[:500]}
    return res


def summary(res):
    out = {}
    for b, r in res.items():
        out[b] = {k: round(v, 4) for k, v in r.items()
                  if k.startswith(("pass@", "syntax@", "build@", "golden@",
                                   "kill_rate", "area_ratio_mean"))
                  and isinstance(v, float)}
    return out


def chat_prompt(tok, system, user, thinking=None):
    """The rendered prompt. ``user`` is a string, or a list of chat
    messages after the system one (multi-turn repair, v5.229).
    ``thinking`` (True / False) is passed as ``enable_thinking`` to the
    template; None leaves the template's default."""
    msgs = user if isinstance(user, list) else [{"role": "user", "content": user}]
    if not tok.chat_template:
        return system + "\n" + "\n".join(m["content"] for m in msgs)
    kw = {} if thinking is None else {"enable_thinking": thinking}
    try:
        return tok.apply_chat_template(
            [{"role": "system", "content": system}, *msgs],
            tokenize=False, add_generation_prompt=True, **kw)
    except Exception:       # templates without a system role
        first = dict(msgs[0], content=system + "\n\n" + msgs[0]["content"])
        return tok.apply_chat_template([first, *msgs[1:]], tokenize=False,
                                       add_generation_prompt=True, **kw)


REASONING_END = ("<|channel|>final<|message|>", "<channel|>", "</think>")
SPECIAL = re.compile(r"<\|[^<>\s]*>|<[^<>\s|]*\|>")


def strip_reasoning(text):
    """The answer after a thinking-mode response's reasoning (v5.233):
    everything after the last end-of-reasoning marker (Gemma 4
    ``<channel|>``, gpt-oss ``<|channel|>final<|message|>`` (v5.235),
    ``</think>``), with leftover chat special tokens
    removed. A response with no marker is kept whole."""
    for mark in REASONING_END:
        if mark in text:
            text = text.rsplit(mark, 1)[1]
            break
    return SPECIAL.sub("", text).lstrip()


class VllmGen:
    """One vLLM engine reused across generation rounds. With thinking on,
    special tokens are kept so the reasoning can be cut off
    (``strip_reasoning``); vLLM otherwise drops the Gemma 4 channel
    markers and the reasoning ends up in the scored answer."""

    def __init__(self, model, max_model_len, gpu_mem, thinking=None):
        self.thinking = thinking
        from vllm import LLM
        self.llm = LLM(model=model, dtype="bfloat16", max_model_len=max_model_len,
                       gpu_memory_utilization=gpu_mem, trust_remote_code=False)
        self.tok = self.llm.get_tokenizer()
        self.max_model_len = max_model_len

    def __call__(self, jobs, n, temperature, top_p, max_new_tokens,
                 uvm_new_tokens=None):
        from vllm import SamplingParams
        prompts, sps = [], []
        for bench, _, system, user in jobs:
            new = (max(max_new_tokens, uvm_new_tokens or 0) if bench == "uvm"
                   else max_new_tokens)
            sps.append(SamplingParams(n=n, temperature=temperature,
                                      top_p=top_p if temperature > 0 else 1.0,
                                      max_tokens=new,
                                      skip_special_tokens=not self.thinking))
            ids = self.tok(chat_prompt(self.tok, system, user, self.thinking),
                           add_special_tokens=False).input_ids
            prompts.append({"prompt_token_ids": ids[-(self.max_model_len - new):]})
        outs = self.llm.generate(prompts, sps)
        if self.thinking:
            return [[strip_reasoning(c.text) for c in o.outputs] for o in outs]
        return [[c.text for c in o.outputs] for o in outs]


def vllm_generate(model, jobs, n, temperature, top_p, max_new_tokens,
                  max_model_len, gpu_mem, uvm_new_tokens=None):
    return VllmGen(model, max_model_len, gpu_mem)(
        jobs, n, temperature, top_p, max_new_tokens, uvm_new_tokens)


REPAIR_BENCHES = ("verilog-eval", "rtllm")
REPAIR_ASK = ("The open-source tools reported these problems with your "
              "answer:\n\n{feedback}\n\nFix them. Return the complete "
              "corrected Verilog module, with no other text.")


def repair(gen, jobs, outputs, rounds, check, benches=REPAIR_BENCHES,
           temperature=0.0, top_p=1.0, max_new_tokens=2048, workers=8,
           bench_checks=None):
    """Agent loop with open-tool feedback only (v5.229): every sample of
    a ``benches`` job whose extracted code fails ``check`` (``hb.
    tool_feedback``: compile alone, no testbench) is sent back to the
    model with the tool messages, up to ``rounds`` times. Returns the new
    outputs (same shape) and per-round stats {"round", "failing",
    "fixed"}. ``gen(jobs, n, temperature, top_p, max_new_tokens)`` is a
    ``VllmGen`` or any stand-in. ``bench_checks`` (v5.234) maps more
    benches to their own ``check(key, raw answer)``, e.g. CVDP
    (``cvdp_check``)."""
    from concurrent.futures import ThreadPoolExecutor
    outputs = [list(o) for o in outputs]
    hist = {}                                       # (job, sample) -> msgs
    stats = []

    bench_checks = bench_checks or {}

    def judge(ij):
        b, k = jobs[ij[0]][:2]
        text = outputs[ij[0]][ij[1]]
        if b in bench_checks:
            return bench_checks[b](k, text)
        return check(hb.extract_code(text))
    todo = [(i, j) for i, (b, _, _, _) in enumerate(jobs)
            if b in benches or b in bench_checks
            for j in range(len(outputs[i]))]
    with ThreadPoolExecutor(max(1, workers)) as ex:
        verdict = dict(zip(todo, ex.map(judge, todo)))
        for r in range(1, rounds + 1):
            failing = [ij for ij in todo if not verdict[ij]["ok"]]
            if not failing:
                stats.append({"round": r, "failing": 0, "fixed": 0})
                break
            rjobs = []
            for i, j in failing:
                b, k, system, user = jobs[i]
                msgs = hist.get((i, j)) or [{"role": "user", "content": user}]
                msgs = msgs + [
                    {"role": "assistant", "content": outputs[i][j]},
                    {"role": "user", "content": REPAIR_ASK.format(
                        feedback=verdict[(i, j)]["feedback"] or "compile failed")}]
                hist[(i, j)] = msgs
                rjobs.append((b, k, system, msgs))
            new = gen(rjobs, 1, temperature, top_p, max_new_tokens)
            for (i, j), o in zip(failing, new):
                outputs[i][j] = o[0]
            again = list(ex.map(judge, failing))
            for ij, v in zip(failing, again):
                verdict[ij] = v
            stats.append({"round": r, "failing": len(failing),
                          "fixed": sum(v["ok"] for v in again)})
    return outputs, stats


def consensus_pick(codes, ok, equiv=None):
    """Index of the candidate to submit (v5.241): among the ``codes`` whose
    compile check passed (``ok``), the one random-simulation equivalent
    (``hdl_reward.simulate`` of one candidate against another, no golden
    design or benchmark testbench) to the most other candidates; the
    first such on a tie. Without any compiling candidate, 0."""
    good = [i for i, o in enumerate(ok) if o]
    if len(good) <= 1:
        return good[0] if good else 0
    votes = consensus_votes(codes, good, equiv)
    return max(good, key=lambda i: (votes[i], -i))


def sim_equiv(a, b):
    """True when two candidate designs agree under ``hdl_reward``'s
    random-stimulus simulation (same top module and ports)."""
    from helioslm_v5.eval import hdl_reward as hr
    try:
        name = hr.top_module(a)
        if hr.top_module(b) != name:
            return False
        ports = hr.top_ports(a, name)
        if hr.top_ports(b, name) != ports:
            return False
        return hr.simulate(b, a, name, ports)["ok"]
    except Exception:
        return False


def consensus_votes(codes, good, equiv=None):
    """{i: number of other ``good`` candidates equivalent to candidate i}."""
    equiv = equiv or sim_equiv
    votes = {i: 0 for i in good}
    for x, i in enumerate(good):
        for j in good[x + 1:]:
            if equiv(codes[i], codes[j]):
                votes[i] += 1
                votes[j] += 1
    return votes


TB_ASK = """Write a self-checking testbench for the hardware design described below. Do not write the design itself.

Requirements:
- One Verilog module named `tb` with no ports. It instantiates the design module with the exact module name and port names given in the description.
- Work out the expected output values yourself from the specification (for a clocked design, model the expected behaviour cycle by cycle inside the testbench). Apply directed corner cases and at least 50 random stimuli.
- Count every comparison in an integer `checks` and every mismatch in an integer `errors`.
- At the end print exactly: $display("ERRORS %0d CHECKS %0d", errors, checks); then call $finish.
- Plain Verilog-2005 / SystemVerilog-2012 that Icarus Verilog (iverilog -g2012) compiles.

Design description:

{spec}

Return only the testbench, in one ```verilog block."""

_TB_RESULT = re.compile(r"ERRORS\s+(\d+)\s+CHECKS\s+(\d+)")


def run_tb(code, tb, timeout=20):
    """(errors, checks) of design ``code`` under the model-written
    testbench ``tb`` (Icarus), or None when it does not compile, run or
    report."""
    import shutil
    import subprocess
    import tempfile
    iverilog, vvp = shutil.which("iverilog"), shutil.which("vvp")
    if not (iverilog and vvp and code.strip() and tb.strip()):
        return None
    with tempfile.TemporaryDirectory(prefix="hdltb_") as td:
        Path(td, "dut.v").write_text(code, encoding="utf-8")
        Path(td, "tb.v").write_text(tb, encoding="utf-8")
        try:
            c = subprocess.run([iverilog, "-g2012", "-s", "tb", "-o", "sim", "dut.v", "tb.v"],
                               cwd=td, capture_output=True, text=True, timeout=60)
            if c.returncode:
                return None
            r = subprocess.run([vvp, "-n", "sim"], cwd=td, capture_output=True, text=True,
                               timeout=timeout)
        except subprocess.TimeoutExpired:
            return None
    m = _TB_RESULT.findall(r.stdout + r.stderr)
    if not m:
        return None
    e, n = map(int, m[-1])
    return (e, n) if n > 0 else None


def tb_pick(codes, ok, tbs, run=run_tb, equiv=None):
    """Index of the candidate to submit (v5.242, CodeT-style): candidates
    that compile are ranked by the number of model-written testbenches
    they pass with zero errors, then by the fraction of checks passed,
    then by :func:`consensus_votes`. A testbench counts only if at least
    one candidate runs under it. Returns (index, {"valid_tbs", "best_pass"})."""
    good = [i for i, o in enumerate(ok) if o]
    if not good:
        return 0, {"valid_tbs": 0, "best_pass": 0}
    res = {(i, t): run(codes[i], tb) for i in good for t, tb in enumerate(tbs)}
    valid = [t for t in range(len(tbs)) if any(res[(i, t)] for i in good)]
    def score(i):
        rs = [res[(i, t)] for t in valid]
        full = sum(1 for r in rs if r and r[0] == 0)
        frac = sum(1 - r[0] / r[1] for r in rs if r)
        return full, frac
    sc = {i: score(i) for i in good}
    top = max(sc.values())
    tied = [i for i in good if sc[i] == top]
    if len(tied) > 1:
        votes = consensus_votes(codes, tied, equiv)
        pick = max(tied, key=lambda i: (votes[i], -i))
    else:
        pick = tied[0]
    return pick, {"valid_tbs": len(valid), "best_pass": top[0]}


def select_tb(gen, jobs, outputs, check, n_tb=2, temperature=0.8, top_p=0.95,
              max_new_tokens=4096, benches=REPAIR_BENCHES, bench_checks=None, workers=8):
    """Like :func:`select_consensus`, but VerilogEval / RTLLM jobs are
    picked with :func:`tb_pick` on ``n_tb`` testbenches the model writes
    from the problem text alone (no golden design, no benchmark
    testbench). Returns (outputs with one sample each, picks, stats)."""
    from concurrent.futures import ThreadPoolExecutor
    idx = [i for i, (b, _, _, u) in enumerate(jobs) if b in benches and isinstance(u, str)]
    tb_jobs = [(jobs[i][0], jobs[i][1], "", TB_ASK.format(spec=jobs[i][3])) for i in idx]
    from helioslm_v5.eval import hdl_reward as hr
    tbs = {i: [hr.extract_answer(t) for t in o]
           for i, o in zip(idx, gen(tb_jobs, n_tb, temperature, top_p, max_new_tokens))} \
        if tb_jobs else {}
    out, picks = select_consensus(jobs, outputs, check, benches=(), bench_checks=bench_checks,
                                  workers=workers)
    info = {}

    def one(i):
        codes = [hb.extract_code(o) for o in outputs[i]]
        return i, tb_pick(codes, [check(c)["ok"] for c in codes], tbs[i])
    with ThreadPoolExecutor(max(1, workers)) as ex:
        for i, (j, st) in ex.map(one, idx):
            picks[i] = j
            out[i] = [outputs[i][j]]
            info[i] = st
    stats = {"tb_jobs": len(idx), "with_valid_tb": sum(1 for v in info.values() if v["valid_tbs"]),
             "picked_full_pass": sum(1 for v in info.values() if v["best_pass"] > 0)}
    return out, picks, stats, {jobs[i][1]: tbs[i] for i in idx}


def select_consensus(jobs, outputs, check, benches=REPAIR_BENCHES, bench_checks=None,
                     workers=8):
    """Best-of-n without the answer key (v5.241): one output per job. A
    ``benches`` job keeps :func:`consensus_pick` of its samples' extracted
    code (``check`` = ``hb.tool_feedback`` compile); a ``bench_checks``
    job (CVDP) keeps its first sample that passes its own check; any
    other job keeps sample 0. Returns (outputs with one sample each,
    picked indices)."""
    from concurrent.futures import ThreadPoolExecutor
    bench_checks = bench_checks or {}

    def one(i):
        b, k = jobs[i][:2]
        outs = outputs[i]
        if b in bench_checks:
            for j, o in enumerate(outs):
                if bench_checks[b](k, o)["ok"]:
                    return j
            return 0
        if b not in benches:
            return 0
        codes = [hb.extract_code(o) for o in outs]
        return consensus_pick(codes, [check(c)["ok"] for c in codes])
    with ThreadPoolExecutor(max(1, workers)) as ex:
        picks = list(ex.map(one, range(len(jobs))))
    return [[o[j]] for o, j in zip(outputs, picks)], picks


HDL_SUFFIXES = (".sv", ".v", ".svh", ".vh")


def cvdp_check(rows, compile_fn=None, max_lines=12):
    """Open-tool check for single-file CVDP answers (v5.234): the file the
    upstream harness would write (``cb.parse_response``) is compiled with
    iverilog together with the RTL context files given in the prompt
    (never the test harness). Only messages about the answer's own file
    count; multi-file (JSON) answers and non-HDL targets always pass.
    ``compile_fn(dir, target, others) -> log or None`` replaces iverilog
    in tests."""
    import subprocess
    import tempfile

    def run_iverilog(td, target, others):
        iverilog = hb.need_tool("iverilog")
        try:
            c = subprocess.run([iverilog, *hb.REPAIR_FLAGS, "-o", "out",
                                *sorted({f"-I{Path(td, o).parent}"
                                         for o in [*others, target]}),
                                *others, target], cwd=td, capture_output=True,
                               text=True, timeout=hb.COMPILE_TIMEOUT_S)
        except subprocess.TimeoutExpired:
            return f"{target}: iverilog timed out (check for infinite loops)"
        return None if c.returncode == 0 else c.stdout + c.stderr
    compile_fn = compile_fn or run_iverilog

    def check(key, text):
        row = rows.get(key)
        files = list(row["output"]["context"]) if row else []
        if len(files) != 1 or not files[0].endswith(HDL_SUFFIXES):
            return {"ok": True, "feedback": ""}
        target = files[0]
        code = cb.parse_response(text, files).get(target, "")
        if "module" not in code and "package" not in code:
            return {"ok": False, "feedback": f"No Verilog was found for {target}. "
                    "Write the complete file contents only."}
        with tempfile.TemporaryDirectory(prefix="cvdpr_") as td:
            others = []
            for name, body in row["input"]["context"].items():
                if (name == target or not name.endswith(HDL_SUFFIXES)
                        or not cb.safe_name(name) or not name.startswith("rtl/")):
                    continue
                Path(td, name).parent.mkdir(parents=True, exist_ok=True)
                Path(td, name).write_text(body, encoding="utf-8")
                if name.endswith((".sv", ".v")):
                    others.append(name)
            Path(td, target).parent.mkdir(parents=True, exist_ok=True)
            Path(td, target).write_text(code, encoding="utf-8")
            others.sort(key=lambda n: "pkg" not in n)      # packages first
            log = compile_fn(td, target, others)
        if not log:
            return {"ok": True, "feedback": ""}
        mine = [ln.replace(target + ":", "line ") for ln in log.splitlines()
                if ln.startswith(target + ":")]
        bad = [ln for ln in mine if "error" in ln.lower()
               or any(h in ln for h in hb._FEEDBACK_HITS)]
        return {"ok": not bad, "feedback": "\n".join(bad[:max_lines])}
    return check


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--model", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--benches", default=",".join(DEFAULT_BENCHES))
    ap.add_argument("--ve-cache", default="data/hdl_bench/verilog_eval_v2")
    ap.add_argument("--rtllm-cache", default="data/hdl_bench/rtllm_v2")
    ap.add_argument("--cvdp-file", default=str(Path("data/hdl_bench/cvdp") / cb.FILE))
    ap.add_argument("--uvm-cache", default="data/hdl_bench/uvm")
    ap.add_argument("--uvm-new-tokens", type=int, default=6144)
    ap.add_argument("--extra", action="store_true",
                    help="yosys gate counts on VerilogEval (area_ratio_mean)")
    ap.add_argument("--rtllm-exclude", default="")
    ap.add_argument("--n", type=int, default=1)
    ap.add_argument("--temperature", type=float, default=0.0)
    ap.add_argument("--top-p", type=float, default=0.95)
    ap.add_argument("--max-new-tokens", type=int, default=2048)
    ap.add_argument("--max-model-len", type=int, default=16384)
    ap.add_argument("--gpu-mem", type=float, default=0.9)
    ap.add_argument("--samples", default=None,
                    help="score the generations of an earlier <out>.samples.json "
                         "instead of generating (no GPU, v5.194)")
    ap.add_argument("--thinking", choices=("on", "off"), default=None,
                    help="pass enable_thinking to the chat template "
                         "(default: the template's own default)")
    ap.add_argument("--repair-rounds", type=int, default=0,
                    help="agent loop: resend answers that fail an iverilog "
                         "compile (no testbench) with the tool messages, up "
                         "to N times (VerilogEval / RTLLM, v5.229; single-file "
                         "CVDP with its RTL context, v5.234)")
    ap.add_argument("--tb-n", type=int, default=2,
                    help="--select tb: testbenches the model writes per problem")
    ap.add_argument("--select", choices=("none", "consensus", "tb"), default="none",
                    help="consensus: from the --n samples (after any repair) submit "
                         "one per problem, the compiling candidate simulation-"
                         "equivalent to the most other candidates (no answer "
                         "key; scored as pass@1 of the picked answer, v5.241)")
    ap.add_argument("--workers", type=int, default=ub.cpu_limit(),
                    help="simulation jobs (default: the container's CPU limit)")
    a = ap.parse_args(argv)
    if a.n > 1 and a.temperature == 0:
        ap.error("--n > 1 needs --temperature > 0")
    benches = [b for b in a.benches.split(",") if b]
    bad = set(benches) - set(BENCHES)
    if bad:
        ap.error(f"unknown benches {sorted(bad)}")
    exclude = tuple(x for x in a.rtllm_exclude.split(",") if x)
    jobs, data = collect(benches, a.ve_cache, a.rtllm_cache, a.cvdp_file, exclude,
                         a.uvm_cache)
    t0 = time.time()
    Path(a.out).parent.mkdir(parents=True, exist_ok=True)
    if a.repair_rounds and a.samples:
        ap.error("--repair-rounds needs the model (not --samples)")
    repair_stats = None
    if a.samples:
        got = {(s["bench"], s["key"]): s["outputs"]
               for s in json.loads(Path(a.samples).read_text(encoding="utf-8"))}
        outputs = [got[(b, k)] for b, k, _, _ in jobs]
    else:
        gen = VllmGen(a.model, a.max_model_len, a.gpu_mem,
                      None if a.thinking is None else a.thinking == "on")
        outputs = gen(jobs, a.n, a.temperature, a.top_p, a.max_new_tokens,
                      a.uvm_new_tokens)
        if a.repair_rounds:
            first = outputs
            Path(a.out).with_suffix(".round0.samples.json").write_text(
                json.dumps([{"bench": b, "key": k, "outputs": o}
                            for (b, k, _, _), o in zip(jobs, first)]) + "\n",
                encoding="utf-8")
            outputs, repair_stats = repair(
                gen, jobs, outputs, a.repair_rounds, hb.tool_feedback,
                temperature=a.temperature, top_p=a.top_p,
                max_new_tokens=a.max_new_tokens, workers=a.workers,
                bench_checks={"cvdp": cvdp_check(
                    {r["id"]: r for r in data.get("cvdp", [])})}
                if "cvdp" in benches else None)
            print("repair:", json.dumps(repair_stats), flush=True)
    picks = select_stats = None
    if a.select != "none":
        if a.n < 2:
            ap.error("--select needs --n >= 2")
        Path(a.out).parent.mkdir(parents=True, exist_ok=True)
        Path(a.out).with_suffix(".all.samples.json").write_text(
            json.dumps([{"bench": b, "key": k, "outputs": o}
                        for (b, k, _, _), o in zip(jobs, outputs)]) + "\n", encoding="utf-8")
        bc = {"cvdp": cvdp_check({r["id"]: r for r in data.get("cvdp", [])})} \
            if "cvdp" in benches else None
        if a.select == "tb":
            if a.samples:
                ap.error("--select tb needs the model (it writes the testbenches)")
            outputs, picks, select_stats, tbs = select_tb(
                gen, jobs, outputs, hb.tool_feedback, a.tb_n, a.temperature, a.top_p,
                a.max_new_tokens, bench_checks=bc, workers=a.workers)
            Path(a.out).with_suffix(".tbs.json").write_text(json.dumps(tbs) + "\n",
                                                            encoding="utf-8")
            print("select tb:", json.dumps(select_stats), flush=True)
        else:
            outputs, picks = select_consensus(jobs, outputs, hb.tool_feedback,
                                              workers=a.workers, bench_checks=bc)
    n_scored = 1 if picks is not None else a.n
    gen_s = time.time() - t0
    samples = [{"bench": b, "key": k, "outputs": o}
               for (b, k, _, _), o in zip(jobs, outputs)]
    if not a.samples:
        Path(a.out).with_suffix(".samples.json").write_text(   # before scoring
            json.dumps(samples) + "\n", encoding="utf-8")
    ks = tuple(k for k in (1, 5, 10) if k <= n_scored)
    res = score_all(jobs, outputs, data, n_scored, ks, exclude, a.workers, a.extra)
    out = {"model": a.model, "n": a.n, "temperature": a.temperature, "select": a.select, "select_stats": select_stats,
           "generate_seconds": round(gen_s), "summary": summary(res),
           "thinking": a.thinking, "repair_rounds": a.repair_rounds, "repair": repair_stats,
           "results": res, "samples": samples}
    Path(a.out).write_text(json.dumps(out, indent=1) + "\n", encoding="utf-8")
    print(json.dumps({"model": a.model, **out["summary"]}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
