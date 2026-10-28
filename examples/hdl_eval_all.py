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
The raw generations are written to ``<out>.samples.json`` before any
scoring, so a scoring failure never costs the GPU time (v5.179).
"""
import argparse
import json
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


def vllm_generate(model, jobs, n, temperature, top_p, max_new_tokens,
                  max_model_len, gpu_mem, uvm_new_tokens=None):
    from vllm import LLM, SamplingParams
    llm = LLM(model=model, dtype="bfloat16", max_model_len=max_model_len,
              gpu_memory_utilization=gpu_mem, trust_remote_code=False)
    tok = llm.get_tokenizer()
    def params(bench):
        new = max(max_new_tokens, uvm_new_tokens or 0) if bench == "uvm" else max_new_tokens
        return new, SamplingParams(n=n, temperature=temperature,
                                   top_p=top_p if temperature > 0 else 1.0,
                                   max_tokens=new)
    prompts, sps = [], []
    for bench, _, system, user in jobs:
        new, sp = params(bench)
        sps.append(sp)
        if tok.chat_template:
            try:
                p = tok.apply_chat_template(
                    [{"role": "system", "content": system},
                     {"role": "user", "content": user}],
                    tokenize=False, add_generation_prompt=True)
            except Exception:       # templates without a system role
                p = tok.apply_chat_template(
                    [{"role": "user", "content": system + "\n\n" + user}],
                    tokenize=False, add_generation_prompt=True)
        else:
            p = system + "\n" + user
        ids = tok(p, add_special_tokens=False).input_ids
        limit = max_model_len - new
        prompts.append({"prompt_token_ids": ids[-limit:]})
    outs = llm.generate(prompts, sps)
    return [[c.text for c in o.outputs] for o in outs]


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
    if a.samples:
        got = {(s["bench"], s["key"]): s["outputs"]
               for s in json.loads(Path(a.samples).read_text(encoding="utf-8"))}
        outputs = [got[(b, k)] for b, k, _, _ in jobs]
    else:
        outputs = vllm_generate(a.model, jobs, a.n, a.temperature, a.top_p,
                                a.max_new_tokens, a.max_model_len, a.gpu_mem,
                                a.uvm_new_tokens)
    gen_s = time.time() - t0
    samples = [{"bench": b, "key": k, "outputs": o}
               for (b, k, _, _), o in zip(jobs, outputs)]
    if not a.samples:
        Path(a.out).with_suffix(".samples.json").write_text(   # before scoring
            json.dumps(samples) + "\n", encoding="utf-8")
    ks = tuple(k for k in (1, 5, 10) if k <= a.n)
    res = score_all(jobs, outputs, data, a.n, ks, exclude, a.workers, a.extra)
    out = {"model": a.model, "n": a.n, "temperature": a.temperature,
           "generate_seconds": round(gen_s), "summary": summary(res),
           "results": res, "samples": samples}
    Path(a.out).write_text(json.dumps(out, indent=1) + "\n", encoding="utf-8")
    print(json.dumps({"model": a.model, **out["summary"]}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
