"""spec_bench.py — speed-up from speculative decoding in llama.cpp (v5.209).

Starts llama-server on the target GGUF alone, then with the draft model,
sends the same greedy prompts to both and reports generated tokens per
second, the draft acceptance rate and the speed-up. Greedy outputs are
also compared: speculative decoding must not change them (small
differences can come from batched vs single-token numerics).

  python examples/spec_bench.py --server ~/llama.cpp/build/bin/llama-server \\
      --target L24/helioslm-l24-Q4_K.gguf --draft L24/helioslm-l24-draft.gguf \\
      --spec-type draft-mtp --out spec_l24.json

--spec-type ngram-mod (no --draft) measures llama.cpp's drafter-free
n-gram speculation instead.
"""
import argparse
import json
import subprocess
import sys
import time
import urllib.request
from pathlib import Path

_REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_REPO))

from helioslm_v5.src.export.speculative import (  # noqa: E402
    SPEC_TYPES, draft_mismatch, server_args)

PROMPTS = [
    "請用繁體中文介紹台灣的夜市文化，並舉三個有名的夜市。",
    "用繁體中文說明光合作用的過程。",
    "Write a Python function that returns the n-th Fibonacci number iteratively, with a docstring.",
    "Explain the difference between TCP and UDP in a short paragraph.",
    "寫一段 Verilog：4 位元同步重設上數計數器。",
    "請把下面這句翻成英文：今天下午三點在台北車站東三門集合。",
]


def summarize(results):
    """Totals over per-prompt ``timings`` dicts from llama-server."""
    n = sum(r["predicted_n"] for r in results)
    ms = sum(r["predicted_ms"] for r in results)
    drafted = sum(r.get("draft_n", 0) for r in results)
    accepted = sum(r.get("draft_n_accepted", 0) for r in results)
    return {"tokens": n, "tok_per_s": round(1000 * n / ms, 2) if ms else None,
            "draft_tokens": drafted,
            "acceptance": round(accepted / drafted, 3) if drafted else None}


def _post(port, body):
    req = urllib.request.Request(f"http://127.0.0.1:{port}/completion",
                                 data=json.dumps(body).encode(),
                                 headers={"Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=1800) as r:
        return json.loads(r.read())


def _wait(port, proc, timeout=1800):
    t0 = time.time()
    while time.time() - t0 < timeout:
        if proc.poll() is not None:
            raise RuntimeError(f"llama-server exited with {proc.returncode}")
        try:
            with urllib.request.urlopen(f"http://127.0.0.1:{port}/health", timeout=5) as r:
                if r.status == 200:
                    return
        except Exception:
            pass
        time.sleep(2)
    raise TimeoutError("llama-server did not become healthy")


def run(server, target, spec, args):
    cmd = [server, "-m", target, "-c", str(args.ctx), "-ngl", str(args.ngl),
           "--port", str(args.port), "-np", "1"]
    if spec and args.draft:
        cmd += server_args(args.draft, args.spec_type, args.ngl, args.draft_max)
    elif spec:
        cmd += ["--spec-type", args.spec_type]
    print("+", " ".join(cmd), flush=True)
    proc = subprocess.Popen(cmd, stdout=subprocess.DEVNULL, stderr=open(args.log, "a"))
    try:
        _wait(args.port, proc)
        _post(args.port, {"prompt": "warm up", "n_predict": 8})
        out = []
        for p in PROMPTS[:args.prompts]:
            r = _post(args.port, {"prompt": p, "n_predict": args.n, "temperature": 0,
                                  "top_k": 1, "cache_prompt": False})
            out.append({"prompt": p, "content": r["content"], **r["timings"]})
            print(f"  {r['timings'].get('predicted_per_second', 0):.1f} tok/s", flush=True)
        return out
    finally:
        proc.terminate()
        proc.wait(60)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--server", required=True, help="llama-server binary")
    ap.add_argument("--target", required=True)
    ap.add_argument("--draft", help="drafter GGUF (not needed for ngram-* types)")
    ap.add_argument("--spec-type", default="draft-mtp",
                    help="draft-mtp, draft-simple or a drafter-free type such as ngram-mod")
    ap.add_argument("--ctx", type=int, default=8192)
    ap.add_argument("--ngl", type=int, default=99)
    ap.add_argument("--n", type=int, default=256, help="tokens per prompt")
    ap.add_argument("--prompts", type=int, default=len(PROMPTS))
    ap.add_argument("--draft-max", type=int, help="--spec-draft-n-max (default per type)")
    ap.add_argument("--port", type=int, default=8090)
    ap.add_argument("--log", default="spec_bench_server.log")
    ap.add_argument("--out", default="spec_bench.json")
    args = ap.parse_args()
    if args.spec_type in SPEC_TYPES:
        if not args.draft:
            ap.error(f"--spec-type {args.spec_type} needs --draft")
        why = draft_mismatch(args.target, args.draft, args.spec_type)
        if why:
            sys.exit(f"draft does not fit the target: {why}")
    base = run(args.server, args.target, False, args)
    spec = run(args.server, args.target, True, args)
    b, s = summarize(base), summarize(spec)
    report = {"target": args.target, "draft": args.draft, "spec_type": args.spec_type,
              "ctx": args.ctx,
              "draft_max": args.draft_max, "baseline": b, "speculative": s,
              "speedup": round(s["tok_per_s"] / b["tok_per_s"], 3)
              if b["tok_per_s"] and s["tok_per_s"] else None,
              "same_output": sum(x["content"] == y["content"] for x, y in zip(base, spec)),
              "prompts": len(base), "runs": {"baseline": base, "speculative": spec}}
    Path(args.out).write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n")
    print(json.dumps({k: report[k] for k in ("baseline", "speculative", "speedup",
                                             "same_output", "prompts")}, indent=2))


if __name__ == "__main__":
    main()
