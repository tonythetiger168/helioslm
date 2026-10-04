"""probe_api.py - run the HeliosLM calibration probe suite against any
OpenAI-compatible API endpoint.

v5.39: the price-war thesis instrument. When inference is $0.14/M and
1M context is standard, cheapness is not a differentiator -- but no
frontier vendor publishes calibration-on-wrong-answers curves. This
script measures exactly that on the cheapest frontier tiers for less
than one cent.

Output format is IDENTICAL to our own model evals (mid_agent_eval_v4
schema): per-task {task, expected, final, correct, steps, conf} --
directly comparable with our 8.5M/360M/Qwen3-0.6B numbers.

Usage:
  set OPENAI_API_KEY
  python examples/probe_api.py                    # GPT-5.6 Luna, 12 tasks
  python examples/probe_api.py --model gpt-5.6-luna --tasks 24
"""
import argparse
import json
import os
import sys
import time
import urllib.request
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "helioslm_v5" / "agent"))

from envs import make_envs, make_long_envs
from gate import FixedGate, Route
from schema import ToolCallError, parse_chat_turn
from tools import build_default_registry

FEWSHOT = """Solved example:
Task: Compute the value of: 2 + 3
Assistant: @@tool@@{\"calls\":[{\"name\":\"calc\",\"args\":{\"expr\":\"2 + 3\"}}]}@@end@@
step 0: 5
Assistant: @@tool@@{\"calls\":[{\"name\":\"finish\",\"args\":{\"answer\":\"5\"}}]}@@end@@
Final answer: 5

"""


def chat_once(base, key, model, prompt, max_tokens=200, temperature=0.0):
    """Single chat completion; returns (text, confidence). Confidence is
    logprob of the most likely FIRST token -- the same functional
    definition as our local softmax-max (greedy first-token peakedness),
    adjusted: we use the API-returned top-logprob when available."""
    body = {"model": model,
            "messages": [{"role": "user", "content": prompt}],
            "max_tokens": max_tokens, "temperature": temperature,
            "logprobs": True, "top_logprobs": 1}
    req = urllib.request.Request(
        base.rstrip("/") + "/chat/completions",
        data=json.dumps(body).encode(),
        headers={"Content-Type": "application/json",
                 "Authorization": f"Bearer {key}"})
    try:
        r = json.loads(urllib.request.urlopen(req, timeout=60).read())
    except urllib.error.HTTPError as e:
        return f"API_ERROR {e.code}: {e.read().decode()[:100]}", None
    choice = r["choices"][0]
    text = choice["message"]["content"] or ""
    import os as _os
    if _os.environ.get("PROBE_DEBUG"):
        _os.makedirs("probe_debug", exist_ok=True)
        n = len(_os.listdir("probe_debug"))
        open(f"probe_debug/resp_{n:03d}.json", "w").write(json.dumps(r, indent=1))
    conf = None
    lp = choice.get("logprobs")
    if lp and lp.get("content"):
        first = lp["content"][0]
        if first.get("top_logprobs"):
            conf = 2.718281828 ** first["top_logprobs"][0]["logprob"]
    return text, conf


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--base", default="https://api.openai.com/v1")
    ap.add_argument("--model", default="gpt-5.6-luna")
    ap.add_argument("--tasks", type=int, default=12)
    ap.add_argument("--out", default="api_probe_results.json")
    args = ap.parse_args()
    key = os.environ.get("OPENAI_API_KEY")
    if not key:
        sys.exit("set OPENAI_API_KEY first")

    n_tasks = args.tasks
    import random
    rng = random.Random(20260928)   # SAME seed as our local evals
    results = []
    total_cost_est = 0.0
    for env in make_envs() + make_long_envs():
        reg, impls = build_default_registry()
        for _ in range(max(1, n_tasks // 4)):
            task = env.sample(rng)
            state = {"conf": None, "steps": 0}

            def model_fn(prompt, seed, step, _base=args.base, _key=key,
                         _model=args.model, _state=state):
                text, conf = chat_once(_base, _key, _model,
                                       FEWSHOT + prompt)
                _state["conf"] = conf
                _state["steps"] = step + 1
                return text

            # drive a minimal tool loop (no ChatSession -- API models
            # use their own reasoning; we parse their tool intent)
            transcript = f"Task: {task.text}"
            final = None
            parsed_ok = 0
            for step in range(task.step_budget):
                out_text, conf = chat_once(args.base, key, args.model,
                                           FEWSHOT + transcript, max_tokens=300)
                state["conf"] = conf
                # DEBUG: record raw model output (smoke-test only)
                print(f"  RAW[{step}]: {out_text[:200]!r}", flush=True)
                try:
                    kind, payload = parse_chat_turn(out_text, reg)
                except ToolCallError:
                    transcript += f"\nstep {step}: PARSE_ERROR"
                    continue
                if kind == "text":
                    final = payload
                    break
                call = payload[0]
                parsed_ok += 1
                from tools import execute
                try:
                    obs = execute(call, reg, impls)
                except ToolCallError as e:
                    obs = f"TOOL_ERROR: {e}"
                transcript += f"\nAssistant: {out_text}\nstep {step}: {obs}"
                if call.name == "finish":
                    final = call.args["answer"]
                    break
            ok = final is not None and env.verify(task, final)
            results.append({"family": getattr(task, "family",
                                              type(task).__name__),
                            "task": task.text[:80],
                            "expected": task.answer,
                            "final": final, "correct": ok,
                            "steps": state["steps"],
                            "parsed_tool_calls": parsed_ok,
                            "conf": state["conf"]})
            print(f"{results[-1]['family']} correct={ok} "
                  f"conf={state['conf']}", flush=True)

    wrong = [r["conf"] for r in results
             if not r["correct"] and r["conf"] is not None]
    summary = {"model": args.model, "n": len(results),
               "correct": f"{sum(r['correct'] for r in results)}/{len(results)}",
               "conf_on_wrong_mean": (round(sum(wrong)/len(wrong), 4)
                                       if wrong else None),
               "conf_on_wrong_max": (round(max(wrong), 4) if wrong else None),
               "note": "same probe suite + same eval seed as our local "
                       "models -- directly comparable",
               "results": results}
    Path(args.out).write_text(json.dumps(summary, indent=2))
    print(json.dumps({k: v for k, v in summary.items() if k != "results"},
                     indent=2))


if __name__ == "__main__":
    main()
