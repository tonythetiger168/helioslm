"""probe_api.py - run the HeliosLM calibration probe suite against any
OpenAI-compatible API endpoint.

v5.39: the price-war thesis instrument. When inference is $0.14/M and
1M context is standard, cheapness is not a differentiator -- but no
frontier vendor publishes calibration-on-wrong-answers curves. This
script measures exactly that on the cheapest frontier tiers for less
than one cent.

v5.51 (2026-10-13 refresh): the 2026-10 model generation roster
(GPT-6 Astra / GPT-6.1 Sol / Claude Opus 5.5 / Claude Sonnet 5.5 /
Gemini 4 Argon) as named presets, per-run USD cost accounting from the
response `usage` block (OpenAI's own framing is now "price per
COMPLETED TASK" -- so we record cost_per_correct_usd), and the output
carries verification_tier="measured" per the M2 artifact registry.
Meta Muse Wonder is NOT probe-able: the full model is locked inside
the Muse Code subscription and has no API -- recorded here, not
silently skipped.

Output format is IDENTICAL to our own model evals (mid_agent_eval_v4
schema): per-task {task, expected, final, correct, steps, conf} --
directly comparable with our 8.5M/360M/Qwen3-0.6B numbers.

Usage:
  set OPENAI_API_KEY            (for OpenRouter presets: the OR key)
  python examples/probe_api.py --preset astra --tasks 12
  python examples/probe_api.py --preset opus-5.5 --tasks 12
  python examples/probe_api.py --list-models --preset astra
  python examples/probe_api.py --model gpt-5.6-luna --tasks 24  # legacy
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

ANTI_SHORTCUT = """IMPORTANT: you MUST solve every task by calling tools.
Do NOT compute answers mentally and reply directly -- always emit a
tool call first, read the observation, and only then finish.

"""

FEWSHOT = """Solved example:
Task: Compute the value of: 2 + 3
Assistant: @@tool@@{\"calls\":[{\"name\":\"calc\",\"args\":{\"expr\":\"2 + 3\"}}]}@@end@@
step 0: 5
Assistant: @@tool@@{\"calls\":[{\"name\":\"finish\",\"args\":{\"answer\":\"5\"}}]}@@end@@
Final answer: 5

"""

# Named probe targets. price = USD per 1M tokens (input, output) from
# public pricing pages AS OF 2026-10-13 -- vendors reprice without
# notice, VERIFY before a paid run; price=None means "unknown, cost
# accounting disabled" (recorded honestly, not estimated). Non-OpenAI
# models route through OpenRouter's OpenAI-compatible endpoint
# (Anthropic has no /chat/completions; Google's openai-compat endpoint
# is an alternative -- either way, --list-models prints the live IDs;
# confirm the model string there before a paid run).
PRESETS = {
    # --- 2026-10 generation ------------------------------------------
    "astra":      {"base": "https://api.openai.com/v1",
                   "model": "gpt-6-astra", "price": (10.0, 50.0)},
    "sol":        {"base": "https://api.openai.com/v1",
                   "model": "gpt-6.1-sol", "price": (2.0, 10.0)},
    "opus-5.5":   {"base": "https://openrouter.ai/api/v1",
                   "model": "anthropic/claude-opus-5.5",
                   "price": (4.0, 20.0)},
    "sonnet-5.5": {"base": "https://openrouter.ai/api/v1",
                   "model": "anthropic/claude-sonnet-5.5",
                   "price": (2.0, 10.0)},
    "argon":      {"base": "https://openrouter.ai/api/v1",
                   "model": "google/gemini-4-argon", "price": None},
    # --- legacy 2026-09 generation (the 2026-10-04 probe row) --------
    "luna":       {"base": "https://api.openai.com/v1",
                   "model": "gpt-5.6-luna", "price": (0.20, 1.20)},
}
PRICES_ASOF = "2026-10-13"
PROBE_GENERATION = "2026-10"


def list_models(base, key):
    """Print the endpoint's live model IDs -- confirm a preset's model
    string here before spending money on it."""
    req = urllib.request.Request(
        base.rstrip("/") + "/models",
        headers={"Authorization": f"Bearer {key}"})
    r = json.loads(urllib.request.urlopen(req, timeout=60).read())
    ids = sorted(m.get("id", "?") for m in r.get("data", []))
    for i in ids:
        print(i)
    return ids


def chat_once(base, key, model, prompt, max_tokens=200, temperature=0.0,
              meter=None):
    """Single chat completion; returns (text, confidence). Confidence is
    logprob of the most likely FIRST token -- the same functional
    definition as our local softmax-max. Two 2026-API realities are
    handled by graceful degradation: max_tokens may be rejected (retry
    as max_completion_tokens then without), logprobs may be unsupported
    (conf stays None, recorded honestly). When `meter` (a dict) is
    given, token usage from the response `usage` block is accumulated
    into it -- the cost accounting behind cost_per_correct_usd."""
    # reasoning-tier models (gpt-5.6-luna et al.) reject temperature
    # != 1 -- it is part of the degradation ladder below
    body = {"model": model,
            "messages": [{"role": "user", "content": prompt}]}

    def attempt(extra):
        b = {**body, **extra}
        req = urllib.request.Request(
            base.rstrip("/") + "/chat/completions",
            data=json.dumps(b).encode(),
            headers={"Content-Type": "application/json",
                     "Authorization": f"Bearer {key}"})
        # network jitter is the norm, not the exception -- retry transient
        # failures (SSL handshake timeouts, resets) without burning the
        # parameter-degradation ladder on them
        import time as _time
        last_err = None
        for wait in (2, 4, 8, 16):
            try:
                return json.loads(urllib.request.urlopen(req, timeout=60).read())
            except urllib.error.HTTPError as e:
                # HTTPError subclasses URLError: without this branch every
                # 400 the degradation ladder relies on was retried 4x with
                # 30 s of sleeps. Only rate limits / server errors are
                # transient; anything else goes straight to the caller.
                if e.code not in (429, 500, 502, 503, 504):
                    raise
                last_err = e
                _time.sleep(wait)
            except (urllib.error.URLError, TimeoutError, OSError) as e:
                last_err = e
                _time.sleep(wait)
        raise last_err

    r = None
    combos = [{"temperature": temperature, "max_tokens": max_tokens,
               "logprobs": True, "top_logprobs": 1},
              {"temperature": temperature,
               "max_completion_tokens": max_tokens, "logprobs": True,
               "top_logprobs": 1},
              {"temperature": temperature, "max_tokens": max_tokens},
              {"max_tokens": max_tokens, "logprobs": True,
               "top_logprobs": 1},
              {"max_completion_tokens": max_tokens, "logprobs": True,
               "top_logprobs": 1},
              {"max_tokens": max_tokens},
              {}]
    for extra in combos:
        try:
            r = attempt(extra)
            break
        except urllib.error.HTTPError as e:
            code = e.code
            if code not in (400, 422):
                return f"API_ERROR {code}: {e.read().decode()[:100]}", None
            # remember the LAST 400 body for diagnosis
            import os as _os
            _os.environ["PROBE_LAST_400"] = e.read().decode()[:300]
    if r is None:
        import os as _os
        return ("API_ERROR: all combos rejected | last 400: "
                + _os.environ.get("PROBE_LAST_400", "?")), None
    choice = r["choices"][0]
    text = choice["message"]["content"] or ""
    if meter is not None:
        usage = r.get("usage") or {}
        meter["prompt_tokens"] = (meter.get("prompt_tokens", 0)
                                  + usage.get("prompt_tokens", 0))
        meter["completion_tokens"] = (meter.get("completion_tokens", 0)
                                      + usage.get("completion_tokens", 0))
        meter["calls"] = meter.get("calls", 0) + 1
    # fallback confidence for reasoning-tier models (no logprobs): the
    # structural peakedness of the first content token is not available,
    # so we record None -- HONESTLY. The calibration comparison then
    # uses only models whose APIs expose logprobs, which is itself a
    # publishable observation about API transparency.
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


def run_episode(args, key, task, reg, impls, meter=None):
    """One full tool-loop episode; returns the final answer or None."""
    transcript = f"Task: {task.text}"
    for step in range(task.step_budget):
        out_text, _ = chat_once(args.base, key, args.model,
                                ANTI_SHORTCUT + FEWSHOT + transcript,
                                max_tokens=300, meter=meter)
        try:
            kind, payload = parse_chat_turn(out_text, reg)
        except ToolCallError:
            transcript += f"\nstep {step}: PARSE_ERROR"
            continue
        if kind == "text":
            return payload
        call = payload[0]
        from tools import execute
        if call.name not in reg._specs:
            obs = (f"TOOL_ERROR: unknown tool '{call.name}'. Available: "
                   "calc, str_op, file_read, file_write, finish")
        else:
            try:
                obs = execute(call, reg, impls)
            except ToolCallError as e:
                obs = f"TOOL_ERROR: {e}"
        transcript += f"\nAssistant: {out_text}\nstep {step}: {obs}"
        if call.name == "finish":
            return call.args["answer"]
    return None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--preset", choices=sorted(PRESETS), default=None,
                    help="named 2026-10-generation target (sets "
                         "--base/--model and enables cost accounting)")
    ap.add_argument("--base", default=None)
    ap.add_argument("--model", default=None)
    ap.add_argument("--tasks", type=int, default=12)
    ap.add_argument("--out", default="api_probe_results.json")
    ap.add_argument("--samples", type=int, default=3)
    ap.add_argument("--list-models", action="store_true",
                    help="print the endpoint's live model IDs and exit")
    args = ap.parse_args()

    price = None
    if args.preset:
        p = PRESETS[args.preset]
        args.base = args.base or p["base"]
        args.model = args.model or p["model"]
        price = p["price"]
    args.base = args.base or "https://api.openai.com/v1"
    args.model = args.model or "gpt-5.6-luna"

    key = os.environ.get("OPENAI_API_KEY")
    if not key:
        sys.exit("set OPENAI_API_KEY first")
    if args.list_models:
        list_models(args.base, key)
        return

    n_tasks = args.tasks
    import random
    rng = random.Random(20260928)   # SAME seed as our local evals
    results = []
    meter = {}
    for env in make_envs() + make_long_envs():
        reg, impls = build_default_registry()
        for _ in range(max(1, n_tasks // 4)):
            task = env.sample(rng)
            state = {"conf": None, "steps": 0}

            final = run_episode(args, key, task, reg, impls, meter=meter)
            parsed_ok = 0
            finals = []
            if final is not None:
                finals.append(str(final).strip())
            for _rep in range(max(0, args.samples - 1)):
                rep = run_episode(args, key, task, reg, impls, meter=meter)
                if rep is not None:
                    finals.append(str(rep).strip())
            from collections import Counter
            if finals:
                majority, cnt = Counter(finals).most_common(1)[0]
                consistency = cnt / len(finals)
            else:
                majority, consistency = None, None
            ok = majority is not None and env.verify(task, majority)
            results.append({"family": getattr(task, "family",
                                              type(task).__name__),
                            "task": task.text[:80],
                            "expected": task.answer,
                            "final": majority, "correct": ok,
                            "steps": state["steps"],
                            "parsed_tool_calls": parsed_ok,
                            "conf": consistency,
                            "conf_type": "self_consistency",
                            "n_samples": len(finals)})
            print(f"{results[-1]['family']} correct={ok} "
                  f"consistency={consistency}", flush=True)

    wrong = [r["conf"] for r in results
             if not r["correct"] and r["conf"] is not None]
    n_correct = sum(r["correct"] for r in results)
    cost_usd = None
    if price is not None and meter.get("calls"):
        cost_usd = round((meter.get("prompt_tokens", 0) * price[0]
                          + meter.get("completion_tokens", 0) * price[1])
                         / 1_000_000, 6)
    summary = {"model": args.model, "n": len(results),
               "correct": f"{n_correct}/{len(results)}",
               "conf_on_wrong_mean": (round(sum(wrong)/len(wrong), 4)
                                       if wrong else None),
               "conf_on_wrong_max": (round(max(wrong), 4) if wrong else None),
               "verification_tier": "measured",
               "probe_generation": PROBE_GENERATION,
               "prices_asof": PRICES_ASOF if price else None,
               "usage": meter or None,
               "cost_usd": cost_usd,
               "cost_per_correct_usd": (round(cost_usd / n_correct, 6)
                                        if cost_usd and n_correct else None),
               "note": "same probe suite + same eval seed as our local "
                       "models -- directly comparable; cost from the "
                       "response usage block, prices unverified since "
                       + PRICES_ASOF,
               "results": results}
    Path(args.out).write_text(json.dumps(summary, indent=2))
    print(json.dumps({k: v for k, v in summary.items() if k != "results"},
                     indent=2))


if __name__ == "__main__":
    main()
