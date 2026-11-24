"""Route C — K3.1 Pro external reference row (docs/K31_DISTILL_PLAN.md).

Runs the hosted Kimi K3.1 Pro API (OpenAI-compatible chat completions)
through the *same* five-track interfaces used by
examples/agentic_candidate_run.py, so the results slot into
docs/AGENTIC_BENCH.md as an *external reference row* next to our own
candidates (toy / mid).

Hard rules (from K31_DISTILL_PLAN.md):

- Vendor-reported numbers stay labeled as such; this script only
  *produces our own measurements* of the external model.
- Teacher output is untrusted data — raw replies are stored verbatim
  and hashed; nothing here feeds training data directly.
- Cost is recorded per API call from the ``usage`` object; the run
  aborts when ``--max-cost-usd`` would be exceeded (default $1.00).
- The API key comes from the environment only and is never written to
  any file (results store only ``key_fingerprint`` = sha256(key)[:12]
  for provenance without disclosure).
- HDL generated code is untrusted; iverilog/vvp run only in the local
  sandbox, same discipline as the candidate runs.

Usage:

    export MOONSHOT_API_KEY=sk-...
    python examples/k31_reference_run.py --smoke --skip-hdl   # 1 episode
    python examples/k31_reference_run.py --episodes 50 \
        --out results/reference/2026-11-24_k31-pro/
"""
import argparse
import hashlib
import json
import os
import random
import sys
import time
import urllib.error
import urllib.request
from datetime import datetime, timezone
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

# Kimi platform rates (vendor-reported, USD per 1M tokens, checked
# 2026-11-24). Flat pricing, no context-length tiers. These are cost
# *assumptions* for budget accounting, not verified claims.
RATE_INPUT_PER_M = 2.0
RATE_OUTPUT_PER_M = 10.0

DEFAULT_BASE_URL = "https://api.moonshot.ai/v1"
DEFAULT_MODEL = "kimi-k3.1-pro"

# Reference-row sampling: weight-proportional subset of each track that
# keeps the total near 50 episodes and the cost under $1.
DEFAULT_EPISODES = {
    "terminal": 16,
    "swe": 8,
    "long_horizon": 5,
    "tool_recovery": 13,
    "hdl": 8,
}


def _post_chat(base_url, key, model, messages, reasoning_effort="low",
               max_retries=2, timeout=300):
    """Minimal OpenAI-compatible chat.completions call (stdlib only)."""
    body = json.dumps({
        "model": model,
        "reasoning_effort": reasoning_effort,
        "messages": messages,
    }).encode()
    last = None
    for attempt in range(max_retries + 1):
        req = urllib.request.Request(
            base_url.rstrip("/") + "/chat/completions",
            data=body,
            headers={
                "Content-Type": "application/json",
                "Authorization": "Bearer " + key,
            },
        )
        try:
            with urllib.request.urlopen(req, timeout=timeout) as r:
                return json.load(r)
        except urllib.error.HTTPError as e:
            payload = e.read().decode("utf-8", "replace")[:400]
            last = f"HTTP {e.code}: {payload}"
            if e.code in (401, 403):  # not retryable (circuit-breaker rule)
                raise RuntimeError(f"auth failure (not retrying): {last}")
            if e.code == 429 or e.code >= 500:
                time.sleep(5 * (attempt + 1))
                continue
            raise RuntimeError(last)
        except (urllib.error.URLError, TimeoutError) as e:
            last = f"transport: {e}"
            time.sleep(5 * (attempt + 1))
    raise RuntimeError(f"chat completion failed after retries: {last}")


class K31Candidate:
    """External reference candidate exposing the same interfaces as
    MidCandidate: episode(prompt, sample), policy(scenario, observations),
    hdl(system, user, i), plus generate(prompt, max_new) for tool_recovery.
    Every API call records its usage for cost accounting."""

    def __init__(self, key, base_url=DEFAULT_BASE_URL, model=DEFAULT_MODEL,
                 reasoning_effort="low"):
        self.key = key
        self.base_url = base_url
        self.model = model
        self.reasoning_effort = reasoning_effort
        self.usage_events = []
        self.total_cost = 0.0
        self._system = None

    # -- internal ------------------------------------------------------
    def _chat_system(self):
        if self._system is None:
            from helioslm_v5.agent.chat import CHAT_SYSTEM
            from helioslm_v5.agent.tools import build_default_registry
            from helioslm_v5.agent.loop import render_tool_docs
            reg, _ = build_default_registry()
            self._system = CHAT_SYSTEM.replace(
                "%%TOOLS%%", render_tool_docs(reg))
        return self._system

    def _call(self, messages):
        resp = _post_chat(self.base_url, self.key, self.model, messages,
                          self.reasoning_effort)
        usage = resp.get("usage") or {}
        pt = usage.get("prompt_tokens", 0)
        ct = usage.get("completion_tokens", 0)
        cost = pt * RATE_INPUT_PER_M / 1e6 + ct * RATE_OUTPUT_PER_M / 1e6
        self.usage_events.append({
            "prompt_tokens": pt,
            "completion_tokens": ct,
            "reasoning_tokens": (usage.get("completion_tokens_details") or {})
            .get("reasoning_tokens"),
            "cost_usd": round(cost, 6),
        })
        self.total_cost += cost
        msg = resp["choices"][0]["message"]
        return msg.get("content") or ""

    # -- track interfaces (same shapes as MidCandidate) ----------------
    def episode(self, prompt, sample):
        return self._call([{"role": "user", "content": prompt}])

    def policy(self, scenario, observations):
        tools = "calc, str_op, file_read, file_write, finish"
        prompt = (
            "You are an agent that replies with exactly one tool call.\n"
            "Reply with ONLY this exact format (no spaces between the "
            "markers and the JSON, compact JSON, no prose):\n"
            '@@tool@@{"calls":[{"name":"<tool>","args":{...}}]}@@end@@\n'
            f"Available tools: {tools}.\n"
            "When the task is fully complete, reply instead with "
            "'DONE <answer>'.\n"
            f"Task: {scenario.task}\n"
            + "".join(f"observation: {o}\n" for o in observations)
            + "Your reply:")
        return self._call([{"role": "user", "content": prompt}])

    def hdl(self, system, user, i):
        return [self._call([{"role": "system", "content": system},
                            {"role": "user", "content": user}])]

    def generate(self, prompt, max_new=256):
        del max_new  # server-side default
        return self._call([{"role": "user", "content": prompt}])


# ---------------------------------------------------------------- tracks

def run_terminal(cand, limit, out):
    from helioslm_v5.eval import term_bench as tb
    tasks = tb.load_tasks()[:limit]
    rows = tb.run_term_bench(tasks, cand.episode)
    (out / "terminal.json").write_text(
        json.dumps(rows, indent=1, ensure_ascii=False) + "\n", encoding="utf-8")
    return rows


def run_swe(cand, limit, out):
    from helioslm_v5.eval import swe_bench as sb
    tasks = sb.load_tasks()[:limit]
    rows = sb.run_swe_bench(tasks, cand.episode)
    (out / "swe.json").write_text(
        json.dumps(rows, indent=1, ensure_ascii=False) + "\n", encoding="utf-8")
    return rows


def run_long_horizon(cand, limit, out):
    from helioslm_v5.eval import long_horizon_bench as lh
    scenarios = list(lh.SCENARIOS)[:limit]
    name = "k31-reference"
    lh.POLICIES[name] = cand.policy
    try:
        rows = [lh.run_one(sc, name) for sc in scenarios]
    finally:
        del lh.POLICIES[name]
    n = len(rows)
    result = {"policy": name, "rows": rows,
              "success_rate": sum(r["success"] for r in rows) / n,
              "violation_rate": sum(bool(r["violations"]) for r in rows) / n,
              "false_completion_rate": sum(r["false_completion"]
                                           for r in rows) / n,
              "injection_rate": sum(r["injection_followed"]
                                    for r in rows) / n}
    (out / "long_horizon.json").write_text(
        json.dumps(result, indent=1, ensure_ascii=False) + "\n",
        encoding="utf-8")
    return result


def run_tool_recovery(cand, limit, out):
    """Same corruption corpus as agentic_candidate_run.run_tool_recovery,
    subsampled deterministically to `limit` items."""
    from helioslm_v5.tests.test_tool_recovery import (
        REG, REPAIRABLE, _sample_calls)
    from helioslm_v5.agent.schema import ToolCallError, parse_tool_call
    rows = []
    rng = random.Random(20261123)
    corpus = []
    for cls, (corrupt, _repair) in sorted(REPAIRABLE.items()):
        for i in range(10):
            corpus.append((cls, i, corrupt))
    corpus = corpus[:limit]
    for cls, i, corrupt in corpus:
        calls = _sample_calls(rng)
        text = corrupt(calls)
        prompt = ("Repair the following malformed tool-call block into a "
                  "valid one. Reply with only the repaired block, nothing "
                  f"else.\n\n{text}\n")
        reply = cand.generate(prompt, max_new=256)
        try:
            got = parse_tool_call(reply, REG)
            ok = got == calls
        except ToolCallError:
            ok = False
        rows.append({"task": f"{cls}:{i}", "correct": bool(ok),
                     "reply_sha256": hashlib.sha256(
                         reply.encode()).hexdigest()})
    artifact = {"schema": "tool-recovery-run/1", "rows": rows}
    (out / "tool_recovery.json").write_text(
        json.dumps(artifact, indent=1, ensure_ascii=False) + "\n",
        encoding="utf-8")
    return rows


def run_hdl(cand, limit, out, cache):
    from helioslm_v5.eval import hdl_bench as hb
    import agentic_candidate_run as acr
    cache = Path(cache)
    if not (cache / "MANIFEST.json").is_file():
        cache.mkdir(parents=True, exist_ok=True)
        hb.fetch(cache, get=acr._tarball_get(cache))
    probs = hb.load(cache)[:limit]
    result = hb.score(probs, cand.hdl, n=1, ks=(1,))
    (out / "hdl.json").write_text(
        json.dumps(result, indent=1, ensure_ascii=False) + "\n", encoding="utf-8")
    return result


TRACKS = {
    "terminal": run_terminal,
    "swe": run_swe,
    "long_horizon": run_long_horizon,
    "tool_recovery": run_tool_recovery,
    "hdl": run_hdl,
}


def main(argv=None):
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--episodes", type=int, default=50,
                    help="total episode budget (default 50)")
    ap.add_argument("--smoke", action="store_true",
                    help="run exactly one terminal episode and stop")
    ap.add_argument("--tracks", default=",".join(DEFAULT_EPISODES))
    ap.add_argument("--skip-hdl", action="store_true",
                    help="skip the hdl track (needs iverilog + upstream cache)")
    ap.add_argument("--hdl-cache", default="results/_hdl_upstream_cache")
    ap.add_argument("--out", default=None)
    ap.add_argument("--model", default=DEFAULT_MODEL)
    ap.add_argument("--base-url", default=DEFAULT_BASE_URL)
    ap.add_argument("--reasoning-effort", default="low",
                    choices=["low", "high", "max"])
    ap.add_argument("--max-cost-usd", type=float, default=1.00)
    a = ap.parse_args(argv)

    key = os.environ.get("MOONSHOT_API_KEY", "")
    if not key:
        raise SystemExit("MOONSHOT_API_KEY not set in environment")
    fingerprint = hashlib.sha256(key.encode()).hexdigest()[:12]

    out = Path(a.out) if a.out else Path(
        f"results/reference/{datetime.now(timezone.utc):%Y-%m-%d}_{a.model}")
    out.mkdir(parents=True, exist_ok=True)

    cand = K31Candidate(key, a.base_url, a.model, a.reasoning_effort)

    plan = dict(DEFAULT_EPISODES)
    if a.smoke:
        plan = {"terminal": 1}
    else:
        chosen = [t for t in a.tracks.split(",") if t in plan]
        if a.skip_hdl and "hdl" in chosen:
            chosen.remove("hdl")
        scale = a.episodes / sum(DEFAULT_EPISODES[t] for t in chosen)
        plan = {t: max(1, round(DEFAULT_EPISODES[t] * scale)) for t in chosen}

    manifest = {
        "schema": "k31-reference-run/1",
        "model": a.model,
        "base_url": a.base_url,
        "reasoning_effort": a.reasoning_effort,
        "key_fingerprint": fingerprint,
        "rates_assumed": {"input_per_1m": RATE_INPUT_PER_M,
                          "output_per_1m": RATE_OUTPUT_PER_M,
                          "vendor_reported": True, "checked": "2026-11-24"},
        "plan": plan,
        "max_cost_usd": a.max_cost_usd,
        "started_utc": datetime.now(timezone.utc).isoformat(),
        "tracks": {},
    }
    for track, n in plan.items():
        runner = TRACKS[track]
        if cand.total_cost >= a.max_cost_usd:
            print(f"[budget] stopping before {track} at ${cand.total_cost:.4f}")
            break
        print(f"[track] {track} x{n}")
        if track == "hdl":
            res = runner(cand, n, out, a.hdl_cache)
            rows = res.get("results") or res.get("rows") or []
            passed = sum(1 for r in rows
                         if r.get("pass") or r.get("ok") or r.get("correct"))
            total = len(rows) or n
        elif track == "long_horizon":
            res = runner(cand, n, out)
            rows = res["rows"]
            passed = sum(1 for r in rows if r.get("success"))
            total = len(rows)
        else:
            res = runner(cand, n, out)
            rows = res if isinstance(res, list) else res.get("rows", [])
            passed = sum(1 for r in rows
                         if r.get("correct") or r.get("ok"))
            total = len(rows)
        manifest["tracks"][track] = {
            "episodes": total, "passed": passed,
            "pass_rate": round(passed / total, 4) if total else None,
        }
        print(f"[track] {track}: {passed}/{total} "
              f"cost=${cand.total_cost:.4f}")

    manifest["finished_utc"] = datetime.now(timezone.utc).isoformat()
    manifest["total_cost_usd"] = round(cand.total_cost, 6)
    manifest["usage_events"] = len(cand.usage_events)
    manifest["usage"] = cand.usage_events
    (out / "manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=1) + "\n",
        encoding="utf-8")
    print(json.dumps({"out": str(out),
                      "cost": manifest["total_cost_usd"],
                      "tracks": manifest["tracks"]},
                     ensure_ascii=False, indent=1))


if __name__ == "__main__":
    main()
