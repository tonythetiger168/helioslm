"""Teacher data generation for sequence-level distillation (v5.111, D1).

The DeepSeek track (docs/DISTILL_DEEPSEEK.md): an open-weight teacher,
DeepSeek-V4.1-Flash, served by vLLM on a rented pod (or reached through
an OpenAI-compatible API), answers a prompt file; every answer is kept
with its reasoning, its token usage and how it ended, so the filter (D2)
and the student SFT (D3) can decide later what to train on. Nothing here
needs torch.

  licence gate   a teacher is used only when its weights' licence review
                 is pinned and allows it (``teacher_license_audit.usable``);
                 closed models are not in the review table at all;
  records        one JSON line per (prompt, sample): messages, reasoning,
                 content, finish_reason, usage, teacher and sampling;
                 the reasoning is always kept here (OS2 keep_thinking
                 decides at SFT time what the student sees);
  keep_thinking  for multi-turn prompts, whether earlier assistant turns
                 are sent with their reasoning ("all") or without it
                 ("none" / "turn": in a prompt every assistant turn is an
                 earlier turn);
  resume         ids already in the output file are skipped, so a pod
                 that dies costs only the in-flight requests; failures
                 go to ``<out>.errors.jsonl`` and are retried next run;
  limits         stop submitting at a cost, output-token or wall-clock
                 limit; in-flight requests still finish and are written;
  token stats    (OS1) totals, per-answer percentiles, reasoning share
                 and truncation rate, overall and per domain.

Prompts can come from the teacher itself (``meta_prompts`` +
``extract_prompts``): asked with thinking off to write questions for a
domain and topic, so no prompt text comes from a non-commercial dataset.
"""
import hashlib
import json
import math
import os
import socket
import time
import urllib.error
from concurrent.futures import FIRST_COMPLETED, ThreadPoolExecutor, wait
from dataclasses import asdict, dataclass, field
from typing import Callable, Dict, Optional

from ...eval.teacher_license_audit import usable
from ...hsh.llm import OpenAICompatAdapter

# alias -> weights repo (the licence row) and the model name a vLLM server
# started with ``vllm serve <repo>`` answers to.
TEACHERS = {
    "deepseek-v4.1-flash": {"repo": "deepseek-ai/DeepSeek-V4.1-Flash",
                            "served_model": "deepseek-ai/DeepSeek-V4.1-Flash"},
}

KEEP_THINKING = ("none", "turn", "all")
RETRY_HTTP = (408, 429, 500, 502, 503, 504)


class TeacherNotAllowed(ValueError):
    pass


def check_teacher(alias):
    """The teacher's row in TEACHERS, if its licence review allows
    distillation today; raise otherwise."""
    t = TEACHERS.get(alias)
    if t is None:
        raise TeacherNotAllowed(f"unknown teacher {alias!r}; known: {sorted(TEACHERS)}")
    if not usable(t["repo"]):
        raise TeacherNotAllowed(
            f"{t['repo']}: licence review missing, pending or not allowing "
            "distillation (eval/teacher_license_audit.py)")
    return t


@dataclass
class Pricing:
    """Per-token (API) or per-hour (rented pod) pricing; one currency."""
    currency: str = "USD"
    input_per_m: float = 0.0          # cache-miss input, per 1M tokens
    cached_input_per_m: float = 0.0
    output_per_m: float = 0.0
    per_hour: float = 0.0             # pod: every GPU-hour of the run

    def token_cost(self, usage):
        hit = usage.get("cached_input_tokens", 0)
        miss = usage.get("input_tokens", 0) - hit
        return (miss * self.input_per_m + hit * self.cached_input_per_m +
                usage.get("output_tokens", 0) * self.output_per_m) / 1e6

    def cost(self, totals, seconds):
        return self.token_cost(totals) + self.per_hour * seconds / 3600


@dataclass
class GenConfig:
    teacher: str = "deepseek-v4.1-flash"
    endpoint: str = "vllm"                # "vllm" | "openai" (a hosted API)
    base_url: str = "http://localhost:8000/v1"
    api_key_env: Optional[str] = None     # None: self-hosted, no key
    model: Optional[str] = None           # default: the served model name
    temperature: float = 0.6
    top_p: float = 0.95
    max_tokens: int = 8192
    thinking: bool = True
    extra: Dict = field(default_factory=dict)   # e.g. chat_template_kwargs
    samples: int = 1                      # answers per prompt
    workers: int = 16
    retries: int = 4
    timeout: float = 600.0
    keep_thinking: str = "none"
    max_cost: Optional[float] = None
    max_output_tokens: Optional[int] = None
    max_seconds: Optional[float] = None

    def body_config(self, teacher):
        cfg = {"provider": "teacher", "model": self.model or teacher["served_model"],
               "temperature": self.temperature, "top_p": self.top_p,
               "max_tokens": self.max_tokens}
        if self.endpoint == "vllm":
            # DeepSeek's chat template switches thinking on this kwarg
            cfg["chat_template_kwargs"] = {"thinking": self.thinking}
        elif self.endpoint != "openai":
            raise ValueError("endpoint must be 'vllm' or 'openai'")
        cfg.update(self.extra)
        return cfg


# ---------------------------------------------------------------- prompts

def prompt_id(text):
    return hashlib.sha256(text.encode("utf-8")).hexdigest()[:16]


def normalize_prompt(rec):
    """``{"id"?, "prompt" | "messages", "domain"?}`` -> canonical form."""
    if "messages" in rec:
        msgs = [dict(m) for m in rec["messages"]]
    elif "prompt" in rec:
        msgs = [{"role": "user", "content": rec["prompt"]}]
    else:
        raise ValueError("prompt record needs 'prompt' or 'messages'")
    if not msgs or msgs[-1]["role"] != "user":
        raise ValueError("a prompt must end with a user message")
    pid = rec.get("id") or prompt_id(json.dumps(msgs, ensure_ascii=False, sort_keys=True))
    return {"id": str(pid), "domain": rec.get("domain", "general"),
            "messages": msgs, "meta": rec.get("meta", {})}


def prompt_messages(messages, keep_thinking):
    if keep_thinking not in KEEP_THINKING:
        raise ValueError(f"keep_thinking must be one of {KEEP_THINKING}")
    out = []
    for m in messages:
        m = dict(m)
        if m["role"] == "assistant" and keep_thinking != "all":
            m.pop("reasoning", None)
        out.append(m)
    return out


META_TEMPLATE = (
    "請用台灣繁體中文（台灣用語）寫出 {n} 個彼此不同、可以獨立回答的{kind}，"
    "主題：{topic}。難度從基礎到進階都要有。只輸出一個 JSON 字串陣列，"
    "不要編號、不要答案、不要其他文字。")

DOMAIN_KIND = {
    "math": "數學題（需要多步推理，答案是一個明確的數值或式子）",
    "code": "程式設計題（Python，題目寫清楚輸入輸出）",
    "knowledge": "知識問答題",
    "reasoning": "邏輯推理題",
    "writing": "寫作或改寫任務",
    "chat": "日常對話中使用者會提出的問題或請求",
}


def meta_prompts(seeds, per_call=10):
    """Prompt records asking the teacher to write questions. ``seeds``:
    [{"domain", "topic", "n"}]; each seed becomes ceil(n / per_call)
    requests, told apart by a batch number so their ids differ."""
    out = []
    for s in seeds:
        kind = DOMAIN_KIND.get(s["domain"], "問題")
        for b in range(math.ceil(s.get("n", per_call) / per_call)):
            text = META_TEMPLATE.format(n=per_call, kind=kind, topic=s["topic"])
            out.append({"id": prompt_id(f"{s['domain']}|{s['topic']}|{b}"),
                        "domain": "meta",
                        "messages": [{"role": "user", "content": text}],
                        "meta": {"target_domain": s["domain"], "topic": s["topic"],
                                 "batch": b}})
    return out


def _json_array(text):
    a, b = text.find("["), text.rfind("]")
    if a < 0 or b <= a:
        return None
    try:
        v = json.loads(text[a:b + 1])
    except json.JSONDecodeError:
        return None
    return v if isinstance(v, list) else None


def extract_prompts(records, min_chars=8, max_chars=2000):
    """Questions from meta-prompt answers, as prompt records; exact
    duplicates (after whitespace strip) are dropped. Returns (prompts,
    stats)."""
    seen, out = set(), []
    stats = {"answers": 0, "unparsed": 0, "items": 0, "kept": 0,
             "duplicates": 0, "bad_length": 0}
    for r in records:
        stats["answers"] += 1
        items = _json_array(r.get("content", ""))
        if items is None:
            stats["unparsed"] += 1
            continue
        meta = r.get("meta", {})
        for q in items:
            stats["items"] += 1
            if not isinstance(q, str) or not (min_chars <= len(q.strip()) <= max_chars):
                stats["bad_length"] += 1
                continue
            q = q.strip()
            if q in seen:
                stats["duplicates"] += 1
                continue
            seen.add(q)
            out.append({"id": prompt_id(q), "domain": meta.get("target_domain", "general"),
                        "messages": [{"role": "user", "content": q}],
                        "meta": {"topic": meta.get("topic"), "source": "teacher_meta"}})
    stats["kept"] = len(out)
    return out, stats


# ---------------------------------------------------------------- generation

def read_jsonl(path):
    if not os.path.exists(path):
        return []
    rows = []
    with open(path, encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if not line:
                continue
            try:
                rows.append(json.loads(line))
            except json.JSONDecodeError:
                break          # a torn last line from a killed run
    return rows


def _retryable(exc):
    if isinstance(exc, urllib.error.HTTPError):
        return exc.code in RETRY_HTTP
    return isinstance(exc, (urllib.error.URLError, TimeoutError, ConnectionError,
                            socket.timeout))


class Generator:
    """Runs a prompt list against the teacher endpoint into a JSONL file.

    ``complete(request) -> adapter dict`` and ``sleep`` / ``clock`` are
    injectable so the whole loop is testable offline."""

    def __init__(self, cfg: GenConfig, complete: Optional[Callable] = None,
                 sleep=time.sleep, clock=time.monotonic):
        self.cfg = cfg
        self.teacher = check_teacher(cfg.teacher)
        if complete is None:
            complete = OpenAICompatAdapter(cfg.base_url, api_key_env=cfg.api_key_env,
                                           timeout=cfg.timeout).complete
        self.complete = complete
        self.sleep = sleep
        self.clock = clock

    def request(self, prompt):
        return {"header": {"config": self.cfg.body_config(self.teacher), "tools": []},
                "messages": prompt_messages(prompt["messages"], self.cfg.keep_thinking)}

    def _call(self, prompt, k):
        req = self.request(prompt)
        delay = 2.0
        for attempt in range(self.cfg.retries + 1):
            t0 = self.clock()
            try:
                out = self.complete(req)
            except Exception as e:  # noqa: BLE001 - classified below
                if attempt < self.cfg.retries and _retryable(e):
                    self.sleep(delay)
                    delay *= 2
                    continue
                return None, {"id": f"{prompt['id']}#{k}", "error": repr(e)[:500],
                              "attempts": attempt + 1}
            return self._record(prompt, k, out, self.clock() - t0, attempt + 1), None

    def _record(self, prompt, k, out, latency, attempts):
        usage = {kk: int((out.get("usage") or {}).get(kk, 0)) for kk in
                 ("input_tokens", "output_tokens", "reasoning_tokens",
                  "cached_input_tokens")}
        c = self.cfg
        return {"id": f"{prompt['id']}#{k}", "prompt_id": prompt["id"], "sample": k,
                "domain": prompt["domain"], "meta": prompt.get("meta", {}),
                "messages": prompt["messages"],
                "reasoning": out.get("reasoning") or "",
                "content": out.get("content") or "",
                "finish_reason": out.get("finish_reason"),
                "usage": usage,
                "teacher": {"alias": c.teacher, "repo": self.teacher["repo"],
                            "model": out.get("model") or c.body_config(self.teacher)["model"]},
                "gen": {"temperature": c.temperature, "top_p": c.top_p,
                        "max_tokens": c.max_tokens, "thinking": c.thinking,
                        "keep_thinking": c.keep_thinking},
                "latency_s": round(latency, 3), "attempts": attempts}

    def run(self, prompts, out_path, pricing: Optional[Pricing] = None, log=None):
        """Generate every (prompt, sample) not yet in ``out_path``.
        Returns a summary dict (written, errors, stop reason, totals,
        cost)."""
        prompts = [normalize_prompt(p) for p in prompts]
        done = {r["id"] for r in read_jsonl(out_path)}
        todo = [(p, k) for p in prompts for k in range(self.cfg.samples)
                if f"{p['id']}#{k}" not in done]
        pricing = pricing or Pricing()
        totals = {"input_tokens": 0, "output_tokens": 0, "reasoning_tokens": 0,
                  "cached_input_tokens": 0}
        start, written, errors, stop = self.clock(), 0, 0, None

        def limit_hit():
            secs = self.clock() - start
            if self.cfg.max_seconds is not None and secs >= self.cfg.max_seconds:
                return "max_seconds"
            if self.cfg.max_output_tokens is not None and \
                    totals["output_tokens"] >= self.cfg.max_output_tokens:
                return "max_output_tokens"
            if self.cfg.max_cost is not None and \
                    pricing.cost(totals, secs) >= self.cfg.max_cost:
                return "max_cost"
            return None

        err_path = out_path + ".errors.jsonl"
        with open(out_path, "a", encoding="utf-8") as out_fh, \
                open(err_path, "a", encoding="utf-8") as err_fh, \
                ThreadPoolExecutor(max_workers=self.cfg.workers) as pool:
            it, pending = iter(todo), set()
            while True:
                stop = stop or limit_hit()
                while stop is None and len(pending) < self.cfg.workers:
                    nxt = next(it, None)
                    if nxt is None:
                        break
                    pending.add(pool.submit(self._call, *nxt))
                if not pending:
                    break
                finished, pending = wait(pending, return_when=FIRST_COMPLETED)
                for f in finished:
                    rec, err = f.result()
                    if rec is not None:
                        out_fh.write(json.dumps(rec, ensure_ascii=False) + "\n")
                        written += 1
                        for kk in totals:
                            totals[kk] += rec["usage"][kk]
                    else:
                        err_fh.write(json.dumps(err, ensure_ascii=False) + "\n")
                        errors += 1
                out_fh.flush()
                err_fh.flush()
                if log is not None and written and written % 100 == 0:
                    log(f"{written} written, {errors} errors, "
                        f"{totals['output_tokens']} output tokens")
        secs = self.clock() - start
        return {"queued": len(todo), "written": written, "errors": errors,
                "skipped_done": len(prompts) * self.cfg.samples - len(todo),
                "stop": stop or "done", "seconds": round(secs, 1),
                "totals": totals, "cost": round(pricing.cost(totals, secs), 4),
                "currency": pricing.currency,
                "output_tokens_per_s": round(totals["output_tokens"] / secs, 1) if secs else 0.0}


# ---------------------------------------------------------------- stats

def _pct(sorted_vals, q):
    if not sorted_vals:
        return 0
    i = min(len(sorted_vals) - 1, max(0, math.ceil(q * len(sorted_vals)) - 1))
    return sorted_vals[i]


def reasoning_tokens(rec):
    """Reported reasoning tokens, or (when the server reports none) the
    output tokens split by the character share of the reasoning text.
    Returns (tokens, estimated?)."""
    u = rec["usage"]
    if u.get("reasoning_tokens"):
        return u["reasoning_tokens"], False
    r, c = len(rec.get("reasoning") or ""), len(rec.get("content") or "")
    if not r:
        return 0, False
    return round(u.get("output_tokens", 0) * r / (r + c)), True


def _block(recs):
    outs = sorted(r["usage"].get("output_tokens", 0) for r in recs)
    reas, est = 0, 0
    for r in recs:
        t, e = reasoning_tokens(r)
        reas += t
        est += e
    out_total = sum(outs)
    fin = {}
    for r in recs:
        fin[str(r.get("finish_reason"))] = fin.get(str(r.get("finish_reason")), 0) + 1
    n = len(recs)
    return {"n": n,
            "input_tokens": sum(r["usage"].get("input_tokens", 0) for r in recs),
            "output_tokens": out_total, "reasoning_tokens": reas,
            "reasoning_share": round(reas / out_total, 4) if out_total else 0.0,
            "reasoning_estimated": est,
            "output_mean": round(out_total / n, 1) if n else 0.0,
            "output_p50": _pct(outs, 0.5), "output_p95": _pct(outs, 0.95),
            "output_max": outs[-1] if outs else 0,
            "finish_reason": fin,
            "truncated_rate": round(fin.get("length", 0) / n, 4) if n else 0.0}


def token_stats(records):
    """OS1 token accounting over generated records: overall and per domain."""
    by = {}
    for r in records:
        by.setdefault(r.get("domain", "general"), []).append(r)
    return {"all": _block(list(records)),
            "by_domain": {d: _block(v) for d, v in sorted(by.items())}}


# ---------------------------------------------------------------- estimate

def estimate(n_prompts, input_tokens, output_tokens, pricing: Pricing,
             samples=1, output_tokens_per_s=None, setup_hours=0.0):
    """Cost before a run. Per-token pricing needs only token counts;
    per-hour pricing also needs the server's aggregate output throughput
    (measure it on the first few hundred prompts) and the setup time
    (weights download, server start) that is paid too."""
    n = n_prompts * samples
    usage = {"input_tokens": n * input_tokens, "output_tokens": n * output_tokens}
    hours = 0.0
    if pricing.per_hour:
        if not output_tokens_per_s:
            raise ValueError("per-hour pricing needs output_tokens_per_s")
        hours = usage["output_tokens"] / output_tokens_per_s / 3600 + setup_hours
    cost = pricing.token_cost(usage) + pricing.per_hour * hours
    return {"answers": n, **usage, "hours": round(hours, 2),
            "cost": round(cost, 2), "currency": pricing.currency,
            "pricing": asdict(pricing)}
