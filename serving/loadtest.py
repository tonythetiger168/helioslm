#!/usr/bin/env python3
"""Streaming load test for an OpenAI-compatible vLLM server (production sizing).

Runs a concurrency sweep with two scenarios and prints one JSON line per level:

* ``chat``: short zh-TW / English prompts, 512 new tokens.
* ``rag``: a ~3k-token product document plus a question, 256 new tokens.

Each line has TTFT, inter-token latency and end-to-end latency
(p50/p95/p99), per-request decode tok/s, aggregate output tok/s, requests
per second and the error count. ``--soak SECONDS`` repeats the first
concurrency level until the time is up (stability / leak check).

Needs ``aiohttp`` (present in the vLLM image). The API key, if any, is read
from ``SERVING_API_KEY`` and sent as a bearer token; it is never printed.

    python serving/loadtest.py --base-url http://127.0.0.1:8000/v1 \\
        --model helioslm --scenario chat --conc 1,4,8,16,32,64

Measured numbers: docs/SERVING_GPU_TEST.md.
"""

from __future__ import annotations

import argparse
import asyncio
import json
import os
import statistics
import time

API_KEY_ENV = "SERVING_API_KEY"
MAX_TOKENS = {"chat": 512, "rag": 256}

PROMPTS = [
 "請用繁體中文解釋什麼是推測解碼（speculative decoding），並舉一個生活化的比喻。",
 "台灣的全民健保制度有哪些優點與挑戰？請條列說明。",
 "請寫一封給客戶的道歉信，說明出貨延誤三天的原因與補償方案。",
 "某商品原價 1,250 元，打八五折後再減 100 元，最後售價多少？請一步一步計算。",
 "請把下列句子翻譯成英文：「這份報告需要在週五下班前交給主管審核。」並說明用詞選擇。",
 "用 Python 寫一個函式，找出字串中最長的不重複字元子字串，並附上時間複雜度分析。",
 "請比較太陽能與離岸風電在台灣發展的優缺點。",
 "Explain the difference between TCP and UDP with examples of when to use each.",
 "請以國中老師的口吻，向學生說明光合作用的過程。",
 "寫一段 Verilog 程式碼，實作一個 4 位元同步上數計數器，含非同步重置。",
 "請為一間位於台中的咖啡廳寫三則 Instagram 貼文文案。",
 "如果一個等差數列首項為 3，公差為 4，求前 20 項和，請列出算式。",
 "請說明 Transformer 模型中的注意力機制，並解釋為什麼需要多頭注意力。",
 "請規劃一個三天兩夜的花東旅遊行程，包含交通與住宿建議。",
 "Write a SQL query that returns the top 3 customers by total order amount in 2025, with ties handled.",
 "請說明勞基法中加班費的計算方式，並舉例平日加班 3 小時的情況。",
]
DOC = ("以下是公司內部的產品規格與客服紀錄摘要。" + "".join(
    f"第{i}條：型號 HX-{100+i} 的保固期為 {12 + i % 3 * 12} 個月，若於保固期內發生非人為損壞，客戶可至全台 {20+i} 家服務據點免費維修；"
    f"送修平均需要 {3 + i % 5} 個工作天，期間可申請備用機。客服紀錄顯示此型號最常見的問題是電池續航與螢幕閃爍，佔比約 {10 + i % 7}%。"
    for i in range(60)))


def messages(i: int, scenario: str) -> list:
    """The i-th request body messages for a scenario."""
    if scenario == "rag":
        q = f"\n\n根據上述資料，回答：型號 HX-{100 + i % 60} 的保固與送修規定是什麼？最常見問題是什麼？"
        return [{"role": "user", "content": DOC + q}]
    return [{"role": "user", "content": PROMPTS[i % len(PROMPTS)]}]


def pct(values, q):
    """Nearest-rank percentile (q in 0..100); None for an empty list."""
    if not values:
        return None
    v = sorted(values)
    return v[min(len(v) - 1, int(round(q / 100 * (len(v) - 1))))]


def summarize(results: list, wall: float, scenario: str, conc: int) -> dict:
    """One sweep level: results are dicts from one_request (or {"err": ...})."""
    ok = [r for r in results if "err" not in r]
    errs = [r["err"] for r in results if "err" in r]
    itl = [x for r in ok for x in r["itl"]]

    def p3(k):
        return {q: pct([r[k] for r in ok], q) for q in (50, 95, 99)}

    return {
        "scenario": scenario, "conc": conc, "n": len(results), "ok": len(ok),
        "errors": len(errs), "err_samples": errs[:3], "wall_s": wall,
        "agg_out_tps": sum(r["tok"] for r in ok) / wall if wall > 0 else 0.0,
        "req_per_s": len(ok) / wall if wall > 0 else 0.0,
        "mean_prompt_tok": statistics.mean(r["pin"] for r in ok) if ok else None,
        "mean_out_tok": statistics.mean(r["tok"] for r in ok) if ok else None,
        "ttft_s": p3("ttft"), "e2e_s": p3("e2e"), "per_req_tps": p3("tps"),
        "itl_ms": {q: (pct(itl, q) or 0) * 1000 for q in (50, 95, 99)},
    }


async def one_request(session, url, headers, model, i, scenario):
    import aiohttp
    t0 = time.perf_counter()
    ttft = last = None
    itl, ntok, pin = [], 0, 0
    body = {"model": model, "messages": messages(i, scenario),
            "max_tokens": MAX_TOKENS[scenario], "temperature": 0.7,
            "stream": True, "stream_options": {"include_usage": True}}
    try:
        async with session.post(url, json=body, headers=headers,
                                timeout=aiohttp.ClientTimeout(total=900)) as r:
            if r.status != 200:
                return {"err": f"http {r.status}"}
            async for raw in r.content:
                line = raw.decode().strip()
                if not line.startswith("data:") or line == "data: [DONE]":
                    continue
                d = json.loads(line[5:])
                if d.get("usage"):
                    ntok = d["usage"]["completion_tokens"]
                    pin = d["usage"]["prompt_tokens"]
                ch = d.get("choices") or []
                if ch and ch[0].get("delta", {}).get("content"):
                    now = time.perf_counter()
                    if ttft is None:
                        ttft = now - t0
                    else:
                        itl.append(now - last)
                    last = now
    except Exception as e:  # network / timeout: count as an error
        return {"err": type(e).__name__}
    e2e = time.perf_counter() - t0
    if ttft is None:
        return {"err": "no tokens"}
    return {"ttft": ttft, "e2e": e2e, "itl": itl, "tok": ntok, "pin": pin,
            "tps": (ntok - 1) / max(e2e - ttft, 1e-6)}


async def run_level(url, headers, model, scenario, conc, n):
    import aiohttp
    sem = asyncio.Semaphore(conc)
    async with aiohttp.ClientSession(
            connector=aiohttp.TCPConnector(limit=conc + 8)) as s:
        async def w(i):
            async with sem:
                return await one_request(s, url, headers, model, i, scenario)
        t = time.perf_counter()
        res = await asyncio.gather(*[w(i) for i in range(n)])
        wall = time.perf_counter() - t
    return summarize(res, wall, scenario, conc)


async def amain(a):
    url = a.base_url.rstrip("/") + "/chat/completions"
    key = os.environ.get(API_KEY_ENV)
    headers = {"Authorization": f"Bearer {key}"} if key else {}
    concs = [int(x) for x in a.conc.split(",")]
    await run_level(url, headers, a.model, a.scenario, 2, 4)  # warm-up
    if a.soak:
        t_end, k = time.time() + a.soak, 0
        while time.time() < t_end:
            r = await run_level(url, headers, a.model, a.scenario, concs[0], concs[0] * 2)
            r.update(arm=a.arm, soak_round=k, t=time.time())
            k += 1
            print(json.dumps(r, ensure_ascii=False), flush=True)
            if r["ok"] == 0:
                await asyncio.sleep(5)
        return
    for c in concs:
        r = await run_level(url, headers, a.model, a.scenario, c, max(8, c * a.mult))
        r["arm"] = a.arm
        print(json.dumps(r, ensure_ascii=False), flush=True)
        if r["errors"] > r["n"] // 2:
            break


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("--base-url", required=True, help="ends with /v1")
    p.add_argument("--model", default="helioslm")
    p.add_argument("--scenario", choices=sorted(MAX_TOKENS), default="chat")
    p.add_argument("--conc", default="1,4,8,16,32,64")
    p.add_argument("--mult", type=int, default=3, help="requests per level = conc x mult (min 8)")
    p.add_argument("--soak", type=int, default=0, help="seconds; repeat the first level")
    p.add_argument("--arm", default="run", help="label copied into each JSON line")
    asyncio.run(amain(p.parse_args(argv)))


if __name__ == "__main__":
    main()
