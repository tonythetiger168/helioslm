"""grounded_report_run.py — run JSON -> report prompts on any number of
models behind OpenAI-compatible endpoints and grade every report with
the rule-based grounding grader (v5.214, helioslm_v5/eval/grounded_report.py).

  python examples/grounded_report_run.py \\
      --cases data/pulmoguard/findings_example.json \\
      --model helios=http://localhost:8080/v1/helios \\
      --model qwen=http://localhost:8081/v1/qwen3-14b \\
      --lang zh --out runs/grounded_report.json

``--cases`` is one findings JSON or a JSONL file with one findings dict
per line (synthetic or de-identified only). Each ``--model`` is
``name=BASE_URL/MODEL_ID``: the request goes to
``BASE_URL/chat/completions`` with ``model`` MODEL_ID; the key comes from
``--api-key-env`` (default none, for local servers). Every model gets the
same prompt (``--lang`` zh or en) at temperature 0. The output JSON has
one row per (model, case) with the report text and its grade, plus a
per-model summary.
"""
import argparse
import json
import os
import sys
import urllib.request
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from helioslm_v5.eval.grounded_report import grade, summarize  # noqa: E402

PROMPTS = {
    "zh": ("你是放射科報告撰寫助理。只根據下列 JSON 撰寫繁體中文（台灣）"
           "低劑量胸部電腦斷層報告的發現與結論段落。每個結節寫出肺葉、"
           "直徑（公釐）、體積（立方公釐）、類型與 Lung-RADS 類別；JSON "
           "沒有的數值寫「未測量」，不可自行補上任何數字或發現。\n\nJSON：\n{json}"),
    "en": ("You are a radiology report assistant. Using only the JSON below, "
           "write the findings and impression of a low-dose chest CT report "
           "in English. For each nodule state the lobe, diameter (mm), volume "
           "(mm3), type and Lung-RADS category; write \"not measured\" for any "
           "value missing from the JSON and never add numbers or findings.\n\n"
           "JSON:\n{json}"),
}


def load_cases(path):
    text = Path(path).read_text(encoding="utf-8")
    try:
        obj = json.loads(text)
        return obj if isinstance(obj, list) else [obj]
    except json.JSONDecodeError:
        return [json.loads(line) for line in text.splitlines() if line.strip()]


def parse_model(spec):
    name, _, url = spec.partition("=")
    base, _, model_id = url.rstrip("/").rpartition("/")
    if not name or not base or not model_id:
        raise SystemExit(f"--model wants name=BASE_URL/MODEL_ID, got {spec!r}")
    return name, base, model_id


def make_client(base, model_id, api_key_env=None, timeout=300, opener=None):
    """complete(prompt) -> text via POST {base}/chat/completions."""
    opener = opener or urllib.request.urlopen

    def complete(prompt, max_tokens=1024):
        body = json.dumps({"model": model_id, "temperature": 0,
                           "max_tokens": max_tokens,
                           "messages": [{"role": "user", "content": prompt}]}).encode()
        headers = {"Content-Type": "application/json"}
        key = os.environ.get(api_key_env) if api_key_env else None
        if key:
            headers["Authorization"] = f"Bearer {key}"
        req = urllib.request.Request(base + "/chat/completions", data=body,
                                     headers=headers, method="POST")
        with opener(req, timeout=timeout) as resp:
            out = json.loads(resp.read().decode())
        return out["choices"][0]["message"].get("content") or ""
    return complete


def run(cases, clients, lang="zh"):
    """clients: {name: complete(prompt) -> text}. Returns the result dict."""
    rows, by_model = [], {}
    for name in sorted(clients):
        for i, case in enumerate(cases):
            prompt = PROMPTS[lang].format(json=json.dumps(case, ensure_ascii=False, indent=2))
            try:
                text, err = clients[name](prompt), None
            except Exception as e:     # one failed call must not end the run
                text, err = "", f"{type(e).__name__}: {e}"
            g = grade(case, text)
            rows.append({"model": name, "case": case.get("exam_id", i),
                         "text": text, "error": err, "grade": g})
            by_model.setdefault(name, []).append(g)
    return {"lang": lang, "rows": rows,
            "summary": {m: summarize(gs) for m, gs in by_model.items()}}


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--cases", required=True)
    ap.add_argument("--model", action="append", required=True)
    ap.add_argument("--lang", choices=sorted(PROMPTS), default="zh")
    ap.add_argument("--api-key-env")
    ap.add_argument("--out", required=True)
    a = ap.parse_args(argv)
    clients = {}
    for spec in a.model:
        name, base, model_id = parse_model(spec)
        clients[name] = make_client(base, model_id, a.api_key_env)
    res = run(load_cases(a.cases), clients, a.lang)
    Path(a.out).parent.mkdir(parents=True, exist_ok=True)
    Path(a.out).write_text(json.dumps(res, ensure_ascii=False, indent=2))
    print(json.dumps(res["summary"], indent=2))
    return res


if __name__ == "__main__":
    main()
