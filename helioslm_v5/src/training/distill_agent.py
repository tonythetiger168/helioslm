"""Agent-trajectory distillation through the hsh harness (v5.114, D4).

Learning from DeepSeek Harness, literally: the teacher (DeepSeek-V4.1-Flash
on the vLLM pod) is run as an hsh agent with the ``minimal`` preset's
tools (bash, str_replace_editor) on tasks whose outcome a program can
check. Only trajectories that pass the check are kept (rejection
sampling), and each of their model steps becomes one SFT example in the
exact text the student sees when hsh drives it through
``TextModelAdapter``:

  prompt   ``render_transcript(request)`` of the request the teacher was
           sent at that step (tools line, system, user, earlier steps
           and tool results), rebuilt from the session log with
           ``rebuild_requests`` (the log is the only source, so this is
           byte-for-byte what was sent);
  target   ``assistant_text(message)``: a ``@@tool@@{"calls": ...}@@end@@``
           block or the final text reply.

Thinking: the teacher runs with thinking on and ``keep_thinking="turn"``
(a task is one turn, so its reasoning stays visible across tool calls, as
DeepSeek's tool use expects); the student examples carry no reasoning,
because the text protocol has no place for it.

Tasks (``make_tasks``) are generated with their answers, in Traditional
Chinese, from five templates: sum a CSV column, fix a bug so a test
passes (the test file must stay unchanged), rewrite Mainland terms to
Taiwanese ones, find the file holding a code word, filter and sort JSON.

The tools run real processes in a per-task temporary workspace with the
harness's rlimits; that is confinement, not a sandbox, so run this on the
disposable pod, not on a machine that matters.
"""
import hashlib
import json
import os
import random
import subprocess
import sys
import tempfile
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

from ...eval.teacher_license_audit import usable
from ...hsh import (LLM, Agents, Context, OpenAICompatAdapter, Sessions, Tools,
                    Workspace, create_agent, rebuild_requests, render_transcript,
                    usage_summary)
from ...hsh.llm import assistant_text

TEACHER_REPO = {"deepseek-v4.1-flash": "deepseek-ai/DeepSeek-V4.1-Flash"}

SYSTEM = ("你在一個工作目錄裡完成使用者交代的任務。先用工具查看檔案，完成後再用工具"
          "驗證結果。最後用台灣繁體中文簡短回報做了什麼。")

TW_TERMS = {"軟件": "軟體", "信息": "資訊", "視頻": "影片", "鼠標": "滑鼠",
            "程序": "程式", "網絡": "網路", "打印": "列印", "數據庫": "資料庫",
            "默認": "預設", "屏幕": "螢幕"}
STORES = ["台北信義店", "台中逢甲店", "高雄巨蛋店", "台南永康店", "新竹竹北店"]
NAMES = ["陳怡君", "林志明", "黃雅婷", "張家豪", "李佳穎", "王俊傑", "吳淑芬", "劉冠宇"]
BUGS = [  # (buggy body, function, test cases)
    ("    return sum(xs) / len(xs) + 1", "average", [([1, 2, 3], 2.0), ([4], 4.0)]),
    ("    return max(xs[1:])", "largest", [([5, 1, 2], 5), ([1, 9], 9)]),
    ("    return [x for x in xs if x % 2 == 1]", "evens", [([1, 2, 3, 4], [2, 4]), ([7], [])]),
    ("    return s[::-1].upper()", "reverse", [("abc", "cba"), ("台灣", "灣台")]),
]


def _sha(text):
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def make_tasks(n, seed=0):
    """``n`` tasks cycling through the five templates. Each task carries
    its files, its prompt and what ``check_task`` compares against."""
    rng = random.Random(seed)
    makers = (_csv_sum, _fix_bug, _term_replace, _find_file, _json_filter)
    tasks = []
    for i in range(n):
        t = makers[i % len(makers)](rng)
        t["id"] = f"{t['kind']}-{seed}-{i}"
        tasks.append(t)
    return tasks


def _csv_sum(rng):
    rows = [(f"2026-10-{rng.randint(1, 28):02d}", rng.choice(STORES), rng.randint(100, 9999))
            for _ in range(rng.randint(8, 30))]
    store = rng.choice(sorted({r[1] for r in rows}))
    text = "日期,門市,金額\n" + "".join(f"{d},{s},{a}\n" for d, s, a in rows)
    return {"kind": "csv_sum", "files": {"sales.csv": text},
            "prompt": f"sales.csv 是各門市的銷售紀錄。請算出「{store}」的金額總和，"
                      "只把這個整數寫進 answer.txt。",
            "check": {"type": "file_equals", "path": "answer.txt",
                      "value": str(sum(a for _, s, a in rows if s == store))}}


def _fix_bug(rng):
    body, fn, cases = rng.choice(BUGS)
    arg = "s" if fn == "reverse" else "xs"
    src = f"def {fn}({arg}):\n{body}\n"
    test = (f"from lib import {fn}\n\n" +
            "".join(f"assert {fn}({a!r}) == {b!r}, {fn}({a!r})\n" for a, b in cases) +
            "print('ok')\n")
    return {"kind": "fix_bug", "files": {"lib.py": src, "test_lib.py": test},
            "prompt": f"lib.py 裡的 {fn} 有錯，`python3 test_lib.py` 會失敗。"
                      "請修正 lib.py 讓測試通過，不要修改 test_lib.py。",
            "check": {"type": "command", "command": [sys.executable, "test_lib.py"],
                      "unchanged": {"test_lib.py": _sha(test)}}}


def _term_replace(rng):
    terms = rng.sample(sorted(TW_TERMS), 4)
    lines = [f"- 第{i + 1}點：請檢查{t}的設定，確認{t}可以正常使用。" for i, t in enumerate(terms)]
    text = "# 會議筆記\n\n" + "\n".join(lines) + "\n"
    want = text
    for t in terms:
        want = want.replace(t, TW_TERMS[t])
    return {"kind": "term_replace", "files": {"notes.md": text},
            "prompt": "notes.md 用了中國大陸的用語。請改成台灣慣用語（例如「軟件」改成「軟體」），"
                      "其他文字保持不變。",
            "check": {"type": "file_equals", "path": "notes.md", "value": want}}


def _find_file(rng):
    code = f"{rng.choice('ABCDEFGHJK')}{rng.randint(100, 999)}"
    names = [f"report_{k}.txt" for k in rng.sample(range(10, 99), 6)]
    target = rng.choice(names)
    files = {}
    for nm in names:
        filler = "\n".join(f"第{j}行：例行紀錄，沒有特別事項。" for j in range(rng.randint(5, 20)))
        files[f"docs/{nm}"] = filler + (f"\n專案代號 {code} 已核准。\n" if nm == target else "\n")
    return {"kind": "find_file", "files": files,
            "prompt": f"docs/ 底下哪一個檔案提到專案代號 {code}？最後回覆檔名。",
            "check": {"type": "answer_names", "value": target,
                      "others": [n for n in names if n != target]}}


def _json_filter(rng):
    users = [{"name": nm, "age": rng.randint(10, 60)} for nm in rng.sample(NAMES, 6)]
    adults = sorted((u for u in users if u["age"] >= 18), key=lambda u: u["age"])
    return {"kind": "json_filter",
            "files": {"users.json": json.dumps(users, ensure_ascii=False, indent=1)},
            "prompt": "users.json 是使用者清單。請建立 adults.json：只保留 age 至少 18 的人，"
                      "依 age 由小到大排序，格式和原本一樣是 JSON 陣列。",
            "check": {"type": "json_equals", "path": "adults.json", "value": adults}}


def setup_workspace(task, root):
    for rel, text in task["files"].items():
        p = Path(root) / rel
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text(text, encoding="utf-8")


def check_task(task, root, final_text):
    """Did the agent do the task? Compares the workspace (or the final
    reply) with what the task was generated with."""
    c = task["check"]
    root = Path(root)

    def read(rel):
        p = root / rel
        return p.read_text(encoding="utf-8") if p.is_file() else None

    if c["type"] == "file_equals":
        got = read(c["path"])
        return got is not None and got.strip() == c["value"].strip()
    if c["type"] == "json_equals":
        try:
            return json.loads(read(c["path"]) or "null") == c["value"]
        except json.JSONDecodeError:
            return False
    if c["type"] == "answer_names":
        text = final_text or ""
        return c["value"] in text and not any(o in text for o in c["others"])
    if c["type"] == "command":
        for rel, sha in c.get("unchanged", {}).items():
            if read(rel) is None or _sha(read(rel)) != sha:
                return False
        # a fresh bytecode cache: an edit within the same second as an
        # earlier run, at the same size (`% 2 == 1` -> `% 2 == 0`), would
        # otherwise import the stale lib.pyc from the workspace
        with tempfile.TemporaryDirectory() as cache:
            env = dict(os.environ, PYTHONPYCACHEPREFIX=cache)
            try:
                r = subprocess.run(c["command"], cwd=root, capture_output=True,
                                   timeout=30, env=env)
            except subprocess.TimeoutExpired:
                return False
        return r.returncode == 0
    raise ValueError(f"unknown check type {c['type']!r}")


def agent_config(model, temperature=0.6, max_tokens=4096, thinking=True, endpoint="vllm"):
    cfg = {"provider": "teacher", "model": model, "temperature": temperature,
           "max_tokens": max_tokens}
    if endpoint == "vllm":
        cfg["chat_template_kwargs"] = {"thinking": thinking}
    return cfg


def run_task(task, adapter, config, max_steps=12, workdir=None):
    """Run one task with a fresh hsh context and workspace. Returns the
    trajectory record (the session log included)."""
    with tempfile.TemporaryDirectory(dir=workdir) as root:
        setup_workspace(task, root)
        ctx = Context()
        for plug in (Sessions, Tools, LLM, Agents):
            ctx.plugin(plug)
        ctx.plugin(Workspace, {"root": root})
        ctx.llm.register("teacher", adapter)
        agent = create_agent(ctx, task["id"], config, "minimal", system_prompt=SYSTEM,
                             keep_thinking="turn", max_steps=max_steps)
        error = None
        try:
            final = agent.send(task["prompt"])
        except Exception as e:  # noqa: BLE001 - recorded, the task just fails
            final, error = None, repr(e)[:500]
        events = agent.session.events
        ends = [e["data"]["reason"] for e in events if e["type"] == "turn/end"]
        reason = ends[-1] if ends else "none"
        ok = error is None and reason == "completed" and check_task(task, root, final)
        steps = sum(e["type"] == "assistant/message" for e in events)
        return {"id": task["id"], "kind": task["kind"], "verified": bool(ok),
                "reason": reason, "error": error, "final": final, "steps": steps,
                "usage": usage_summary(events)["total"], "events": events}


def trajectory_examples(rec):
    """SFT records (prompt_text / target_text) for every model step of a
    trajectory, in the student's text protocol."""
    events = rec["events"]
    by_seq = {e["seq"]: e for e in events}
    out = []
    for k, (seq, request) in enumerate(rebuild_requests(events, keep_thinking="none")):
        ev = by_seq[seq]
        if ev["type"] != "assistant/message":
            continue                       # a failed attempt has no answer
        out.append({"id": f"{rec['id']}@{k}", "prompt_id": rec["id"], "domain": "agent",
                    "prompt_text": render_transcript(request),
                    "target_text": assistant_text(ev["data"]["message"])})
    return out


def _read_jsonl(path):
    if not os.path.exists(path):
        return []
    rows = []
    with open(path, encoding="utf-8") as fh:
        for line in fh:
            try:
                rows.append(json.loads(line))
            except json.JSONDecodeError:
                break
    return rows


def run_tasks(tasks, out_path, teacher="deepseek-v4.1-flash", base_url="http://localhost:8000/v1",
              model=None, api_key_env=None, workers=16, adapter=None, log=None, **cfg):
    """Run tasks into ``out_path`` (JSONL, one trajectory per line,
    resumable by id). Returns a summary with the verified rate per kind."""
    repo = TEACHER_REPO.get(teacher)
    if repo is None or not usable(repo):
        raise ValueError(f"teacher {teacher!r} is not an allowed, pinned open-weight teacher")
    adapter = adapter or OpenAICompatAdapter(base_url, api_key_env=api_key_env, timeout=600)
    config = agent_config(model or repo, **cfg)
    done = {r["id"] for r in _read_jsonl(out_path)}
    todo = [t for t in tasks if t["id"] not in done]
    n = 0
    with open(out_path, "a", encoding="utf-8") as fh, ThreadPoolExecutor(workers) as pool:
        futs = [pool.submit(run_task, t, adapter, config) for t in todo]
        for f in as_completed(futs):
            fh.write(json.dumps(f.result(), ensure_ascii=False) + "\n")
            fh.flush()
            n += 1
            if log is not None and n % 50 == 0:
                log(f"{n}/{len(todo)} trajectories")
    return summarize(_read_jsonl(out_path))


def summarize(records):
    kinds = {}
    for r in records:
        k = kinds.setdefault(r["kind"], {"n": 0, "verified": 0, "steps": 0, "output_tokens": 0})
        k["n"] += 1
        k["verified"] += r["verified"]
        k["steps"] += r["steps"]
        k["output_tokens"] += r["usage"].get("output_tokens", 0)
    for k in kinds.values():
        k["verified_rate"] = round(k["verified"] / k["n"], 4) if k["n"] else 0.0
        k["mean_steps"] = round(k["steps"] / k["n"], 2) if k["n"] else 0.0
    n = len(records)
    v = sum(r["verified"] for r in records)
    return {"n": n, "verified": v, "verified_rate": round(v / n, 4) if n else 0.0,
            "by_kind": dict(sorted(kinds.items()))}
