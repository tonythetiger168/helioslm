"""Verifiable reward for multiple-choice RL (v5.207).

The DeepSeek-R1 recipe on our exam-style tasks: the policy reasons, then
states one letter; the reward checks the letter against the key, so no
reward model is needed. ``examples/hdl_rl_hf.py train --reward mcq``
runs the same GRPO loop as the HDL track on these tasks.

  tasks      ``make_task`` writes the zh-TW prompt (question, A-D, "think
             step by step, last line 答案：X"); ``tasks_from_mcq`` takes
             ``tw_bench.load_mcq`` rows (train / dev splits of public
             sets); ``tasks_from_distill`` takes teacher answers of
             ``distill_gen`` MCQ prompts and uses the teacher's letter as
             the key, keeping only questions where ``min_agree`` of its
             samples chose the same letter;
  decontam   ``drop_contaminated`` removes tasks that overlap an
             evaluation set (``decontam.EvalIndex``, 13-gram), so RL never
             trains on a scored question;
  reward     1 for the key's letter, 0 for a wrong or missing one. The
             letter is read from the text after any ``</think>``, last
             「答案：X」 first, so a letter mentioned while reasoning does
             not count. A zh-TW task whose answer is Simplified Chinese
             loses ``lang_penalty`` (keeps the model writing Taiwan
             Traditional Chinese while it learns to reason).
"""
import re
from collections import Counter

from ..src.training.data_quality import zh_variant
from . import tw_bench as tb
from .decontam import EvalIndex

LETTERS = tb.LETTERS
LANG_PENALTY = 0.2
INSTRUCTION = "以下是單選題。請一步一步思考，最後一行只寫「答案：X」（X 為 A、B、C 或 D）。"
_FINAL = re.compile(r"答案\s*(?:是|為|为|:|：)\s*[（(]?\s*([A-D])\b")
_THINK_END = re.compile(r"</think>", re.I)


def make_task(question, choices, answer, *, task_id=None, subject=None, lang="tw"):
    if len(choices) != 4:
        raise ValueError("need 4 choices")
    answer = str(answer).strip().upper()
    if answer not in LETTERS:
        raise ValueError(f"answer {answer!r} is not A-D")
    body = "\n".join([question.strip()] + [f"{k}. {c}" for k, c in zip(LETTERS, choices)])
    return {"kind": "mcq", "task_id": task_id, "subject": subject, "lang": lang,
            "question": question.strip(), "choices": list(choices), "answer": answer,
            "prompt": f"{INSTRUCTION}\n\n{body}"}


def tasks_from_mcq(rows, lang="tw"):
    """Tasks from ``tw_bench.load_mcq`` rows."""
    return [make_task(r["question"], r["choices"], r["answer"], task_id=r["task_id"],
                      subject=r.get("subject"), lang=lang) for r in rows]


def final_letter(text):
    """The letter the reply commits to, or None: the last 「答案：X」 after
    ``</think>`` (or in the whole text when there is no think block),
    else ``tw_bench.extract_letter`` on the last line."""
    text = (text or "").translate(tb._FULLWIDTH)
    parts = _THINK_END.split(text)
    tail = parts[-1] if len(parts) > 1 else text
    found = _FINAL.findall(tail)
    if found:
        return found[-1]
    lines = [x for x in tail.strip().splitlines() if x.strip()]
    return tb.extract_letter(lines[-1]) if lines else None


_CHOICE_LINE = re.compile(r"^\s*([A-D])\s*[.．、:：)）]\s*(.+?)\s*$")


def parse_mcq_prompt(text):
    """(question, [A..D]) out of a prompt that lists its choices one per
    line as ``A. ...``; None when it does not."""
    q, ch = [], {}
    for line in (text or "").splitlines():
        m = _CHOICE_LINE.match(line)
        if m and m.group(1) not in ch:
            ch[m.group(1)] = m.group(2)
        elif not ch:
            q.append(line)
    if sorted(ch) != list(LETTERS):
        return None
    question = "\n".join(q).strip()
    return (question, [ch[k] for k in LETTERS]) if question else None


def tasks_from_distill(records, min_agree=1.0, lang="tw"):
    """Tasks from teacher answers (``distill_gen`` records, any number of
    samples per prompt). The key is the teacher's majority letter; a
    prompt is kept when that letter has at least ``min_agree`` of its
    answered samples (1.0: all agree). Returns (tasks, stats)."""
    by = {}
    for r in records:
        if r.get("finish_reason") != "stop":
            continue
        user = [m["content"] for m in r.get("messages", []) if m.get("role") == "user"]
        if not user:
            continue
        g = by.setdefault(r.get("prompt_id") or r["id"], {"prompt": user[-1], "letters": []})
        g["letters"].append(final_letter(r.get("content")))
    tasks, st = [], {"prompts": len(by), "no_letter": 0, "disagree": 0, "unparsed": 0}
    for pid, g in sorted(by.items()):
        got = [x for x in g["letters"] if x]
        if not got:
            st["no_letter"] += 1
            continue
        letter, n = Counter(got).most_common(1)[0]
        if n / len(got) < min_agree:
            st["disagree"] += 1
            continue
        parsed = parse_mcq_prompt(g["prompt"])
        if parsed is None:
            st["unparsed"] += 1
            continue
        tasks.append(make_task(*parsed, letter, task_id=pid, lang=lang))
    st["tasks"] = len(tasks)
    return tasks, st


def drop_contaminated(tasks, eval_items):
    """Tasks with no 13-gram overlap with ``eval_items`` ({id: text}, e.g.
    ``tw_bench.eval_texts(load_mcq(...))``). Returns (kept, n_dropped)."""
    index = EvalIndex(eval_items)
    kept = [t for t in tasks
            if not index.scan("\n".join([t["question"], *t["choices"]]))]
    return kept, len(tasks) - len(kept)


def reward(text, task, lang_penalty=LANG_PENALTY):
    """``(reward, info)`` for one sampled reply to ``task``."""
    pred = final_letter(text)
    ok = pred == task["answer"]
    r = 1.0 if ok else 0.0
    info = {"ok": ok, "pred": pred}
    if pred is None:
        info["why"] = "no answer"
    parts = _THINK_END.split(text or "")
    if task.get("lang") == "tw" and lang_penalty and zh_variant(parts[-1]) == "cn":
        r = max(0.0, r - lang_penalty)
        info["simplified"] = True
    return r, info
