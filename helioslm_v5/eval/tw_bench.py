"""Traditional-Chinese multiple-choice evaluation pack (plan E3 / roadmap TW3).

Scores four-option multiple-choice benchmarks without lm-evaluation-harness,
so a checkpoint can be evaluated where lm-eval is not installed and every
step is pinned by an oracle test:

- TMMLU+ (``ikala/tmmluplus``, card: MIT; question copyright is the
  sources' own): one CSV per subject, ``<subject>_{dev,val,test}.csv``,
  columns ``question, A, B, C, D, answer``;
- TW-JusticeBench (``JerryWu/TW-JusticeBench``, card: CC BY 4.0): the
  column layout was not readable from the sandbox, so its loader takes a
  ``columns`` map; check it against the files before trusting a score;
- C-Eval (``<subject>_val.csv``: ``id, question, A..D, answer,
  explanation``) and CMMLU (``<subject>.csv``: ``Question, A..D, Answer``)
  as Simplified-Chinese controls.

TMLU (``miulab/tmlu``) is not on the Hub any more (TW1 audit, 2026-10-20).

Licence rule (TW1, docs/TW_DATASETS_2026-10-20.md): the question text is
evaluated locally and never redistributed. The files stay under
``data/tw_bench/`` (git-ignored) and results carry task ids, letters and
scores only, never question text.

Two scoring modes, both model-agnostic:

  loglikelihood  ``score_fn(context, continuations) -> [log p]``; the
                 prediction is the letter with the highest log p, the
                 confidence its softmax over the four letters (the
                 lm-eval ``acc`` metric for TMMLU+ / C-Eval / CMMLU);
  generate       ``generate_fn(prompt, i) -> text`` (or ``(text, conf)``);
                 the letter is extracted from the reply, and a reply with
                 no letter is wrong.

Phase A gates (``self_check``): on every loaded set the reference-answer
oracle scores 100% in both modes and the empty answer scores 0%.
"""
import csv
import json
import math
import re
from collections import defaultdict
from pathlib import Path

SCHEMA_ID = "tw-bench/1"
LETTERS = ("A", "B", "C", "D")
BENCHES = ("tmmluplus", "tw_justice", "ceval", "cmmlu")
_SPLIT_SUFFIX = re.compile(r"_(dev|val|validation|test|train)$")
_FULLWIDTH = str.maketrans("ＡＢＣＤ", "ABCD")
_ANSWER_PATTERNS = [
    re.compile(r"答案\s*(?:是|為|为|應為|应为|:|：)?\s*[（(]?\s*([A-D])\b"),
    re.compile(r"(?:選|选|選擇|选择)\s*[（(]?\s*([A-D])\b"),
    re.compile(r"^\s*[（(]?\s*([A-D])\s*[)）.．、:：]?(?:\s|$)"),
]
TEMPLATES = {
    # zh-TW wording; the default
    "zh": "{question}\nA. {A}\nB. {B}\nC. {C}\nD. {D}\n答案：",
    # lm-eval's tmmluplus / ceval-valid / cmmlu doc_to_text
    "lm_eval": "{question}\nA. {A}\nB. {B}\nC. {C}\nD. {D}\nAnswer:",
}
DEFAULT_COLUMNS = {
    "question": ("question", "Question", "題目", "问题"),
    "A": ("A", "a", "option_a"),
    "B": ("B", "b", "option_b"),
    "C": ("C", "c", "option_c"),
    "D": ("D", "d", "option_d"),
    "answer": ("answer", "Answer", "label", "答案"),
}


class TWBenchError(ValueError):
    """Raised for any malformed dataset record. Never swallowed."""


# ------------------------------------------------------------------ loading

def _rows(path):
    path = Path(path)
    if path.suffix == ".jsonl":
        out = []
        for i, line in enumerate(path.read_text(encoding="utf-8").splitlines()):
            if line.strip():
                try:
                    out.append(json.loads(line))
                except json.JSONDecodeError as e:
                    raise TWBenchError(f"{path.name}:{i + 1}: not JSON: {e}") from e
        return out
    with path.open(encoding="utf-8-sig", newline="") as f:
        return list(csv.DictReader(f))


def _pick(row, names, key, where):
    for n in names:
        if n in row and row[n] is not None and str(row[n]).strip() != "":
            return str(row[n]).strip()
    raise TWBenchError(f"{where}: missing {key!r} (looked for {list(names)})")


def _answer_letter(raw, where):
    a = str(raw).strip().translate(_FULLWIDTH).upper()
    if a in LETTERS:
        return a
    if a in ("0", "1", "2", "3"):
        return LETTERS[int(a)]
    raise TWBenchError(f"{where}: answer {raw!r} is not A-D or 0-3")


def subject_of(path):
    """'accounting_test.csv' -> 'accounting'."""
    return _SPLIT_SUFFIX.sub("", Path(path).stem)


def load_mcq(path, bench, split="test", columns=None):
    """MCQ records from a file or a directory of per-subject files.

    Returns [{task_id, bench, subject, question, choices, answer}] with
    ``answer`` a letter. A directory is read file by file in sorted order;
    only files whose name ends in ``_<split>`` are used when any file
    carries a split suffix (CMMLU's ``test/<subject>.csv`` has none).
    """
    if bench not in BENCHES:
        raise TWBenchError(f"unknown bench {bench!r}; one of {BENCHES}")
    cols = dict(DEFAULT_COLUMNS)
    for k, v in (columns or {}).items():
        if k not in cols:
            raise TWBenchError(f"unknown column key {k!r}")
        cols[k] = (v,) if isinstance(v, str) else tuple(v)
    path = Path(path)
    if path.is_dir():
        files = sorted(p for p in path.iterdir()
                       if p.suffix in (".csv", ".jsonl"))
        if any(_SPLIT_SUFFIX.search(p.stem) for p in files):
            files = [p for p in files if p.stem.endswith("_" + split)]
    elif path.is_file():
        files = [path]
    else:
        raise TWBenchError(f"dataset not found: {path}")
    if not files:
        raise TWBenchError(f"{path}: no {split} files")
    out = []
    for f in files:
        subject = subject_of(f)
        rows = _rows(f)
        if not rows:
            raise TWBenchError(f"{f.name}: no records")
        for i, row in enumerate(rows):
            where = f"{bench}/{f.name}[{i}]"
            out.append({
                "task_id": f"{bench}/{subject}/{i}",
                "bench": bench, "subject": subject,
                "question": _pick(row, cols["question"], "question", where),
                "choices": [_pick(row, cols[k], k, where) for k in LETTERS],
                "answer": _answer_letter(
                    _pick(row, cols["answer"], "answer", where), where),
            })
    return out


def eval_texts(tasks):
    """{task_id: question + choices}, the input of decontam.EvalIndex (E2)."""
    return {t["task_id"]: "\n".join([t["question"], *t["choices"]])
            for t in tasks}


# ------------------------------------------------------------------ prompts

def format_question(task, template="zh"):
    fields = dict(zip(LETTERS, task["choices"]), question=task["question"])
    return TEMPLATES[template].format(**fields)


def build_prompt(task, shots=(), template="zh"):
    """k-shot prompt: each shot is a solved question, then the task."""
    parts = [format_question(s, template) + " " + s["answer"] for s in shots]
    parts.append(format_question(task, template))
    return "\n\n".join(parts)


def extract_letter(text):
    """The answer letter of a generated reply, or None."""
    text = (text or "").translate(_FULLWIDTH)
    for pat in _ANSWER_PATTERNS:
        m = pat.search(text)
        if m:
            return m.group(1)
    return None


def shots_by_subject(dev_tasks, k):
    """First k dev questions of every subject (the MMLU / C-Eval convention)."""
    out = defaultdict(list)
    for t in dev_tasks:
        if len(out[t["subject"]]) < k:
            out[t["subject"]].append(t)
    return out


# --------------------------------------------------------------------- runs

def _row(task, final, conf, conf_type, mode):
    return {"family": task["bench"], "task": task["task_id"],
            "subject": task["subject"], "mode": mode,
            "expected": task["answer"], "final": final,
            "correct": final == task["answer"],
            "conf": conf, "conf_type": conf_type}


def run_loglik(tasks, score_fn, shots=None, template="zh"):
    """Highest log p(" X" | prompt) over the four letters."""
    results = []
    for t in tasks:
        prompt = build_prompt(t, (shots or {}).get(t["subject"], ()), template)
        lps = [float(x) for x in score_fn(prompt, [" " + L for L in LETTERS])]
        if len(lps) != 4 or not all(math.isfinite(x) for x in lps):
            raise TWBenchError(f"{t['task_id']}: score_fn returned {lps}")
        best = max(range(4), key=lambda j: lps[j])
        m = max(lps)
        z = sum(math.exp(x - m) for x in lps)
        results.append(_row(t, LETTERS[best], math.exp(lps[best] - m) / z,
                            "letter_softmax", "loglik"))
    return results


def run_generate(tasks, generate_fn, shots=None, template="zh"):
    """Letter extracted from generate_fn's reply; no letter is wrong."""
    results = []
    for i, t in enumerate(tasks):
        prompt = build_prompt(t, (shots or {}).get(t["subject"], ()), template)
        out = generate_fn(prompt, i)
        text, conf = out if isinstance(out, tuple) else (out, None)
        if conf is not None and not 0.0 <= float(conf) <= 1.0:
            raise TWBenchError(f"generate_fn returned conf {conf} outside [0, 1]")
        results.append(_row(t, extract_letter(text), conf,
                            "model" if conf is not None else None, "generate"))
    return results


def summarize(results, categories=None):
    """Micro accuracy, macro over subjects, and per subject.

    ``categories`` ({subject: category}) adds a macro over categories,
    the TMMLU+ headline number (STEM / humanities / social / other)."""
    if not results:
        raise TWBenchError("summarize: no results")
    by_subj = defaultdict(list)
    for r in results:
        by_subj[r["subject"]].append(r["correct"])
    subj_acc = {s: sum(v) / len(v) for s, v in sorted(by_subj.items())}
    c = sum(r["correct"] for r in results)
    out = {"n": len(results), "correct": f"{c}/{len(results)}",
           "accuracy": round(c / len(results), 4),
           "macro_subject": round(sum(subj_acc.values()) / len(subj_acc), 4),
           "subjects": {s: round(a, 4) for s, a in subj_acc.items()}}
    if categories:
        by_cat = defaultdict(list)
        for s, a in subj_acc.items():
            if s not in categories:
                raise TWBenchError(f"subject {s!r} has no category")
            by_cat[categories[s]].append(a)
        cat = {k: sum(v) / len(v) for k, v in sorted(by_cat.items())}
        out["categories"] = {k: round(a, 4) for k, a in cat.items()}
        out["macro_category"] = round(sum(cat.values()) / len(cat), 4)
    return out


def self_check(tasks, template="zh"):
    """Phase A gates on a loaded set; raises if any gate fails.

    Reference answer = 100% (generate and loglik), empty answer = 0%."""
    gold = {build_prompt(t, (), template): t["answer"] for t in tasks}

    def oracle_gen(prompt, i):
        return f"答案：{tasks[i]['answer']}"

    def oracle_score(prompt, conts):
        return [0.0 if c.strip() == gold[prompt] else -10.0 for c in conts]

    acc = {
        "reference_generate": summarize(run_generate(tasks, oracle_gen, template=template))["accuracy"],
        "reference_loglik": summarize(run_loglik(tasks, oracle_score, template=template))["accuracy"],
        "empty_generate": summarize(run_generate(tasks, lambda p, i: "", template=template))["accuracy"],
    }
    if acc["reference_generate"] != 1.0 or acc["reference_loglik"] != 1.0 \
            or acc["empty_generate"] != 0.0:
        raise TWBenchError(f"Phase A gate failed: {acc}")
    return acc


def model_score_fn(model, tokenizer, max_length=1024, device=None):
    """score_fn for a HeliosLMv5 model and a ``tokenizers`` tokenizer.

    Sum of log p(continuation | context), the context cut from the left to
    fit ``max_length``; the same arithmetic as lm_harness.HeliosHarnessLM."""
    import torch
    import torch.nn.functional as F
    dev = device or next(model.parameters()).device

    @torch.no_grad()
    def score(context, continuations):
        ctx = tokenizer.encode(context).ids
        out = []
        for cont in continuations:
            full = tokenizer.encode(context + cont).ids
            # tokens past the shared prefix belong to the continuation
            k = 0
            while k < min(len(ctx), len(full)) and ctx[k] == full[k]:
                k += 1
            k = min(k, len(full) - 1)
            ids = full[-(max_length + 1):]
            n = min(len(full) - k, len(ids) - 1)
            x = torch.tensor([ids[:-1]], dtype=torch.long, device=dev)
            logp = F.log_softmax(model(x)[0][0].float(), dim=-1)[-n:]
            target = torch.tensor(ids[-n:], device=dev)
            out.append(logp.gather(-1, target[:, None]).sum().item())
        return out
    return score


def _single_token_continuations(ctx, fulls):
    """(prefix, [token]) when the context tokenises to the same ids alone
    and inside every continuation, and each continuation adds exactly one
    token; then the per-continuation scores equal the last-position
    log-probs of one pass. Else None (a merge across the boundary)."""
    k = len(ctx)
    if k < 1 or any(len(f) != k + 1 or f[:k] != ctx for f in fulls):
        return None
    return ctx, [f[k] for f in fulls]


def hf_score_fn(model, tokenizer, max_length=2048, device=None):
    """score_fn for a Hugging Face causal LM and its tokenizer (route B
    base / baseline gap eval, v5.128).

    Same arithmetic as ``model_score_fn``. When every continuation is one
    token after a shared prefix (the four letters, usually) the four
    log p come from ONE forward pass over the prefix; otherwise each
    continuation gets its own pass."""
    import torch
    import torch.nn.functional as F
    dev = device or next(model.parameters()).device

    def enc(text):
        return list(tokenizer(text, add_special_tokens=True)["input_ids"])

    def logits(ids):
        x = torch.tensor([ids], dtype=torch.long, device=dev)
        out = model(input_ids=x)
        return (out.logits if hasattr(out, "logits") else out[0])[0].float()

    @torch.no_grad()
    def score(context, continuations):
        ctx = enc(context)
        fulls = [enc(context + c) for c in continuations]
        fast = _single_token_continuations(ctx, fulls)
        if fast is not None:
            prefix, toks = fast
            lp = F.log_softmax(logits(prefix[-max_length:])[-1], dim=-1)
            return [lp[t].item() for t in toks]
        out = []
        for full in fulls:
            k = 0
            while k < min(len(ctx), len(full)) and ctx[k] == full[k]:
                k += 1
            k = min(k, len(full) - 1)
            ids = full[-(max_length + 1):]
            n = min(len(full) - k, len(ids) - 1)
            lp = F.log_softmax(logits(ids[:-1]), dim=-1)[-n:]
            target = torch.tensor(ids[-n:], device=dev)
            out.append(lp.gather(-1, target[:, None]).sum().item())
        return out
    return score


def write_results(path, bench, results, summary, meta=None):
    """Results JSON; question text is never written (licence rule)."""
    for r in results:
        if set(r) - {"family", "task", "subject", "mode", "expected", "final",
                     "correct", "conf", "conf_type"}:
            raise TWBenchError("result row carries extra fields")
    payload = {"schema": SCHEMA_ID, "bench": bench, "summary": summary,
               "meta": meta or {}, "results": results}
    Path(path).write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n",
                          encoding="utf-8")
    return payload
