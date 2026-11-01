"""PulmoGuard report bake-off gate (v5.218, docs/prereg/pulmoguard-report.json).

Turns per-model reports and an external physician rating sheet into a
results file for :func:`helioslm_v5.eval.prereg.check_gates`:

* ``report_grounding / unverified_facts``: facts the findings JSON does
  not support, i.e. the grader's unverified facts plus contradictions (a
  value that differs from the nodule it is stated for is not supported
  either), summed over every report and every seed of ours;
* ``report_omission / omission_rate``: omitted required fields over
  required fields;
* ``missing_field_abstention / accuracy``: null measurement fields written
  as 未測量 / not measured (a fabricated value counts wrong), via
  src/training/report_abstain.py;
* ``physician_rating / score``: mean rating of ours, plus a paired
  bootstrap of (ours - baseline) per report for every relative gate, with
  ours averaged over the plan's seeds (prereg.paired_bootstrap).

Inputs are de-identified summaries only: ``reports`` maps a model name to
``{seed: [{"report_id", "findings", "text"}]}`` (one seed for a
baseline), and ratings are rows ``{"model", "report_id", "score"}`` with
an optional ``seed`` and ``rater`` (several raters are averaged).
"""
import json
from pathlib import Path

from helioslm_v5.eval.grounded_report import grade
from helioslm_v5.eval.prereg import PreregError, lock, paired_bootstrap
from helioslm_v5.src.training.report_abstain import REMOVABLE, score_report


def report_metrics(rows):
    """Grounding, omission and missing-field metrics over one list of
    {"report_id", "findings", "text"} rows."""
    unv = con = om = req = miss = ok = 0
    for r in rows:
        g = grade(r["findings"], r["text"])
        unv += g["counts"]["unverified_facts"]
        con += g["counts"]["contradictions"]
        om += g["counts"]["omissions"]
        req += g["counts"]["required"]
        missing = [[n["id"], k] for n in r["findings"]["nodules"]
                   for k in REMOVABLE if n.get(k) is None]
        if missing:
            s = score_report({"findings": r["findings"], "missing": missing}, r["text"])
            miss += len(missing)
            ok += len(s["abstained"])
    return {"unverified_facts": unv, "contradictions": con,
            "unsupported_facts": unv + con, "omissions": om, "required": req,
            "omission_rate": om / req if req else 0.0,
            "missing_fields": miss, "abstained": ok,
            "missing_field_accuracy": ok / miss if miss else 1.0}


def load_ratings(path_or_rows):
    """{(model, report_id, seed or None): mean score} from a JSONL path or
    a list of rows."""
    if isinstance(path_or_rows, (str, Path)):
        text = Path(path_or_rows).read_text(encoding="utf-8")
        rows = [json.loads(x) for x in text.splitlines() if x.strip()]
    else:
        rows = list(path_or_rows)
    acc = {}
    for r in rows:
        key = (r["model"], str(r["report_id"]), r.get("seed"))
        acc.setdefault(key, []).append(float(r["score"]))
    return {k: sum(v) / len(v) for k, v in acc.items()}


def _scores(ratings, model, ids, seed=None):
    out = []
    for i in ids:
        key = (model, str(i), seed)
        if key not in ratings:
            key = (model, str(i), None)
        if key not in ratings:
            raise PreregError(f"no physician rating for {model} report {i} (seed {seed})")
        out.append(ratings[key])
    return out


def build_results(plan, ours, reports, ratings, n_boot=10000, boot_seed=0):
    """Results dict citing ``lock(plan)``; ``ours`` names our model in
    ``reports``. Every seed of the plan must be present for ours."""
    seeds = plan["seeds"]
    mine = reports[ours]
    absent = [s for s in seeds if s not in mine]
    if absent:
        raise PreregError(f"{ours}: no reports for seeds {absent}")
    ids = [str(r["report_id"]) for r in mine[seeds[0]]]
    for s in seeds:
        if [str(r["report_id"]) for r in mine[s]] != ids:
            raise PreregError(f"{ours}: seed {s} covers different reports")
    per_seed = [report_metrics(mine[s]) for s in seeds]
    rated = [_scores(ratings, ours, ids, s) for s in seeds]
    n = len(seeds)
    scores = {
        "report_grounding": {"unverified_facts": sum(m["unsupported_facts"] for m in per_seed)},
        "report_omission": {"omission_rate": sum(m["omission_rate"] for m in per_seed) / n},
        "missing_field_abstention": {
            "accuracy": sum(m["missing_field_accuracy"] for m in per_seed) / n},
        "physician_rating": {"score": sum(sum(x) / len(x) for x in rated) / n},
    }
    comparisons, baseline_metrics = {}, {}
    for g in plan["gates"]:
        if "vs" not in g:
            continue
        if g["task"] != "physician_rating":
            raise PreregError(f"relative gate on {g['task']!r} is not supported here")
        name = g["vs"]
        theirs = reports.get(name)
        if not theirs:
            raise PreregError(f"no reports for baseline {name}")
        trows = next(iter(theirs.values()))
        by_id = {str(r["report_id"]): r for r in trows}
        if set(by_id) != set(ids):
            raise PreregError(f"baseline {name} covers different reports")
        baseline_metrics[name] = report_metrics([by_id[i] for i in ids])
        c = paired_bootstrap(rated, _scores(ratings, name, ids), n_boot=n_boot, seed=boot_seed)
        comparisons.setdefault(name, {}).setdefault(g["task"], {})[g["metric"]] = c
    return {"prereg_sha256": lock(plan), "model": ours, "scores": scores,
            "comparisons": comparisons, "per_seed": dict(zip(map(str, seeds), per_seed)),
            "baseline_metrics": baseline_metrics}
