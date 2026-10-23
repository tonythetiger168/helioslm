"""Pre-registered evaluation plans (plan E5, docs/GPU_AND_EVAL_PLAN_2026-10.md).

Before a model generation is trained, its evaluation is written down and
committed: which tasks (with dataset revision, split, few-shot count and
metric), which harness version, which baselines, and the pass / fail
gates. The plan is then locked by the sha256 of its canonical JSON.

A results file must cite that sha256 (``prereg_sha256``). ``check_gates``
refuses results that cite another plan, so editing the gates after seeing
the scores invalidates every result filed against the old plan; and it
refuses results missing any registered metric, so a weak task cannot be
dropped quietly.

Plan fields (all required unless marked optional):
  release        e.g. "v1.0"            model        e.g. "0.5B dense"
  harness        {"name", "version"}    seeds        list of ints
  tasks          [{"name", "revision", "split", "n_shot", "metric"}]
  gates          [{"task", "metric", "op": ">=" | "<=", "threshold"}]
                 or, relative to a registered baseline (v5.114):
                 [{"task", "metric", "vs": baseline name, "margin"}]
  baselines      optional: [{"name", "revision"}] run under the same plan
  notes          optional free text

A relative gate passes when the lower end of the 95% paired-bootstrap
interval of (ours - baseline), over the same questions, is above
``margin``: "beat the baseline" means beat it beyond sampling noise, not
by a point estimate. ``paired_bootstrap`` computes that interval from
per-question correctness, averaged over the plan's seeds.

A revision may be written as "PENDING" while the dataset or baseline
weights are not pinned yet (e.g. they are only reachable from the pod).
Such a plan validates and locks, but ``check_gates`` refuses it: pin the
revisions, commit, and that lock is the one results must cite.

Results fields:
  prereg_sha256  the plan's lock()
  scores         {task: {metric: value}}
  comparisons    for relative gates: {baseline: {task: {metric:
                 {"ours", "theirs", "ci95_low", "ci95_high"}}}}
"""
import hashlib
import json

OPS = {">=": lambda v, t: v >= t, "<=": lambda v, t: v <= t}
_TASK_KEYS = ("name", "revision", "split", "n_shot", "metric")
_GATE_KEYS = ("task", "metric", "op", "threshold")
_REL_GATE_KEYS = ("task", "metric", "vs", "margin")
PENDING = "PENDING"


class PreregError(ValueError):
    pass


def _canonical(plan):
    return json.dumps(plan, sort_keys=True, ensure_ascii=False,
                      separators=(",", ":")).encode("utf-8")


def lock(plan):
    """sha256 of the plan's canonical JSON (validated first)."""
    validate(plan)
    return hashlib.sha256(_canonical(plan)).hexdigest()


def validate(plan):
    for key in ("release", "model", "harness", "seeds", "tasks", "gates"):
        if key not in plan:
            raise PreregError(f"plan is missing {key!r}")
    if not {"name", "version"} <= set(plan["harness"]):
        raise PreregError("harness needs name and version")
    seeds = plan["seeds"]
    if not seeds or len(set(seeds)) != len(seeds) or \
            not all(isinstance(s, int) and s >= 0 for s in seeds):
        raise PreregError(f"seeds must be distinct non-negative ints, got {seeds}")
    tasks = {}
    for t in plan["tasks"]:
        missing = [k for k in _TASK_KEYS if k not in t]
        if missing:
            raise PreregError(f"task {t.get('name')!r} is missing {missing}")
        if t["name"] in tasks:
            raise PreregError(f"task {t['name']!r} registered twice")
        if not isinstance(t["n_shot"], int) or t["n_shot"] < 0:
            raise PreregError(f"task {t['name']!r}: n_shot must be an int >= 0")
        tasks[t["name"]] = t
    if not tasks:
        raise PreregError("plan registers no tasks")
    if not plan["gates"]:
        raise PreregError("plan registers no gates")
    baselines = {b["name"] for b in plan.get("baselines", [])}
    for g in plan["gates"]:
        relative = "vs" in g
        keys = _REL_GATE_KEYS if relative else _GATE_KEYS
        missing = [k for k in keys if k not in g]
        if missing:
            raise PreregError(f"gate {g} is missing {missing}")
        if g["task"] not in tasks:
            raise PreregError(f"gate on unregistered task {g['task']!r}")
        if g["metric"] != tasks[g["task"]]["metric"]:
            raise PreregError(f"gate metric {g['metric']!r} is not the "
                              f"registered metric of {g['task']!r}")
        if relative:
            if g["vs"] not in baselines:
                raise PreregError(f"gate vs unregistered baseline {g['vs']!r}")
            if isinstance(g["margin"], bool) or not isinstance(g["margin"], (int, float)):
                raise PreregError(f"gate margin must be a number, got {g['margin']!r}")
            continue
        if g["op"] not in OPS:
            raise PreregError(f"gate op must be one of {sorted(OPS)}, got {g['op']!r}")
        if isinstance(g["threshold"], bool) or not isinstance(g["threshold"], (int, float)):
            raise PreregError(f"gate threshold must be a number, got {g['threshold']!r}")
    return plan


def pending(plan):
    """Names of the tasks and baselines whose revision is still PENDING."""
    return [x["name"] for x in plan["tasks"] + plan.get("baselines", [])
            if x.get("revision") == PENDING]


def check_gates(plan, results):
    """Gate verdicts for results filed against this plan."""
    sha = lock(plan)
    if pending(plan):
        raise PreregError(f"plan is not final: revisions PENDING for {pending(plan)}")
    if results.get("prereg_sha256") != sha:
        raise PreregError("results cite a different (or no) pre-registration: "
                          f"{results.get('prereg_sha256')!r} != {sha!r}")
    scores = results.get("scores", {})
    missing = [(t["name"], t["metric"]) for t in plan["tasks"]
               if t["metric"] not in scores.get(t["name"], {})]
    comps = results.get("comparisons", {})
    missing += [(g["vs"], g["task"], g["metric"]) for g in plan["gates"]
                if "vs" in g and g["metric"] not in comps.get(g["vs"], {}).get(g["task"], {})]
    if missing:
        raise PreregError(f"results omit registered task metrics: {missing}")
    rows = []
    for g in plan["gates"]:
        if "vs" in g:
            c = comps[g["vs"]][g["task"]][g["metric"]]
            rows.append({**g, "value": c["ci95_low"], "ours": c["ours"],
                         "theirs": c["theirs"], "pass": c["ci95_low"] > g["margin"]})
            continue
        value = scores[g["task"]][g["metric"]]
        rows.append({**g, "value": value, "pass": OPS[g["op"]](value, g["threshold"])})
    return {"prereg_sha256": sha, "release": plan["release"],
            "passed": all(r["pass"] for r in rows), "gates": rows}


def paired_bootstrap(ours, theirs, n_boot=10000, seed=0):
    """95% interval of mean(ours) - mean(theirs) over the same questions.

    ``ours`` is one 0/1 (or score) list per seed, ``theirs`` one list (a
    released baseline is a single model); all lists are per question in
    the same order. Questions are resampled with replacement, keeping each
    question's pair together."""
    import numpy as np
    a = np.asarray(ours, dtype=np.float64)
    if a.ndim == 1:
        a = a[None]
    b = np.asarray(theirs, dtype=np.float64)
    if a.shape[1] != b.shape[0] or b.ndim != 1 or not b.size:
        raise PreregError(f"paired_bootstrap: shapes {a.shape} vs {b.shape}")
    d = a.mean(0) - b
    rng = np.random.default_rng(seed)
    boots = np.concatenate([                 # chunks keep memory at ~n x 500
        d[rng.integers(0, d.size, size=(min(500, n_boot - i), d.size))].mean(1)
        for i in range(0, n_boot, 500)])
    lo, hi = np.percentile(boots, [2.5, 97.5])
    return {"ours": float(a.mean()), "theirs": float(b.mean()),
            "diff": float(d.mean()), "ci95_low": float(lo), "ci95_high": float(hi),
            "n": int(d.size), "seeds": int(a.shape[0]), "n_boot": int(n_boot)}
