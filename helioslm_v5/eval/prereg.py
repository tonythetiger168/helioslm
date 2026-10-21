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
  baselines      optional: [{"name", "revision"}] run under the same plan
  notes          optional free text

Results fields:
  prereg_sha256  the plan's lock()
  scores         {task: {metric: value}}
"""
import hashlib
import json

OPS = {">=": lambda v, t: v >= t, "<=": lambda v, t: v <= t}
_TASK_KEYS = ("name", "revision", "split", "n_shot", "metric")
_GATE_KEYS = ("task", "metric", "op", "threshold")


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
    for g in plan["gates"]:
        missing = [k for k in _GATE_KEYS if k not in g]
        if missing:
            raise PreregError(f"gate {g} is missing {missing}")
        if g["task"] not in tasks:
            raise PreregError(f"gate on unregistered task {g['task']!r}")
        if g["metric"] != tasks[g["task"]]["metric"]:
            raise PreregError(f"gate metric {g['metric']!r} is not the "
                              f"registered metric of {g['task']!r}")
        if g["op"] not in OPS:
            raise PreregError(f"gate op must be one of {sorted(OPS)}, got {g['op']!r}")
        if isinstance(g["threshold"], bool) or not isinstance(g["threshold"], (int, float)):
            raise PreregError(f"gate threshold must be a number, got {g['threshold']!r}")
    return plan


def check_gates(plan, results):
    """Gate verdicts for results filed against this plan."""
    sha = lock(plan)
    if results.get("prereg_sha256") != sha:
        raise PreregError("results cite a different (or no) pre-registration: "
                          f"{results.get('prereg_sha256')!r} != {sha!r}")
    scores = results.get("scores", {})
    missing = [(t["name"], t["metric"]) for t in plan["tasks"]
               if t["metric"] not in scores.get(t["name"], {})]
    if missing:
        raise PreregError(f"results omit registered task metrics: {missing}")
    rows = []
    for g in plan["gates"]:
        value = scores[g["task"]][g["metric"]]
        rows.append({**g, "value": value, "pass": OPS[g["op"]](value, g["threshold"])})
    return {"prereg_sha256": sha, "release": plan["release"],
            "passed": all(r["pass"] for r in rows), "gates": rows}
