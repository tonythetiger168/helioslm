"""Cost per completed task (v5.52, M6).

OpenAI's 2026 pricing narrative asks for cost per COMPLETED TASK, not
cost per token. HeliosLM already has the two sides of that metric:

- completion evidence: benchmark records with an oracle-checked final
  answer (``correct`` plus ``final == expected``),
- serving cost: the disagg simulator's worker-seconds and Pareto front,
- quality audit: ``score_stream`` scores the same final answers under
  the toy reference checkpoint.

This module combines them without inventing a market price. USD values
are reported at an explicit ``worker_hour_usd`` normalization (the
shipped artifact uses $1/worker-hour), alongside worker-seconds per
completed task so readers can substitute their own infrastructure price.
"""
import hashlib
import json
import math
from pathlib import Path

from evastar.agent.disagg import DisaggConfig, Request, evaluate

SCHEMA_ID = "cost-per-completed-task/1"


class CostSchemaError(ValueError):
    """Raised for any missing/dishonest input or vacuous cost claim."""


def sha256_file(path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def canonical_sha256(payload: dict) -> str:
    """Hash canonical JSON (used for results_sha256 provenance)."""
    blob = json.dumps(payload, sort_keys=True, separators=(",", ":"),
                      ensure_ascii=False).encode("utf-8")
    return hashlib.sha256(blob).hexdigest()


def load_completion_records(path) -> list:
    """Load oracle-checked task completions from a mid_agent_eval-style JSON.

    Every row must carry task/expected/final/correct. A row is completed
    only when the source eval marked it correct AND its final answer still
    equals the expected answer — the latter check prevents a stale or
    hand-edited ``correct`` flag from silently becoming a cost denominator.
    """
    path = Path(path)
    if not path.is_file():
        raise CostSchemaError(f"completion evidence not found: {path}")
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as e:
        raise CostSchemaError(f"completion evidence is not JSON: {e}") from e
    rows = payload.get("results")
    if not isinstance(rows, list) or not rows:
        raise CostSchemaError(f"{path}: results must be a non-empty list")

    records = []
    for i, row in enumerate(rows):
        rid = f"task-{i:03d}"
        if not isinstance(row, dict):
            raise CostSchemaError(f"{rid}: result row must be an object")
        for key in ("task", "expected", "final", "correct"):
            if key not in row:
                raise CostSchemaError(f"{rid}: missing {key!r}")
        if not isinstance(row["correct"], bool):
            raise CostSchemaError(f"{rid}: correct must be a bool")
        prompt = str(row["task"])
        expected = str(row["expected"])
        final = str(row["final"]) if row["final"] is not None else None
        completed = bool(row["correct"] and final == expected)
        records.append({
            "id": rid,
            "family": str(row.get("family", "")),
            "prompt": prompt,
            "expected": expected,
            "final": final,
            "completed": completed,
        })
    return records


def requests_from_records(records: list) -> list:
    """Map scored records to disagg Requests.

    The shipped toy checkpoint is char-level (token id == ord(char)), so
    character counts are the honest token counts for this baseline.
    """
    if not records:
        raise CostSchemaError("no completion records")
    requests = []
    for i, rec in enumerate(records):
        if rec["final"] is None:
            raise CostSchemaError(f"{rec['id']}: final is None")
        prefill = len(rec["prompt"])
        decode = len(rec["final"])
        if prefill <= 0 or decode <= 0:
            raise CostSchemaError(
                f"{rec['id']}: prompt/final must both be non-empty")
        requests.append(Request(i, prefill, decode,
                                rec["family"] or None))
    return requests


def pareto_front(points: list) -> list:
    """Latency-vs-cost front from (makespan, worker_seconds, workers, cfg)."""
    best = {}
    for ms, ws, workers, cfg in points:
        if workers not in best or ms < best[workers][0]:
            best[workers] = (ms, ws, workers, cfg)
    front = []
    for workers in sorted(best):
        ms, ws, _, cfg = best[workers]
        if not front or ms < front[-1][0] - 1e-12:
            front.append((ms, ws, workers, cfg))
    return front


def cost_front(records: list, configs: list = None,
               worker_hour_usd: float = 1.0) -> dict:
    """Build the disagg Pareto cost front per completed task."""
    if worker_hour_usd <= 0:
        raise CostSchemaError("worker_hour_usd must be positive")
    requests = requests_from_records(records)
    completed = [r for r in records if r["completed"]]
    if not completed:
        raise CostSchemaError(
            "zero completed tasks: cost-per-completed-task is undefined")
    if configs is None:
        configs = [DisaggConfig(p, d)
                   for p in range(1, 7) for d in range(1, 7)]
    if not configs:
        raise CostSchemaError("configs must be non-empty")

    points = []
    for cfg in configs:
        makespan, worker_seconds = evaluate(requests, cfg)
        points.append((makespan, worker_seconds,
                       cfg.n_prefill + cfg.n_decode, cfg))
    front = pareto_front(points)
    if len(front) < 2:
        raise CostSchemaError("Pareto front has no cost/latency trade-off")

    rows = []
    for makespan, worker_seconds, workers, cfg in front:
        ws_per_completed = worker_seconds / len(completed)
        rows.append({
            "cfg": f"{cfg.n_prefill}P+{cfg.n_decode}D",
            "workers": workers,
            "makespan": round(makespan, 6),
            "worker_seconds_total": round(worker_seconds, 6),
            "worker_seconds_per_completed": round(ws_per_completed, 9),
            "usd_per_completed_task": round(
                ws_per_completed * worker_hour_usd / 3600.0, 12),
        })
    return {
        "n_records": len(records),
        "n_completed": len(completed),
        "completion_rate": round(len(completed) / len(records), 6),
        "worker_hour_usd": worker_hour_usd,
        "front": rows,
    }


def attach_score_audit(records: list, score_fn) -> dict:
    """Score every completed final answer with score_stream.score_record.

    ``score_fn`` is injected so tests can use a small fake while the CLI
    uses the real toy checkpoint. Score-stream NLL is an audit signal,
    never a substitute for the completion oracle.
    """
    rows = []
    for rec in records:
        if not rec["completed"]:
            continue
        score = score_fn(rec["prompt"], rec["final"])
        nll = float(score["nll_per_token"])
        if not math.isfinite(nll):
            raise CostSchemaError(f"{rec['id']}: non-finite score_stream NLL")
        rows.append({"id": rec["id"],
                     "family": rec["family"],
                     "n_tokens": int(score["n_tokens"]),
                     "nll_per_token": round(nll, 6),
                     "ppl": score["ppl"]})
    if not rows:
        raise CostSchemaError("score audit is empty")
    nlls = [r["nll_per_token"] for r in rows]
    return {
        "scoring": "score_stream loglikelihood under the toy checkpoint",
        "role": "quality audit only; completion is decided by final == expected",
        "n_scored": len(rows),
        "mean_nll_per_token": round(sum(nlls) / len(nlls), 6),
        "max_nll_per_token": round(max(nlls), 6),
        "rows": rows,
    }
