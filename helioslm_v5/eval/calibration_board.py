"""Calibration leaderboard (v5.54, roadmap Phase A).

Public leaderboards rank models by how often they are right. None of
them report whether a model KNOWS when it is wrong — the question the
HeliosLM decision layer exists to answer. This module scores any eval
artifact in the shared probe schema (``results[*].correct`` plus a
per-task ``conf``) on that question:

- ``auroc``  : P(conf on a correct task > conf on a wrong task), ties
               count 1/2. ``None`` when only one class is present — a
               12/12 or 0/12 run carries no ranking signal, and the
               board says so instead of printing 0.5 or 1.0.
- ``brier``  : mean (conf - correct)^2. A proper scoring rule defined on
               every run, so it is the ranking key.
- ``ece``    : 10-bin equal-width ECE (``quant_calib.ece``).
- ``overconfident_wrong``: wrong answers given with conf >= 0.9 — the
               failure mode a TrustGate has to catch.

The same probe suite and eval seed feed both our local models
(``mid_agent_eval_*``) and API models (``examples/probe_api.py``), so
rows are comparable on the task side. The confidence SOURCE differs
(token probability vs. self-consistency over n samples), and every row
records which — a board that hid that would overstate comparability.
"""
import hashlib
import json
from pathlib import Path

from helioslm_v5.eval.quant_calib import ece

SCHEMA_ID = "calibration-board/1"
OVERCONFIDENT = 0.9


class BoardSchemaError(ValueError):
    """Raised for any malformed input artifact. Never swallowed."""


def sha256_file(path) -> str:
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def canonical_sha256(payload) -> str:
    blob = json.dumps(payload, sort_keys=True, separators=(",", ":"),
                      ensure_ascii=False).encode("utf-8")
    return hashlib.sha256(blob).hexdigest()


def auroc(confs, correct):
    """Mann-Whitney AUROC of conf as a correct-vs-wrong detector.

    Returns None when either class is empty (undefined, not 0.5)."""
    pos = [c for c, y in zip(confs, correct) if y]
    neg = [c for c, y in zip(confs, correct) if not y]
    if not pos or not neg:
        return None
    wins = 0.0
    for p in pos:
        for n in neg:
            wins += 1.0 if p > n else 0.5 if p == n else 0.0
    return wins / (len(pos) * len(neg))


def brier(confs, correct):
    if not confs:
        raise BoardSchemaError("brier: no scored rows")
    return sum((c - float(y)) ** 2 for c, y in zip(confs, correct)) / len(confs)


def load_rows(path):
    """(correct flags, confidences, conf_type) from one probe artifact.

    ``correct`` must be a real bool and ``conf`` a number in [0, 1] or
    null. Null confidences (e.g. an episode that never produced a final
    answer) count toward accuracy but not toward calibration metrics;
    the row count of each is reported.
    """
    path = Path(path)
    if not path.is_file():
        raise BoardSchemaError(f"artifact not found: {path}")
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as e:
        raise BoardSchemaError(f"{path.name}: not JSON: {e}") from e
    rows = payload.get("results") if isinstance(payload, dict) else None
    if not isinstance(rows, list) or not rows:
        raise BoardSchemaError(f"{path.name}: results must be a non-empty list")
    flags, confs, types = [], [], set()
    for i, row in enumerate(rows):
        if not isinstance(row, dict):
            raise BoardSchemaError(f"{path.name}[{i}]: row must be an object")
        if not isinstance(row.get("correct"), bool):
            raise BoardSchemaError(f"{path.name}[{i}]: correct must be a bool")
        conf = row.get("conf")
        if conf is not None:
            if isinstance(conf, bool) or not isinstance(conf, (int, float)) \
                    or not 0.0 <= conf <= 1.0:
                raise BoardSchemaError(
                    f"{path.name}[{i}]: conf must be a number in [0, 1]")
            conf = float(conf)
        flags.append(row["correct"])
        confs.append(conf)
        types.add(row.get("conf_type", "token_prob"))
    model = payload.get("model") or path.stem
    return model, flags, confs, sorted(types)


def score_artifact(path):
    """One board row for one artifact (metrics rounded to 4 places)."""
    model, flags, confs, types = load_rows(path)
    scored = [(c, y) for c, y in zip(confs, flags) if c is not None]
    if not scored:
        raise BoardSchemaError(f"{Path(path).name}: no row carries a conf")
    sc = [c for c, _ in scored]
    sy = [y for _, y in scored]
    wrong = [c for c, y in scored if not y]
    au = auroc(sc, sy)
    r4 = lambda x: None if x is None else round(x, 4)  # noqa: E731
    return {
        "artifact": Path(path).name,
        "model": model,
        "conf_type": "+".join(types),
        "n": len(flags),
        "n_scored": len(scored),
        "accuracy": r4(sum(flags) / len(flags)),
        "auroc": r4(au),
        "brier": r4(brier(sc, sy)),
        "ece": r4(ece(sc, [int(y) for y in sy])),
        "conf_on_wrong_mean": r4(sum(wrong) / len(wrong)) if wrong else None,
        "overconfident_wrong": sum(1 for c in wrong if c >= OVERCONFIDENT),
        "n_wrong_scored": len(wrong),
    }


def build_board(paths, created, repo_root=None):
    """Score every artifact and rank by Brier (lower is better).

    Ties break on artifact name so the output is deterministic."""
    if not paths:
        raise BoardSchemaError("build_board: no artifacts given")
    root = Path(repo_root) if repo_root else None
    rows = sorted((score_artifact(p) for p in paths),
                  key=lambda r: (r["brier"], r["artifact"]))
    for rank, row in enumerate(rows, 1):
        row["rank"] = rank
    inputs, seen = {}, {}
    for p in paths:
        p = Path(p)
        key = str(p.relative_to(root)) if root else p.name
        digest = sha256_file(p)
        if digest in seen:
            # a byte-identical copy would count one run twice
            raise BoardSchemaError(
                f"duplicate input: {key} is identical to {seen[digest]}")
        seen[digest] = key
        inputs[key] = digest
    board = {
        "schema": SCHEMA_ID,
        "created": created,
        "verification_tier": "measured",
        "ranking_key": "brier (ascending)",
        "overconfident_threshold": OVERCONFIDENT,
        "caveats": [
            "n is 12 tasks per artifact: rankings are directional, not "
            "significant",
            "auroc is null when a run is all-correct or all-wrong (one "
            "class carries no ranking signal)",
            "conf sources differ (token_prob vs self_consistency); "
            "compare within a conf_type before across",
            "rows with a null conf (no final answer) count toward "
            "accuracy only; 'scored' shows how many rows the "
            "calibration metrics cover",
        ],
        "inputs": inputs,
        "board": rows,
    }
    board["results_sha256"] = canonical_sha256(board["board"])
    return board


def render_markdown(board):
    """Human-readable table for the .md companion artifact."""
    fmt = lambda x: "—" if x is None else f"{x:.4f}"  # noqa: E731
    lines = [
        f"# Calibration board ({board['created']})",
        "",
        f"Schema `{board['schema']}`, tier `{board['verification_tier']}`, "
        f"ranked by {board['ranking_key']}. "
        f"`overconf. wrong` = wrong answers with conf ≥ "
        f"{board['overconfident_threshold']}.",
        "",
        "| # | model / artifact | conf type | scored | acc | AUROC | Brier "
        "| ECE | conf on wrong | overconf. wrong |",
        "|---|---|---|---|---|---|---|---|---|---|",
    ]
    for r in board["board"]:
        lines.append(
            f"| {r['rank']} | {r['model']} (`{r['artifact']}`) "
            f"| {r['conf_type']} | {r['n_scored']}/{r['n']} "
            f"| {fmt(r['accuracy'])} | {fmt(r['auroc'])} "
            f"| {fmt(r['brier'])} | {fmt(r['ece'])} "
            f"| {fmt(r['conf_on_wrong_mean'])} "
            f"| {r['overconfident_wrong']}/{r['n_wrong_scored']} |")
    lines += ["", "Caveats:", ""]
    lines += [f"- {c}" for c in board["caveats"]]
    lines += ["", f"results_sha256: `{board['results_sha256']}`", ""]
    return "\n".join(lines)
