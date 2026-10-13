"""cost_per_completed_task.py — M6: $/completed-task baseline.

Combines three existing HeliosLM assets:

1. ``benchmarks/mid_agent_eval_v5.json`` supplies completion evidence
   (12/12 grounded task completions; this script re-checks
   ``final == expected`` instead of trusting the flag),
2. the disagg simulator/Pareto machinery behind
   ``benchmarks/disagg_pareto_2026-09-25.json`` supplies worker-seconds
   and the latency/cost front,
3. ``helioslm_v5.eval.score_stream`` scores the same final answers under
   the toy checkpoint as a quality-audit channel.

The USD number is deliberately a UNIT-PRICE NORMALIZATION
($1/worker-hour by default), not a market-price claim. The artifact also
reports worker-seconds per completed task so any real infrastructure
price can be substituted without rerunning the model.

Run from repo root:
    python examples/cost_per_completed_task.py
"""
import argparse
import json
import sys
from pathlib import Path

import torch

_REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_REPO))

from helioslm_v5.configs.config_v5 import HeliosLMv5Config  # noqa: E402
from helioslm_v5.eval.artifact_tiers import load_registry, tier_of  # noqa: E402
from helioslm_v5.eval.cost_per_task import (SCHEMA_ID, attach_score_audit,  # noqa: E402
                                            cost_front, load_completion_records,
                                            sha256_file, canonical_sha256)
from helioslm_v5.eval.score_stream import score_record  # noqa: E402
from helioslm_v5.src.model_v5 import HeliosLMv5  # noqa: E402

DATE = "2026-10-13"
DEFAULT_EVAL = "benchmarks/mid_agent_eval_v5.json"
DEFAULT_PARETO = "benchmarks/disagg_pareto_2026-09-25.json"
DEFAULT_CHECKPOINT = "checkpoints/toy_v5.13.pt"
DEFAULT_OUT = "benchmarks/cost_per_completed_task_2026-10-13.json"


def load_toy_scorer(checkpoint_path: Path):
    ckpt = torch.load(checkpoint_path, map_location="cpu",
                      weights_only=False)
    model = HeliosLMv5(HeliosLMv5Config(size="lite"))
    model.load_state_dict(ckpt["state_dict"])
    model.eval()
    vocab = int(ckpt["vocab_size"])

    def score(prompt, output):
        return score_record(model, prompt, output, vocab)

    return score, vocab


def build_report(eval_path: Path, pareto_path: Path, checkpoint_path: Path,
                 worker_hour_usd: float = 1.0) -> dict:
    for path, label in ((eval_path, "completion evidence"),
                        (pareto_path, "disagg Pareto artifact"),
                        (checkpoint_path, "toy checkpoint")):
        if not path.is_file():
            raise FileNotFoundError(f"{label} not found: {path}")

    records = load_completion_records(eval_path)
    registry = load_registry(_REPO / "benchmarks" / "verification_tiers.json")
    score_fn, vocab = load_toy_scorer(checkpoint_path)
    cost = cost_front(records, worker_hour_usd=worker_hour_usd)
    audit = attach_score_audit(records, score_fn)

    payload = {
        "experiment": "M6 cost per completed task baseline",
        "schema": SCHEMA_ID,
        "verification_tier": "measured",
        "created": DATE,
        "scope": ("toy-scale local baseline; USD is a unit-price "
                  "normalization, not a market quote"),
        "inputs": {
            "completion_evidence": {
                "path": eval_path.relative_to(_REPO).as_posix(),
                "sha256": sha256_file(eval_path),
                "verification_tier": tier_of(eval_path.name, registry),
            },
            "disagg_pareto_reference": {
                "path": pareto_path.relative_to(_REPO).as_posix(),
                "sha256": sha256_file(pareto_path),
                "verification_tier": tier_of(pareto_path.name, registry),
            },
            "score_checkpoint": {
                "path": checkpoint_path.relative_to(_REPO).as_posix(),
                "sha256": sha256_file(checkpoint_path),
                "vocab_size": vocab,
            },
        },
        "assumptions": {
            "worker_hour_usd": worker_hour_usd,
            "usd_is_unit_price_normalization": True,
            "token_accounting": ("char-level toy checkpoint: one character "
                                 "is one token for these ASCII records"),
            "completed_task": ("source eval correct=true AND "
                               "final == expected, re-checked here"),
            "score_stream_role": ("quality audit only; NLL never decides "
                                  "task completion"),
        },
        "results": {
            "completion": {
                "n_records": cost["n_records"],
                "n_completed": cost["n_completed"],
                "completion_rate": cost["completion_rate"],
            },
            "cost_front": cost["front"],
            "quality_audit": audit,
        },
    }
    payload["results_sha256"] = canonical_sha256(payload)
    return payload


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--eval-json", default=DEFAULT_EVAL,
                    help="mid_agent_eval-style completion evidence")
    ap.add_argument("--pareto-json", default=DEFAULT_PARETO,
                    help="oracle-tested disagg Pareto reference artifact")
    ap.add_argument("--checkpoint", default=DEFAULT_CHECKPOINT)
    ap.add_argument("--worker-hour-usd", type=float, default=1.0,
                    help="unit-price normalization for USD reporting")
    ap.add_argument("--out", default=DEFAULT_OUT)
    args = ap.parse_args()

    report = build_report(_REPO / args.eval_json,
                          _REPO / args.pareto_json,
                          _REPO / args.checkpoint,
                          worker_hour_usd=args.worker_hour_usd)
    out = Path(args.out)
    if not out.is_absolute():
        out = _REPO / out
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")

    comp = report["results"]["completion"]
    cheapest = min(report["results"]["cost_front"],
                   key=lambda r: r["usd_per_completed_task"])
    print(json.dumps({
        "completed": f"{comp['n_completed']}/{comp['n_records']}",
        "mean_nll_per_token":
            report["results"]["quality_audit"]["mean_nll_per_token"],
        "cheapest_cfg": cheapest["cfg"],
        "worker_seconds_per_completed":
            cheapest["worker_seconds_per_completed"],
        "usd_per_completed_task_at_1usd_worker_hour":
            cheapest["usd_per_completed_task"],
        "results_sha256": report["results_sha256"],
        "out": str(out),
    }, indent=2))


if __name__ == "__main__":
    main()
