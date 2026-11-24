"""Route A offline preparation and verified teacher-trajectory collection.

    python examples/agentic_distill.py tasks --per-protocol 25 \
        --out runs/route-a/tasks.jsonl
    python examples/agentic_distill.py eval-items --out runs/route-a/eval.jsonl

Collection calls a hosted teacher but never rents one.  Run it only in an
already-confirmed disposable container; the API key is read from the named
environment variable and is never serialized:

    python examples/agentic_distill.py collect --tasks runs/route-a/tasks.jsonl \
        --out runs/route-a/trajectories.jsonl --container-confirmed \
        --max-cost-usd 1.0
    python examples/agentic_distill.py export \
        --trajectories runs/route-a/trajectories.jsonl --out runs/route-a/sft.jsonl

Export always indexes the four built-in formal agentic eval sets and any
additional ``--eval`` JSONL files.  It writes data and a report only; SFT
training is a separate, explicitly authorized operation.
"""
import argparse
import hashlib
import json
import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from helioslm_v5.src.training import agentic_distill as ad  # noqa: E402
from helioslm_v5.src.training.distill_filter import load_eval_items  # noqa: E402


def _write_jsonl(path, rows):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="\n") as fh:
        for row in rows:
            fh.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n")


def _read_jsonl(path):
    with Path(path).open(encoding="utf-8") as fh:
        return [json.loads(line) for line in fh if line.strip()]


def _eval_items(paths):
    extra = load_eval_items(paths or [], fields=("text", "question", "prompt", "query"))
    return ad.combined_eval_items(extra)


def main(argv=None):
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = parser.add_subparsers(dest="cmd", required=True)

    tasks = sub.add_parser("tasks", help="generate fresh synthetic training tasks")
    tasks.add_argument("--per-protocol", type=int, default=25)
    tasks.add_argument("--seed", type=int, default=0)
    tasks.add_argument("--out", required=True)

    items = sub.add_parser("eval-items", help="export the built-in eval text index")
    items.add_argument("--out", required=True)

    collect = sub.add_parser("collect", help="collect verified teacher trajectories")
    collect.add_argument("--tasks", required=True)
    collect.add_argument("--out", required=True)
    collect.add_argument("--eval", action="append", default=[],
                         help="additional eval JSONL; repeatable")
    collect.add_argument("--container-confirmed", action="store_true",
                         help="confirm this process is already in a disposable container")
    collect.add_argument("--api-key-env", default="MOONSHOT_API_KEY")
    collect.add_argument("--base-url", default="https://api.moonshot.ai/v1")
    collect.add_argument("--model", default="kimi-k3.1-pro")
    collect.add_argument("--max-cost-usd", type=float, default=1.0)

    export = sub.add_parser("export", help="filter trajectories into SFT JSONL")
    export.add_argument("--trajectories", required=True)
    export.add_argument("--out", required=True)
    export.add_argument("--eval", action="append", default=[],
                        help="additional eval JSONL; repeatable")

    args = parser.parse_args(argv)
    if args.cmd == "tasks":
        rows = ad.make_tasks(args.per_protocol, args.seed)
        _write_jsonl(args.out, rows)
        print(json.dumps({"tasks": len(rows), "task_set": ad.TRAIN_TASK_SET,
                          "task_set_sha256": ad.task_sha256(rows)}, indent=2))
        return

    if args.cmd == "eval-items":
        rows = [{"id": key, "text": text}
                for key, text in sorted(ad.builtin_eval_items().items())]
        _write_jsonl(args.out, rows)
        print(json.dumps({"eval_items": len(rows)}, indent=2))
        return

    eval_items = _eval_items(args.eval)
    if args.cmd == "collect":
        if not args.container_confirmed:
            parser.error("collect requires --container-confirmed; teacher commands "
                         "and benchmark samples are untrusted code")
        if args.max_cost_usd <= 0:
            parser.error("--max-cost-usd must be positive")
        tasks = _read_jsonl(args.tasks)
        ad.assert_tasks_disjoint(tasks, eval_items)  # before any API call
        key = os.environ.get(args.api_key_env)
        if not key:
            parser.error(f"environment variable {args.api_key_env} is not set")
        from k31_reference_run import K31Candidate
        teacher = K31Candidate(key, base_url=args.base_url, model=args.model)
        done = {row["id"] for row in _read_jsonl(args.out)} if Path(args.out).exists() else set()
        output = Path(args.out)
        output.parent.mkdir(parents=True, exist_ok=True)
        summary = {"tasks": len(tasks), "already_present": len(done),
                   "collected": 0, "verified": 0, "cost_usd": 0.0,
                   "training_started": False}
        with output.open("a", encoding="utf-8", newline="\n") as fh:
            for task in tasks:
                if task["id"] in done:
                    continue
                before = teacher.total_cost
                record = ad.collect_task(task, teacher)
                record["teacher"] = {"model": args.model,
                                     "cost_usd": teacher.total_cost - before}
                fh.write(json.dumps(record, ensure_ascii=False, sort_keys=True) + "\n")
                fh.flush()
                summary["collected"] += 1
                summary["verified"] += record["verified"]
                summary["cost_usd"] = teacher.total_cost
                if teacher.total_cost > args.max_cost_usd:
                    raise RuntimeError(
                        f"teacher cost {teacher.total_cost:.6f} exceeded the authorized "
                        f"cap {args.max_cost_usd:.6f}; collected rows remain resumable")
        print(json.dumps(summary, ensure_ascii=False, indent=2))
        return

    records = _read_jsonl(args.trajectories)
    rows, report = ad.to_sft(records, eval_items)
    if not rows:
        raise RuntimeError("no SFT rows survived the gates; refusing to write a placeholder")
    _write_jsonl(args.out, rows)
    report["training_started"] = False
    report["output_sha256"] = hashlib.sha256(
        Path(args.out).read_bytes()).hexdigest()
    Path(str(args.out) + ".report.json").write_text(
        json.dumps(report, ensure_ascii=False, sort_keys=True, indent=2) + "\n",
        encoding="utf-8")
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
