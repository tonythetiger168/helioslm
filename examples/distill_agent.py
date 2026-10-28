"""Agent-trajectory distillation through hsh (D4, docs/DISTILL_DEEPSEEK.md).

The teacher runs as an hsh agent (bash + str_replace_editor) on generated
tasks; verified trajectories become SFT steps in the student's text
protocol. The vLLM server needs tool calling on:
``--enable-auto-tool-choice --tool-call-parser deepseek_v41``.

    python examples/distill_agent.py tasks --n 3000 --out runs/agent/tasks.jsonl
    python examples/distill_agent.py run --tasks runs/agent/tasks.jsonl \\
        --out runs/agent/trajectories.jsonl --workers 32
    python examples/distill_agent.py to-sft --trajectories runs/agent/trajectories.jsonl \\
        --out runs/agent/sft.jsonl
    # then: examples/distill_sft.py prepare --records runs/agent/sft.jsonl ...

Run it on the disposable pod: the tools execute real commands.
"""
import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from helioslm_v5.src.training import distill_agent as da  # noqa: E402


def write_jsonl(path, rows):
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as fh:
        for r in rows:
            fh.write(json.dumps(r, ensure_ascii=False) + "\n")


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    t = sub.add_parser("tasks")
    t.add_argument("--n", type=int, default=3000)
    t.add_argument("--seed", type=int, default=0)
    t.add_argument("--out", required=True)
    r = sub.add_parser("run")
    r.add_argument("--tasks", required=True)
    r.add_argument("--out", required=True)
    r.add_argument("--teacher", default="deepseek-v4.1-flash")
    r.add_argument("--base-url", default="http://localhost:8000/v1")
    r.add_argument("--model", default=None)
    r.add_argument("--workers", type=int, default=32)
    r.add_argument("--max-tokens", type=int, default=4096)
    r.add_argument("--temperature", type=float, default=0.6)
    s = sub.add_parser("to-sft")
    s.add_argument("--trajectories", required=True)
    s.add_argument("--out", required=True)
    s.add_argument("--all", action="store_true", help="keep unverified trajectories too")
    a = ap.parse_args(argv)

    if a.cmd == "tasks":
        write_jsonl(a.out, da.make_tasks(a.n, a.seed))
    elif a.cmd == "run":
        tasks = [json.loads(x) for x in Path(a.tasks).read_text(encoding="utf-8").splitlines() if x]
        Path(a.out).parent.mkdir(parents=True, exist_ok=True)
        summary = da.run_tasks(tasks, a.out, teacher=a.teacher, base_url=a.base_url,
                               model=a.model, workers=a.workers, max_tokens=a.max_tokens,
                               temperature=a.temperature, log=print)
        print(json.dumps(summary, ensure_ascii=False, indent=1))
    else:
        rows, n = [], 0
        with open(a.trajectories, encoding="utf-8") as fh:
            for line in fh:
                rec = json.loads(line)
                n += 1
                if rec["verified"] or a.all:
                    rows.extend(da.trajectory_examples(rec))
        write_jsonl(a.out, rows)
        print(json.dumps({"trajectories": n, "steps": len(rows)}))


if __name__ == "__main__":
    main()
