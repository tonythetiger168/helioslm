"""eval_harness.py — score a pilot checkpoint with lm-evaluation-harness (plan E1).

    python examples/eval_harness.py --run runs/dense --data data/pilot \\
        --tasks arc_easy,hellaswag,tmmluplus --device cuda --out results/eval.json

Runs lm_eval.simple_evaluate on the `helioslm` model type
(helioslm_v5/eval/lm_harness.py). Harness datasets are downloaded from the
Hugging Face Hub, so run it where the Hub's file hosts are reachable (the
GPU pod, not the development sandbox). --limit N scores only the first N
documents of each task (a smoke run; such scores are not comparable).
The output JSON keeps the harness's results, configs and versions, plus
the run metadata and the checkpoint step.
"""
import argparse
import json
import sys
from pathlib import Path

import torch

_REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_REPO))

from helioslm_v5.eval.lm_harness import HeliosHarnessLM  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", required=True, help="pilot_100m.py output directory")
    ap.add_argument("--data", default=None, help="pilot data dir (default: run.json)")
    ap.add_argument("--checkpoint", default="ckpt.pt")
    ap.add_argument("--tasks", required=True, help="comma-separated harness tasks")
    ap.add_argument("--num-fewshot", type=int, default=None)
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    from lm_eval import simple_evaluate
    lm = HeliosHarnessLM(run=args.run, data=args.data, device=args.device,
                         checkpoint=args.checkpoint)
    res = simple_evaluate(model=lm, tasks=args.tasks.split(","),
                          num_fewshot=args.num_fewshot, limit=args.limit,
                          log_samples=False)
    state = torch.load(Path(args.run) / args.checkpoint, map_location="cpu",
                       weights_only=False)
    out = {"run": json.loads((Path(args.run) / "run.json").read_text()),
           "checkpoint_step": state.get("step"), "limit": args.limit,
           "results": res["results"], "configs": res.get("configs"),
           "versions": res.get("versions"), "n-shot": res.get("n-shot")}
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out).write_text(json.dumps(out, indent=2, default=str) + "\n")
    for task, r in res["results"].items():
        print(task, {k: v for k, v in r.items() if isinstance(v, (int, float))})


if __name__ == "__main__":
    main()
