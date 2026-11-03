"""ot_bugfix_run.py — OpenTitan bug-fix / completion benchmark CLI
(v5.228; ``helioslm_v5/eval/ot_bugfix_bench.py``).

  # on a pod with yosys, sv2v (and verilator), git; vLLM for generate
  python examples/ot_bugfix_run.py build --ot /workspace/opentitan \\
      --manifest manifest.json --out ot_tasks.jsonl
  python examples/ot_bugfix_run.py generate --tasks ot_tasks.jsonl \\
      --model google/gemma-4-E4B-it --out e4b_it.samples.json
  python examples/ot_bugfix_run.py score --tasks ot_tasks.jsonl \\
      --ot /workspace/opentitan --samples e4b_it.samples.json --out e4b_it.json

  python examples/ot_bugfix_run.py pairs --ot /workspace/opentitan \\
      --tasks ot_tasks.jsonl --docs stack.jsonl --out edit_pairs.jsonl

``pairs`` (v5.231) writes edit / repair SFT records (``system``,
``prompt``, ``response``) for ``examples/sft_hf.py`` from files outside
every evaluation IP. ``build`` checks out the pinned OpenTitan commit when ``--ot`` is empty.
``generate`` reuses ``hdl_eval_all.vllm_generate`` (chat template,
greedy by default) and writes {task id: [answers]}. ``score`` writes the
per-task results and the summary (by kind, seen/unseen split, operator).
"""
import argparse
import json
import sys
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "examples"))

from helioslm_v5.eval import ot_bugfix_bench as ob  # noqa: E402


def read_tasks(path):
    return [json.loads(line) for line in Path(path).read_text().splitlines()
            if line.strip()]


def cmd_build(a):
    root = ob.checkout(a.ot)
    t0 = time.time()
    tasks = ob.build(root, seed=a.seed, per_ip=a.per_ip,
                     max_tasks=a.max_tasks, manifest=a.manifest,
                     workers=a.workers)
    with open(a.out, "w") as f:
        for t in tasks:
            f.write(json.dumps(t, ensure_ascii=False) + "\n")
    print(f"build: {len(tasks)} tasks in {time.time() - t0:.0f}s -> {a.out}")


def cmd_pairs(a):
    """Edit / repair SFT records (v5.231) from OpenTitan files outside the
    evaluation tasks' IPs, plus packed project documents (``--docs``,
    jsonl with a ``text`` field, e.g. the H5 data)."""
    import random
    ips, paths = ob.eval_exclusions(read_tasks(a.tasks))
    texts = ob.repo_texts(ob.checkout(a.ot))
    recs = ob.edit_pairs(texts, exclude_ips=ips, exclude_paths=paths,
                         per_file=a.per_file, seed=a.seed)
    n_ot = len(recs)
    for path in a.docs or []:
        for i, line in enumerate(Path(path).read_text().splitlines()):
            if not line.strip():
                continue
            files = ob.split_packed(json.loads(line).get("text", ""))
            # project files only; an OpenTitan path is already covered
            files = {f"{Path(path).stem}/{i}/{k}": v for k, v in files.items()
                     if not k.startswith("hw/ip/")}
            recs += ob.edit_pairs(files, per_file=a.per_file,
                                  seed=a.seed + i + 1, source=Path(path).stem)
    random.Random(a.seed).shuffle(recs)
    recs = recs[:a.max_records]
    with open(a.out, "w") as f:
        for r in recs:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    print(f"pairs: {len(recs)} records ({n_ot} from OpenTitan before the cap; "
          f"{len(ips)} eval IPs excluded) -> {a.out}")


def cmd_generate(a):
    from hdl_eval_all import vllm_generate
    tasks = read_tasks(a.tasks)
    jobs = [("ot_bugfix", t["id"], ob.SYSTEM, t["prompt"]) for t in tasks]
    outs = vllm_generate(a.model, jobs, a.n, a.temperature, a.top_p,
                         a.max_new_tokens, a.max_model_len, a.gpu_mem)
    Path(a.out).write_text(json.dumps(
        {t["id"]: o for t, o in zip(tasks, outs)}, ensure_ascii=False))
    print(f"generate: {len(tasks)} tasks x {a.n} -> {a.out}")


def cmd_score(a):
    tasks = read_tasks(a.tasks)
    samples = json.loads(Path(a.samples).read_text())
    scorer = ob.Scorer(ob.checkout(a.ot))
    for t in tasks:          # fill the gold cache before threads share it
        scorer.gold(t)
    jobs = [(t, s) for t in tasks for s in samples.get(t["id"], [])]
    with ThreadPoolExecutor(a.workers) as ex:
        scored = list(ex.map(lambda ts: scorer.score(*ts), jobs))
    results = {}
    for (t, _), r in zip(jobs, scored):
        results.setdefault(t["id"], []).append(r)
    summary = ob.summarize(tasks, results)
    Path(a.out).write_text(json.dumps(
        {"benchmark": "ot-bugfix/1", "commit": ob.OT_COMMIT,
         "samples": a.samples, "summary": summary, "results": results},
        indent=1, ensure_ascii=False))
    print("score:", json.dumps(summary.get("all")),
          {k: v["pass"] for k, v in summary.items() if "/" in k})


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    sub = ap.add_subparsers(dest="cmd", required=True)
    b = sub.add_parser("build")
    b.add_argument("--ot", default="data/opentitan")
    b.add_argument("--manifest", default=None,
                   help="H5 data manifest.json (seen/unseen split)")
    b.add_argument("--out", default="ot_tasks.jsonl")
    b.add_argument("--seed", type=int, default=0)
    b.add_argument("--per-ip", type=int, default=10)
    b.add_argument("--max-tasks", type=int, default=200)
    b.add_argument("--workers", type=int, default=8)
    pr = sub.add_parser("pairs", help="edit / repair SFT records (v5.231)")
    pr.add_argument("--ot", default="data/opentitan")
    pr.add_argument("--tasks", required=True,
                    help="evaluation tasks: their IPs and files are excluded")
    pr.add_argument("--docs", nargs="*", help="packed project jsonl files")
    pr.add_argument("--per-file", type=int, default=2)
    pr.add_argument("--max-records", type=int, default=20000)
    pr.add_argument("--seed", type=int, default=0)
    pr.add_argument("--out", default="edit_pairs.jsonl")
    g = sub.add_parser("generate")
    g.add_argument("--tasks", required=True)
    g.add_argument("--model", required=True)
    g.add_argument("--out", required=True)
    g.add_argument("--n", type=int, default=1)
    g.add_argument("--temperature", type=float, default=0.0)
    g.add_argument("--top-p", type=float, default=1.0)
    g.add_argument("--max-new-tokens", type=int, default=6144)
    g.add_argument("--max-model-len", type=int, default=16384)
    g.add_argument("--gpu-mem", type=float, default=0.9)
    s = sub.add_parser("score")
    s.add_argument("--tasks", required=True)
    s.add_argument("--samples", required=True)
    s.add_argument("--ot", default="data/opentitan")
    s.add_argument("--out", required=True)
    s.add_argument("--workers", type=int, default=8)
    a = ap.parse_args(argv)
    {"build": cmd_build, "pairs": cmd_pairs, "generate": cmd_generate,
     "score": cmd_score}[a.cmd](a)


if __name__ == "__main__":
    main()
