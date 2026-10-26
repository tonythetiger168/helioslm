"""api_distill_probe.py — distillation from an imperfect API teacher
(helioslm_v5/eval/api_distill_probe.py): sequence-level KD with and
without a verifier, and sparse top-k KD across tokenizers.

Usage:
  python examples/api_distill_probe.py --output benchmarks/api_distill_probe_2026-10-20.json
"""
import argparse
import hashlib
import json
import sys
import time
from pathlib import Path

import torch

_REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_REPO))

from helioslm_v5.eval.api_distill_probe import (  # noqa: E402
    TextAPI, TopKAPI, collect_sequences, exact_match, prompt_pool,
    topk_coverage, topk_targets, train_student, train_teacher, verify_rev)

ARMS = ("ground_truth", "seq_kd", "seq_kd_verified", "topk_kd",
        "topk_kd_verified")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--prompts", type=int, default=1000)
    ap.add_argument("--teacher-steps", type=int, default=100)
    ap.add_argument("--steps", type=int, default=600)
    ap.add_argument("--k", type=int, default=5)
    ap.add_argument("--seed", type=int, default=0,
                    help="teacher, prompts and teacher sampling")
    ap.add_argument("--student-seeds", type=int, nargs="+", default=[0, 1, 2],
                    help="one student per seed and arm (training from "
                         "scratch on this task has a sudden jump in "
                         "accuracy, so one seed is not enough)")
    ap.add_argument("--threads", type=int, default=4)
    ap.add_argument("--output")
    args = ap.parse_args()
    torch.set_num_threads(args.threads)
    t0 = time.time()
    teacher = train_teacher(args.teacher_steps, args.seed)
    prompts, truths = prompt_pool(args.prompts, args.seed)
    text, topk = TextAPI(teacher), TopKAPI(teacher, k=args.k)
    torch.manual_seed(7000 + args.seed)               # teacher sampling
    sampled, _ = collect_sequences(prompts, text.sample)
    verified = [(p, r) for p, r in sampled if verify_rev(p, r)]
    soft = [(p, *topk_targets(topk, p)) for p in prompts]
    soft_verified = [ex for ex in soft if verify_rev(ex[0], ex[1])]
    data = {"ground_truth": (list(zip(prompts, truths)), False),
            "seq_kd": (sampled, False),
            "seq_kd_verified": (verified, False),
            "topk_kd": (soft, True),
            "topk_kd_verified": (soft_verified, True)}
    queries = {"ground_truth": 0, "seq_kd": len(prompts),
               "seq_kd_verified": len(prompts), "topk_kd": topk.calls,
               "topk_kd_verified": topk.calls}
    teacher_row = {"steps": args.teacher_steps,
                   "greedy_exact_match": exact_match(teacher, "rev"),
                   "sample_accept_rate": round(len(verified) / len(prompts), 4),
                   "greedy_path_accept_rate": round(
                       len(soft_verified) / len(prompts), 4),
                   "topk_mass_first_token": round(
                       topk_coverage(TopKAPI(teacher, k=args.k),
                                     prompts[:100]), 4)}
    print("teacher", json.dumps(teacher_row), flush=True)
    results = []
    for arm in ARMS:
        examples, is_soft = data[arm]
        per_seed = [exact_match(train_student(examples, args.steps, s,
                                              soft=is_soft), "rev")
                    for s in args.student_seeds]
        row = {"arm": arm, "examples": len(examples),
               "teacher_queries": queries[arm],
               "exact_match_by_seed": per_seed,
               "exact_match_mean": round(sum(per_seed) / len(per_seed), 4)}
        results.append(row)
        print(json.dumps(row), flush=True)
    out = {"experiment": "API-teacher distillation probe: imperfect teacher, "
                         "text-only vs top-k API, verifier filtering",
           "teacher": teacher_row,
           "train": {k: v for k, v in vars(args).items()
                     if k not in ("output", "threads")},
           "results": results,
           "results_sha256": hashlib.sha256(
               json.dumps(results, sort_keys=True).encode()).hexdigest(),
           "wall_seconds": round(time.time() - t0, 1)}
    if args.output:
        Path(args.output).write_text(json.dumps(out, indent=2) + "\n")
        print("wrote", args.output)


if __name__ == "__main__":
    main()
