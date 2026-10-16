"""std_bench_run.py — roadmap Phase A: GSM8K / HumanEval / terminal runs,
plus the Phase AC mini SWE tasks.

Two modes:

1. Harness validation (no model, no network) — the committed artifact:
       python examples/std_bench_run.py --reference
   Scores synthetic generators on the committed subsets and the
   terminal and SWE tasks, and enforces 15 gates: gold answers /
   canonical solutions / scripted reference solutions must score 100%;
   empty, perturbed, do-nothing, prose-only and trap-patch replies must
   score 0%. A failing
   gate exits
   non-zero and writes nothing, so the artifact is "oracle-tested".
   Add --check to verify the committed artifact instead of writing it.

2. A model run over an OpenAI-compatible API (needs OPENAI_API_KEY;
   presets and the request ladder come from probe_api.py):
       python examples/std_bench_run.py --preset opus-5.5 --bench gsm8k --samples 3
   Output is in the shared probe schema, so it can go straight onto the
   calibration board. Any API error aborts the run — a partial run is
   not a measurement. HumanEval executes the model's code: see the
   safety note in helioslm_v5/eval/std_bench.py.

Default data is the committed CI subset; --full uses data/std_bench/full/
(fetch it first with scripts/fetch_std_bench.py).
"""
import argparse
import json
import os
import re
import sys
from pathlib import Path

_REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_REPO))

from helioslm_v5.eval.calibration_board import (  # noqa: E402
    canonical_sha256, sha256_file)
from helioslm_v5.eval.std_bench import (  # noqa: E402
    SCHEMA_ID, gsm8k_prompt, humaneval_prompt, load_gsm8k, load_humaneval,
    run_gsm8k, run_humaneval, summarize)
from helioslm_v5.eval import swe_bench  # noqa: E402
from helioslm_v5.eval.term_bench import load_tasks, run_term_bench  # noqa: E402

DATE = "2026-10-15"
DATA = {
    "gsm8k": ("data/std_bench/gsm8k_test_first50.jsonl",
              "data/std_bench/full/gsm8k_test.jsonl", load_gsm8k, run_gsm8k),
    "humaneval": ("data/std_bench/humaneval_first20.jsonl",
                  "data/std_bench/full/HumanEval.jsonl", load_humaneval,
                  run_humaneval),
    # the terminal tasks are defined in code; their "data file" is the
    # module itself, hashed like any other input
    "terminal": ("helioslm_v5/eval/term_bench.py",
                 "helioslm_v5/eval/term_bench.py",
                 lambda _path: load_tasks(), run_term_bench),
    "swe": ("helioslm_v5/eval/swe_bench.py", "helioslm_v5/eval/swe_bench.py",
            lambda _path: swe_bench.load_tasks(), swe_bench.run_swe_bench),
}
REFERENCE_OUT = "benchmarks/std_bench_reference_2026-10-15.json"


def reference_generators(bench, tasks):
    """(name, generate_fn, required accuracy) for harness validation."""
    if bench == "gsm8k":
        gold = {gsm8k_prompt(t): t["answer"] for t in tasks}
        bumped = {p: str(int(float(a)) + 1) for p, a in gold.items()}
        return [
            ("gold_marker", lambda p, i: f"Reasoning.\n#### {gold[p]}", 1.0),
            ("gold_last_number",
             lambda p, i: f"so the answer is {gold[p]}.", 1.0),
            ("empty", lambda p, i: "", 0.0),
            ("off_by_one", lambda p, i: f"#### {bumped[p]}", 0.0),
        ]
    if bench in ("terminal", "swe"):
        def scripted_from(solutions):
            def scripted(prompt, i):
                m = re.search(r"\[((?:term|swe)/[^,]+), step (\d+) of",
                              prompt)
                steps, k = solutions[m.group(1)], int(m.group(2)) - 1
                return (f"```bash\n{steps[k]}\n```" if k < len(steps)
                        else "DONE confidence=0.9")
            return scripted
        gens = [
            ("reference_solution",
             scripted_from({t["id"]: t["solution"] for t in tasks}), 1.0),
            ("done_immediately", lambda p, i: "DONE confidence=0.5", 0.0),
            ("prose_no_command", lambda p, i: "This looks easy.", 0.0),
        ]
        if bench == "swe":
            traps = {k: [v] for k, v in swe_bench.trap_commands().items()}
            gens.append(("trap_patch", scripted_from(traps), 0.0))
        return gens
    canon = {humaneval_prompt(t): t for t in tasks}
    return [
        ("canonical_body", lambda p, i: canon[p]["canonical_solution"], 1.0),
        ("canonical_fenced_function",
         lambda p, i: ("```python\n" + canon[p]["prompt"]
                       + canon[p]["canonical_solution"] + "```"), 1.0),
        ("empty", lambda p, i: "", 0.0),
        ("pass_body", lambda p, i: "    pass\n", 0.0),
    ]


def build_reference():
    report = {"schema": SCHEMA_ID, "created": DATE,
              "verification_tier": "oracle-tested",
              "purpose": "harness validation: answer extraction, the "
                         "HumanEval sandbox and the terminal and SWE "
                         "task verifiers, scored against known-right and "
                         "known-wrong generators",
              "inputs": {}, "gates": {}}
    failures = []
    for bench, (subset, _, load, run) in DATA.items():
        report["inputs"][subset] = sha256_file(_REPO / subset)
        tasks = load(_REPO / subset)
        for name, fn, want in reference_generators(bench, tasks):
            got = summarize(run(tasks, fn))["accuracy"]
            report["gates"][f"{bench}/{name}"] = {"required": want,
                                                  "accuracy": got}
            if got != want:
                failures.append(f"{bench}/{name}: {got} != {want}")
    if failures:
        sys.exit("reference gate FAILED: " + "; ".join(failures))
    report["results_sha256"] = canonical_sha256(report["gates"])
    return report


def api_run(args):
    sys.path.insert(0, str(_REPO / "examples"))
    from probe_api import PRESETS, PRICES_ASOF, chat_once  # noqa: E402
    price = None
    if args.preset:
        p = PRESETS[args.preset]
        args.base, args.model = args.base or p["base"], args.model or p["model"]
        price = p["price"]
    if not (args.base and args.model):
        sys.exit("give --preset, or --base and --model")
    key = os.environ.get("OPENAI_API_KEY")
    if not key:
        sys.exit("set OPENAI_API_KEY first")
    temperature = 0.7 if args.samples > 1 else 0.0
    meter, errors = {}, []

    def generate(prompt, i):
        text, conf = chat_once(args.base, key, args.model, prompt,
                               max_tokens=args.max_tokens,
                               temperature=temperature, meter=meter)
        if text.startswith("API_ERROR"):
            errors.append(text[:200])
            sys.exit(f"aborting: {text[:200]}")
        # first-token logprob says little about a multi-step answer;
        # use it only on request, else self-consistency (samples > 1)
        return (text, conf) if args.model_conf else text

    for bench in args.bench:
        subset, full, load, run = DATA[bench]
        path = _REPO / (full if args.full else subset)
        tasks = load(path)[:args.limit] if args.limit else load(path)
        results = run(tasks, generate, n_samples=args.samples)
        summary = summarize(results)
        cost = None
        if price and meter.get("calls"):
            cost = round((meter.get("prompt_tokens", 0) * price[0]
                          + meter.get("completion_tokens", 0) * price[1])
                         / 1_000_000, 6)
        n_correct = sum(r["correct"] for r in results)
        out = {"schema": SCHEMA_ID, "bench": bench, "model": args.model,
               "created": DATE, "verification_tier": "measured",
               "data": {"path": str(path.relative_to(_REPO)),
                        "sha256": sha256_file(path), "n": len(tasks)},
               "samples": args.samples, "temperature": temperature,
               **summary, "usage": dict(meter) or None, "cost_usd": cost,
               "cost_per_correct_usd": (round(cost / n_correct, 6)
                                        if cost and n_correct else None),
               "prices_asof": PRICES_ASOF if price else None,
               "results": results}
        name = args.model.split("/")[-1]
        dest = Path(args.out_dir) / f"std_bench_{bench}_{name}_{DATE}.json"
        dest.write_text(json.dumps(out, indent=2) + "\n", encoding="utf-8")
        print(json.dumps({k: v for k, v in out.items() if k != "results"},
                         indent=2))
        meter.clear()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--reference", action="store_true")
    ap.add_argument("--check", action="store_true",
                    help="with --reference: verify the committed artifact")
    ap.add_argument("--bench", nargs="+", choices=sorted(DATA),
                    default=sorted(DATA))
    ap.add_argument("--preset", default=None)
    ap.add_argument("--base", default=None)
    ap.add_argument("--model", default=None)
    ap.add_argument("--samples", type=int, default=1)
    ap.add_argument("--max-tokens", type=int, default=1024)
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--full", action="store_true")
    ap.add_argument("--model-conf", action="store_true",
                    help="use the API's first-token logprob as conf")
    ap.add_argument("--out-dir", default=".")
    args = ap.parse_args()

    if args.reference:
        text = json.dumps(build_reference(), indent=2) + "\n"
        dest = _REPO / REFERENCE_OUT
        if args.check:
            if dest.read_text(encoding="utf-8") != text:
                sys.exit(f"stale reference artifact: {REFERENCE_OUT}")
            print("std_bench reference artifact up to date")
            return
        dest.write_text(text, encoding="utf-8")
        print(text)
        return
    api_run(args)


if __name__ == "__main__":
    main()
