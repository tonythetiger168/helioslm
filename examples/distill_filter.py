"""Filter D1 teacher answers before SFT (D2, docs/DISTILL_DEEPSEEK.md).

    python examples/distill_filter.py --answers runs/distill/answers.jsonl \\
        --out runs/distill/kept.jsonl \\
        --eval data/std_bench/gsm8k_test_first50.jsonl data/std_bench/humaneval_first20.jsonl

Prints the report (drops per reason) and writes it next to the output as
``<out>.report.json``. Standard library only.
"""
import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from helioslm_v5.src.training import distill_filter as df  # noqa: E402
from helioslm_v5.src.training.distill_gen import read_jsonl  # noqa: E402


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--answers", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--eval", nargs="*", default=[],
                    help="evaluation JSONL files to decontaminate against")
    ap.add_argument("--max-reasoning-tokens", type=int,
                    default=df.DEFAULTS["max_reasoning_tokens"])
    ap.add_argument("--max-answer-tokens", type=int,
                    default=df.DEFAULTS["max_answer_tokens"])
    ap.add_argument("--max-simplified-answer", type=float,
                    default=df.DEFAULTS["max_simplified_answer"])
    ap.add_argument("--max-simplified-reasoning", type=float,
                    default=df.DEFAULTS["max_simplified_reasoning"])
    a = ap.parse_args(argv)
    flt = df.Filter(eval_items=df.load_eval_items(a.eval),
                    max_reasoning_tokens=a.max_reasoning_tokens,
                    max_answer_tokens=a.max_answer_tokens,
                    max_simplified_answer=a.max_simplified_answer,
                    max_simplified_reasoning=a.max_simplified_reasoning)
    kept, report = flt.run(read_jsonl(a.answers))
    Path(a.out).parent.mkdir(parents=True, exist_ok=True)
    with open(a.out, "w", encoding="utf-8") as fh:
        for r in kept:
            fh.write(json.dumps(r, ensure_ascii=False) + "\n")
    Path(a.out + ".report.json").write_text(json.dumps(report, ensure_ascii=False, indent=1),
                                            encoding="utf-8")
    print(json.dumps({k: v for k, v in report.items() if k != "config"},
                     ensure_ascii=False, indent=1))


if __name__ == "__main__":
    main()
