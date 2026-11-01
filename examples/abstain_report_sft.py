"""abstain_report_sft.py — synthetic JSON -> report SFT records with
removed measurement fields (v5.216, src/training/report_abstain.py).

  python examples/abstain_report_sft.py --n 1000 --out report_abstain.jsonl
  python examples/sft_hf.py prepare --records report_abstain.jsonl ...

Each record has system / prompt / response (the template report, which
writes 未測量 / not measured for each removed field) plus ``missing`` and
``lang``. Synthetic findings only, no patient data.
"""
import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from helioslm_v5.src.training.report_abstain import make_sft_records  # noqa: E402


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--n", type=int, default=1000)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--p-missing", type=float, default=0.3)
    ap.add_argument("--lang", choices=("zh", "en"), default=None,
                    help="default: both at random")
    ap.add_argument("--out", required=True)
    a = ap.parse_args(argv)
    recs = make_sft_records(a.n, seed=a.seed, p_missing=a.p_missing, lang=a.lang)
    Path(a.out).parent.mkdir(parents=True, exist_ok=True)
    with open(a.out, "w", encoding="utf-8") as fh:
        for r in recs:
            fh.write(json.dumps(r, ensure_ascii=False) + "\n")
    st = {"records": len(recs), "missing_fields": sum(len(r["missing"]) for r in recs)}
    print(json.dumps(st))
    return st


if __name__ == "__main__":
    main()
