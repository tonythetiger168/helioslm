"""Offline cost/provenance analysis; reads files only, never calls an API.

python examples/analyze_api_usage.py --records saved_records.json \
    --corpus tasks.jsonl --harness-revision <commit-used-for-evaluation> \
    --tokenizer <declared-name-and-revision> --rates rates.json --out report.json

records: JSON list of {provider, requested_model, response, task_id?}.
response is a saved complete OpenAI or Anthropic response envelope.
rates: {provider, model, as_of, source, rates: {input, output, cache_read,
cache_write, cache_write_30m, cache_write_12h, cache_write_5m, cache_write_1h}}.
All rates are explicit USD/1M token assumptions. Supply only relevant keys;
nonzero counts with missing rates produce unknown cost. Optional rate-card
service_tier/max_input_tokens constrain applicability. --outcomes is a JSON
mapping of task ids to booleans, used for cost per successful task.
--settings records declared seed, shots, temperature, sample count and
other evaluation conditions from a JSON object; it does not infer them.

Existing probe_debug/resp_*.json envelopes can be wrapped into records.
Legacy aggregated probe summaries cannot recover missing cache counters.
"""
import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from evastar.eval.api_accounting import build_report  # noqa: E402


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--records", required=True)
    ap.add_argument("--corpus", required=True)
    ap.add_argument("--harness-revision", required=True)
    ap.add_argument("--tokenizer")
    ap.add_argument("--rates")
    ap.add_argument("--outcomes")
    ap.add_argument("--settings")
    ap.add_argument("--out", required=True)
    a = ap.parse_args(argv)

    def read(path):
        return json.loads(Path(path).read_text(encoding="utf-8"))

    report = build_report(read(a.records), corpus=Path(a.corpus).read_bytes(),
                          harness_revision=a.harness_revision, tokenizer=a.tokenizer,
                          rate_card=read(a.rates) if a.rates else None,
                          outcomes=read(a.outcomes) if a.outcomes else None,
                          evaluation_settings=read(a.settings) if a.settings else None)
    Path(a.out).write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n",
                           encoding="utf-8")
    return report


if __name__ == "__main__":
    main()
