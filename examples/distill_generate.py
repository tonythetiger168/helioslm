"""Teacher data generation for distillation (D1, docs/DISTILL_DEEPSEEK.md).

Talks to an OpenAI-compatible endpoint: by default a vLLM server on the
same pod (``vllm serve deepseek-ai/DeepSeek-V4.1-Flash ...``, no key).
Standard library only; torch is not needed.

    # 1. the teacher writes the prompts (thinking off, short answers)
    python examples/distill_generate.py seed --seeds data/distill/seeds_zhtw.json \\
        --per-topic 350 --out runs/distill/meta.jsonl
    python examples/distill_generate.py generate --prompts runs/distill/meta.jsonl \\
        --out runs/distill/meta_answers.jsonl --no-thinking --max-tokens 2048
    python examples/distill_generate.py extract --answers runs/distill/meta_answers.jsonl \\
        --out runs/distill/prompts.jsonl
    # 2. the teacher answers them (thinking on); safe to re-run after a crash
    python examples/distill_generate.py generate --prompts runs/distill/prompts.jsonl \\
        --out runs/distill/answers.jsonl --max-tokens 8192 --per-hour 15 --max-cost 120
    # 3. token statistics (OS1) and cost before / after
    python examples/distill_generate.py stats --answers runs/distill/answers.jsonl
    python examples/distill_generate.py estimate --prompts 20000 --in-tokens 300 \\
        --out-tokens 2000 --per-hour 15 --tok-per-s 3000 --setup-hours 1
"""
import argparse
import json
import random
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from helioslm_v5.src.training import distill_gen as dg  # noqa: E402


def write_jsonl(path, rows):
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as fh:
        for r in rows:
            fh.write(json.dumps(r, ensure_ascii=False) + "\n")


def pricing(a):
    return dg.Pricing(currency=a.currency, input_per_m=a.input_per_m,
                      cached_input_per_m=a.cached_input_per_m,
                      output_per_m=a.output_per_m, per_hour=a.per_hour)


def add_pricing(p):
    p.add_argument("--currency", default="USD")
    p.add_argument("--input-per-m", type=float, default=0.0)
    p.add_argument("--cached-input-per-m", type=float, default=0.0)
    p.add_argument("--output-per-m", type=float, default=0.0)
    p.add_argument("--per-hour", type=float, default=0.0,
                   help="pod price per hour (all GPUs), for self-hosting")


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)

    s = sub.add_parser("seed", help="meta-prompts asking the teacher for questions")
    s.add_argument("--seeds", default="data/distill/seeds_zhtw.json")
    s.add_argument("--per-topic", type=int, default=350)
    s.add_argument("--per-call", type=int, default=10)
    s.add_argument("--out", required=True)

    g = sub.add_parser("generate", help="run prompts through the teacher")
    g.add_argument("--prompts", required=True)
    g.add_argument("--out", required=True)
    g.add_argument("--teacher", default="deepseek-v4.1-flash")
    g.add_argument("--endpoint", choices=("vllm", "openai"), default="vllm")
    g.add_argument("--base-url", default="http://localhost:8000/v1")
    g.add_argument("--api-key-env", default=None,
                   help="env var with the key (hosted API only)")
    g.add_argument("--model", default=None)
    g.add_argument("--temperature", type=float, default=0.6)
    g.add_argument("--top-p", type=float, default=0.95)
    g.add_argument("--max-tokens", type=int, default=8192)
    g.add_argument("--no-thinking", action="store_true")
    g.add_argument("--reasoning-effort", default=None,
                   help="passed through, e.g. low / high (DeepSeek V4.1)")
    g.add_argument("--keep-thinking", choices=dg.KEEP_THINKING, default="none")
    g.add_argument("--samples", type=int, default=1)
    g.add_argument("--workers", type=int, default=64)
    g.add_argument("--limit", type=int, default=None, help="first N prompts only")
    g.add_argument("--shuffle-seed", type=int, default=None)
    g.add_argument("--max-cost", type=float, default=None)
    g.add_argument("--max-output-tokens", type=int, default=None)
    g.add_argument("--max-hours", type=float, default=None)
    add_pricing(g)

    x = sub.add_parser("extract", help="questions out of meta-prompt answers")
    x.add_argument("--answers", required=True)
    x.add_argument("--out", required=True)

    t = sub.add_parser("stats", help="token statistics of generated answers")
    t.add_argument("--answers", required=True)

    e = sub.add_parser("estimate", help="cost before a run")
    e.add_argument("--prompts", type=int, required=True)
    e.add_argument("--in-tokens", type=int, default=300)
    e.add_argument("--out-tokens", type=int, default=2000)
    e.add_argument("--samples", type=int, default=1)
    e.add_argument("--tok-per-s", type=float, default=None)
    e.add_argument("--setup-hours", type=float, default=1.0)
    add_pricing(e)

    a = ap.parse_args(argv)
    if a.cmd == "seed":
        topics = json.loads(Path(a.seeds).read_text(encoding="utf-8"))
        seeds = [{"domain": d, "topic": tp, "n": a.per_topic}
                 for d, tps in topics.items() if not d.startswith("_") for tp in tps]
        rows = dg.meta_prompts(seeds, per_call=a.per_call)
        write_jsonl(a.out, rows)
        print(json.dumps({"topics": len(seeds), "meta_prompts": len(rows)}))
    elif a.cmd == "generate":
        prompts = dg.read_jsonl(a.prompts)
        if a.shuffle_seed is not None:
            random.Random(a.shuffle_seed).shuffle(prompts)
        if a.limit:
            prompts = prompts[:a.limit]
        extra = {"reasoning_effort": a.reasoning_effort} if a.reasoning_effort else {}
        cfg = dg.GenConfig(
            teacher=a.teacher, endpoint=a.endpoint, base_url=a.base_url,
            api_key_env=a.api_key_env, model=a.model, temperature=a.temperature,
            top_p=a.top_p, max_tokens=a.max_tokens, thinking=not a.no_thinking,
            extra=extra, samples=a.samples, workers=a.workers,
            keep_thinking=a.keep_thinking, max_cost=a.max_cost,
            max_output_tokens=a.max_output_tokens,
            max_seconds=a.max_hours * 3600 if a.max_hours else None)
        Path(a.out).parent.mkdir(parents=True, exist_ok=True)
        summary = dg.Generator(cfg).run(prompts, a.out, pricing=pricing(a),
                                        log=lambda m: print(m, flush=True))
        print(json.dumps(summary, ensure_ascii=False, indent=1))
    elif a.cmd == "extract":
        rows, st = dg.extract_prompts(dg.read_jsonl(a.answers))
        write_jsonl(a.out, rows)
        print(json.dumps(st, ensure_ascii=False))
    elif a.cmd == "stats":
        print(json.dumps(dg.token_stats(dg.read_jsonl(a.answers)), ensure_ascii=False,
                         indent=1))
    elif a.cmd == "estimate":
        print(json.dumps(dg.estimate(a.prompts, a.in_tokens, a.out_tokens, pricing(a),
                                     samples=a.samples, output_tokens_per_s=a.tok_per_s,
                                     setup_hours=a.setup_hours if a.per_hour else 0.0),
                         indent=1))


if __name__ == "__main__":
    main()
