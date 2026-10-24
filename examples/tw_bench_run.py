"""tw_bench_run.py — score a pilot checkpoint on TMMLU+ / TW-JusticeBench /
C-Eval / CMMLU without lm-eval (plan E3, evastar/eval/tw_bench.py).

    # files fetched where the Hub is reachable (the pod), kept local only
    python examples/tw_bench_run.py --bench tmmluplus \\
        --data data/tw_bench/tmmluplus/data --run runs/dense --shots 5 \\
        --device cuda --out results/tmmluplus.json

    # a Hugging Face model (route B gap eval, v5.129; needs transformers)
    python examples/tw_bench_run.py --bench tmmluplus \\
        --data data/tw_bench/tmmluplus/data --hf google/gemma-4-E4B \\
        --shots 5 --device cuda --out results/gemma4_tmmluplus.json

    # no model: load the set and run the Phase A gates only
    python examples/tw_bench_run.py --bench ceval --data data/tw_bench/ceval/val \\
        --split val --check-only

The Phase A gates (reference answer 100%, empty answer 0%) run on the
loaded set before any model is scored. The output JSON holds task ids,
letters and scores, never question text. --limit N scores the first N
questions (a smoke run; not comparable).
"""
import argparse
import json
import sys
from pathlib import Path

_REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_REPO))

from evastar.eval import tw_bench  # noqa: E402


def load_hf(model_id, revision=None, device="cuda"):
    """(model, tokenizer, resolved commit sha) for a Hugging Face causal LM
    in bf16 (fp32 on CPU). Multimodal checkpoints (Gemma 4, Breeze 2) are
    scored on text only; trust_remote_code covers models that ship their
    own modelling code."""
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer
    kw = dict(revision=revision, trust_remote_code=True)
    tok = AutoTokenizer.from_pretrained(model_id, **kw)
    dtype = torch.float32 if device == "cpu" else torch.bfloat16
    model = AutoModelForCausalLM.from_pretrained(
        model_id, torch_dtype=dtype,
        device_map=None if device == "cpu" else "auto", **kw).eval()
    sha = getattr(model.config, "_commit_hash", None) or revision
    return model, tok, sha


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--bench", choices=tw_bench.BENCHES, required=True)
    ap.add_argument("--data", required=True, help="file or per-subject directory")
    ap.add_argument("--split", default="test")
    ap.add_argument("--dev-split", default="dev", help="split the shots come from")
    ap.add_argument("--shots", type=int, default=0)
    ap.add_argument("--columns", default=None,
                    help='JSON column map, e.g. {"question": "prompt"}')
    ap.add_argument("--categories", default=None,
                    help="JSON file {subject: category} for the category macro")
    ap.add_argument("--template", choices=sorted(tw_bench.TEMPLATES), default="zh")
    ap.add_argument("--run", default=None, help="pilot_100m.py output directory")
    ap.add_argument("--hf", default=None,
                    help="Hugging Face model id or path instead of --run "
                         "(needs transformers; route B gap eval, v5.129)")
    ap.add_argument("--hf-revision", default=None)
    ap.add_argument("--max-length", type=int, default=2048,
                    help="context cut for --hf models")
    ap.add_argument("--checkpoint", default="ckpt.pt")
    ap.add_argument("--device", default="cpu")
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--check-only", action="store_true")
    ap.add_argument("--out", default=None)
    args = ap.parse_args()

    columns = json.loads(args.columns) if args.columns else None
    tasks = tw_bench.load_mcq(args.data, args.bench, args.split, columns)
    if args.limit:
        tasks = tasks[:args.limit]
    gates = tw_bench.self_check(tasks, args.template)
    print(json.dumps({"n": len(tasks), "phase_a": gates}, ensure_ascii=False))
    if args.check_only:
        return
    if not args.run and not args.hf:
        ap.error("--run or --hf is required unless --check-only")

    shots = None
    if args.shots:
        dev = tw_bench.load_mcq(args.data, args.bench, args.dev_split, columns)
        shots = tw_bench.shots_by_subject(dev, args.shots)
    if args.hf:
        model, tok, revision = load_hf(args.hf, args.hf_revision, args.device)
        score = tw_bench.hf_score_fn(model, tok, args.max_length)
    else:
        from evastar.eval.lm_harness import load_pilot
        model, tok, max_len = load_pilot(args.run, device=args.device,
                                         checkpoint=args.checkpoint)
        score = tw_bench.model_score_fn(model, tok, max_len)
    results = tw_bench.run_loglik(tasks, score, shots, args.template)
    cats = json.loads(Path(args.categories).read_text()) if args.categories else None
    summary = tw_bench.summarize(results, cats)
    print(json.dumps({k: v for k, v in summary.items() if k != "subjects"},
                     ensure_ascii=False, indent=2))
    if args.out:
        tw_bench.write_results(args.out, args.bench, results, summary, meta={
            "run": args.run, "checkpoint": args.checkpoint,
            "hf": args.hf, "hf_revision": revision if args.hf else None,
            "split": args.split,
            "shots": args.shots, "template": args.template, "limit": args.limit,
            "phase_a": gates})


if __name__ == "__main__":
    main()
