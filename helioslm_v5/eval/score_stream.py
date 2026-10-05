"""Standalone token-stream scorer (v5.17, roadmap #6 component 3).

Scores (prompt, output) records produced by ANY engine — colibri, vLLM,
llama.cpp, or HeliosLM itself — under a reference model, so quality
regressions from container formats, pruning, or tiering become comparable
numbers (the audit-side complement to serving engines).

Input JSONL, one record per line:
    {"id": "...", "prompt": "...", "output": "...", "meta": {...}}

Scores emitted per record:
    sum_logprob  — total logprob of output under the reference model
    n_tokens     — number of scored tokens
    nll_per_token— mean negative log-likelihood (lower = more likely)
    ppl          — exp(nll_per_token)

Comparison mode (`--compare`): two files of paired records (same ids) —
e.g. gs64-container outputs vs per-row outputs — emits per-id deltas and a
summary with a bootstrap CI, in the spirit of colibri's container A/B
findings (quality deltas belong in tables, not anecdotes).

Char-level note: with the shipped toy checkpoint (token id == ord(char)),
prompts/outputs must be ASCII. The CLI errors loudly otherwise — the *file
format* is the deliverable; swap the checkpoint for a real model to score
real text.

CLI:
    python -m helioslm_v5.eval.score_stream \
        --checkpoint checkpoints/toy_v5.13.pt \
        --input colibri_outputs.jsonl --output scores.jsonl

    python -m helioslm_v5.eval.score_stream \
        --checkpoint checkpoints/toy_v5.13.pt \
        --compare gs64.jsonl perrow.jsonl --output compare.json
"""

import argparse
import json
import math
import sys

import torch

from helioslm_v5.eval.harness import loglikelihood


def _encode(text, vocab_size):
    bad = sorted({c for c in text if ord(c) >= vocab_size})
    if bad:
        raise ValueError(
            f"text contains {len(bad)} distinct character(s) with "
            f"ord >= vocab_size={vocab_size} (e.g. {bad[:5]!r}); the toy "
            "checkpoint scores char-level ASCII text — swap the checkpoint "
            "to score text outside that range"
        )
    ids = [ord(c) for c in text]
    if not ids:
        raise ValueError("text encodes to zero tokens (empty string?)")
    return ids


def score_record(model, prompt, output, vocab_size):
    """Score one (prompt, output) pair. Returns the score dict.

    ``ppl`` is exp(nll_per_token), computed faithfully: it saturates at
    ``math.inf`` when the exponent overflows (nll > ~709), instead of
    clamping nll — an nll of 20 and an nll of 60 must not report the
    same perplexity.
    """
    ctx = _encode(prompt, vocab_size)
    cont = _encode(output, vocab_size)
    lp = loglikelihood(model, ctx, cont)
    n = len(cont)
    nll = -lp / max(n, 1)
    try:
        ppl = math.exp(nll)
    except OverflowError:
        ppl = math.inf
    return {
        "sum_logprob": round(lp, 4),
        "n_tokens": n,
        "nll_per_token": round(nll, 4),
        "ppl": round(ppl, 2),
    }


def iter_records(path):
    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                yield json.loads(line)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--checkpoint", required=True)
    ap.add_argument("--input", help="JSONL of {id, prompt, output}")
    ap.add_argument("--compare", nargs=2, metavar=("FILE_A", "FILE_B"),
                    help="paired records (same ids) for A/B comparison")
    ap.add_argument("--output", required=True)
    args = ap.parse_args()

    from helioslm_v5.configs.config_v5 import HeliosLMv5Config
    from helioslm_v5.src.model_v5 import HeliosLMv5

    ckpt = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
    model = HeliosLMv5(HeliosLMv5Config(size="lite"))
    model.load_state_dict(ckpt["state_dict"])
    model.eval()
    vocab = ckpt["vocab_size"]

    if args.compare:
        a_path, b_path = args.compare
        a = {r.get("id", str(i)): r for i, r in enumerate(iter_records(a_path))}
        b = {r.get("id", str(i)): r for i, r in enumerate(iter_records(b_path))}
        rows, deltas = [], []
        for rid in a:
            if rid not in b:
                continue
            sa = score_record(model, a[rid]["prompt"], a[rid]["output"], vocab)
            sb = score_record(model, b[rid]["prompt"], b[rid]["output"], vocab)
            d = sb["nll_per_token"] - sa["nll_per_token"]  # + => B worse
            deltas.append(d)
            rows.append({"id": rid, "nll_a": sa["nll_per_token"],
                         "nll_b": sb["nll_per_token"], "delta_b_minus_a": round(d, 4)})
        n = len(deltas)
        if n == 0:
            raise ValueError(
                f"no common record ids between {a_path} and {b_path}; "
                "compare mode needs paired records (same ids in both files)"
            )
        mean = sum(deltas) / n
        # bootstrap CI (deterministic seed)
        g = torch.Generator().manual_seed(0)
        boots = sorted(float(torch.tensor(deltas)[
            torch.randint(0, n, (n,), generator=g)].mean()) for _ in range(1000))
        out = {
            "n_paired": n,
            "mean_delta_nll": round(mean, 4),
            "ci95": [round(boots[25], 4), round(boots[975], 4)],
            "interpretation": "positive delta => B has higher NLL (worse)",
            "rows": rows,
        }
        with open(args.output, "w") as f:
            json.dump(out, f, indent=1)
        print(json.dumps({k: v for k, v in out.items() if k != "rows"}, indent=1))
        return

    n_scored = 0
    with open(args.output, "w", encoding="utf-8") as f:
        for rec in iter_records(args.input):
            scores = score_record(model, rec["prompt"], rec["output"], vocab)
            f.write(json.dumps({"id": rec.get("id"), **scores,
                                "meta": rec.get("meta")}) + "\n")
            n_scored += 1
    print(f"scored {n_scored} records -> {args.output}")


if __name__ == "__main__":
    main()
