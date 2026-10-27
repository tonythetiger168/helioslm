"""teacher_topk.py — LB4 teacher side: next-token top-k of an open model
(v5.120). Runs on the GPU pod; needs ``transformers`` (not a repo
dependency, install it on the pod).

For each document of --text (.jsonl field --field, or .txt) writes one
JSON line {"text", "pieces", "topk"}: the (possibly truncated) text, the
teacher's token pieces (``logit_kd.pieces_from_offsets``) and, for every
piece, its k most likely tokens as [decoded text, log-prob]. Feed the
file to ``build_kd_targets.py``. Teachers must be open-weight models
(no closed-model teachers).

  python3 examples/teacher_topk.py --model Qwen/Qwen3-8B \\
      --text corpus.jsonl --out topk.jsonl --k 20 --max-len 2048
"""
import argparse
import json
import sys
from functools import lru_cache
from pathlib import Path

_REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_REPO))

from helioslm_v5.src.training.logit_kd import (  # noqa: E402
    pieces_from_offsets, topk_from_logprobs)


def teacher_record(text, ids, offsets, logprobs, decode, k, has_bos):
    """One output record. ``logprobs`` are the model's next-token
    log-probs for the input ([BOS] + ids when ``has_bos``, else ids):
    with a BOS row j predicts ids[j]; without one there is no
    distribution for the first piece (an empty top-k list)."""
    end = offsets[-1][1] if offsets else 0
    text = text[:max(end, 0)] if len(ids) and end < len(text) else text
    pieces = pieces_from_offsets(text, offsets)
    rows = logprobs[:len(ids)] if has_bos else logprobs[:len(ids) - 1]
    topk = topk_from_logprobs(rows, k, decode) if len(rows) else []
    if not has_bos:
        topk = [[]] + topk
    return {"text": text, "pieces": pieces, "topk": topk}


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--text", nargs="+", required=True)
    ap.add_argument("--field", default="text")
    ap.add_argument("--out", required=True)
    ap.add_argument("--k", type=int, default=20)
    ap.add_argument("--max-len", type=int, default=2048)
    ap.add_argument("--device", default="cuda")
    a = ap.parse_args(argv)
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer
    from examples.data_quality import read_docs
    tok = AutoTokenizer.from_pretrained(a.model)
    model = AutoModelForCausalLM.from_pretrained(
        a.model, torch_dtype=torch.bfloat16).to(a.device).eval()
    bos = tok.bos_token_id

    @lru_cache(maxsize=None)
    def decode(i):
        return tok.decode([i])

    n = 0
    with open(a.out, "w", encoding="utf-8") as f:
        for text in read_docs(a.text, a.field):
            enc = tok(text, add_special_tokens=False, truncation=True,
                      max_length=a.max_len - 1, return_offsets_mapping=True)
            ids, offs = enc["input_ids"], enc["offset_mapping"]
            if not ids:
                continue
            inp = ([bos] if bos is not None else []) + ids
            with torch.no_grad():
                logits = model(torch.tensor([inp], device=a.device)).logits[0]
            lp = torch.log_softmax(logits.float(), -1).cpu().numpy()
            rec = teacher_record(text, ids, offs, lp, decode, a.k,
                                 bos is not None)
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")
            n += 1
    print(json.dumps({"docs": n, "out": a.out, "k": a.k}))


if __name__ == "__main__":
    main()
