"""build_kd_targets.py — LB4 student-side KD shard builder (v5.120).

Reads the teacher's top-k file (``teacher_topk.py`` output, one JSON per
document: {"text", "pieces", "topk"}), tokenises each text with the
student tokenizer exactly as ``pilot_prepare_data.py`` does (ids + EOS),
moves the teacher distributions onto student tokens
(``logit_kd.doc_targets``) and writes, into --out:

  <name>.bin                       the token shard (uint16, write_shards
                                   layout)
  <name>.kd_idx.npy / .kd_p.npy    the targets, one row per token
  <name>.kd.json                   rows with a target, aligned share

Train on it with ``pilot_100m.py --data <pilot data> --kd-train
<out>/<name>.bin`` (the tokenizer hash must match the pilot data's). No torch
needed.

  python3 examples/build_kd_targets.py --teacher topk.jsonl \\
      --tokenizer pilot_data/tokenizer.json --out kd_data --k 8
"""
import argparse
import hashlib
import json
import sys
from pathlib import Path

import numpy as np

_REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_REPO))

from evastar.src.training.logit_kd import (  # noqa: E402
    FirstTokenMap, KDTargetWriter, doc_targets)


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--teacher", nargs="+", required=True,
                    help="JSONL from teacher_topk.py")
    ap.add_argument("--tokenizer", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--name", default="train")
    ap.add_argument("--k", type=int, default=8)
    a = ap.parse_args(argv)
    from tokenizers import Tokenizer
    tok = Tokenizer.from_file(a.tokenizer)
    eot = tok.token_to_id("<|endoftext|>")
    if eot is None:
        eot = tok.get_vocab_size() - 1
    first = FirstTokenMap(lambda t: tok.encode(t).ids)
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    w = KDTargetWriter(a.k)
    docs = bad = 0
    with open(out / f"{a.name}.bin", "wb") as f:
        for path in a.teacher:
            for line in Path(path).read_text(encoding="utf-8").splitlines():
                if not line.strip():
                    continue
                r = json.loads(line)
                enc = tok.encode(r["text"])
                try:
                    tg = doc_targets(enc.offsets, r["pieces"], r["topk"],
                                     r["text"], first, a.k)
                except ValueError:
                    tg, bad = {}, bad + 1      # keep the text, no targets
                ids = enc.ids + [eot]
                f.write(np.asarray(ids, dtype=np.uint16).tobytes())
                w.add(ids, tg)
                docs += 1
    stats = w.save(out, a.name)
    stats.update(docs=docs, docs_without_targets=bad,
                 tokenizer_sha256=hashlib.sha256(
                     Path(a.tokenizer).read_bytes()).hexdigest())
    (out / f"{a.name}.kd.json").write_text(json.dumps(stats) + "\n")
    print(json.dumps(stats))
    return stats


if __name__ == "__main__":
    main()
