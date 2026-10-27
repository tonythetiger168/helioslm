"""Cross-tokenizer logit distillation targets for pretraining (v5.120, LB4).

The 2026 small-model leaders pretrain on a large teacher's logits rather
than one-hot tokens (Gemma 2/3/4, MiniCPM5's distilled 1B). HeliosLM's
student keeps its own Big5-seeded BPE, so a teacher's next-token
distribution is over a DIFFERENT vocabulary and must be moved onto the
student's tokens. ``api_distill.py`` (v5.82) does this for a
character-level student; this module does it for a BPE student.

Alignment. A teacher distribution and a student prediction can only be
compared where both condition on the same text: the student predicts
token m from the prefix ending at token m's start offset; the teacher's
``topk[j]`` predicts piece j from the prefix ``"".join(pieces[:j])``.
Where the two prefixes end at the same character (a shared boundary)
the position is aligned; elsewhere (a teacher token cut inside a student
token or the reverse, or a byte-level piece inside one character) the
student gets no KD target and trains on plain cross-entropy.

Projection. Each teacher candidate t (its decoded text) is mapped to the
first student token of ``student_encode(t)``; probabilities of teacher
tokens with the same first student token add up, and the result is
renormalised over the covered support and cut to the student's top k.
This is the first-token marginal of the teacher's next-token
distribution: exact when a student token never spans a teacher boundary,
an approximation otherwise (the student's real next token may extend
beyond t). Candidates that decode to nothing or to U+FFFD (byte pieces of
a split character) are skipped.

Storage. Targets live beside a ``write_shards`` shard as two arrays with
one row per stream position i: row i is the distribution of token i + 1
(``<name>.kd_idx.npy`` uint16 [N, k] student ids, ``<name>.kd_p.npy``
float16 [N, k] probabilities; an all-zero row is "no target"). That is
k * 4 bytes per token, fine for the pilot and the #41 60B-token gate run
(about 2 TB at k = 8); at 1T tokens run the teacher online instead.
``pretrain.KDShard`` reads them and ``pretrain(kd_alpha=...)`` trains on
them. No torch here, so the targets are built on CPU.
"""
import json
from pathlib import Path

import numpy as np

REPLACEMENT = "�"


class FirstTokenMap:
    """Teacher token text -> first student token id, cached."""

    def __init__(self, student_encode):
        self.encode = student_encode
        self.cache = {}

    def __call__(self, text):
        if text not in self.cache:
            ids = (self.encode(text)
                   if text and REPLACEMENT not in text else [])
            self.cache[text] = int(ids[0]) if len(ids) else None
        return self.cache[text]


def project_topk(topk, first_token, k):
    """[(teacher text, logprob)] -> (ids, probs) over student tokens:
    summed by first student token, renormalised, the k largest (ties by
    id). Returns None when no candidate maps to a student token."""
    mass = {}
    for text, lp in topk:
        sid = first_token(text)
        if sid is not None:
            mass[sid] = mass.get(sid, 0.0) + float(np.exp(lp))
    z = sum(mass.values())
    if z <= 0:
        return None
    top = sorted(mass.items(), key=lambda kv: (-kv[1], kv[0]))[:k]
    zt = sum(p for _, p in top)
    return [i for i, _ in top], [p / zt for _, p in top]


def pieces_from_offsets(text, offsets):
    """Teacher piece texts from a tokenizer's (start, end) offsets: piece
    j is the text from the previous end to its own end, so a character
    split over byte tokens is spelled by its FIRST piece and the rest are
    empty; text after the last token goes on the last piece."""
    pieces, prev = [], 0
    for _, end in offsets:
        end = max(end, prev)
        pieces.append(text[prev:end])
        prev = end
    if pieces:
        pieces[-1] += text[prev:]
    return pieces


def teacher_prefix_ends(pieces, text):
    """{prefix end offset: j} for the teacher's topk[j]; checks that the
    pieces spell the text. Several pieces start at one offset only after
    empty pieces (``pieces_from_offsets``: the byte tokens completing a
    character); the last of them is the one conditioned on the whole
    character, so it wins."""
    if "".join(pieces) != text:
        raise ValueError("teacher pieces do not concatenate to the text")
    ends, pos = {}, 0
    for j, p in enumerate(pieces):
        ends[pos] = j
        pos += len(p)
    return ends


def topk_from_logprobs(logprobs, k, decode):
    """Rows of next-token log-probs ([n, V] array; row j is the
    distribution of piece j) -> per-row [(text, logprob)] of the k most
    likely tokens, ``decode(id)`` giving each token's text."""
    lp = np.asarray(logprobs, dtype=np.float64)
    kk = min(k, lp.shape[-1])
    top = np.argpartition(-lp, kk - 1, axis=-1)[:, :kk]
    out = []
    for row, ids in zip(lp, top):
        ids = ids[np.argsort(-row[ids], kind="stable")]
        out.append([(decode(int(i)), float(row[i])) for i in ids])
    return out


def student_prefix_ends(offsets):
    """For each student token m: the prefix end it is predicted from, or
    None when token m does not start on a clean boundary (it starts
    inside the previous token's span, as byte pieces of one character
    do)."""
    out = []
    for m, (a, _) in enumerate(offsets):
        if m == 0:
            out.append(0 if a == 0 else None)
        else:
            pa, pb = offsets[m - 1]
            out.append(pb if a == pb and pb > pa else None)
    return out


def doc_targets(offsets, pieces, topk, text, first_token, k):
    """KD targets for one document: {student token m: (ids, probs)}."""
    if len(topk) != len(pieces):
        raise ValueError("need one top-k list per teacher piece")
    t_end = teacher_prefix_ends(pieces, text)
    out = {}
    for m, end in enumerate(student_prefix_ends(offsets)):
        j = t_end.get(end) if end is not None else None
        if j is None:
            continue
        r = project_topk(topk[j], first_token, k)
        if r is not None:
            out[m] = r
    return out


class KDTargetWriter:
    """Builds a shard's target arrays alongside its token stream.

    ``add(ids, targets)`` appends one document's token ids (already
    ending in EOS, as ``write_shards`` takes them) and its ``doc_targets``;
    a target for the document's token m lands on stream row
    ``start + m - 1`` (the row of the token it is predicted from), so the
    very first token of the shard has none."""

    def __init__(self, k):
        self.k = k
        self.idx, self.p, self.n = [], [], 0
        self.aligned = self.predictable = 0

    def add(self, ids, targets):
        n = len(ids)
        idx = np.zeros((n, self.k), dtype=np.uint16)
        p = np.zeros((n, self.k), dtype=np.float16)
        prev = None if not self.idx else (self.idx[-1], self.p[-1])
        for m, (ti, tp) in targets.items():
            if not 0 <= m < n:
                raise ValueError(f"target for token {m} of a {n}-token doc")
            if max(ti) > 65535:
                raise ValueError("student ids must fit uint16")
            r = m - 1
            if r < 0:
                if prev is None:
                    continue
                dst_i, dst_p, r = prev[0], prev[1], len(prev[0]) - 1
            else:
                dst_i, dst_p = idx, p
            dst_i[r, :len(ti)] = ti
            dst_p[r, :len(tp)] = tp
            self.aligned += 1
        self.idx.append(idx)
        self.p.append(p)
        self.n += n
        self.predictable += n - 1          # text tokens; EOS has no target
        return self

    def save(self, out_dir, name):
        out = Path(out_dir)
        idx = (np.concatenate(self.idx) if self.idx
               else np.zeros((0, self.k), np.uint16))
        p = (np.concatenate(self.p) if self.p
             else np.zeros((0, self.k), np.float16))
        np.save(out / f"{name}.kd_idx.npy", idx)
        np.save(out / f"{name}.kd_p.npy", p)
        stats = {"tokens": self.n, "k": self.k,
                 "rows_with_target": int((p.astype(np.float32).sum(1) > 0)
                                         .sum()),
                 "aligned_share": round(self.aligned /
                                        max(self.predictable, 1), 4)}
        (out / f"{name}.kd.json").write_text(json.dumps(stats) + "\n")
        return stats


def load_targets(out_dir, name):
    """Memory-mapped (idx, p) arrays of a shard."""
    out = Path(out_dir)
    return (np.load(out / f"{name}.kd_idx.npy", mmap_mode="r"),
            np.load(out / f"{name}.kd_p.npy", mmap_mode="r"))
