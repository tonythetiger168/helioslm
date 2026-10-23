"""T125 — LB4 cross-tokenizer logit KD for pretraining (v5.120).

Pins:

1. teacher pieces: offsets -> pieces spell the text, a character split
   over byte tokens is spelled by its first piece, and the prefix map
   points at the piece conditioned on the whole character;
2. student boundaries: a token starting inside the previous token's span
   (byte pieces of one character) gets no target;
3. projection: teacher candidates add up by first student token, are
   renormalised and cut to k; U+FFFD / empty candidates are skipped;
4. doc_targets: with one shared tokenization and the full teacher
   vocabulary the targets ARE the teacher distribution; with a coarser
   teacher only shared boundaries get targets;
5. KDTargetWriter: a target for token m lands on row m - 1, the first
   token of a later document on the previous EOS row, the shard's first
   token nowhere; stats count aligned positions;
6. topk_from_logprobs and teacher_topk.teacher_record (with and without
   BOS, truncation);
7. build_kd_targets.py end to end on a trained BPE (needs
   ``tokenizers``): a teacher sharing the student's tokenization aligns
   every text position and its best candidate is the true next token;
8. torch: sparse_kd_loss is CE for a one-hot target and 0 without
   targets; KDShard rows line up with the targets they predict; pretrain
   with KD logs kd_loss, kd_alpha=0 matches plain CE, kd_alpha=1 pulls
   the student toward the teacher; bad kd_alpha is refused.

Run from repo root: python3 helioslm_v5/tests/test_logit_kd.py
"""
import json
import math
import subprocess
import sys
import tempfile
from pathlib import Path

import numpy as np

_REPO = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(_REPO))

from examples.teacher_topk import teacher_record  # noqa: E402
from helioslm_v5.src.training.logit_kd import (  # noqa: E402
    FirstTokenMap, KDTargetWriter, doc_targets, pieces_from_offsets,
    project_topk, student_prefix_ends, teacher_prefix_ends,
    topk_from_logprobs)

LN = math.log


def test_teacher_pieces():
    text = "a字b"
    # byte-level teacher: 字 is two byte tokens sharing offsets (1, 2)
    offs = [(0, 1), (1, 2), (1, 2), (2, 3)]
    pieces = pieces_from_offsets(text, offs)
    assert pieces == ["a", "字", "", "b"], pieces
    ends = teacher_prefix_ends(pieces, text)
    # prefix "a字" (end 2) is conditioned on the whole character: piece 3
    assert ends == {0: 0, 1: 1, 2: 3}, ends
    assert pieces_from_offsets("ab ", [(0, 1), (1, 2)]) == ["a", "b "]
    try:
        teacher_prefix_ends(["a", "x"], "ab")
    except ValueError:
        pass
    else:
        raise AssertionError("mismatched pieces accepted")
    print("PASS test_teacher_pieces")


def test_student_boundaries():
    assert student_prefix_ends([(0, 1), (1, 3), (3, 4)]) == [0, 1, 3]
    # tokens 1 and 2 are byte pieces of one character at (1, 2)
    assert student_prefix_ends([(0, 1), (1, 2), (1, 2), (2, 3)]) == \
        [0, 1, None, 2]
    assert student_prefix_ends([(1, 2)]) == [None]
    print("PASS test_student_boundaries")


VOCAB = {"a": 1, "b": 2, "c": 3, "ab": 4, " ": 5}


def _greedy(text):
    """Longest-match student tokenizer over VOCAB: ids and offsets."""
    ids, offs, i = [], [], 0
    while i < len(text):
        for n in (2, 1):
            if text[i:i + n] in VOCAB:
                ids.append(VOCAB[text[i:i + n]])
                offs.append((i, i + n))
                i += n
                break
        else:
            raise ValueError(text[i])
    return ids, offs


def test_projection():
    first = FirstTokenMap(lambda t: _greedy(t)[0])
    ids, p = project_topk([("ab", LN(0.4)), ("a", LN(0.2)), ("b", LN(0.2)),
                           ("�", LN(0.1)), ("", LN(0.1))], first, 8)
    # "ab" -> 4, "a" -> 1, "b" -> 2; the rest skipped; renormalised
    assert ids == [4, 1, 2] and np.allclose(p, [0.5, 0.25, 0.25]), (ids, p)
    ids, p = project_topk([("ab", LN(0.4)), ("abc", LN(0.4)),
                           ("c", LN(0.2))], first, 1)
    assert ids == [4] and p == [1.0]           # 0.8 on 4 ("ab", "ab"+"c")
    assert project_topk([("�", 0.0)], first, 4) is None
    print("PASS test_projection")


def test_doc_targets():
    text = "abab c"
    ids, offs = _greedy(text)                   # ab ab " " c
    first = FirstTokenMap(lambda t: _greedy(t)[0])
    inv = {v: k for k, v in VOCAB.items()}
    # same tokenization, full-vocab teacher: targets equal the teacher
    dist = {4: 0.5, 1: 0.2, 2: 0.1, 3: 0.1, 5: 0.1}
    topk = [[(inv[i], LN(q)) for i, q in dist.items()] for _ in ids]
    pieces = [text[a:b] for a, b in offs]
    tg = doc_targets(offs, pieces, topk, text, first, 8)
    assert sorted(tg) == list(range(len(ids)))
    for i, q in tg.values():
        assert sorted(i) == sorted(dist)
        assert all(abs(dist[t] - v) < 1e-9 for t, v in zip(i, q))
    # coarser teacher: "abab" is one piece, so student token 1 (the second
    # "ab", predicted from "ab") has no aligned teacher distribution
    tp = ["abab", " ", "c"]
    tk = [[("ab", 0.0)], [(" ", 0.0)], [("c", 0.0)]]
    tg = doc_targets(offs, tp, tk, text, first, 4)
    assert sorted(tg) == [0, 2, 3], sorted(tg)
    assert tg[3] == ([3], [1.0])
    try:
        doc_targets(offs, tp, tk[:2], text, first, 4)
    except ValueError:
        pass
    else:
        raise AssertionError("short top-k accepted")
    print("PASS test_doc_targets")


def test_writer():
    w = KDTargetWriter(2)
    w.add([7, 8, 0], {0: ([7], [1.0]), 1: ([8, 9], [0.75, 0.25])})
    w.add([5, 0], {0: ([5], [1.0])})
    with tempfile.TemporaryDirectory() as d:
        st = w.save(d, "train")
        idx = np.load(Path(d) / "train.kd_idx.npy")
        p = np.load(Path(d) / "train.kd_p.npy")
    assert idx.shape == (5, 2) and idx.dtype == np.uint16
    # row 0 predicts token 1 (8); row 2 (doc 1's EOS) predicts doc 2's 5
    assert idx[0].tolist() == [8, 9] and p[0].tolist() == [0.75, 0.25]
    assert idx[2].tolist() == [5, 0] and p[2].tolist() == [1.0, 0.0]
    assert not p[1].any() and not p[3].any() and not p[4].any()
    # doc 1's token 0 is the first token of the shard: no row
    assert st == {"tokens": 5, "k": 2, "rows_with_target": 2,
                  "aligned_share": round(2 / 3, 4)}, st
    print("PASS test_writer")


def test_teacher_side():
    lp = np.log(np.array([[0.1, 0.6, 0.3], [0.5, 0.2, 0.3]]))
    out = topk_from_logprobs(lp, 2, lambda i: "xyz"[i])
    assert [[t for t, _ in r] for r in out] == [["y", "z"], ["x", "z"]]
    assert abs(out[0][0][1] - LN(0.6)) < 1e-12
    text = "ab c"
    offs = [(0, 2), (2, 3), (3, 4)]
    v = np.log(np.full((4, 3), 1 / 3))
    r = teacher_record(text, [1, 2, 3], offs, v, lambda i: "xyz"[i], 2, True)
    assert r["pieces"] == ["ab", " ", "c"] and len(r["topk"]) == 3
    r = teacher_record(text, [1, 2, 3], offs, v, lambda i: "xyz"[i], 2,
                       False)
    assert r["topk"][0] == [] and len(r["topk"]) == 3
    # truncated: the text is cut at the last token
    r = teacher_record("ab cd", [1, 2], offs[:2], v, lambda i: "xyz"[i], 2,
                       True)
    assert r["text"] == "ab " and r["pieces"] == ["ab", " "]
    print("PASS test_teacher_side")


DOCS = ["the cat sat on the mat and the dog sat on the log",
        "a bird in the hand is worth two in the bush",
        "the quick brown fox jumps over the lazy dog"] * 4


def _bpe(d):
    from tokenizers import (Tokenizer, decoders, models, pre_tokenizers,
                            trainers)
    tok = Tokenizer(models.BPE(unk_token="[UNK]"))
    tok.pre_tokenizer = pre_tokenizers.ByteLevel(add_prefix_space=False)
    tok.decoder = decoders.ByteLevel()
    tok.train_from_iterator(DOCS, trainers.BpeTrainer(
        vocab_size=300, min_frequency=2, show_progress=False,
        initial_alphabet=pre_tokenizers.ByteLevel.alphabet(),
        special_tokens=["[UNK]", "<|endoftext|>"]))
    tok.save(str(d / "tokenizer.json"))
    return tok


def _shared_teacher(tok, path):
    """A teacher with the student's own tokenization: 0.7 on the true
    next piece, 0.3 on " the"."""
    with open(path, "w") as f:
        for t in DOCS:
            enc = tok.encode(t)
            pieces = pieces_from_offsets(t, enc.offsets)
            topk = [[[pc, LN(0.7)], [" the", LN(0.3)]] for pc in pieces]
            f.write(json.dumps({"text": t, "pieces": pieces,
                                "topk": topk}) + "\n")


def test_build_script():
    try:
        import tokenizers  # noqa: F401
    except ImportError:
        print("SKIP test_build_script (needs tokenizers)")
        return
    with tempfile.TemporaryDirectory() as d:
        d = Path(d)
        tok = _bpe(d)
        _shared_teacher(tok, d / "topk.jsonl")
        r = subprocess.run(
            [sys.executable, str(_REPO / "examples" / "build_kd_targets.py"),
             "--teacher", str(d / "topk.jsonl"), "--tokenizer",
             str(d / "tokenizer.json"), "--out", str(d / "kd"), "--k", "2"],
            capture_output=True, text=True, check=True)
        st = json.loads(r.stdout)
        data = np.fromfile(d / "kd" / "train.bin", dtype=np.uint16)
        idx = np.load(d / "kd" / "train.kd_idx.npy")
        p = np.load(d / "kd" / "train.kd_p.npy").astype(np.float32)
    n = sum(len(tok.encode(t).ids) for t in DOCS)
    assert st["docs"] == len(DOCS) and st["tokens"] == n + len(DOCS)
    # every text token but the shard's first is predicted from a row
    assert st["aligned_share"] == round((n - 1) / n, 4), st
    assert st["rows_with_target"] == n - 1
    has = np.nonzero(p.sum(1) > 0)[0]
    the = tok.encode(" the").ids[0]
    for i in has:
        assert idx[i, 0] == data[i + 1], i          # best = true next token
        if data[i + 1] != the:
            assert abs(p[i, 0] - 0.7) < 1e-3 and idx[i, 1] == the
    print("PASS test_build_script")


# ------------------------------------------------------------------ torch
def test_sparse_kd_loss():
    import torch
    import torch.nn.functional as F
    from helioslm_v5.src.training.pretrain import sparse_kd_loss
    torch.manual_seed(0)
    logits = torch.randn(2, 3, 11)
    y = torch.randint(0, 11, (2, 3))
    idx = torch.zeros(2, 3, 4, dtype=torch.long)
    idx[..., 0] = y
    p = torch.zeros(2, 3, 4)
    p[..., 0] = 1.0
    kd, n = sparse_kd_loss(logits, idx, p)
    ce = F.cross_entropy(logits.reshape(-1, 11), y.reshape(-1))
    assert n == 6 and torch.allclose(kd, ce)
    kd, n = sparse_kd_loss(logits, idx, torch.zeros_like(p))
    assert n == 0 and kd.item() == 0.0
    p[0, 1] = 0                       # one position without a target
    kd, n = sparse_kd_loss(logits, idx, p)
    keep = torch.ones(6, dtype=torch.bool)
    keep[1] = False
    ce = F.cross_entropy(logits.reshape(-1, 11)[keep], y.reshape(-1)[keep])
    assert n == 5 and torch.allclose(kd, ce)
    print("PASS test_sparse_kd_loss")


def _kd_shard(d, n=4000, vocab=64, target=7, k=2):
    from helioslm_v5.src.training.pretrain import write_shards
    rng = np.random.default_rng(0)
    toks = rng.integers(2, vocab, n).tolist()
    write_shards([toks], d, "train")
    w = KDTargetWriter(k)
    # teacher: 0.9 on `target`, 0.1 on the true next token, every row
    w.add(toks, {m: ([target, toks[m]], [0.9, 0.1]) for m in range(1, n)})
    w.save(d, "train")
    write_shards([rng.integers(2, vocab, 600).tolist()], d, "val")
    return toks


def test_kd_shard_rows():
    from helioslm_v5.src.training.pretrain import KDShard, TokenShard
    with tempfile.TemporaryDirectory() as d:
        _kd_shard(d)
        sh = KDShard(Path(d) / "train.bin")
        x, y, m, idx, p = sh.batch(3, 16, 5, 0)
        x0, y0 = TokenShard(Path(d) / "train.bin").batch(3, 16, 5, 0)
        assert x.equal(x0) and y.equal(y0) and m.sum() == 48
        assert idx.shape == (3, 16, 2) and p.shape == (3, 16, 2)
        # the second candidate is the true next token y
        assert idx[..., 1].equal(y) and (idx[..., 0] == 7).all()
        np.save(Path(d) / "train.kd_p.npy", np.zeros((5, 2), np.float16))
        try:
            KDShard(Path(d) / "train.bin")
        except ValueError:
            pass
        else:
            raise AssertionError("row count mismatch accepted")
    print("PASS test_kd_shard_rows")


def _cfg():
    from helioslm_v5.src.training.pretrain import pilot_config
    return pilot_config("dense", vocab_size=64, hidden_size=32, num_layers=1,
                        intermediate_size=64)


def test_pretrain_kd():
    import torch
    from helioslm_v5.src.model_v5 import HeliosLMv5
    from helioslm_v5.src.training.pretrain import (KDShard, TokenShard,
                                                   pretrain)
    with tempfile.TemporaryDirectory() as d:
        _kd_shard(d)
        kd = KDShard(Path(d) / "train.bin")
        plain = TokenShard(Path(d) / "train.bin")
        val = TokenShard(Path(d) / "val.bin")
        kw = dict(steps=3, batch=4, seq=16, lr=3e-3, warmup=1,
                  eval_every=3, eval_windows=2, ckpt_every=100,
                  log=lambda *_: None)
        pretrain(_cfg(), plain, val, Path(d) / "a", **kw)
        pretrain(_cfg(), kd, val, Path(d) / "b", kd_alpha=0.0, **kw)
        la = [json.loads(x) for x in
              (Path(d) / "a" / "log.jsonl").read_text().splitlines()]
        lb = [json.loads(x) for x in
              (Path(d) / "b" / "log.jsonl").read_text().splitlines()]
        assert all("kd_loss" in r for r in lb)
        for ra, rb in zip(la, lb):
            assert abs(ra["train_loss"] - rb["train_loss"]) < 1e-4, (ra, rb)
        kw.update(steps=40, eval_every=40, lr=1e-2)
        pretrain(_cfg(), kd, val, Path(d) / "c", kd_alpha=1.0, **kw)
        lc = [json.loads(x) for x in
              (Path(d) / "c" / "log.jsonl").read_text().splitlines()]
        assert lc[-1]["kd_loss"] < lc[0]["kd_loss"] - 0.5, (lc[0], lc[-1])
        model = HeliosLMv5(_cfg())
        state = torch.load(Path(d) / "c" / "ckpt.pt", weights_only=False)
        model.load_state_dict(state["model"])
        model.eval()
        x, *_ = kd.batch(4, 16, 99, 1)
        with torch.no_grad():
            probs = torch.softmax(model(x)[0].float(), -1)[..., 7]
        assert probs.mean() > 0.3, probs.mean()   # teacher's favourite
        try:
            pretrain(_cfg(), kd, val, Path(d) / "e", kd_alpha=1.5, **kw)
        except ValueError:
            pass
        else:
            raise AssertionError("kd_alpha 1.5 accepted")
    print("PASS test_pretrain_kd")


if __name__ == "__main__":
    test_teacher_pieces()
    test_student_boundaries()
    test_projection()
    test_doc_targets()
    test_writer()
    test_teacher_side()
    test_build_script()
    test_sparse_kd_loss()
    test_kd_shard_rows()
    test_pretrain_kd()
