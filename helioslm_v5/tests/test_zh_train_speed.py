"""T118 — zh-TW training speed: Big5-seeded byte-level BPE, v5.80.

Pins:

1. big5_common_chars: exactly the 5401 Big5 level-1 characters, all CJK,
   Traditional (no Simplified-only forms), round-tripping through the
   big5 codec, and covering every CJK character of the TW4 held-out text;
2. gb2312_common_chars: the 3755 GB2312 level-1 Simplified characters,
   2434 of them shared with Big5; the "both" seed is their 6722-character
   union;
3. seeding: a byte-level BPE trained on English-only text plus the
   "both" seed makes every one of the 6722 characters a single token
   (Traditional and Simplified text alike), and the unseeded one makes
   none of them; examples/train_toy_bpe.py --zh-seed does the same and
   refuses a vocabulary that cannot hold the seed (needs ``tokenizers``);
4. bits_per_char divides the summed token NLL by characters, so a model
   that is certain of every token scores 0 and a uniform one scores
   tokens * log2(V) / chars;
5. the committed artifact: seeding takes a tokenizer that never saw
   Chinese from 3 tokens per CJK character to exactly 1 (100% single-token
   coverage), puts > 1.7x as many characters into each training token,
   and reaches a lower held-out bits per character than the unseeded
   arm's best; with Chinese already in the corpus, seeding still lowers
   held-out tokens per CJK character and completes coverage; hash.

Run from repo root: python3 helioslm_v5/tests/test_zh_train_speed.py
"""
import hashlib
import json
import math
import sys
from pathlib import Path

import torch

_REPO = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(_REPO))

from helioslm_v5.eval.zh_tokenizer_coverage import (  # noqa: E402
    _CJK, big5_common_chars, eval_split, gb2312_common_chars, zh_seed_chars,
    zh_seed_text)
from helioslm_v5.eval.zh_train_speed import bits_per_char  # noqa: E402

_ART = _REPO / "benchmarks" / "zh_train_speed_2026-10-20.json"


def test_big5_common_chars():
    chars = big5_common_chars()
    assert len(chars) == len(set(chars)) == 5401
    assert chars[0] == "一" and chars[-1] == "籲"
    assert all(_CJK.fullmatch(c) for c in chars)
    assert all(c.encode("big5").decode("big5") == c for c in chars)
    s = set(chars)
    assert not s & set("这们说为时会与体档训练测试发现")   # Simplified-only
    assert set("這們說為時會與體檔訓練測試發現") <= s
    zh, _ = eval_split(_REPO)
    held = set("".join(_CJK.findall("\n".join(zh))))
    assert held <= s, held - s
    print("PASS test_big5_common_chars")


def test_gb2312_and_both_scripts():
    cn = gb2312_common_chars()
    assert len(cn) == len(set(cn)) == 3755
    assert cn[0] == "啊" and cn[-1] == "座"
    assert all(_CJK.fullmatch(c) for c in cn)
    assert set("这们说为时会与体档训练测试发现") <= set(cn)   # Simplified
    tw = set(big5_common_chars())
    assert len(set(cn) & tw) == 2434                     # shared forms
    both = zh_seed_chars("both")
    assert len(both) == len(set(both)) == 6722
    assert set(both) == tw | set(cn)
    assert zh_seed_chars("tw") == big5_common_chars()
    assert zh_seed_chars("cn") == cn
    assert zh_seed_text("both", repeats=3).count("\n") == 3 * 6722
    try:
        zh_seed_chars("hk")
    except ValueError:
        pass
    else:
        raise AssertionError("unknown seed scripts accepted")
    print("PASS test_gb2312_and_both_scripts")


def test_seeding_makes_single_tokens():
    try:
        import tokenizers  # noqa: F401
    except ImportError:
        print("SKIP test_seeding_makes_single_tokens (no tokenizers)")
        return
    from helioslm_v5.eval.zh_tokenizer_coverage import Bpe, train_bpe
    english = "\n".join(f"the quick brown fox {i} jumps over the lazy dog"
                        for i in range(200))
    chars = zh_seed_chars("both")
    plain = Bpe(train_bpe(english, 32000))
    seeded = Bpe(train_bpe(english + "\n" + zh_seed_text("both"), 32000))
    assert sum(plain.covers(c) for c in chars) == 0
    assert all(seeded.covers(c) for c in chars)
    assert len(seeded.tok.encode("繁體中文訓練").ids) == 6
    assert len(seeded.tok.encode("简体中文训练").ids) == 6
    # the training tool's flag: same seed, refused without room for it
    sys.path.insert(0, str(_REPO / "examples"))
    import train_toy_bpe
    tok = Bpe(train_toy_bpe.train_tokenizer(english, 32000, zh_seed="both"))
    assert all(tok.covers(c) for c in chars)
    for bad in ((1024, "both"), (6000, "tw"), (32000, "hk")):
        try:
            train_toy_bpe.train_tokenizer(english, *bad)
        except ValueError:
            pass
        else:
            raise AssertionError(f"--zh-seed accepted {bad}")
    print("PASS test_seeding_makes_single_tokens")


def test_bits_per_char():
    V, chars = 7, 10

    class Uniform(torch.nn.Module):
        def forward(self, ids):
            return (torch.zeros(*ids.shape, V),)

    class Oracle(torch.nn.Module):
        def forward(self, ids):            # next token = current + 1
            return (torch.nn.functional.one_hot((ids + 1) % V, V) * 1e4,)

    ids = [i % V for i in range(9)]       # 8 predicted tokens
    u = Uniform()
    u.train()
    got = bits_per_char(u, ids, chars, seq=4)
    assert abs(got - 8 * math.log2(V) / chars) < 1e-5, got
    assert u.training                       # mode restored
    assert bits_per_char(Oracle(), ids, chars, seq=4) < 1e-6
    print("PASS test_bits_per_char")


def test_speed_artifact():
    art = json.loads(_ART.read_text())
    r = {x["arm"]: x for x in art["results"]}
    plain, seeded = r["code_only"], r["code_only_big5"]
    assert plain["coverage"]["tokens_per_cjk_char"] > 2.99
    assert plain["coverage"]["cjk_single_token_coverage"] == 0.0
    assert seeded["coverage"]["tokens_per_cjk_char"] == 1.0
    assert seeded["coverage"]["cjk_single_token_coverage"] == 1.0
    # English is not made more expensive
    assert seeded["coverage"]["tokens_per_char_en_lines"] \
        <= plain["coverage"]["tokens_per_char_en_lines"]
    assert seeded["chars_per_token"] > 1.7 * plain["chars_per_token"]
    # adding the Simplified seed costs Traditional nothing
    both = r["code_only_both"]
    assert both["coverage"]["tokens_per_cjk_char"] == 1.0
    assert both["coverage"]["cjk_single_token_coverage"] == 1.0
    assert both["chars_per_token"] >= seeded["chars_per_token"] - 0.01

    def best(x):
        return min(c["bits_per_char"] for c in x["bits_per_char_curve"])
    assert best(seeded) < best(plain) - 0.5, (best(seeded), best(plain))
    repo, repo_s = r["repo"], r["repo_big5"]
    assert repo_s["coverage"]["tokens_per_cjk_char"] \
        < repo["coverage"]["tokens_per_cjk_char"]
    assert repo["coverage"]["cjk_single_token_coverage"] < 0.8
    assert repo_s["coverage"]["cjk_single_token_coverage"] == 1.0
    assert art["results_sha256"] == hashlib.sha256(
        json.dumps(art["results"], sort_keys=True).encode()).hexdigest()
    print("PASS test_speed_artifact")


if __name__ == "__main__":
    test_big5_common_chars()
    test_gb2312_and_both_scripts()
    test_seeding_makes_single_tokens()
    test_bits_per_char()
    test_speed_artifact()
