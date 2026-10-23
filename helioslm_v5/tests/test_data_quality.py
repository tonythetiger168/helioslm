"""T123 — LB3 pretraining data quality and mixture manifests (v5.116).

Pins:

1. zh_profile / zh_variant: Traditional-only, Simplified-only and shared
   characters are told apart by Big5 / GB2312 encodability; documents
   label tw / cn / mixed / none;
2. heuristic_check: a clean document passes; short, duplicated-line,
   ellipsis and symbol-heavy, and non-text documents fail with the named
   reason; Chinese length is counted in characters;
3. HashedNgramClassifier: features are deterministic (crc32), training
   separates a toy curated-vs-spam set, save / load round-trips the
   scores exactly, and fitting without both classes is refused;
4. filter_docs: heuristics, variants and the classifier drop documents
   in order and the report counts each reason;
5. mixture manifests: the committed lb3_0p5b_v1.json validates, its
   token budget adds up per phase, nothing in it is usable until a
   licence is verified; every validation rule refuses its bad case;
6. examples/data_quality.py train / score / profile / budget end to end,
   and pilot_prepare_data.py --heuristics --zh-variants writes the drop
   counts to meta.json (needs ``tokenizers``).

Run from repo root: python3 helioslm_v5/tests/test_data_quality.py
"""
import copy
import json
import subprocess
import sys
import tempfile
from pathlib import Path

_REPO = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(_REPO))

from helioslm_v5.src.training.data_mixture import (  # noqa: E402
    load_mixture, mixture_sha256, token_budget, usable_sources,
    validate_mixture)
from helioslm_v5.src.training.data_quality import (  # noqa: E402
    HashedNgramClassifier, features, filter_docs, heuristic_check,
    zh_profile, zh_variant)

TW = ("這是一篇關於臺灣歷史與地理的文章，說明國家公園的生態保育、"
      "鐵路發展與傳統飲食文化，並且討論學校教育如何讓學生認識鄉土。") * 3
CN = ("这是一篇关于历史与地理的文章，说明国家公园的生态保护、"
      "铁路发展与传统饮食文化，并且讨论学校教育如何让学生认识乡土。") * 3
EN = ("The river delta supports a wide range of birds, fish and plants. "
      "Researchers measured salinity at twelve stations over three years "
      "and found that seasonal floods restore the wetland soil. ") * 6

MIX = _REPO / "data" / "mixtures" / "lb3_0p5b_v1.json"


def test_zh_variant():
    p = zh_profile("龜這國龟这国山的")
    assert p == {"cjk": 8, "trad_only": 3, "simp_only": 3}, p
    assert zh_variant(TW) == "tw"
    assert zh_variant(CN) == "cn"
    assert zh_variant(TW + CN) == "mixed"
    assert zh_variant("山水日月的人大小上下中" * 3) == "mixed"  # shared only
    assert zh_variant(EN) == "none"
    print("PASS test_zh_variant")


def test_heuristics():
    assert heuristic_check(EN) == (True, [])
    assert heuristic_check(TW)[0]          # ~100 CJK chars count as units
    assert "too_short" in heuristic_check("short text")[1]
    dup = "\n".join(["buy now cheap"] * 40 + [EN])
    assert "duplicate_lines" in heuristic_check(dup)[1]
    ell = "\n".join(f"read more about item {i} ..." for i in range(40))
    r = heuristic_check(ell)[1]
    assert "ellipsis_lines" in r and "symbols" in r, r
    junk = EN + " " + "@@ %% && ** ^^ ~~ " * 200
    assert "non_text" in heuristic_check(junk)[1]
    assert heuristic_check("short", {"min_units": 1})[0]
    print("PASS test_heuristics")


def _toy_sets():
    good = [EN.replace("twelve", w) for w in
            ("ten", "eleven", "twelve", "nine", "eight", "seven")] + [TW, TW[5:]]
    bad = [("click here free prize winner casino bonus " * 20) + str(i)
           for i in range(6)] + [("免費 下載 點擊 中獎 " * 30) + str(i)
                                 for i in range(2)]
    return good, bad


def test_classifier():
    assert features(EN, 1 << 18) == features(EN, 1 << 18)
    assert all(0 <= f < 64 for f in features(TW, 64))
    good, bad = _toy_sets()
    clf = HashedNgramClassifier(1 << 16).fit(good, bad, epochs=10, seed=0)
    assert all(clf.score(d) > 0.5 for d in good)
    assert all(clf.score(d) < 0.5 for d in bad)
    with tempfile.TemporaryDirectory() as d:
        clf.save(Path(d) / "q")
        back = HashedNgramClassifier.load(Path(d) / "q")
        assert all(back.score(x) == clf.score(x) for x in good + bad)
    for pos, neg in (([], bad), (good, [])):
        try:
            HashedNgramClassifier(64).fit(pos, neg)
        except ValueError:
            pass
        else:
            raise AssertionError("one-class fit accepted")
    print("PASS test_classifier")


def test_filter_docs():
    good, bad = _toy_sets()
    clf = HashedNgramClassifier(1 << 16).fit(good, bad, epochs=10, seed=0)
    docs = [EN, TW, CN, "tiny", bad[0]]
    kept, rep = filter_docs(docs, variants={"tw", "none"}, classifier=clf)
    assert kept == [EN, TW], rep
    assert rep["in"] == 5 and rep["kept"] == 2
    assert rep["dropped"] == {"zh_variant": 1, "heuristic:too_short": 1,
                              "quality": 1}, rep
    kept, rep = filter_docs(docs, heuristics=False)
    assert kept == docs and rep["dropped"] == {}
    print("PASS test_filter_docs")


def test_mixture():
    mix = load_mixture(MIX)
    b = token_budget(mix, 100_000_000_000, 0.2)
    main = sum(v for s in mix["sources"] for k, v in b.items()
               if s["name"] == k and s.get("phase", "main") == "main")
    assert abs(main - 80_000_000_000) <= len(b)
    assert abs(sum(b.values()) - 100_000_000_000) <= len(b)
    assert usable_sources(mix) == []        # nothing verified yet
    assert mixture_sha256(mix) == mixture_sha256(copy.deepcopy(mix))
    ok = {"name": "m", "sources": [
        {"name": "a", "source": "path:x", "license": "MIT", "weight": 1.0,
         "license_verified": True}]}
    assert [s["name"] for s in usable_sources(validate_mixture(ok))] == ["a"]
    assert token_budget(ok, 1000) == {"a": 1000}   # no decay phase
    bad_cases = [
        {"name": "", "sources": ok["sources"]},
        {"name": "m", "sources": []},
        {"name": "m", "sources": ok["sources"] * 2},          # dup names
    ]
    for patch in ({"weight": 0.5}, {"weight": 0}, {"license": "GPL-3.0"},
                  {"license": "review"}, {"phase": "warm"},
                  {"zh_variants": ["hk"]}, {"quality_threshold": 2},
                  {"source": ""},
                  {"license": "review", "license_note": "x",
                   "license_verified": True}):
        bad_cases.append({"name": "m",
                          "sources": [dict(ok["sources"][0], **patch)]})
    bad_cases.append({"name": "m", "sources": [
        dict(ok["sources"][0], phase="decay")]})          # no main phase
    for bad in bad_cases:
        try:
            validate_mixture(bad)
        except ValueError:
            pass
        else:
            raise AssertionError(f"accepted {bad}")
    print("PASS test_mixture")


def _cli(*args):
    r = subprocess.run([sys.executable, str(_REPO / "examples" /
                                            "data_quality.py"), *args],
                       capture_output=True, text=True, check=True)
    return json.loads(r.stdout)


def test_cli():
    good, bad = _toy_sets()
    with tempfile.TemporaryDirectory() as d:
        d = Path(d)
        for name, docs in (("pos", good), ("neg", bad)):
            (d / f"{name}.jsonl").write_text("".join(
                json.dumps({"text": t}, ensure_ascii=False) + "\n"
                for t in docs), encoding="utf-8")
        out = _cli("train", "--pos", str(d / "pos.jsonl"), "--neg",
                   str(d / "neg.jsonl"), "--out", str(d / "q"),
                   "--dim", "65536", "--epochs", "10")
        assert out["train_accuracy"] == 1.0 and out["pos"] == len(good)
        out = _cli("score", "--model", str(d / "q"), "--text",
                   str(d / "pos.jsonl"), str(d / "neg.jsonl"),
                   "--thresholds", "0.5")
        assert out["docs"] == len(good) + len(bad)
        assert out["kept_share"]["0.5"] == round(len(good) / out["docs"], 4)
        out = _cli("profile", "--text", str(d / "pos.jsonl"))
        assert out["zh_variant"] == {"none": 6, "tw": 2}, out
        out = _cli("budget", "--mixture", str(MIX), "--tokens", "1e9")
        assert out["name"] == "lb3-0.5b-v1" and out["usable"] == []
    print("PASS test_cli")


def test_prepare_data_filters():
    try:
        import tokenizers  # noqa: F401
    except ImportError:
        print("SKIP test_prepare_data_filters (needs tokenizers)")
        return
    with tempfile.TemporaryDirectory() as d:
        d = Path(d)
        docs = [EN, TW, CN, "tiny"] * 30
        (d / "c.jsonl").write_text("".join(
            json.dumps({"text": t}, ensure_ascii=False) + "\n"
            for t in docs), encoding="utf-8")
        subprocess.run([sys.executable, str(_REPO / "examples" /
                                            "pilot_prepare_data.py"),
                        "--text", str(d / "c.jsonl"), "--vocab", "600",
                        "--zh-seed", "none", "--heuristics",
                        "--zh-variants", "tw", "none", "--out",
                        str(d / "out")], check=True, capture_output=True)
        meta = json.loads((d / "out" / "meta.json").read_text())
        q = meta["quality_filter"]
        assert q["heuristics"] and q["zh_variants"] == ["tw", "none"]
        n = meta["splits"]["train"]["docs"]
        assert q["kept"] == n and q["in"] == n + sum(q["dropped"].values())
        assert set(q["dropped"]) == {"zh_variant", "heuristic:too_short"}
    print("PASS test_prepare_data_filters")


if __name__ == "__main__":
    test_zh_variant()
    test_heuristics()
    test_classifier()
    test_filter_docs()
    test_mixture()
    test_cli()
    test_prepare_data_filters()
