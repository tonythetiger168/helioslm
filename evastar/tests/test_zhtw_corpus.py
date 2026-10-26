"""T148 — roadmap TW2 zh-TW pretraining corpus (v5.154).

Pins:

1. dedup keys: NFKC + whitespace-free normalisation makes spacing and
   full-width variants exact duplicates; MinHash signatures are seeded
   (identical across instances) and estimate Jaccard similarity;
2. Deduper: exact copies -> exact_dup, a one-sentence edit of a long
   document -> near_dup, unrelated documents kept, the first copy wins;
3. process: a data.gov.tw record without a provider is dropped, with one
   its attribution names the agency; the converter runs before the
   variant check (so converted Simplified text is kept) and a convert
   source without a converter is refused; FineWeb-2 keeps only already
   Traditional text; heuristics drop short pages; stats add up;
4. examples/tw2_prepare_corpus.py end to end on local files: budgets
   stop a source, a document seen in datagov is not kept again from
   fineweb2, shard sha256 and corpus_sha256 are reproducible, every
   record keeps licence and attribution;
5. with OpenCC installed (skipped otherwise): s2twp gives Taiwan
   phrasing and the zhwiki source runs through the CLI.

Run from repo root: python3 evastar/tests/test_zhtw_corpus.py
"""
import json
import subprocess
import sys
import tempfile
from pathlib import Path

import numpy as np

_REPO = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(_REPO))

from evastar.src.training.zhtw_corpus import (  # noqa: E402
    SOURCES, Deduper, MinHash, exact_key, new_stats, process, shingles,
    trad_share)

SCRIPT = _REPO / "examples" / "tw2_prepare_corpus.py"
# Traditional-only characters (Big5, not GB2312) and shared ones.
_TRAD = "這國們說個對時會來學與發後經樣點過質實現體關還麼開員條頭際傳"
_SHARED = "山水日月人大小上中下天地心手口目田石木火土"
_SIMP = "这国们说个对时会来学与发后经样点过质实现体关还么开员条头际传"


def doc(seed, n=40, chars=_TRAD):
    """A distinct multi-line document of ``n`` 12-character lines."""
    rng = np.random.default_rng(seed)
    pool = chars + _SHARED
    return "\n".join("".join(rng.choice(list(pool), 11)) + "。"
                     for _ in range(n))


def to_trad(text):
    return text.translate(str.maketrans(_SIMP, _TRAD))


def test_keys_and_minhash():
    a = doc(1)
    assert exact_key(a) == exact_key(a.replace("\n", "  \n "))
    assert exact_key("ＡＢＣ１２３") == exact_key("ABC123")    # NFKC
    assert exact_key(a) != exact_key(doc(2))
    s1 = shingles(a)
    m1, m2 = MinHash(128, 0), MinHash(128, 0)
    assert np.array_equal(m1.signature(s1), m2.signature(s1))
    assert m1.signature(s1).dtype == np.uint32
    assert not np.array_equal(m1.signature(s1), MinHash(128, 1).signature(s1))
    # half the shingles shared: J = |A & B| / |A | B|
    ids = list(s1)
    x, y = set(ids[:600]), set(ids[300:900])
    true = len(x & y) / len(x | y)
    est = float(np.mean(m1.signature(x) == m1.signature(y)))
    assert abs(est - true) < 0.12, (est, true)
    assert len(m1.signature(set())) == 128
    print("PASS test_keys_and_minhash")


def test_deduper():
    d = Deduper()
    a, b = doc(1), doc(2)
    edit = a.replace(a.splitlines()[5], "這是改過的一行內容。")
    assert d.check(a) is None
    assert d.check(a + "\n") == "exact_dup"
    assert d.check(edit) == "near_dup"
    assert d.check(b) is None
    assert d.check(doc(3)) is None
    assert len(d.sigs) == 3
    try:
        Deduper(num_perm=100, bands=16)
        raise AssertionError("num_perm % bands accepted")
    except ValueError:
        pass
    print("PASS test_deduper")


def test_process():
    gov, fw = SOURCES["datagov"], SOURCES["fineweb2"]
    wiki = SOURCES["zhwiki"]
    d, st = Deduper(), new_stats()
    assert process({"text": doc(1)}, "datagov", gov, d, st) is None
    assert st["dropped"] == {"no_provider": 1}
    r = process({"text": doc(1), "provider": "數位發展部", "url": "u"},
                "datagov", gov, d, st)
    assert r["attribution"].startswith("提供機關：數位發展部")
    assert r["license"] == "OGDL-Taiwan-1.0" and r["url"] == "u"
    assert r["id"] == "datagov-00000000"

    st = new_stats()
    simp = doc(4, chars=_SIMP)
    assert process({"text": simp}, "fineweb2", fw, d, st) is None
    assert st["dropped"] == {"zh_variant": 1}
    assert process({"text": "太短"}, "fineweb2", fw, d, st) is None
    assert "heuristic:too_short" in st["dropped"]
    assert process({"text": doc(1)}, "fineweb2", fw, d, st) is None
    assert st["dropped"]["exact_dup"] == 1          # seen in datagov
    assert process({"text": doc(5)}, "fineweb2", fw, d, st)["source"] \
        == "fineweb2"
    assert st["in"] == 4 and st["kept"] == 1
    assert st["in"] == st["kept"] + sum(st["dropped"].values())
    assert trad_share(st) > 0.95 and trad_share(new_stats()) is None

    st = new_stats()
    try:
        process({"text": simp}, "zhwiki", wiki, d, st)
        raise AssertionError("convert source ran without a converter")
    except ValueError:
        pass
    st = new_stats()
    r = process({"text": simp, "title": "t"}, "zhwiki", wiki, d, st,
                converter=to_trad, count_tokens=len)
    assert r is not None and r["title"] == "t"
    assert trad_share(st) == 1.0 and st["tokens"] == len(r["text"])
    print("PASS test_process")


def _write(path, rows):
    path.write_text("".join(json.dumps(r, ensure_ascii=False) + "\n"
                            for r in rows), encoding="utf-8")


def _run(*args):
    p = subprocess.run([sys.executable, str(SCRIPT), *map(str, args)],
                       capture_output=True, text=True)
    assert p.returncode == 0, p.stderr
    return p


def test_cli():
    with tempfile.TemporaryDirectory() as t:
        t = Path(t)
        _write(t / "gov.jsonl", [{"text": doc(i), "provider": "內政部"}
                                 for i in range(3)]
               + [{"text": doc(99)}])
        fw = [{"text": doc(0)}] + [{"text": doc(i)} for i in range(10, 40)] \
            + [{"text": doc(50, chars=_SIMP)}]
        _write(t / "fw.jsonl", fw)
        common = ["--sources", "datagov", "fineweb2", "--datagov",
                  t / "gov.jsonl", "--local", f"fineweb2={t / 'fw.jsonl'}",
                  "--tokens", 8000, "--shares", "datagov=0.5",
                  "fineweb2=0.5", "--shard-docs", 4]
        _run("--out", t / "a", *common)
        _run("--out", t / "b", *common)
        ma = json.loads((t / "a" / "manifest.json").read_text(encoding="utf-8"))
        mb = json.loads((t / "b" / "manifest.json").read_text(encoding="utf-8"))
        assert ma["corpus_sha256"] == mb["corpus_sha256"]
        assert ma["budget_unit"] == "units"
        g, f = ma["sources"]["datagov"], ma["sources"]["fineweb2"]
        assert g["stats"]["kept"] == 3
        assert g["stats"]["dropped"] == {"no_provider": 1}
        assert not g["budget_reached"]
        assert f["budget"] == 4000 and f["budget_reached"]
        assert f["stats"]["dropped"].get("exact_dup") == 1   # doc(0)
        assert f["stats"]["kept"] < 30                       # budget stop
        assert len(f["shards"]) == -(-f["stats"]["kept"] // 4)
        assert ma["total"]["docs"] == 3 + f["stats"]["kept"]
        texts = set()
        for src in (g, f):
            for sh in src["shards"]:
                p = t / "a" / sh["path"]
                rows = [json.loads(x) for x in p.read_text(encoding="utf-8").splitlines()]
                assert len(rows) == sh["docs"]
                assert all(r["license"] and r["attribution"] for r in rows)
                texts |= {r["text"] for r in rows}
        assert len(texts) == ma["total"]["docs"]
        assert "OGDL-Taiwan-1.0" in (t / "a" / "ATTRIBUTION.md").read_text(encoding="utf-8")
    print("PASS test_cli")


def test_opencc():
    try:
        import opencc  # noqa: F401
    except ImportError:
        print("SKIP test_opencc (pip install opencc)")
        return
    from evastar.src.training.zhtw_corpus import opencc_converter
    conv = opencc_converter()
    assert conv("这个软件的内存和网络") == "這個軟體的記憶體和網路"
    with tempfile.TemporaryDirectory() as t:
        t = Path(t)
        _write(t / "wiki.jsonl", [{"text": doc(i, chars=_SIMP),
                                   "title": f"t{i}"} for i in range(3)])
        _run("--out", t / "o", "--sources", "zhwiki", "--local",
             f"zhwiki={t / 'wiki.jsonl'}", "--tokens", 1e6)
        m = json.loads((t / "o" / "manifest.json").read_text(encoding="utf-8"))
        w = m["sources"]["zhwiki"]
        assert w["stats"]["kept"] == 3 and w["trad_share"] > 0.95
        assert w["convert"] == "s2twp"
    print("PASS test_opencc")


if __name__ == "__main__":
    test_keys_and_minhash()
    test_deduper()
    test_process()
    test_cli()
    test_opencc()
