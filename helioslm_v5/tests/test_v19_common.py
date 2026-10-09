"""T51 - v1.9: common plugins batch 1 (local no-dep)."""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from harness.harness_core import Context
from harness.harness_plugins import (PDFPlugin, PresetPlugin,
                                     SQLitePlugin, SearchPlugin,
                                     TemplatePlugin, WebSearchPlugin)


def test_search_local():
    ctx = Context()
    ctx.use(PresetPlugin("full", model_fn=None))
    ctx.use(SearchPlugin())
    corpus = ["the cat sat on the mat", "dogs are great", "cats and dogs"]
    r = ctx.search.query("cat", corpus)
    assert r["n"] >= 1 and "cat" in r["results"][0].lower()
    print("PASS test_search_local")


def test_web_search_stub():
    ctx = Context()
    ctx.use(WebSearchPlugin(provider="tavily", api_key_env="TAVILY_KEY"))
    r = ctx.web.search("test query")
    assert "error" in r or "stub" in r
    print("PASS test_web_search_stub")


def test_sqlite_roundtrip():
    import tempfile, os
    with tempfile.TemporaryDirectory() as d:
        db = os.path.join(d, "t.db")
        ctx = Context()
        ctx.use(SQLitePlugin(db_path=db))
        ctx.sqlite.query("CREATE TABLE t (id INTEGER, name TEXT)")
        ctx.sqlite.query("INSERT INTO t VALUES (1, 'hello')")
        r = ctx.sqlite.query("SELECT * FROM t")
        assert r["rows"] == [(1, "hello")]
    print("PASS test_sqlite_roundtrip")


def test_template_scaffold():
    ctx = Context()
    ctx.use(TemplatePlugin())
    r = ctx.plugin.scaffold("MyTool", "does a thing")
    assert "class mytoolplugin" in r["code"].lower()
    assert "ctx.register" in r["code"]
    print("PASS test_template_scaffold")


if __name__ == "__main__":
    test_search_local()
    test_web_search_stub()
    test_sqlite_roundtrip()
    test_template_scaffold()
    print("\n4/4 common plugins (local) tests passed")
