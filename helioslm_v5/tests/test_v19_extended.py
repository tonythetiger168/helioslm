"""T51 - v1.9: extended plugins (sqlite/github/search/slack/excel)."""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from harness.harness_core import Context
from harness.harness_plugins import (ExcelPlugin, GitHubPlugin,
                                     PresetPlugin, SearchPlugin,
                                     SQLitePlugin, SlackPlugin)


def test_sqlite():
    ctx = Context()
    ctx.use(SQLitePlugin())
    ctx.sqlite.execute("CREATE TABLE t (a, b)")
    ctx.sqlite.execute("INSERT INTO t VALUES (?, ?)", ["x", 1])
    r = ctx.sqlite.query("SELECT * FROM t")
    # query() returns plain sqlite3 tuples (see test_v19_common);
    # columns are available separately via r["columns"]
    assert r["rows"] == [("x", 1)]
    assert r["columns"] == ["a", "b"]
    print("PASS test_sqlite")


def test_github_no_token():
    ctx = Context()
    ctx.use(GitHubPlugin())
    r = ctx.github.get_repo("tonythetiger168", "helioslm")
    assert "error" in r or "full_name" in r
    print("PASS test_github_no_token (degrades gracefully)")


def test_search_offline():
    ctx = Context()
    ctx.use(SearchPlugin())
    r = ctx.search.web("test query that should fail offline")
    assert "results" in r or "error" in r
    print("PASS test_search_offline")


def test_slack_no_token():
    ctx = Context()
    ctx.use(SlackPlugin())
    r = ctx.slack.post("hello")
    assert "error" in r
    print("PASS test_slack_no_token")


def test_excel_csv_fallback():
    import tempfile, os
    with tempfile.TemporaryDirectory() as root:
        p = os.path.join(root, "test.csv")
        ctx = Context()
        ctx.use(ExcelPlugin(path=p))
        ctx.excel.write([["a", "b"], ["1", "2"]])
        r = ctx.excel.read()
        assert r["rows"] == [["a", "b"], ["1", "2"]]
    print("PASS test_excel_csv_fallback")


if __name__ == "__main__":
    test_sqlite()
    test_github_no_token()
    test_search_offline()
    test_slack_no_token()
    test_excel_csv_fallback()
    print("\n5/5 extended plugin tests passed")
