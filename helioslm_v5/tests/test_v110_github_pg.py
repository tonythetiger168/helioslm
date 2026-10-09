"""T52 - v1.10: GitHub + PostgreSQL plugins."""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from harness.harness_core import Context
from harness.harness_plugins import GitHubPlugin, PostgresPlugin, PresetPlugin


def test_github_requires_token():
    import os
    old = os.environ.pop("GITHUB_TOKEN", None)
    ctx = Context()
    ctx.use(GitHubPlugin())
    r = ctx.github.get_repo("octocat", "hello-world")
    assert "error" in r and "GITHUB_TOKEN" in r["error"]
    if old:
        os.environ["GITHUB_TOKEN"] = old
    print("PASS test_github_requires_token")


def test_postgres_requires_driver():
    ctx = Context()
    ctx.use(PostgresPlugin(dsn="postgresql://fake"))
    r = ctx.postgres.query("SELECT 1")
    assert "error" in r and ("psycopg2" in r["error"] or "pg8000" in r["error"])
    print("PASS test_postgres_requires_driver")


def test_github_registered():
    ctx = Context()
    ctx.use(PresetPlugin("full", model_fn=None))
    ctx.use(GitHubPlugin())
    for name in ("github.get_repo", "github.list_issues", "github.create_issue"):
        assert name in ctx._services
    print("PASS test_github_registered")


if __name__ == "__main__":
    test_github_requires_token()
    test_postgres_requires_driver()
    test_github_registered()
    print("\n3/3 GitHub/Postgres plugin tests passed")
