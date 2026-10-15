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


class _FakeResp:
    def __init__(self, payload):
        self.payload = payload
    def read(self):
        return self.payload


def test_github_api_json_roundtrip():
    """_api must json-encode bodies and json-decode responses.

    Regression: the merged GitHubPlugin used json.dumps/json.loads without
    importing json — caught by the CI syntax gate (F821), invisible to the
    no-token tests above. Monkeypatched urlopen keeps it offline.
    """
    import urllib.request
    ctx = Context()
    ctx.use(GitHubPlugin(token="fake-token"))
    # create_issue fails closed without a TrustGate; wire an always-DIRECT
    # gate so this test keeps exercising the POST body path.
    from gate import FixedGate, Route
    ctx.register("decision.trust", FixedGate(Route.DIRECT))
    seen = {}
    real = urllib.request.urlopen

    def fake_urlopen(req, timeout=0):
        seen["data"] = req.data
        return _FakeResp(b'{"full_name": "octocat/hello-world"}')

    urllib.request.urlopen = fake_urlopen
    try:
        r = ctx.github.get_repo("octocat", "hello-world")
        assert r == {"full_name": "octocat/hello-world"}, r
        assert seen["data"] is None                      # GET has no body
        r = ctx.github.create_issue("o", "r", "t", "b")  # POST body path
        assert r == {"full_name": "octocat/hello-world"}, r
        import json as _json
        assert _json.loads(seen["data"].decode()) == {"title": "t", "body": "b"}
    finally:
        urllib.request.urlopen = real
    print("PASS test_github_api_json_roundtrip")


if __name__ == "__main__":
    test_github_requires_token()
    test_postgres_requires_driver()
    test_github_registered()
    test_github_api_json_roundtrip()
    print("\n4/4 GitHub/Postgres plugin tests passed")
