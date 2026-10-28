"""hsh: agent skills (hsh/skills.py).

Pins:
1. frontmatter: plain, quoted and block scalars, booleans, and the repo's
   own skills/ files parse; an unclosed block is an error;
2. discovery: bundles and flat files, rank order across roots, a scoped
   layer beats the global one, malformed skills are skipped and recorded,
   bodies are re-read on every load;
3. the catalog is a logged user/message (names and descriptions only),
   is added once, is replaced when a skill appears, and every request
   still rebuilds from the log;
4. the skill tool returns the instructions, refuses unknown and
   model-disabled skills; /name expands a user-invocable skill;
5. runtime skills registered on ctx.skills reach the agent.
"""
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from helioslm_v5.hsh import (LLM, Agents, Context, ScriptedAdapter,  # noqa: E402
                         Sessions, SkillError, Skills, Tools, Workspace,
                         create_agent, parse_frontmatter, rebuild_requests,
                         request_hash)
from helioslm_v5.hsh.skills import FilesystemSkills, default_roots  # noqa: E402

CFG = {"provider": "fake", "model": "m"}
REPO = Path(__file__).resolve().parents[2]


def world(tmp_path, adapter=None, shared=False):
    ctx = Context()
    for p in (Sessions, Tools, LLM, Agents):
        ctx.plugin(p)
    if shared:
        ctx.plugin(Skills)
    ctx.plugin(Workspace, {"root": str(tmp_path)})
    ctx.llm.register("fake", adapter or ScriptedAdapter(fn=lambda r, i: "ok"))
    return ctx


def skill(root, name, desc, body="Do the thing.", flat=False, extra=""):
    text = f"---\nname: {name}\ndescription: {desc}\n{extra}---\n{body}\n"
    if flat:
        root.mkdir(parents=True, exist_ok=True)
        (root / f"{name}.md").write_text(text, encoding="utf-8")
    else:
        (root / name).mkdir(parents=True, exist_ok=True)
        (root / name / "SKILL.md").write_text(text, encoding="utf-8")


def agent(ctx, sid="a", **cfg):
    rows = [[{"replace": "skills", "config": dict({"home": False}, **cfg)}]]
    return create_agent(ctx, sid, CFG, "standard", patches=rows)


def test_frontmatter():
    f, body = parse_frontmatter(
        "---\nname: x\ndescription: >-\n  Use when a\n  thing.\n\n  Second.\n"
        "guard: '{\"a\": [1]}'\nflag: false\nq: \"t\\u00e9\"\nlit: |\n  a\n   b\n---\n\nBody\n")
    assert f == {"name": "x", "description": "Use when a thing.\nSecond.",
                 "guard": '{"a": [1]}', "flag": False, "q": "té", "lit": "a\n b\n"}
    assert body == "Body"
    assert parse_frontmatter("no frontmatter") == ({}, "no frontmatter")
    with pytest.raises(SkillError):
        parse_frontmatter("---\nname: x\n")


def test_repo_skills_parse():
    prov = FilesystemSkills([("custom", 300, REPO / "skills")])
    found = {s["name"]: s for s in prov.list()}
    assert prov.errors == {}
    assert "agent-entry" in found and found["agent-entry"]["description"].startswith("Use when")
    assert found["agent-entry"]["metadata"]["type"] == "entry"


def test_discovery_rank_and_errors(tmp_path):
    ws = tmp_path / "ws"
    skill(ws / ".hsh" / "skills", "deploy", "project version", body="P")
    skill(ws / ".agents" / "skills", "deploy", "agents version")
    skill(ws / ".agents" / "skills", "lint", "flat lint", flat=True)
    skill(tmp_path / "home" / ".hsh" / "skills", "notes", "user notes")
    (ws / ".hsh" / "skills" / "Bad_Name").mkdir()
    (ws / ".hsh" / "skills" / "broken").mkdir()
    (ws / ".hsh" / "skills" / "broken" / "SKILL.md").write_text("---\nname: broken\n---\n")
    prov = FilesystemSkills(default_roots(ws, home=tmp_path / "home"))
    reg_ctx = Context()
    reg_ctx.plugin(Skills)
    reg_ctx.skills.register_provider(prov)
    s = {x["name"]: x for x in reg_ctx.skills.list()}
    assert sorted(s) == ["deploy", "lint", "notes"]
    assert s["deploy"]["source"] == "project-hsh" and s["deploy"]["description"] == "project version"
    assert s["notes"]["source"] == "user-hsh" and "content" not in s["deploy"]
    assert list(prov.errors) == [str(ws / ".hsh" / "skills" / "broken" / "SKILL.md")]
    (ws / ".hsh" / "skills" / "deploy" / "SKILL.md").write_text(
        "---\nname: deploy\ndescription: project version\n---\nNEW BODY\n")
    assert reg_ctx.skills.get("deploy")["content"].strip() == "NEW BODY"
    # a scoped layer beats the global one outright, even at a worse rank
    key = object()
    reg_ctx.skills.register({"name": "deploy", "description": "scoped", "content": "S",
                             "rank": 999}, ctx=reg_ctx.scoped(key))
    assert reg_ctx.skills.get("deploy", key)["content"] == "S"
    assert reg_ctx.skills.get("deploy")["content"].strip() == "NEW BODY"
    with pytest.raises(SkillError):
        reg_ctx.skills.register({"name": "Not Kebab", "description": "d"})


def test_catalog_logged_and_rebuilds(tmp_path):
    skill(tmp_path / ".hsh" / "skills", "deploy", "Use when deploying <prod>")
    ad = ScriptedAdapter(fn=lambda r, i: "ok")
    ctx = world(tmp_path, ad)
    a = agent(ctx)
    a.send("hi")
    a.send("again")
    cats = [e for e in a.session.events if e["type"] == "user/message"
            and e["data"]["source"] == "skill-catalog"]
    assert len(cats) == 1
    c = cats[0]["data"]["content"]
    assert "<name>deploy</name>" in c and "&lt;prod&gt;" in c and "Do the thing" not in c
    skill(tmp_path / ".hsh" / "skills", "review", "Use when reviewing")
    a.send("third")
    cats = [e for e in a.session.events if e["data"].get("source") == "skill-catalog"]
    assert len(cats) == 2 and "<name>review</name>" in cats[1]["data"]["content"]
    assert a.sent == [request_hash(r) for _, r in rebuild_requests(a.session.events)]
    # no skills anywhere -> no catalog message at all
    empty = tmp_path / "empty"
    empty.mkdir()
    ctx2 = world(empty)
    b = agent(ctx2, "b")
    b.send("hi")
    assert not [e for e in b.session.events if e["data"].get("source") == "skill-catalog"]


def test_skill_tool_and_slash(tmp_path):
    root = tmp_path / ".hsh" / "skills"
    skill(root, "deploy", "Use when deploying", body="Run make deploy.")
    skill(root, "secret", "hidden from the model", extra="disable-model-invocation: true\n")
    skill(root, "auto", "model only", extra="user-invocable: false\n")
    ad = ScriptedAdapter(fn=lambda r, i: "ok")
    ctx = world(tmp_path, ad)
    a = agent(ctx)
    r = ctx.tools.execute("t", "skill", {"name": "deploy"}, scope=a.key)
    assert not r.is_error and "<skill_instructions>\nRun make deploy." in r.content
    assert str(root / "deploy") in r.content
    for bad in ("secret", "nope", "Bad Name"):
        assert ctx.tools.execute("t", "skill", {"name": bad}, scope=a.key).is_error
    a.send("hi")
    cat = ad.requests[0]["messages"][1]["content"]
    assert "<name>deploy</name>" in cat and "secret" not in cat
    a.send("/deploy to staging")
    last = [e for e in a.session.events if e["type"] == "user/message"][-1]["data"]
    assert last["content"].startswith('<skill_content name="deploy">')
    assert last["content"].endswith("to staging")
    a.send("/secret please")   # user-invocable, model-disabled: still expands
    last = [e for e in a.session.events if e["type"] == "user/message"][-1]["data"]
    assert last["content"].startswith('<skill_content name="secret">')
    a.send("/auto go")         # not user-invocable: left as typed
    last = [e for e in a.session.events if e["type"] == "user/message"][-1]["data"]
    assert last["content"] == "/auto go"
    assert a.sent == [request_hash(r) for _, r in rebuild_requests(a.session.events)]


def test_runtime_skill_on_shared_registry(tmp_path):
    ctx = world(tmp_path, shared=True)
    ctx.skills.register({"name": "house-style", "description": "Use when writing docs",
                         "content": "Write short sentences."})
    a = agent(ctx)
    r = ctx.tools.execute("t", "skill", {"name": "house-style"}, scope=a.key)
    assert "Write short sentences." in r.content
    b = agent(ctx, "b", dirs=["extra"])
    skill(tmp_path / "extra", "only-b", "b's skill")
    assert not ctx.tools.execute("t", "skill", {"name": "only-b"}, scope=b.key).is_error
    assert ctx.tools.execute("t", "skill", {"name": "only-b"}, scope=a.key).is_error
