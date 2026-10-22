"""hsh H5: built-in tools, PTC mode and presets (hsh/builtin.py, presets.py).

Pins:
1. workspace confinement: paths outside the root (.., absolute, symlink)
   are refused before any file is touched;
2. bash: exit code and output, workdir, timeout, scrubbed environment;
3. str_replace_editor: view / create (never over an existing file) /
   str_replace (exactly one match) / insert; fs tools read, write, edit,
   glob, grep; todo_write renders the list;
4. PTC: in code mode the model sees only run_code, a direct native call is
   refused as unknown, and the program's tool calls go through the full
   pipeline (a guard still denies inside the program);
5. presets: each preset gives its agent exactly its tools, agents with
   different presets do not see each other's tools, the persona reaches
   the system prompt, patches insert/replace/remove/disable rows, unknown
   plugins and ids are refused; creator's inspect reads the harness;
6. end to end: a scripted model edits a file through the minimal preset,
   and every request still rebuilds from the log.
"""
import json
import os
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from helioslm_v5.hsh import (LLM, Agents, Context, PresetError,  # noqa: E402
                             ScriptedAdapter, Sessions, Tools, Workspace, compose,
                             create_agent, load_patch, rebuild_requests,
                             request_hash)

CFG = {"provider": "fake", "model": "m"}


def world(tmp_path, adapter=None):
    ctx = Context()
    for p in (Sessions, Tools, LLM, Agents):
        ctx.plugin(p)
    ctx.plugin(Workspace, {"root": str(tmp_path)})
    ctx.llm.register("fake", adapter or ScriptedAdapter(fn=lambda r, i: "ok"))
    return ctx


def names(ctx, agent):
    return [s["name"] for s in ctx.tools.schemas(agent.key)]


def run(ctx, agent, name, args):
    return ctx.tools.execute("t", name, args, scope=agent.key)


def test_workspace_confinement(tmp_path):
    ctx = world(tmp_path)
    a = create_agent(ctx, "s", CFG, "standard")
    outside = tmp_path.parent / "outside.txt"
    (tmp_path / "link").symlink_to(tmp_path.parent)
    for p in ("../outside.txt", str(outside), "link/outside.txt"):
        res = run(ctx, a, "write", {"path": p, "content": "x"})
        assert res.is_error and "outside the workspace" in res.content
    assert not outside.exists()
    assert run(ctx, a, "bash", {"description": "d", "command": "pwd",
                                "workdir": ".."}).is_error


def test_bash(tmp_path):
    ctx = world(tmp_path)
    a = create_agent(ctx, "s", CFG, "minimal")
    (tmp_path / "sub").mkdir()
    r = run(ctx, a, "bash", {"description": "d", "command": "echo hi; pwd; exit 3",
                             "workdir": "sub"})
    assert r.value["exit_code"] == 3 and "hi" in r.content
    assert r.value["output"].split()[-1].endswith("sub")
    os.environ["HSH_SECRET_TEST"] = "leak"
    try:
        r = run(ctx, a, "bash", {"description": "d", "command": "echo ${HSH_SECRET_TEST:-none}"})
    finally:
        del os.environ["HSH_SECRET_TEST"]
    assert r.value["output"].strip() == "none"
    r = run(ctx, a, "bash", {"description": "d", "command": "sleep 5", "timeout_s": 1})
    assert "timed out" in r.content


def test_editor(tmp_path):
    ctx = world(tmp_path)
    a = create_agent(ctx, "s", CFG, "minimal")
    ed = lambda **kw: run(ctx, a, "str_replace_editor", kw)  # noqa: E731
    assert not ed(command="create", path="a.py", file_text="x = 1\ny = 2\n").is_error
    assert ed(command="create", path="a.py", file_text="again").is_error
    assert ed(command="view", path="a.py").content.splitlines()[0] == "     1\tx = 1"
    assert ed(command="str_replace", path="a.py", old_str="= ", new_str="== ").is_error  # 2 hits
    assert not ed(command="str_replace", path="a.py", old_str="y = 2", new_str="y = 3").is_error
    assert not ed(command="insert", path="a.py", insert_line=1, new_str="# mid").is_error
    assert (tmp_path / "a.py").read_text() == "x = 1\n# mid\ny = 3\n"
    assert "a.py" in ed(command="view", path=".").content
    assert ed(command="view", path="a.py", view_range=[2, 2]).content == "     2\t# mid"


def test_fs_and_todo(tmp_path):
    ctx = world(tmp_path)
    a = create_agent(ctx, "s", CFG, "standard")
    assert not run(ctx, a, "write", {"path": "pkg/m.py", "content": "def f():\n    return 1\n"}).is_error
    assert run(ctx, a, "read", {"path": "pkg/m.py"}).content.startswith("     1\tdef f():")
    assert not run(ctx, a, "edit", {"path": "pkg/m.py", "old": "return 1", "new": "return 2"}).is_error
    assert run(ctx, a, "glob", {"pattern": "**/*.py"}).content == "pkg/m.py"
    assert run(ctx, a, "grep", {"pattern": "return 2"}).content == "pkg/m.py:2:    return 2"
    t = run(ctx, a, "todo_write", {"todos": [{"content": "write test", "status": "completed"},
                                             {"content": "ship", "status": "in_progress"}]})
    assert t.content == "[x] write test\n[~] ship"


def test_ptc_code_mode(tmp_path):
    ctx = world(tmp_path)
    a = create_agent(ctx, "s", CFG, "code")
    assert names(ctx, a) == ["run_code"]
    direct = run(ctx, a, "write", {"path": "x", "content": "y"})
    assert direct.is_error and direct.error["code"] == "unknown-tool"
    prog = ("tools.write(path='n.txt', content='42')\n"
            "text = tools.read(path='n.txt')\n"
            "print('read back')\n"
            "result = {'len': len(text)}\n")
    r = run(ctx, a, "run_code", {"program": prog})
    assert not r.is_error, r.content
    assert (tmp_path / "n.txt").read_text() == "42"
    assert r.value["value"] == {"len": len("     1\t42")}
    assert r.value["dispatches"] == [{"name": "write", "is_error": False},
                                     {"name": "read", "is_error": False}]
    assert "read back" in r.content

    # guards apply inside the program; the program sees the denial as an error
    ctx.tools.guard(lambda e: "no bash in programs" if e.name == "bash" else None)
    r = run(ctx, a, "run_code", {"program": "tools.bash(description='d', command='ls')"})
    assert r.is_error and "no bash in programs" in r.content
    r = run(ctx, a, "run_code", {"program": "raise ValueError('mine')"})
    assert r.is_error and "ValueError: mine" in r.content


def test_presets_isolate_agents_and_persona(tmp_path):
    ad = ScriptedAdapter(fn=lambda r, i: "ok")
    ctx = world(tmp_path, ad)
    m = create_agent(ctx, "m", CFG, "minimal", system_prompt="task: fix bug")
    s = create_agent(ctx, "s", CFG, "standard")
    c = create_agent(ctx, "c", CFG, "creator")
    assert names(ctx, m) == ["bash", "str_replace_editor"]
    assert names(ctx, s) == ["bash", "edit", "glob", "grep", "read", "subagent",
                             "todo_write", "write"]
    assert "inspect" in names(ctx, c) and "inspect" not in names(ctx, s)
    assert ctx.tools.schemas() == []  # nothing leaked to the global scope
    m.send("hi")
    sys_msg = ad.requests[0]["messages"][0]
    assert sys_msg["role"] == "system"
    assert sys_msg["content"] == "You are a helpful software engineer assistant.\n\ntask: fix bug"
    s.send("hi")
    assert ad.requests[1]["messages"][0]["content"].startswith("You are a coding agent")


def test_creator_inspect(tmp_path):
    ctx = world(tmp_path)
    c = create_agent(ctx, "c", CFG, "creator")
    c.send("hello")
    assert "tools" in run(ctx, c, "inspect", {"what": "services"}).value
    assert run(ctx, c, "inspect", {"what": "events"}).value["tools/pre-execute"] == "waterfall"
    ev = run(ctx, c, "inspect", {"what": "session", "last": 2}).value
    assert [e["type"] for e in ev] == ["step/end", "turn/end"]
    assert [r["id"] for r in run(ctx, c, "inspect", {"what": "preset"}).value][-1] == "creator-inspect"


def test_patches(tmp_path):
    rows = compose("standard",
                   [{"insert": [{"id": "ed", "name": "tool-editor"}], "after": "tool-bash"},
                    {"replace": "persona", "config": {"prefix": "P"}},
                    {"disable": "tool-todo"}],
                   [{"remove": "tool-fs"}])
    assert [r["id"] for r in rows] == ["persona", "agent-instructions", "tool-bash", "ed",
                                       "tool-todo", "tool-subagent", "compaction"]
    assert rows[0]["config"] == {"prefix": "P"} and rows[4]["disabled"]
    ctx = world(tmp_path)
    a = create_agent(ctx, "a", CFG, rows)
    assert names(ctx, a) == ["bash", "str_replace_editor", "subagent"]
    for bad in ([{"remove": "nope"}], [{"insert": [{"id": "persona", "name": "persona"}]}],
                [{"insert": [{"id": "z", "name": "no-such-plugin"}]}], [{"frobnicate": 1}]):
        with pytest.raises(PresetError):
            compose("standard", bad)
    p = tmp_path / "patch.json"
    p.write_text(json.dumps([{"disable": "tool-bash"}]))
    assert compose("minimal", load_patch(p))[1]["disabled"] is True


def test_preset_unwinds_with_agents_service(tmp_path):
    ctx = world(tmp_path)
    a = create_agent(ctx, "a", CFG, "standard")
    assert names(ctx, a)
    agents_fork = [f for f in ctx.forks if f.name == "Agents"][0]
    agents_fork.dispose()
    assert ctx.tools.schemas(a.key) == []


def test_minimal_agent_edits_file_end_to_end(tmp_path):
    (tmp_path / "app.py").write_text("print('helo')\n")
    calls = [
        {"tool_calls": [{"id": "1", "name": "str_replace_editor", "arguments": json.dumps(
            {"command": "str_replace", "path": "app.py", "old_str": "helo", "new_str": "hello"})}]},
        {"tool_calls": [{"id": "2", "name": "bash", "arguments": json.dumps(
            {"description": "run app", "command": "python3 app.py"})}]},
        "Fixed the typo; the script prints hello.",
    ]
    ad = ScriptedAdapter(calls)
    ctx = world(tmp_path, ad)
    a = create_agent(ctx, "e2e", CFG, "minimal")
    assert a.send("fix the typo in app.py") == "Fixed the typo; the script prints hello."
    assert (tmp_path / "app.py").read_text() == "print('hello')\n"
    assert "[exit 0]\nhello" in ad.requests[2]["messages"][-1]["content"]
    assert a.sent == [request_hash(r) for _, r in rebuild_requests(a.session.events)]
