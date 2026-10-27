"""hsh H5: built-in tool plugins and the workspace seam.

Each tool family is a plugin that injects ``tools`` and ``workspace`` and
registers through the context it is mounted on, so mounting it on an
agent's scoped context gives that agent alone the tools (dsh agent
presets work the same way).

- ``Workspace`` (``ctx.workspace``): the root directory every file path is
  resolved in. A path that escapes the root (``..``, an absolute path
  elsewhere, a symlink out) is refused before any file is touched.
- ``bash_tool``: one-shot ``bash -c`` in the workspace, scrubbed
  environment, timeout, CPU and memory rlimits on Linux, output tail
  truncated. With ``{"jobs": true}`` it also runs background jobs
  (``run_in_background``, ``job_list`` / ``job_output`` / ``job_kill``). This is process confinement, not a security sandbox: dsh
  uses Landlock/Seatbelt for that, which pure Python cannot.
- ``editor_tool``: dsh's ``str_replace_editor`` (view / create /
  str_replace / insert); a replacement must match exactly once.
- ``fs_tools``: ``read``, ``write``, ``edit``, ``glob``, ``grep``.
- ``todo_tool``: ``todo_write``; the list is returned and kept as
  ``meta`` so the log replays it.
- ``run_code``: the code preset's transport (dsh "PTC"). The model writes
  one Python program; inside it every visible tool is a function
  ``tools.<name>(**args)`` and each call goes back through the host's full
  pipeline (``parent`` set). The program runs in a child process with the
  same confinement as ``bash``; tool calls cross a JSON-lines pipe.
"""
from __future__ import annotations

import copy
import fnmatch
import json
import os
import re
import subprocess
import sys
from pathlib import Path

from .kernel import Service
from .tools import ToolDefinition

MAX_OUTPUT = 16_000


class Workspace(Service):
    key = "workspace"

    def __init__(self, ctx, config=None):
        super().__init__(ctx, config)
        self.root = Path((config or {}).get("root", ".")).resolve()

    def resolve(self, path):
        p = Path(path)
        full = (p if p.is_absolute() else self.root / p).resolve()
        if full != self.root and self.root not in full.parents:
            raise PermissionError(f"path {path!r} is outside the workspace")
        return full

    def rel(self, full):
        return str(Path(full).relative_to(self.root)) or "."


def _clip(text, limit=MAX_OUTPUT):
    if len(text) <= limit:
        return text
    return "<response clipped>\n" + text[-limit:]


def _limits(cpu_s, mem_mb):
    def apply():
        try:
            import resource
            resource.setrlimit(resource.RLIMIT_CPU, (cpu_s, cpu_s))
            mem = mem_mb * 1024 * 1024
            resource.setrlimit(resource.RLIMIT_AS, (mem, mem))
        except (ImportError, ValueError, OSError):
            pass
    return apply if sys.platform.startswith("linux") else None


def _env(root):
    keep = {k: os.environ[k] for k in ("PATH", "LANG", "LC_ALL", "TERM")
            if k in os.environ}
    # no .pyc in the workspace: an agent that edits a module and reruns it
    # within the same second, at the same size, would import stale bytecode
    return keep | {"HOME": str(root), "HSH_WORKSPACE": str(root),
                   "PYTHONDONTWRITEBYTECODE": "1"}


def _mount(ctx, *defs):
    for d in defs:
        ctx.tools.register(d, ctx=ctx)


# ----- bash -------------------------------------------------------------

BASH_SCHEMA = {
    "type": "object",
    "properties": {
        "description": {"type": "string", "description": "What the command does, 5-10 words."},
        "command": {"type": "string", "description": "The bash command to execute."},
        "workdir": {"type": "string", "description": "Directory inside the workspace."},
        "timeout_s": {"type": "number", "minimum": 1, "maximum": 600},
    },
    "required": ["description", "command"],
    "additionalProperties": False,
}


class JobTable:
    """Background processes of one ``bash`` row (dsh ``ctx.jobs`` +
    ``tool-jobs``). Output goes to a spill file outside the workspace;
    ``read`` returns what arrived since the previous read."""

    def __init__(self, limit=MAX_OUTPUT):
        import tempfile
        self.dir = Path(tempfile.mkdtemp(prefix="hsh-jobs-"))
        self.jobs = {}
        self.limit = limit
        self._n = 0
        self._files = 0

    def spill_path(self):
        self._files += 1
        return self.dir / f"out-{self._files}.log"

    def add(self, proc, command, description, out_path):
        self._n += 1
        jid = f"job-{self._n}"
        self.jobs[jid] = {"id": jid, "proc": proc, "command": command,
                          "description": description, "path": out_path,
                          "offset": 0, "killed": None}
        return jid

    def _status(self, j):
        code = j["proc"].poll()
        if code is None:
            return "running"
        return "killed" if j["killed"] is not None else "exited"

    def info(self, j):
        return {"job_id": j["id"], "kind": "bash", "status": self._status(j),
                "exit_code": j["proc"].poll(), "description": j["description"],
                "output_file": str(j["path"])}

    def get(self, jid):
        j = self.jobs.get(jid)
        if j is None:
            raise ValueError(f"unknown job {jid!r}")
        return j

    def read(self, jid, wait=False, timeout_s=30):
        j = self.get(jid)
        if wait:
            try:
                j["proc"].wait(timeout=timeout_s)
            except subprocess.TimeoutExpired:
                pass
        with open(j["path"], "rb") as fh:
            fh.seek(j["offset"])
            data = fh.read()
        j["offset"] += len(data)
        return dict(self.info(j), output=_clip(data.decode("utf-8", "replace"), self.limit))

    def kill(self, jid, reason=None):
        j = self.get(jid)
        if j["proc"].poll() is None:
            j["killed"] = reason or "killed"
            _kill_tree(j["proc"])
        return self.info(j)

    def close(self):
        for j in self.jobs.values():
            if j["proc"].poll() is None:
                _kill_tree(j["proc"])
        import shutil
        shutil.rmtree(self.dir, ignore_errors=True)


def _kill_tree(proc):
    try:
        if hasattr(os, "killpg"):
            os.killpg(proc.pid, 9)
        else:
            proc.kill()
    except (ProcessLookupError, PermissionError, OSError):
        pass
    try:
        proc.wait(timeout=5)
    except subprocess.TimeoutExpired:
        pass


JOB_ID = {"type": "string", "description": "Job id returned by the tool that started the "
          "background work."}


def bash_tool(ctx, config=None):
    """``bash``. Config: ``timeout_s`` (default 60), ``mem_mb`` (2048),
    ``jobs`` (default False): with jobs on, ``run_in_background`` starts a
    job and a foreground command that outlives its timeout moves to the
    background instead of being killed; ``job_list``, ``job_output`` and
    ``job_kill`` come with it (dsh ``tool-bash`` + ``tool-jobs``)."""
    cfg = config or {}
    ws = ctx.workspace
    jobs = JobTable() if cfg.get("jobs") else None

    def run(args, exe):
        cwd = ws.resolve(args.get("workdir", "."))
        timeout = min(args.get("timeout_s", cfg.get("timeout_s", 60)), 600)
        if jobs is not None:
            return run_job(args, cwd, timeout)
        try:
            p = subprocess.run(["bash", "-c", args["command"]], cwd=cwd,
                               env=_env(ws.root), capture_output=True, text=True,
                               timeout=timeout,
                               preexec_fn=_limits(int(timeout) + 1, cfg.get("mem_mb", 2048)))
        except subprocess.TimeoutExpired:
            return {"exit_code": None, "output": f"timed out after {timeout}s"}
        return {"exit_code": p.returncode, "output": _clip(p.stdout + p.stderr)}

    def run_job(args, cwd, timeout):
        out_path = jobs.spill_path()
        with open(out_path, "wb") as fh:
            proc = subprocess.Popen(
                ["bash", "-c", args["command"]], cwd=cwd, env=_env(ws.root),
                stdin=subprocess.DEVNULL, stdout=fh, stderr=subprocess.STDOUT,
                start_new_session=hasattr(os, "setsid"),
                preexec_fn=_limits(cfg.get("job_cpu_s", 3600), cfg.get("mem_mb", 2048)))
        background = bool(args.get("run_in_background"))
        if not background:
            try:
                proc.wait(timeout=timeout)
            except subprocess.TimeoutExpired:
                background = True
        jid = jobs.add(proc, args["command"], args.get("description", ""), out_path)
        if background:
            first = jobs.read(jid)
            note = ("started in the background" if args.get("run_in_background")
                    else f"still running after {timeout}s; moved to the background")
            return {"exit_code": None, "job_id": jid, "note": note,
                    "output": first["output"]}
        done = jobs.read(jid)
        del jobs.jobs[jid]
        return {"exit_code": proc.returncode, "output": done["output"]}

    def render(a, v):
        if v.get("job_id"):
            head = f"[{v['job_id']}: {v['note']}; read it with job_output, stop it with job_kill]"
            return head + ("\n" + v["output"] if v["output"] else "")
        return f"[exit {v['exit_code']}]\n{v['output']}"

    schema = BASH_SCHEMA
    if jobs is not None:
        schema = copy.deepcopy(BASH_SCHEMA)
        schema["properties"]["run_in_background"] = {
            "type": "boolean", "description": "Run in the background and return a job id "
            "at once (read it with job_output, stop it with job_kill). No timeout applies."}
    desc = ("Run a bash command (bash -c) in the workspace. Each call is a fresh "
            "shell; pass workdir instead of cd.")
    if jobs is not None:
        desc += (" A command still running at its timeout moves to the background "
                 "as a job instead of being killed.")
    _mount(ctx, ToolDefinition("bash", desc, schema, run, render=render))
    if jobs is None:
        return
    ctx.effect(jobs.close)

    def job_list(a, e):
        return [jobs.info(j) for j in jobs.jobs.values()]

    def job_output(a, e):
        return jobs.read(a["job_id"], a.get("wait", False),
                         min(a.get("timeout_s", 30), cfg.get("max_wait_s", 300)))

    def job_kill(a, e):
        return jobs.kill(a["job_id"], a.get("reason"))

    def render_info(a, v):
        code = "" if v["exit_code"] is None else f", exit {v['exit_code']}"
        return f"{v['job_id']} {v['status']}{code}" + (
            "\n" + v["output"] if v.get("output") else "")

    _mount(ctx, ToolDefinition(
        "job_list", "List your background jobs (running and finished) with their ids "
        "and statuses.", {"type": "object", "properties": {}, "additionalProperties": False},
        job_list, render=lambda a, v: "\n".join(
            f"{j['job_id']} {j['status']}  {j['description'] or j['job_id']}" for j in v)
        or "(no jobs)"),
        ToolDefinition(
        "job_output", "Read a background job: output since the previous read, its status "
        "and exit code.", {"type": "object", "properties": {
            "job_id": JOB_ID,
            "wait": {"type": "boolean", "description": "Block until the job finishes or "
                     "the timeout expires; a timed-out wait leaves the job running."},
            "timeout_s": {"type": "number", "minimum": 0}},
            "required": ["job_id"], "additionalProperties": False}, job_output,
        render=render_info),
        ToolDefinition(
        "job_kill", "Stop a running background job.", {"type": "object", "properties": {
            "job_id": JOB_ID, "reason": {"type": "string"}},
            "required": ["job_id"], "additionalProperties": False}, job_kill,
        render=render_info))


bash_tool.inject = ["tools", "workspace"]


# ----- str_replace_editor ------------------------------------------------

EDITOR_SCHEMA = {
    "type": "object",
    "properties": {
        "command": {"type": "string", "enum": ["view", "create", "str_replace", "insert"]},
        "path": {"type": "string"},
        "file_text": {"type": ["string", "null"]},
        "old_str": {"type": ["string", "null"]},
        "new_str": {"type": ["string", "null"]},
        "insert_line": {"type": ["integer", "null"]},
        "view_range": {"type": ["array", "null"], "items": {"type": "integer"}},
    },
    "required": ["command", "path"],
    "additionalProperties": False,
}


def _numbered(text, start=1):
    return "\n".join(f"{i:6d}\t{line}" for i, line in
                     enumerate(text.split("\n"), start=start))


def editor_tool(ctx, config=None):
    ws = ctx.workspace

    def run(args, exe):
        cmd, path = args["command"], ws.resolve(args["path"])
        if cmd == "view":
            if path.is_dir():
                items = sorted(str(p.relative_to(path)) for p in path.rglob("*")
                               if not any(part.startswith(".") for part in p.relative_to(path).parts)
                               and len(p.relative_to(path).parts) <= 2)
                return "\n".join(items)
            text = path.read_text()
            rng = args.get("view_range")
            if rng:
                lines = text.split("\n")
                lo, hi = rng[0], (rng[1] if rng[1] != -1 else len(lines))
                return _clip(_numbered("\n".join(lines[lo - 1:hi]), lo))
            return _clip(_numbered(text))
        if cmd == "create":
            if path.exists():
                raise FileExistsError(f"{args['path']} already exists")
            if args.get("file_text") is None:
                raise ValueError("create needs file_text")
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text(args["file_text"])
            return f"created {ws.rel(path)}"
        text = path.read_text()
        if cmd == "str_replace":
            old = args.get("old_str")
            if not old:
                raise ValueError("str_replace needs old_str")
            n = text.count(old)
            if n != 1:
                raise ValueError(f"old_str matches {n} times; it must match exactly once")
            path.write_text(text.replace(old, args.get("new_str") or "", 1))
            return f"edited {ws.rel(path)}"
        line, new = args.get("insert_line"), args.get("new_str")
        if line is None or new is None:
            raise ValueError("insert needs insert_line and new_str")
        lines = text.split("\n")
        if not 0 <= line <= len(lines):
            raise ValueError(f"insert_line must be in [0, {len(lines)}]")
        lines[line:line] = new.split("\n")
        path.write_text("\n".join(lines))
        return f"inserted after line {line} of {ws.rel(path)}"

    _mount(ctx, ToolDefinition(
        "str_replace_editor", "View, create and edit files: view (cat -n or a "
        "2-level listing), create (new file only), str_replace (old_str must match "
        "exactly once), insert (after insert_line).", EDITOR_SCHEMA, run))


editor_tool.inject = ["tools", "workspace"]


# ----- fs tools ----------------------------------------------------------

def fs_tools(ctx, config=None):
    ws = ctx.workspace
    def obj(props, req):
        return {"type": "object", "properties": props, "required": req,
                "additionalProperties": False}

    S = {"type": "string"}

    def read(a, e):
        return _clip(_numbered(ws.resolve(a["path"]).read_text()))

    def write(a, e):
        p = ws.resolve(a["path"])
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text(a["content"])
        return f"wrote {len(a['content'])} chars to {ws.rel(p)}"

    def edit(a, e):
        p = ws.resolve(a["path"])
        text = p.read_text()
        n = text.count(a["old"])
        if n != 1:
            raise ValueError(f"old matches {n} times; it must match exactly once")
        p.write_text(text.replace(a["old"], a["new"], 1))
        return f"edited {ws.rel(p)}"

    def glob(a, e):
        base = ws.resolve(a.get("path", "."))
        hits = sorted(ws.rel(p) for p in base.glob(a["pattern"]) if p.is_file())
        return "\n".join(hits[:500]) or "(no matches)"

    def grep(a, e):
        base = ws.resolve(a.get("path", "."))
        rx = re.compile(a["pattern"])
        out = []
        files = [base] if base.is_file() else sorted(p for p in base.rglob("*") if p.is_file())
        for f in files:
            if a.get("glob") and not fnmatch.fnmatch(f.name, a["glob"]):
                continue
            try:
                lines = f.read_text().split("\n")
            except (UnicodeDecodeError, OSError):
                continue
            out += [f"{ws.rel(f)}:{i}:{ln}" for i, ln in enumerate(lines, 1) if rx.search(ln)]
            if len(out) >= 500:
                break
        return _clip("\n".join(out[:500])) or "(no matches)"

    _mount(ctx,
           ToolDefinition("read", "Read a file (cat -n).", obj({"path": S}, ["path"]), read),
           ToolDefinition("write", "Write a whole file.",
                          obj({"path": S, "content": S}, ["path", "content"]), write),
           ToolDefinition("edit", "Replace one exact, unique occurrence of old with new.",
                          obj({"path": S, "old": S, "new": S}, ["path", "old", "new"]), edit),
           ToolDefinition("glob", "List files matching a glob pattern.",
                          obj({"pattern": S, "path": S}, ["pattern"]), glob),
           ToolDefinition("grep", "Search file contents with a regular expression.",
                          obj({"pattern": S, "path": S, "glob": S}, ["pattern"]), grep))


fs_tools.inject = ["tools", "workspace"]


# ----- todo ----------------------------------------------------------------

TODO_SCHEMA = {
    "type": "object",
    "properties": {"todos": {"type": "array", "items": {
        "type": "object",
        "properties": {"content": {"type": "string"},
                       "status": {"enum": ["pending", "in_progress", "completed"]}},
        "required": ["content", "status"], "additionalProperties": False}}},
    "required": ["todos"], "additionalProperties": False,
}


def todo_tool(ctx, config=None):
    mark = {"pending": "[ ]", "in_progress": "[~]", "completed": "[x]"}

    def run(a, e):
        return a["todos"]

    defn = ToolDefinition(
        "todo_write", "Replace the task list for this session.", TODO_SCHEMA, run,
        render=lambda a, v: "\n".join(f"{mark[t['status']]} {t['content']}" for t in v)
        or "(empty)")
    _mount(ctx, defn)


todo_tool.inject = ["tools"]


# ----- run_code (PTC) ------------------------------------------------------

_CHILD = r'''
import io, json, sys, contextlib
_rpc_in, _rpc_out = sys.stdin, sys.__stdout__
class _Tools:
    def __getattr__(self, name):
        def call(**kwargs):
            _rpc_out.write(json.dumps({"call": name, "args": kwargs}) + "\n")
            _rpc_out.flush()
            reply = json.loads(_rpc_in.readline())
            if reply["is_error"]:
                raise RuntimeError(reply["content"])
            return reply["value"] if reply.get("value") is not None else reply["content"]
        return call
tools = _Tools()
src = json.loads(_rpc_in.readline())["program"]
buf = io.StringIO()
result = {"ok": True}
try:
    with contextlib.redirect_stdout(buf):
        ns = {"tools": tools, "__name__": "__program__"}
        exec(compile(src, "<program>", "exec"), ns)
    if "result" in ns:
        json.dumps(ns["result"])
        result["value"] = ns["result"]
except BaseException as exc:
    result = {"ok": False, "error": f"{type(exc).__name__}: {exc}"}
result["stdout"] = buf.getvalue()
_rpc_out.write(json.dumps({"done": result}) + "\n")
_rpc_out.flush()
'''

RUN_CODE_SCHEMA = {
    "type": "object",
    "properties": {"program": {"type": "string", "description":
                               "Python program. Call tools as tools.<name>(**args); "
                               "assign the final answer to `result`; print() is captured."}},
    "required": ["program"], "additionalProperties": False,
}


def run_code_tool(ctx, config=None):
    cfg = config or {}
    tools, ws = ctx.tools, ctx.workspace
    max_calls = cfg.get("max_calls", 64)

    def run(args, exe):
        proc = subprocess.Popen([sys.executable, "-I", "-c", _CHILD], cwd=ws.root,
                                env=_env(ws.root), stdin=subprocess.PIPE,
                                stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True,
                                preexec_fn=_limits(cfg.get("cpu_s", 60), cfg.get("mem_mb", 2048)))
        dispatches = []
        try:
            proc.stdin.write(json.dumps({"program": args["program"]}) + "\n")
            proc.stdin.flush()
            for line in proc.stdout:
                msg = json.loads(line)
                if "done" in msg:
                    done = msg["done"]
                    break
                if len(dispatches) >= max_calls:
                    reply = {"is_error": True, "content": f"more than {max_calls} tool calls"}
                elif msg["call"] == "run_code":
                    reply = {"is_error": True, "content": "run_code cannot call itself"}
                else:
                    sub = tools.execute(f"{exe.call_id}.{len(dispatches)}", msg["call"],
                                        msg["args"], scope=exe.scope, parent=exe)
                    reply = sub.to_json()
                dispatches.append({"name": msg["call"], "is_error": reply["is_error"]})
                proc.stdin.write(json.dumps(reply) + "\n")
                proc.stdin.flush()
            else:
                done = {"ok": False, "error": "program exited without a result",
                        "stdout": ""}
        finally:
            proc.kill()
            proc.wait()
        if not done["ok"]:
            raise RuntimeError(done["error"] + ("\n" + done["stdout"] if done["stdout"] else ""))
        return {"value": done.get("value"), "stdout": _clip(done["stdout"]),
                "dispatches": dispatches}

    _mount(ctx, ToolDefinition(
        "run_code", "Run one Python program that calls the other tools as "
        "functions (tools.<name>(**args)). Use it to batch several tool calls "
        "into one step.", RUN_CODE_SCHEMA, run,
        render=lambda a, v: (v["stdout"] + ("\nresult = " + json.dumps(v["value"], ensure_ascii=False)
                                            if v["value"] is not None else "")).strip()
        or "(no output)"))


run_code_tool.inject = ["tools", "workspace"]
