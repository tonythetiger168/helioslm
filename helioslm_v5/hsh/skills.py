"""hsh: agent skills (``ctx.skills``, the ``skill`` tool, the skills row).

After dsh's skill family (docs/subsystems/skills.md): a registry of
providers, a local filesystem provider, and a consumer that shows the
model a catalog and loads full instructions on demand.

A skill is a Markdown file with a frontmatter block::

    ---
    name: release-notes
    description: Use when the user asks for release notes; reads git log ...
    ---
    # Release notes
    1. Run `git log --oneline <last-tag>..HEAD` ...

It lives as ``<root>/<name>/SKILL.md`` (a bundle: scripts, references and
assets may sit next to it) or as a flat ``<root>/<name>.md``. Names are
kebab-case. Roots, best rank first (a lower rank wins a duplicate name):

=====  ==================  ===========================================
rank   source              root
=====  ==================  ===========================================
100    ``project-hsh``     ``<workspace>/.hsh/skills``
200    ``project-agents``  ``<workspace>/.agents/skills``
250    ``project-claude``  ``<workspace>/.claude/skills``
300    ``custom``          ``config["dirs"]`` (relative to the workspace)
400    ``user-hsh``        ``<home>/.hsh/skills``
500    ``user-agents``     ``<home>/.agents/skills``
=====  ==================  ===========================================

Frontmatter keys read: ``name`` (must equal the directory or file name),
``description`` (required), ``when_to_use``, ``disable-model-invocation``
and ``user-invocable`` (both default to allowing). Every other key is
kept as ``metadata``. Only the frontmatter subset skills use is parsed
(``key: value``, quoted scalars, ``>``/``|`` block scalars, booleans);
PyYAML is not needed.

The consumer (the ``skills`` preset row):

- before each turn's first request it compares the model-invocable
  catalog (names and descriptions only, never bodies or paths) with the
  newest catalog still visible in the log, and when they differ it adds
  a ``user/message`` (source ``skill-catalog``) holding the new one, so
  the catalog is part of the session log and every request still
  rebuilds from it; compaction that hides the old catalog brings it back;
- the ``skill`` tool returns a skill's full instructions, re-read from
  disk on every call, wrapped in ``<skill_content>``;
- a human message ``/<name> <rest>`` for a user-invocable skill is
  expanded to the skill's instructions followed by ``<rest>`` before it
  is logged.
"""
from __future__ import annotations

import hashlib
import json
import re
from pathlib import Path

from .kernel import Service
from .tools import ToolDefinition

NAME_RE = re.compile(r"^[a-z0-9]+(?:-[a-z0-9]+)*$")
DESCRIPTION_MAX = 500


class SkillError(ValueError):
    pass


# ----- frontmatter ---------------------------------------------------------

def _scalar(text):
    t = text.strip()
    if len(t) >= 2 and t[0] == t[-1] == "'":
        return t[1:-1].replace("''", "'")
    if len(t) >= 2 and t[0] == t[-1] == '"':
        return json.loads(t)
    low = t.lower()
    if low in ("true", "yes", "on"):
        return True
    if low in ("false", "no", "off"):
        return False
    if low in ("", "null", "~"):
        return None
    if re.fullmatch(r"-?\d+", t):
        return int(t)
    return t


def _block(lines, style):
    indent = min((len(x) - len(x.lstrip()) for x in lines if x.strip()), default=0)
    body = [x[indent:] for x in lines]
    while body and not body[-1].strip():
        body.pop()
    if style[0] == "|":
        text = "\n".join(body)
    else:  # folded: lines join with spaces, blank lines become newlines
        paras, cur = [], []
        for x in body:
            if x.strip():
                cur.append(x.strip())
            else:
                paras.append(" ".join(cur))
                cur = []
        paras.append(" ".join(cur))
        text = "\n".join(paras)
    return text if style.endswith("-") else text + "\n"


def parse_frontmatter(text):
    """``(fields, body)`` of a Markdown file; ``({}, text)`` without one."""
    if not text.startswith("---"):
        return {}, text
    lines = text.splitlines()
    try:
        end = next(i for i in range(1, len(lines)) if lines[i].strip() == "---")
    except StopIteration:
        raise SkillError("frontmatter is not closed with ---") from None
    fields, i, head = {}, 0, lines[1:end]
    while i < len(head):
        line = head[i]
        i += 1
        if not line.strip() or line.lstrip().startswith("#"):
            continue
        m = re.match(r"^([A-Za-z0-9_-]+):(.*)$", line)
        if not m:
            raise SkillError(f"cannot parse frontmatter line {line!r}")
        key, rest = m.group(1), m.group(2).strip()
        if rest in (">", ">-", ">+", "|", "|-", "|+"):
            block = []
            while i < len(head) and (not head[i].strip() or head[i][:1] in " \t"):
                block.append(head[i])
                i += 1
            fields[key] = _block(block, rest.rstrip("+"))
        else:
            fields[key] = _scalar(rest)
    body = "\n".join(lines[end + 1:]).lstrip("\n")
    return fields, body


def _summary(defn):
    return {k: v for k, v in defn.items() if k not in ("content", "metadata")}


def make_skill(name, description, content, *, source="runtime", provider="runtime",
               path=None, base=None, when_to_use=None, model_invocable=True,
               user_invocable=True, metadata=None, rank=0):
    if not isinstance(name, str) or not NAME_RE.match(name):
        raise SkillError(f"skill name {name!r} is not kebab-case")
    if not isinstance(description, str) or not description.strip():
        raise SkillError(f"skill {name!r} has no description")
    out = {"name": name, "description": " ".join(description.split()),
           "content": content, "source": source, "provider": provider, "rank": rank,
           "invocation": {"model": bool(model_invocable), "user": bool(user_invocable)}}
    for k, v in (("path", path), ("base", base), ("when_to_use", when_to_use),
                 ("metadata", metadata)):
        if v:
            out[k] = v
    return out


def load_skill_file(path, name, source="custom", provider="filesystem", rank=300):
    """Parse one skill file; raises SkillError when it is malformed."""
    path = Path(path)
    fields, body = parse_frontmatter(path.read_text(encoding="utf-8"))
    declared = fields.pop("name", None)
    if declared is not None and declared != name:
        raise SkillError(f"{path}: frontmatter name {declared!r} != {name!r}")
    desc = fields.pop("description", None)
    when = fields.pop("when_to_use", None) or fields.pop("when-to-use", None)
    no_model = fields.pop("disable-model-invocation", False)
    user = fields.pop("user-invocable", True)
    base = str(path.parent) if path.name == "SKILL.md" else None
    return make_skill(name, desc if isinstance(desc, str) else "", body,
                      source=source, provider=provider, path=str(path), base=base,
                      when_to_use=when, model_invocable=not no_model,
                      user_invocable=user is not False, metadata=fields or None,
                      rank=rank)


# ----- providers -----------------------------------------------------------

class FilesystemSkills:
    """Scans skill roots on every ``list()``; bodies are re-read on ``get()``.

    ``roots`` is a list of ``(source, rank, directory)``. A malformed
    skill is skipped and recorded in ``errors`` (path -> reason) instead
    of hiding the rest of the catalog."""

    name = "filesystem"

    def __init__(self, roots):
        self.roots = [(s, r, Path(d)) for s, r, d in roots]
        self.errors = {}

    def _entries(self, root):
        if not root.is_dir():
            return
        for p in sorted(root.iterdir()):
            if p.is_dir() and (p / "SKILL.md").is_file():
                yield p.name, p / "SKILL.md"
            elif p.is_file() and p.suffix == ".md" and p.name != "README.md":
                yield p.stem, p

    def list(self):
        out, self.errors = [], {}
        for source, rank, root in self.roots:
            for name, path in self._entries(root):
                if not NAME_RE.match(name):
                    continue
                try:
                    out.append(load_skill_file(path, name, source, self.name, rank))
                except (SkillError, OSError, UnicodeDecodeError, ValueError) as exc:
                    self.errors[str(path)] = str(exc)
        return out

    def get(self, candidate):
        try:
            return load_skill_file(candidate["path"], candidate["name"],
                                   candidate["source"], self.name, candidate["rank"])
        except (SkillError, OSError, UnicodeDecodeError, ValueError):
            return None


class _Runtime:
    name = "runtime"

    def __init__(self):
        self.skills = []

    def list(self):
        return list(self.skills)

    def get(self, candidate):
        return next((s for s in self.skills if s["name"] == candidate["name"]), None)


# ----- the registry --------------------------------------------------------

class Skills(Service):
    """``ctx.skills``: providers layered by scope.

    A provider or skill registered through an agent's scoped context is
    seen by that agent only and wins a duplicate name over the global
    layer outright; inside one layer the lower rank wins, then the
    earlier registration."""

    key = "skills"

    def start(self):
        self.ctx.declare("skills/change", "emit")
        self._layers = {}     # scope -> [provider]
        self._runtime = {}    # scope -> _Runtime

    def register_provider(self, provider, ctx=None):
        ctx = ctx or self.ctx
        layer = self._layers.setdefault(ctx.scope, [])
        if any(p.name == provider.name for p in layer):
            raise SkillError(f"skill provider {provider.name!r} already registered here")
        layer.append(provider)
        self.ctx.emit("skills/change")

        def dispose():
            if provider in layer:
                layer.remove(provider)
                self.ctx.emit("skills/change")

        return ctx.effect(dispose)

    def register(self, skill, ctx=None):
        """A runtime skill: ``{"name", "description", "content", ...}``."""
        ctx = ctx or self.ctx
        rt = self._runtime.get(ctx.scope)
        if rt is None:
            rt = self._runtime[ctx.scope] = _Runtime()
            self._layers.setdefault(ctx.scope, []).append(rt)
        s = make_skill(skill["name"], skill.get("description", ""),
                       skill.get("content", ""), when_to_use=skill.get("when_to_use"),
                       base=skill.get("base"), rank=skill.get("rank", 150),
                       model_invocable=skill.get("model_invocable", True),
                       user_invocable=skill.get("user_invocable", True))
        if any(x["name"] == s["name"] for x in rt.skills):
            raise SkillError(f"runtime skill {s['name']!r} already registered here")
        rt.skills.append(s)
        self.ctx.emit("skills/change")

        def dispose():
            if s in rt.skills:
                rt.skills.remove(s)
                self.ctx.emit("skills/change")

        return ctx.effect(dispose)

    def _layer(self, scope):
        best = {}
        for order, prov in enumerate(self._layers.get(scope, ())):
            for c in prov.list():
                key = (c["rank"], order)
                if c["name"] not in best or key < best[c["name"]][0]:
                    best[c["name"]] = (key, prov, c)
        return {n: (prov, c) for n, (_, prov, c) in best.items()}

    def _merged(self, scope):
        merged = self._layer(None)
        if scope is not None:
            merged.update(self._layer(scope))
        return merged

    def list(self, scope=None):
        """Summaries (no bodies), sorted by name."""
        return [_summary(c) for _, (_, c) in sorted(self._merged(scope).items())]

    def get(self, name, scope=None):
        """The full definition, re-read from its provider, or None."""
        hit = self._merged(scope).get(name)
        if hit is None:
            return None
        prov, cand = hit
        defn = prov.get(cand)
        if defn is None or defn["name"] != name:
            return None
        return defn


# ----- the consumer --------------------------------------------------------

CATALOG_SOURCE = "skill-catalog"


def _escape(text):
    return text.replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;")


def render_catalog(summaries, max_len=DESCRIPTION_MAX):
    entries = []
    for s in summaries:
        d = s["description"]
        if len(d) > max_len:
            d = d[:max_len - 3] + "..."
        entries.append(f"<skill><name>{s['name']}</name>"
                       f"<description>{_escape(d)}</description></skill>")
    inner = "\n".join(entries)
    return ("<system-reminder>\nSkills hold task instructions. When a task names or "
            "clearly matches one of these skills, call the skill tool with its name "
            "before acting.\n<available_skills>\n" + inner +
            "\n</available_skills>\n</system-reminder>")


def _catalog_digest(text):
    m = re.search(r"<available_skills>\n?(.*?)\n?</available_skills>", text, re.S)
    return hashlib.sha256((m.group(1) if m else "").encode()).hexdigest()


def visible_catalog(events):
    """The newest skill catalog still visible in the log, or None."""
    shadowed = [(e["data"]["first"], e["data"]["last"]) for e in events
                if e["type"] == "context/shadow"]
    for e in reversed(events):
        if (e["type"] == "user/message" and e["data"].get("source") == CATALOG_SOURCE
                and not any(a <= e["seq"] <= b for a, b in shadowed)):
            return e["data"]["content"]
    return None


def render_skill(defn):
    res = (f"Base directory: {defn['base']}. Paths in these instructions are relative "
           "to it; read or run them only as needed." if defn.get("base")
           else "This skill has no resource directory.")
    return (f'<skill_content name="{defn["name"]}">\n<skill_resources>\n{res}\n'
            f"</skill_resources>\n<skill_instructions>\n{defn['content'].rstrip()}\n"
            "</skill_instructions>\n</skill_content>")


def default_roots(workspace_root, dirs=(), home=None):
    ws = Path(workspace_root)
    home = Path(home) if home is not None else Path.home()
    roots = [("project-hsh", 100, ws / ".hsh" / "skills"),
             ("project-agents", 200, ws / ".agents" / "skills"),
             ("project-claude", 250, ws / ".claude" / "skills")]
    roots += [("custom", 300, (ws / d) if not Path(d).is_absolute() else Path(d))
              for d in dirs]
    roots += [("user-hsh", 400, home / ".hsh" / "skills"),
              ("user-agents", 500, home / ".agents" / "skills")]
    return roots


SKILL_SCHEMA = {
    "type": "object",
    "properties": {"name": {"type": "string",
                            "description": "The exact skill name from the available skills list."}},
    "required": ["name"], "additionalProperties": False,
}


def skills_plugin(ctx, config=None):
    """Preset row ``skills``. Config: ``dirs`` (extra roots), ``home``
    (user root base, default ``~``; ``False`` skips user roots),
    ``slash`` (expand ``/<name>``, default on), ``catalog_max`` (per
    description), ``roots`` (replace the whole root list)."""
    cfg = config or {}
    if ctx.has("skills"):
        reg = ctx.skills
    else:  # no shared registry mounted: this agent gets a private one
        reg = Skills(ctx)
        reg.start()
    if "roots" in cfg:
        roots = cfg["roots"]
    else:
        home = cfg.get("home")
        roots = default_roots(ctx.workspace.root, cfg.get("dirs", ()),
                              None if home in (None, True) else (home or "/nonexistent"))
        if cfg.get("home") is False:
            roots = [r for r in roots if not r[0].startswith("user-")]
    provider = FilesystemSkills(roots)
    reg.register_provider(provider, ctx=ctx)
    max_len = cfg.get("catalog_max", DESCRIPTION_MAX)

    def model_view(scope):
        return [s for s in reg.list(scope) if s["invocation"]["model"]]

    def run(args, exe):
        name = args["name"]
        if not NAME_RE.match(name):
            raise ValueError(f"{name!r} is not a skill name (kebab-case)")
        if not any(s["name"] == name for s in model_view(exe.scope)):
            raise ValueError(f"unknown skill {name!r}")
        defn = reg.get(name, exe.scope)
        if defn is None or not defn["invocation"]["model"]:
            raise ValueError(f"skill {name!r} is no longer available")
        return render_skill(defn)

    ctx.tools.register(ToolDefinition(
        "skill", "Load the full instructions for a skill. Call it before acting on a "
        "task that names or clearly matches a skill in the available skills list.",
        SKILL_SCHEMA, run, render=lambda a, v: v), ctx=ctx)

    def expand(agent, msg):
        text = msg["content"]
        m = re.match(r"^/([a-z0-9]+(?:-[a-z0-9]+)*)(?:\s+(.*))?$", text, re.S)
        if not m or msg.get("source") != "human":
            return msg
        defn = reg.get(m.group(1), agent.key)
        if defn is None or not defn["invocation"]["user"]:
            return msg
        rest = (m.group(2) or "").strip()
        content = render_skill(defn) + (f"\n\n{rest}" if rest else "")
        return dict(msg, content=content, skill=defn["name"])

    def pre_step(agent, decision, nxt):
        if decision.get("action") == "enter" and decision.get("messages"):
            msgs = [expand(agent, m) for m in decision["messages"]] \
                if cfg.get("slash", True) else list(decision["messages"])
            view = model_view(agent.key)
            old = visible_catalog(agent.session.events)
            new = render_catalog(view, max_len) if view else None
            if new is not None and (old is None or _catalog_digest(old) != _catalog_digest(new)):
                msgs.insert(0, {"content": new, "source": CATALOG_SOURCE})
            elif new is None and old is not None and _catalog_digest(old) != _catalog_digest(""):
                msgs.insert(0, {"content": render_catalog([]), "source": CATALOG_SOURCE})
            decision = dict(decision, messages=msgs)
        return nxt(agent, decision)

    ctx.on("agent/pre-step", pre_step)
    return provider


skills_plugin.inject = ["tools", "agents", "workspace"]
