"""hsh: asking the user, plan mode, and fetching web pages.

After dsh's ``tool-ask-user``, ``plan-mode`` and ``tool-web``
(docs/subsystems/user-questions.md, plan.md, web.md).

``ctx.questions`` is the user-questions seam, the counterpart of
``ctx.approval``: a host (a terminal, the web UI, a test) provides a
callable ``fn(execution, questions) -> answers``. ``questions`` is the
model's list (``id``, ``question``, optional ``header``, ``options``,
``multi_select``); ``answers`` maps each id to a string or a list of
strings (or is a list of ``{"id", "answer"}``). Returning None, or no
``ctx.questions`` at all, tells the model nobody answered.

- ``ask_user_tool`` (row ``tool-ask-user``): ``ask_user_question``.
- ``plan_mode`` (row ``plan-mode``): logged per-agent state ``plan/mode``
  (``{"active": bool}``, log-only). While active the ``plan`` section
  (config ``section``) joins the system prompt. Like upstream this is
  guidance, not enforcement; ``deny_tools`` (default none) adds a guard
  that refuses the named tools while planning. ``exit_plan_mode(plan)``
  shows the plan through ``ctx.questions``: approve leaves plan mode
  before the next step; anything else keeps planning and returns the
  user's feedback. A human message ``/plan`` enters plan mode, ``/plan
  off`` leaves it, ``/plan <text>`` enters and sends ``<text>``. Hosts
  call :func:`set_plan_mode`; the change is logged when the next turn
  starts.
- ``web_fetch_tool`` (row ``tool-web-fetch``, opt-in): GET one http(s)
  URL, HTML reduced to text, size-capped. Hosts that resolve to
  private, loopback or link-local addresses are refused unless
  ``allow_private`` is set, so a model cannot reach the local network
  or a cloud metadata endpoint. Redirects are followed by hand (at
  most 5) and each hop is checked again.
"""
from __future__ import annotations

import html.parser
import ipaddress
import re
import socket
import urllib.error
import urllib.parse
import urllib.request

from .session import register_event_type
from .tools import ToolDefinition

register_event_type("plan/mode")


# ----- ask_user_question -----------------------------------------------------

QUESTION_SCHEMA = {
    "type": "object",
    "properties": {"questions": {
        "type": "array", "minItems": 1, "maxItems": 4,
        "description": "Questions to ask the user before continuing.",
        "items": {"type": "object", "properties": {
            "id": {"type": "string", "description": "Stable id; echoed in the answer."},
            "question": {"type": "string"},
            "header": {"type": "string", "description": "Short heading, such as \"Confirm\"."},
            "options": {"type": "array", "description": "Choices. Put a recommended one "
                        "first and append \"(Recommended)\" to its label.",
                        "items": {"type": "object", "properties": {
                            "label": {"type": "string"},
                            "description": {"type": "string"}},
                            "required": ["label"]}},
            "multi_select": {"type": "boolean"}},
            "required": ["id", "question"]}}},
    "required": ["questions"], "additionalProperties": False,
}


def _normalize_answers(questions, raw):
    if raw is None:
        return None
    if isinstance(raw, list):
        raw = {a["id"]: a.get("answer") for a in raw}
    out = []
    for q in questions:
        a = raw.get(q["id"])
        if isinstance(a, (list, tuple)):
            a = [str(x) for x in a]
        elif a is not None:
            a = str(a)
        out.append({"id": q["id"], "question": q["question"], "answer": a})
    return out


def ask_user(ctx, exe, questions):
    """Ask through ``ctx.questions``; None when nobody can answer."""
    fn = ctx.get("questions")
    if fn is None:
        return None
    return _normalize_answers(questions, fn(exe, questions))


def _render_answers(a, v):
    if v is None:
        return "The user did not answer. Continue with your best judgement and say what you assumed."
    lines = []
    for x in v:
        ans = x["answer"]
        if ans is None or ans == []:
            ans = "(no answer)"
        elif isinstance(ans, list):
            ans = ", ".join(ans)
        lines.append(f"{x['id']}: {ans}")
    return "\n".join(lines)


def ask_user_tool(ctx, config=None):
    def run(args, exe):
        ids = [q["id"] for q in args["questions"]]
        if len(set(ids)) != len(ids):
            raise ValueError("question ids must be unique")
        return ask_user(ctx, exe, args["questions"])

    ctx.tools.register(ToolDefinition(
        "ask_user_question", "Ask the user a concise question when you need confirmation, "
        "a choice, or missing information before proceeding.", QUESTION_SCHEMA, run,
        render=_render_answers), ctx=ctx)


ask_user_tool.inject = ["tools"]


# ----- plan mode ---------------------------------------------------------------

PLAN_SECTION = (
    "# Plan mode\n\nPlan mode is on. Explore and read, but do not change files, run "
    "commands that modify state, or start long jobs. When you have a plan, call "
    "exit_plan_mode with the complete plan as markdown starting with a # heading. "
    "If the user asks you to keep planning, revise the plan and present it again.")


def plan_active(events):
    for e in reversed(events):
        if e["type"] == "plan/mode":
            return bool(e["data"]["active"])
    return False


def set_plan_mode(agent, active):
    """Select plan mode for ``agent``; logged when its next turn starts."""
    if plan_active(agent.session.events) == bool(active):
        agent._plan_pending = None
    else:
        agent._plan_pending = bool(active)


def plan_state(agent):
    out = {"active": plan_active(agent.session.events)}
    if getattr(agent, "_plan_pending", None) is not None:
        out["pending"] = agent._plan_pending
    return out


def plan_mode(ctx, config=None):
    cfg = config or {}
    section = cfg.get("section", PLAN_SECTION)
    if not isinstance(section, str) or not section.strip():
        raise ValueError("plan-mode needs a non-blank section")
    deny = set(cfg.get("deny_tools", ()))

    def agent_of(scope):
        return next((a for a in ctx.agents._live.values() if a.key is scope), None)

    def log(agent, active):
        agent.session.append("plan/mode", {"active": bool(active)})

    def pre_step(agent, decision, nxt):
        msgs = []
        for m in decision.get("messages") or []:
            text = m.get("content", "")
            hit = re.match(r"^/plan(?:\s+(.*))?$", text, re.S) if m.get("source") == "human" else None
            if not hit:
                msgs.append(m)
                continue
            rest = (hit.group(1) or "").strip()
            set_plan_mode(agent, rest != "off")
            if rest and rest != "off":
                msgs.append(dict(m, content=rest))
        decision = dict(decision, messages=msgs) if decision.get("messages") is not None else decision
        pending = getattr(agent, "_plan_pending", None)
        if pending is not None:
            agent._plan_pending = None
            if plan_active(agent.session.events) != pending:
                log(agent, pending)
        return nxt(agent, decision)

    def assemble(agent, prompt, nxt):
        if plan_active(agent.session.events):
            prompt = "\n\n".join(p for p in (prompt, section) if p)
        return nxt(agent, prompt)

    ctx.on("agent/pre-step", pre_step, prepend=True)
    ctx.on("system-prompt/assemble", assemble)

    if deny:
        def guard(exe):
            agent = agent_of(exe.scope)
            if agent is not None and exe.name in deny and plan_active(agent.session.events):
                return f"{exe.name} is not allowed in plan mode"
            return None

        ctx.tools.guard(guard, ctx=ctx)

    def exit_run(args, exe):
        agent = agent_of(exe.scope)
        if agent is None or not plan_active(agent.session.events):
            raise RuntimeError("exit_plan_mode works only in plan mode")
        answers = ask_user(ctx, exe, [{
            "id": "plan", "header": "Plan", "question": args["plan"],
            "options": [{"label": "Approve", "description": "Leave plan mode and carry out the plan."},
                        {"label": "Keep planning", "description": "Stay in plan mode; "
                         "your feedback goes back to the agent."}]}])
        answer = (answers or [{}])[0].get("answer")
        if answer is None:
            return {"approved": False, "feedback": "Nobody reviewed the plan. Stay in "
                    "plan mode; present it again later or ask a question."}
        if isinstance(answer, list):
            answer = ", ".join(answer)
        if answer.strip().lower() in ("approve", "approved", "yes", "y", "ok"):
            log(agent, False)
            return {"approved": True, "feedback": ""}
        return {"approved": False, "feedback": answer}

    ctx.tools.register(ToolDefinition(
        "exit_plan_mode", "Use only in plan mode. Present your plan for the user's review "
        "and, on approval, leave plan mode. If the user keeps planning, their feedback comes "
        "back in the result; revise and present again.",
        {"type": "object", "properties": {"plan": {"type": "string", "description":
                                                   "The complete plan, as markdown, starting "
                                                   "with a # heading that names it."}},
         "required": ["plan"], "additionalProperties": False}, exit_run,
        render=lambda a, v: "Plan approved. Plan mode is off; carry out the plan."
        if v["approved"] else f"Keep planning. Feedback: {v['feedback']}"), ctx=ctx)


plan_mode.inject = ["tools", "agents"]


# ----- web_fetch ---------------------------------------------------------------

class _Text(html.parser.HTMLParser):
    SKIP = {"script", "style", "noscript", "template", "svg", "head"}
    BLOCK = {"p", "div", "br", "li", "tr", "h1", "h2", "h3", "h4", "h5", "h6",
             "section", "article", "pre", "table", "ul", "ol", "header", "footer"}

    def __init__(self):
        super().__init__(convert_charrefs=True)
        self.out, self.skip, self.title = [], 0, None
        self._in_title = False

    def handle_starttag(self, tag, attrs):
        if tag in self.SKIP:
            self.skip += 1
        if tag == "title":
            self._in_title = True
        if tag in self.BLOCK:
            self.out.append("\n")

    def handle_endtag(self, tag):
        if tag in self.SKIP and self.skip:
            self.skip -= 1
        if tag == "title":
            self._in_title = False
        if tag in self.BLOCK:
            self.out.append("\n")

    def handle_data(self, data):
        if self._in_title and self.title is None:
            self.title = data.strip()
        if not self.skip:
            self.out.append(data)

    def text(self):
        t = "".join(self.out)
        t = re.sub(r"[ \t\r\f\v]+", " ", t)
        return re.sub(r"\n\s*\n\s*(\n\s*)+", "\n\n", t).strip()


def html_to_text(page):
    p = _Text()
    p.feed(page)
    p.close()
    return p.title, p.text()


def _check_host(url, allow_private):
    parts = urllib.parse.urlsplit(url)
    if parts.scheme not in ("http", "https") or not parts.hostname:
        raise ValueError("only http(s) URLs with a host can be fetched")
    if allow_private:
        return
    try:
        infos = socket.getaddrinfo(parts.hostname, parts.port or 443, proto=socket.IPPROTO_TCP)
    except socket.gaierror as exc:
        raise ValueError(f"cannot resolve {parts.hostname}: {exc}") from None
    for info in infos:
        ip = ipaddress.ip_address(info[4][0].split("%")[0])
        if (ip.is_private or ip.is_loopback or ip.is_link_local or ip.is_reserved
                or ip.is_multicast or ip.is_unspecified):
            raise ValueError(f"{parts.hostname} resolves to a private address ({ip})")


class _NoRedirect(urllib.request.HTTPRedirectHandler):
    def redirect_request(self, *a, **kw):
        return None


def web_fetch_tool(ctx, config=None):
    cfg = config or {}
    max_bytes = cfg.get("max_bytes", 2_000_000)
    max_chars = cfg.get("max_chars", 40_000)
    timeout = cfg.get("timeout_s", 30)
    allow_private = cfg.get("allow_private", False)
    opener = cfg.get("opener") or urllib.request.build_opener(_NoRedirect).open
    agent_name = cfg.get("user_agent", "hsh-web-fetch/1")

    def run(args, exe):
        url = args["url"]
        for _ in range(6):
            _check_host(url, allow_private)
            req = urllib.request.Request(url, headers={"User-Agent": agent_name,
                                                       "Accept": "text/html, text/plain, */*"})
            try:
                resp = opener(req, timeout=timeout)
            except urllib.error.HTTPError as exc:
                if exc.code in (301, 302, 303, 307, 308) and exc.headers.get("Location"):
                    url = urllib.parse.urljoin(url, exc.headers["Location"])
                    continue
                raise RuntimeError(f"HTTP {exc.code} for {url}") from None
            with resp:
                ctype = resp.headers.get("Content-Type", "")
                raw = resp.read(max_bytes + 1)
                status = getattr(resp, "status", 200)
            break
        else:
            raise RuntimeError("too many redirects")
        truncated = len(raw) > max_bytes
        m = re.search(r"charset=([\w.-]+)", ctype)
        try:
            body = raw[:max_bytes].decode(m.group(1) if m else "utf-8", "replace")
        except LookupError:
            body = raw[:max_bytes].decode("utf-8", "replace")
        title = None
        if "html" in ctype or (not ctype and body.lstrip()[:1] == "<"):
            title, body = html_to_text(body)
        elif not (ctype.startswith("text/") or "json" in ctype or "xml" in ctype):
            body = f"[{ctype or 'binary'} content, {len(raw)} bytes, not shown]"
        if len(body) > max_chars:
            body, truncated = body[:max_chars], True
        return {"url": url, "status": status, "content_type": ctype, "title": title,
                "text": body, "truncated": truncated}

    def render(a, v):
        head = f"URL: {v['url']}" + (f"\nTitle: {v['title']}" if v["title"] else "")
        tail = "\n[truncated]" if v["truncated"] else ""
        return f"{head}\n\n{v['text']}{tail}"

    ctx.tools.register(ToolDefinition(
        "web_fetch", "Fetch the content of a specific HTTP(S) URL and return it decoded to text.",
        {"type": "object", "properties": {"url": {"type": "string",
                                                  "description": "The HTTP(S) URL to fetch."}},
         "required": ["url"], "additionalProperties": False}, run, render=render), ctx=ctx)


web_fetch_tool.inject = ["tools"]
