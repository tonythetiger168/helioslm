# hsh — the HeliosLM harness after DeepSeek Harness

`helioslm_v5/hsh/` rebuilds the agent harness on the design of
[DeepSeek Harness](https://github.com/deepseek-ai/deepseek-harness) (`dsh`,
MIT, developer preview released 2026-08-13). dsh is a TypeScript/Node
project; hsh is pure Python, written from dsh's design documents
(`docs/architecture.md`, `docs/cordis-primer.md`,
`docs/tool-execution-pipeline.md`, `docs/glossary.md`), not ported from its
source. The older `harness/harness_core.py` (a 50-line Cordis-inspired
service map) stays as it is until hsh replaces its callers.

## What dsh does that hsh copies

| dsh idea | What it means | hsh phase |
|---|---|---|
| Everything is a plugin (Cordis) | Model adapter, tool registry, session log and agent loop are all plugins; there is no privileged core | H1 |
| `inject` | A plugin names the services it needs and waits for them | H1 |
| Reversible effects | Every registration has a disposer; unloading a plugin unwinds it | H1 |
| Typed events with dispatch modes | `emit` / `waterfall` / `parallel` / `serial` / `bail`, declared per event | H1 |
| Agent scopes | `agent.ctx` registrations are visible to, and live as long as, one agent | H1 (kernel), H3 (tool shadowing and restriction) |
| Append-only session log | "Model-visible means logged": every request can be rebuilt from the log; fork, resume and replay derive from it | H2 |
| Tool execution pipeline | `tools/pre-execute` → guards → approval → `tools/execute` → `tools/post-execute` → finalize → `tools/result` | H3 |
| LLM adapter seam, turn/step loop | `ctx.llm` adapters; `agent/pre-step`, `agent/request`, retries recorded as attempts | H4 |
| Presets | `standard`, `minimal` (bash + `str_replace_editor`), `code` (the model writes one program against the tools), `creator` (inspection) | H5 |
| Profiles and patches | Declarative config rows, layered and patched by id | H5 |
| Compaction, subagents, hooks bridge, `AGENTS.md` | Context compaction; child agents; running an existing `hooks.json` | H6 |

Not copied: the Web UI, the Electron app, the Landlock/Seatbelt native
sandbox (hsh keeps HeliosLM's TrustGate and path guards and wraps
subprocesses with resource limits instead), and provider SDKs beyond an
OpenAI-compatible adapter and the local HeliosLM model.

## Differences on purpose

- **Synchronous.** One CPU process, deterministic order. `parallel` calls
  every listener and raises the failures together instead of running
  them concurrently.
- **Oracle tests.** Every phase pins its contract with tests, as the rest
  of the repo does. In H2 the main oracle is that a request rebuilt from
  the log is byte-identical to the request that was sent.
- **知止 hooks.** HeliosLM's decision layer (TrustGate, the stop
  calibration of AC5) plugs in as a tool guard and a turn-stopping
  listener, not as special cases in the loop.

## H1: the kernel (`hsh/kernel.py`)

- `Context.provide(key, value)` / `ctx.<key>`; providing a key twice raises.
- `ctx.plugin(p, config)` mounts a callable, an object with `apply`, or a
  `Service` subclass. A plugin with unmet `inject` keys stays `pending`,
  activates when they appear, and goes back to `pending` (unwound) when
  one disappears.
- `ctx.effect(dispose)` records a disposer on the plugin; disposal is
  last-in first-out and runs each disposer once. Services, listeners and
  child plugins are all effects.
- A failing `apply` rolls back what it registered and leaves the fork
  `failed`. During automatic activation the error is recorded on the fork
  and in `ctx.errors` instead of being raised into an unrelated caller.
- `ctx.declare(event, mode)`, then dispatch with the method of that mode;
  any other method, or an undeclared event, raises `EventModeError`.
- `ctx.scoped(key)` tags registrations and dispatches with `key`.

Tests: `helioslm_v5/tests/test_hsh_kernel.py` (16).

## H2: the session log (`hsh/session.py`)

- The log is the source of truth; `derive_messages(events)` is the model
  history. Surface events: `system/message`, `user/message`,
  `assistant/message`, `tool/result`. Log-only: `turn/*`, `step/*`,
  `assistant/attempt`, `tool/call`, `request/header`, `session/end-seed`.
- **Model-visible means logged.** `build_request(events)` is the only way
  to build a request; `rebuild_requests(events)` replays the log and
  returns the request every attempt saw. A loop that sneaks anything into
  a request without logging it fails the hash comparison.
- `keep_thinking` (OS2): `none`, `turn` (default), `all`.
- `usage_summary(events)` (OS1): tokens per turn, attempts counted apart.
- `Session.repair()` seals an interrupted tail; `Session.fork(after_turn)`
  starts a child at a turn boundary.

Tests: `helioslm_v5/tests/test_hsh_session.py` (15).

## H3: tools (`hsh/tools.py`, `hsh/schema.py`)

- `ToolDefinition(name, description, parameters, execute, output=None,
  render=None, project_content=None, finalize_content=None)`; the body
  gets validated, copied arguments and returns a JSON value.
- Pipeline: arguments -> `tools/pre-execute` -> approval (for `ask`) ->
  guards -> `tools/execute` -> `project_content` -> `tools/post-execute`
  -> `finalize_content` -> `tools/result`. Error codes: `unknown-tool`,
  `invalid-arguments`, `denied`, `tool-error`, `invalid-output`,
  `pipeline-error`.
- `register(defn, ctx=agent_ctx)` scopes a tool to one agent (shadowing a
  global of the same name); `restrict(agent_ctx, allow=..., deny=...)`
  filters the globals that agent sees; `guard(fn, ctx=...)` denies by
  returning a reason. All three are effects of the registering plugin.
- `ToolResult.session_data(turn, step, call_id)` is the `tool/result`
  payload for the log.

Tests: `helioslm_v5/tests/test_hsh_tools.py` (21).

## H4: model seam and agent loop (`hsh/llm.py`, `hsh/loop.py`)

- `ctx.llm.register(provider, adapter)`; `ctx.llm.call(request)` runs the
  `llm/stream` waterfall around the adapter. The request header's
  `config.provider` picks the adapter.
- `ctx.agents.create(session_id, config, system_prompt=..., setup=...)`
  returns an `Agent`; `agent.send(text)` runs turns until the inbox is
  empty; `agent.inject(text)` adds context to the next request.
- Turn flow and events as in dsh: `agent/pre-step`, `agent/request`,
  `system-prompt/assemble`, `agent/request-error`, `agent/turn-stopping`,
  `agent/status`, `agent/created`.
- Oracle: `agent.sent` (hashes of every request sent) equals
  `rebuild_requests(agent.session.events)`.

A real model in five lines:

```python
from helioslm_v5.hsh import *
ctx = Context()
for p in (Sessions, Tools, LLM, Agents): ctx.plugin(p)
ctx.llm.register("deepseek", OpenAICompatAdapter("https://api.deepseek.com", "DEEPSEEK_API_KEY"))
agent = ctx.agents.create("demo", {"provider": "deepseek", "model": "deepseek-v4-flash"})
print(agent.send("hello"))
```

Tests: `helioslm_v5/tests/test_hsh_loop.py` (12).

## H5: presets and built-in tools (`hsh/presets.py`, `hsh/builtin.py`)

| Preset | Tools the model sees | Note |
|---|---|---|
| `minimal` | `bash`, `str_replace_editor` | dsh's minimal mode |
| `standard` | `bash`, `read`, `write`, `edit`, `glob`, `grep`, `todo_write` | |
| `code` | `run_code` only | PTC: the program calls the standard tools |
| `creator` | standard + `inspect` | plugins, services, events, tools, log, preset |

```python
ctx.plugin(Workspace, {"root": "/path/to/repo"})
agent = create_agent(ctx, "fix-1", config, "minimal",
                     patches=[[{"replace": "persona", "config": {"prefix": "..."}}]])
```

`bash` and `run_code` confine processes (workspace cwd, scrubbed
environment, timeout, rlimits) but are not a security sandbox; dsh uses
Landlock/Seatbelt for that. Keep HeliosLM's TrustGate as a guard when the
model is untrusted.

Tests: `helioslm_v5/tests/test_hsh_presets.py` (10).

## H6: compaction, subagents, instructions, hooks

- **Compaction** (`hsh/compaction.py`): the whole operation is log events;
  `context/shadow` removes the old range from the derived history and a
  `user/message` (source `compaction`) carries the summary. Automatic
  compaction runs only before a turn's first request, because the summary
  is appended at the end of the log; a long tool loop inside one turn is
  compacted at the start of the next.
- **Subagents** (`subagent` tool): a child agent on `<parent>/subN` with
  its own preset; the parent's scoped tools are not inherited; depth is
  limited by `max_depth` (default 1).
- **AGENTS.md** (`agent-instructions` row): the first of `AGENTS.md`,
  `CLAUDE.md` found in the workspace root is appended to the system
  prompt, capped at `max_bytes`.
- **Claude Code hooks** (`claude-hooks` row, config `{"path": "hooks.json"}`
  or `{"hooks": {...}}`): PreToolUse, PostToolUse, UserPromptSubmit and
  Stop with matchers, exit code 2 and the JSON decision fields. Argument
  rewriting is not supported: the logged call runs as logged.

Not done in H6 (and why; MCP came in H8, the web UI in H12): desktop app, Landlock-class sandboxing,
goals and schedules.

Tests: `helioslm_v5/tests/test_hsh_extras.py` (8). All hsh tests: 82.

## H7: agent skills (`hsh/skills.py`)

After dsh's skill family (`docs/subsystems/skills.md`, upstream master of
2026-10-23).

- A skill is `<root>/<name>/SKILL.md` (a bundle; files next to it are its
  resources) or a flat `<root>/<name>.md`, with `name` and `description`
  in a frontmatter block. Optional keys: `when_to_use`,
  `disable-model-invocation: true` (only a human can start it with
  `/name`), `user-invocable: false` (only the model can load it). Other
  keys (such as this repo's `type` and `guard`) are kept as metadata.
- Roots, best first: `.hsh/skills`, `.agents/skills`, `.claude/skills` in
  the workspace, then the row's `dirs`, then `~/.hsh/skills` and
  `~/.agents/skills`. A lower rank wins a duplicate name; a skill
  registered on an agent's scoped context beats the global layer.
- `ctx.skills` (`Skills` service, optional): `register_provider`,
  `register` (runtime skills), `list`, `get`. Without it, each agent's
  `skills` row keeps a private registry.
- The `skills` row (in `standard`, `code`, `creator`) adds the `skill`
  tool and, at the start of each turn, logs the catalog (names and
  descriptions only) as a `user/message` with source `skill-catalog`
  when it differs from the newest visible one. The catalog is in the
  log, so requests still rebuild byte for byte. `/name rest` from a
  human expands to the skill's instructions followed by `rest`.
- Bodies are re-read on every load, so editing a skill takes effect on
  the next call without a restart.

Tests: `helioslm_v5/tests/test_hsh_skills.py` (6).

## H8: MCP client (`hsh/mcp.py`)

After dsh's `mcp-client` / `mcp-resources` (`docs/subsystems/mcp.md`), in
the standard library only (no MCP SDK).

- Transports: stdio (newline-delimited JSON-RPC to a child process) and
  Streamable HTTP (POST, JSON or SSE reply, `Mcp-Session-Id` kept).
- The `mcp` row takes `{"servers": {name: {"command", "args", "env",
  "cwd"} | {"url", "headers"}}}` or `{"path": ".mcp.json"}` (Claude Code's
  `mcpServers` file works as is). Tools appear as `mcp__<server>__<tool>`;
  the namespace is the configured name, never the server's own.
- Child processes get a scrubbed environment (KEY/PASSWORD/SECRET/TOKEN
  names dropped); `${VAR}` in `env` and `headers` expands from the host,
  so a secret reaches only the server that names it.
- Server instructions join the system prompt; `list_mcp_resources` and
  `read_mcp_resource` read resources. A server that fails to start is
  recorded in `ctx.errors` and adds no tools (`fail_on_error` raises).
- Not supported (as upstream): prompts, elicitation, sampling,
  subscriptions. Images reach the model as a text note; the full MCP
  result is the tool result's `value`.

Tests: `helioslm_v5/tests/test_hsh_mcp.py` (5).

## H9: background jobs (`bash` with `{"jobs": true}`)

After dsh's `tool-bash` + `tool-jobs`. `standard` turns it on; `minimal`
keeps the plain one-shot `bash`.

- `run_in_background: true` starts the command and returns a job id at
  once; a foreground command still running at its timeout moves to the
  background instead of being killed.
- `job_list`, `job_output` (output since the last read; `wait` blocks up
  to `timeout_s`), `job_kill` (kills the process group).
- Output goes to a spill file outside the workspace; disposing the row
  kills running jobs and removes the files.

Tests: `test_bash_jobs` in `helioslm_v5/tests/test_hsh_presets.py`.

## H10: asking the user, plan mode, web_fetch (`hsh/interaction.py`)

After dsh's `tool-ask-user`, `plan-mode` and `tool-web`.

- `ctx.questions` is the user-questions seam (like `ctx.approval`): the
  host provides `fn(execution, questions) -> answers`. Without it the
  model is told nobody answered.
- `ask_user_question` (row `tool-ask-user`, in `standard`): up to four
  questions with optional options and multi-select.
- Plan mode (row `plan-mode`, in `standard`): logged state `plan/mode`;
  while on, the plan section joins the system prompt. `/plan`,
  `/plan off`, `/plan <text>` from a human; `set_plan_mode(agent, on)`
  from a host (logged when the next turn starts). `exit_plan_mode`
  shows the plan through `ctx.questions`: "Approve" leaves plan mode
  before the next step; any other answer is feedback. Guidance only, as
  upstream; `deny_tools` adds a guard if you want enforcement.
- `web_fetch` (row `tool-web-fetch`, opt-in): one http(s) GET, HTML to
  text, size caps; hosts that resolve to private, loopback or link-local
  addresses are refused (also after each redirect) unless
  `allow_private`. `web_search` is not included: it needs a search
  provider.

Tests: `helioslm_v5/tests/test_hsh_interaction.py` (4).

## H11: streaming, HTTP errors and request retry (`hsh/llm.py`)

- `OpenAICompatAdapter(..., stream=True)` reads server-sent events; text
  and reasoning deltas go to `on_chunk` as they arrive and the loop
  re-emits them as `agent/assistant-stream` (`emit`, live only: the
  log keeps the merged stream on `assistant/message`). Tool-call
  fragments are joined by index; usage includes DeepSeek's
  `prompt_cache_hit_tokens` as `cached_input_tokens`.
- HTTP failures raise `LLMError` with `status`, `transient` (408, 409,
  425, 429, 5xx, dropped connections) and the server's `retry_after`.
- Row `request-retry` (in every preset): on a transient failure it waits
  (exponential backoff with jitter, at least `Retry-After`) and retries
  in the same step, up to `max_retries` (4). Each failed attempt is an
  `assistant/attempt` in the log, so the rebuild oracle still holds and
  `usage_summary` counts them apart.
- `finish_reason` (such as `length`) is kept on the adapter output.

Not copied: dsh's `dsh_session_log` / `dsh_plugin_packages` request
fields and `x-deepseek-harness-*` headers, which send the session log
(code, tool output) and an install id to the API on every request.
hsh sends the chat request only.

Tests: `helioslm_v5/tests/test_hsh_llm.py` (4).

## H12: web UI (`hsh/web.py`, `hsh/webui/`)

Modelled on the DeepSeek Harness web app and written fresh: standard
library server plus one static page (HTML, CSS, JS; no build step, no
third-party code). It replaces `helioslm_v5/harness/web_ui_v2.py` as the
harness UI (that file stays only for the legacy plugin harness).

```
python -m helioslm_v5.hsh.web --workspace /path/to/repo --open        # DeepSeek (DEEPSEEK_API_KEY)
python -m helioslm_v5.hsh.web --workspace . --openai-base http://127.0.0.1:8080/v1 --openai-models my-model
python -m helioslm_v5.hsh.web --workspace . --demo                    # no model, to try the UI
```

- Layout as upstream: sidebar (brand, New Session, sessions of this
  workspace with relative times), centred composer (preset, permission,
  plan and model chips; round send button that becomes stop), streamed
  replies with collapsible thinking, tool cards with input and output,
  todo lists, settings dialog (General, Models, Skills, MCP servers,
  Agent presets), light and dark themes, phone layout.
- Sessions are hsh agents saved as JSONL logs under `--data-dir`
  (`~/.hsh/web`), resumed after a restart; a turn cut off mid-way is
  sealed as `interrupted`.
- Questions, plan review and tool approvals appear as cards
  (`ctx.questions`, `ctx.approval`). Permission modes: `auto`, `ask`
  (approve each command, edit and MCP call), `read-only`.
- `/` lists skills and `/plan`; Enter sends, Shift+Enter is a new line,
  Ctrl/Cmd+K starts a new session.
- Keys come from environment variables only; the page shows whether each
  is set and never accepts one.
- Loopback only by default; non-loopback `Host` and cross-origin or
  non-JSON POSTs are refused; strict CSP (`'self'` only).
- `Agent.cancel()` (new in `loop.py`) stops at the next step boundary;
  the turn ends with reason `cancelled`.

Tests: `helioslm_v5/tests/test_hsh_web.py` (4).

## H13: network egress audit (`hsh/egress.py`, v5.181)

A local model that runs commands should reach the network only where the
operator allows it, and every attempt should be visible. Mount the
`Egress` service (`ctx.plugin(Egress, {"allow": ["*.pypi.org"]})`, the
`egress` preset row, or `web --egress deny --egress-allow HOST`) and the
`bash` and `run_code` children get:

- a proxy on 127.0.0.1 in `HTTP(S)_PROXY` / `ALL_PROXY`: `CONNECT` and
  plain HTTP to an allowed host pass, anything else gets 403;
- a Python audit hook on `socket.getaddrinfo` / `socket.connect`
  (`sitecustomize` on `PYTHONPATH` for Python started from `bash`,
  prepended to the `run_code` child): a host off the allow list is
  refused before DNS in `deny` mode.

`audit` mode records and lets through. Loopback is always allowed.
Each call's attempts come back as `egress` in its value and a
`[network blocked by the harness: host:port]` line the model reads;
`ctx.egress.report()` sums the session per host.

Without `isolate` this is an audit trail, not a sandbox: a non-Python
program that ignores the proxy variables can still connect.
`{"isolate": true}` runs `bash` under `unshare -rn` (Linux user
namespaces) so the child has loopback only. T158 pins it without leaving
the machine.
