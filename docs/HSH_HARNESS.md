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
