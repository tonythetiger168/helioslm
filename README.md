---
license: apache-2.0
tags:
- helioslm
- rlcd
- calibrated-decisions
- decision-layer
- agentic-rl
- deepseek-v3
- reference-implementation
pipeline_tag: text-generation
---

# HeliosLM（羲和）— Verifiable DeepSeek-V3/V4 and Kimi-K3-Style LLM Architectures in Pure PyTorch

> 中文名「羲和」：中國神話中駕日車的太陽神，對應希臘的 Helios。程式套件 `helioslm_v5`；2026-10-24 曾短暫改名 Evastar（v5.133–v5.163），舊的 `import evastar`、`EvastarV5` 類別名和 `EVASTAR_*` 環境變數仍可使用。標語「知止」（《大學》「知止而后有定」）：可驗證的機制，加上知道自己何時可能出錯、該停下來的決策層。

**Every mechanism ships with an oracle test.** HeliosLM is a hackable, CPU-runnable reference for frontier LLM designs — MLA, sigmoid-gated MoE, hybrid linear attention, compressed sparse attention — plus a calibrated decision layer that measures whether a model knows when it is wrong.

A from-scratch PyTorch reference implementation of a modern LLM stack: MLA attention with weight absorption, sigmoid-gated MoE with auxiliary-loss-free load balancing, hybrid linear attention, speculative decoding, FP8 training, a DualPipe schedule simulation, a vLLM-style serving engine, a **verifiable agent layer** (strict tool schema, bitwise-replay oracle), DSA sparse attention, Mooncake-style prefill/decode disaggregation, and tool-tuned checkpoints. Built to be **read, modified, and verified** — every core path is unit-tested and many are checked with bitwise-equivalence tests. Everything runs on CPU.

> **One-liner:** If you want to understand (or hack on) how DeepSeek-V3/K3-class models actually work — without needing a GPU cluster first — this repo is for you.
>
> **Also:** the open-source reference for the **RLCD "decision layer"** paradigm — calibrated confidence on acting models, verified across three scales and on the 7,193-instance RLCDAlignBench. Working paper: `docs/paper_draft_2026-10-03.tex` ([repo](https://github.com/tonythetiger168/helioslm/blob/main/docs/paper_draft_2026-10-03.tex) | [HF](https://huggingface.co/chienhsinlin/helioslm/tree/main/docs)).

[![CI](https://github.com/tonythetiger168/helioslm/actions/workflows/ci.yml/badge.svg)](https://github.com/tonythetiger168/helioslm/actions)
![Tests](https://img.shields.io/badge/tests-300%2B%20unit%20%2B%20integration%20%2B%20oracle-brightgreen)
![License](https://img.shields.io/badge/license-Apache%202.0-blue)
![PyTorch](https://img.shields.io/badge/framework-PyTorch%20(pure)-ee4c2c)

![Demo](https://raw.githubusercontent.com/tonythetiger168/helioslm/main/docs/demo.gif)

## Who is this for?

| You are... | What HeliosLM gives you |
|---|---|
| **A learner** who wants to understand MLA, MoE routing, DualPipe, speculative decoding | Annotated, review-hardened PyTorch with 120+ tests (T11-T75) that act as executable documentation |
| **A researcher** who wants a stack to modify, ablate, and extend quickly | Single-process, CPU-iterable training + serving code — change one file, run one test |
| **A practitioner** evaluating serving/quantization techniques | vLLM-style paged engine, GPTQ/AWQ/FP8/MXFP4 quantization, MTP speculative decoding — all inspectable |

**Honest positioning:** this is a correctness-focused reference implementation, not a throughput-optimized production engine (see [Known Limitations](#known-limitations)).

## Using the harness (hsh)

`helioslm_v5/hsh/` is the agent harness: it runs a model in a loop with tools,
skills, MCP servers, plan mode and a session log, and it has a local web
UI. It follows the design of the open-source DeepSeek Harness and is
pure Python standard library, so it needs nothing beyond Python 3.11+
(PyTorch is only needed for the model code, not for the harness). Design
notes and every contract it keeps: [docs/HSH_HARNESS.md](docs/HSH_HARNESS.md).

### 1. Install

```bash
git clone https://github.com/tonythetiger168/helioslm
cd helioslm
python -m helioslm_v5.hsh.web --help        # no pip install needed for the harness
```

### 2. Connect a model

| Model | How |
|---|---|
| DeepSeek API | `export DEEPSEEK_API_KEY=sk-...` (PowerShell: `$env:DEEPSEEK_API_KEY="sk-..."`). Models `deepseek-v4-flash`, `deepseek-v4-pro`. |
| A local model or any OpenAI-compatible server (llama.cpp, vLLM, SGLang) | Start it, then pass `--openai-base http://127.0.0.1:8080/v1 --openai-models <model-id>`; add `--openai-key-env NAME` if it needs a key. For llama.cpp: `llama-server -m model.gguf --port 8080`. |
| Several providers | `--config providers.json` with `{"providers": {"name": {"label", "base_url", "api_key_env", "models": [...], "stream": true}}}` |
| No model (to try the UI) | `--demo` |

Keys are read from environment variables only. The harness never writes
a key to a file or a session log, and the web UI never asks for one.

### 3. The web UI

```bash
python -m helioslm_v5.hsh.web --workspace /path/to/your/repo --open
# then open http://127.0.0.1:3080
```

- **New Session** (or Ctrl/Cmd+K), type what you want, press Enter
  (Shift+Enter for a new line). Replies stream in; click "Thought for…"
  to see the model's thinking and a tool card to see its input and
  output.
- The chips under the input box choose the **preset** (which tools the
  agent gets, see below), the **permission mode** (`Workspace write`
  runs tools, `Ask first` asks you before each command, edit or MCP
  call, `Read only` refuses them), **Plan** (the agent plans first and
  asks you to approve the plan), and the **model**. The send button
  becomes a stop button while the agent works.
- Questions from the agent, plan reviews and approvals appear as cards
  above the input box.
- Type `/` for skills and commands (`/plan`, `/plan off`, `/<skill>`).
- **Settings** (bottom left) shows the models and whether each key is
  set, the skills found, MCP servers and presets, and switches between
  light and dark.
- Sessions are saved as JSONL logs under `~/.hsh/web` (change with
  `--data-dir`) and come back after a restart.
- Other options: `--port`, `--preset minimal|standard|code|creator`,
  `--skills DIR` (extra skill folder, repeatable), `--mcp .mcp.json`,
  `--egress deny --egress-allow HOST` (block and log network access from
  the agent's commands except to the hosts you allow; `audit` only logs).
  The server listens on 127.0.0.1 only.

### 4. From Python

```python
from helioslm_v5.hsh import (Context, Sessions, Tools, LLM, Agents, Workspace,
                         OpenAICompatAdapter, create_agent)

ctx = Context()
for p in (Sessions, Tools, LLM, Agents):
    ctx.plugin(p)
ctx.plugin(Workspace, {"root": "/path/to/your/repo"})
ctx.llm.register("deepseek", OpenAICompatAdapter(
    "https://api.deepseek.com", "DEEPSEEK_API_KEY", stream=True))

# optional: print the reply as it streams; answer questions and approvals
ctx.on("agent/assistant-stream", lambda agent, chunk: print(chunk["text"], end=""))
ctx.provide("questions", lambda exe, qs: {q["id"]: input(q["question"] + " ") for q in qs})
ctx.provide("approval", lambda exe, question: input(question + " [y/N] ") == "y")

agent = create_agent(ctx, "fix-1", {"provider": "deepseek", "model": "deepseek-v4-flash"},
                     "standard")
print(agent.send("Run the tests and fix the first failure."))
agent.session.save("fix-1.jsonl")     # the whole session, replayable
```

- A local model server: `OpenAICompatAdapter("http://127.0.0.1:8080/v1", None)`.
- A HeliosLM checkpoint or any `generate(prompt) -> str` function:
  `ctx.llm.register("local", TextModelAdapter(generate))`.
- `agent.followup(text)` queues a message, `agent.inject(text)` adds
  context to the next request, `agent.cancel()` stops at the next step.
- Every request can be rebuilt from the log:
  `rebuild_requests(agent.session.events)`; token use per turn:
  `usage_summary(agent.session.events)`.

### 5. Presets and patches

| Preset | Tools the model sees |
|---|---|
| `minimal` | `bash`, `str_replace_editor` |
| `standard` | `bash` (with background jobs: `job_list`, `job_output`, `job_kill`), `read`, `write`, `edit`, `glob`, `grep`, `todo_write`, `subagent`, `skill`, `ask_user_question`, `exit_plan_mode`; plus `AGENTS.md` in the prompt, the skill catalog, plan mode, compaction and request retry |
| `code` | `standard`, but the model writes one Python program (`run_code`) that calls the tools |
| `creator` | `standard` plus `inspect` (look at the running harness) |

A preset is a list of rows; patches change it without copying it:

```python
agent = create_agent(ctx, "a", config, "standard", patches=[[
    {"replace": "persona", "config": {"prefix": "You are a careful reviewer."}},
    {"insert": [{"id": "web", "name": "tool-web-fetch"}]},     # opt-in web_fetch
    {"replace": "plan-mode", "config": {"deny_tools": ["write", "edit", "bash"]}},
    {"remove": "tool-subagent"},
]])
```

Patches can also live in a JSON file: `load_patch("my-profile.json")`.

### 6. Add an agent skill

A skill is a folder with a `SKILL.md`. The agent sees each skill's name
and description, loads the full instructions with the `skill` tool when a
task matches, and you can start one yourself with `/<name>`.

1. Create the folder in your workspace:

   ```
   .hsh/skills/release-notes/SKILL.md
   ```

2. Write the frontmatter and the instructions:

   ```markdown
   ---
   name: release-notes
   description: Use when the user asks for release notes or a changelog entry.
   ---
   # Release notes

   1. Run `git log --oneline $(git describe --tags --abbrev=0)..HEAD`.
   2. Group the commits by area (model, harness, data, docs).
   3. Write one line per change, newest first, in the CHANGELOG style.
   ```

   `name` must match the folder name and be kebab-case; `description`
   is what the model uses to decide when the skill applies, so say when
   to use it. Scripts or reference files can sit next to `SKILL.md`; the
   agent is told the folder path and reads them when the instructions
   say so.

3. Use it: start a new session (the catalog is refreshed at the start of
   every turn, and edits to a skill take effect on its next use), then
   either ask for the task in plain words or type `/release-notes`.
   Settings → Skills in the web UI lists what was found.

Where skills are looked for (the first one wins a duplicate name):
`.hsh/skills`, `.agents/skills`, `.claude/skills` in the workspace, then
extra folders (`--skills DIR`, or `{"dirs": [...]}` on the `skills` row),
then `~/.hsh/skills` and `~/.agents/skills`. A single file
`<name>.md` works too. This repository's own [`skills/`](skills/) folder
loads with `--skills skills`.

Optional frontmatter keys: `disable-model-invocation: true` (only you can
start it, with `/name`), `user-invocable: false` (only the agent can load
it), `when_to_use`. From Python you can also register a skill at runtime:
`ctx.plugin(Skills)` then `ctx.skills.register({"name": ..., "description": ..., "content": ...})`.

### 7. MCP servers

Put a Claude-style `.mcp.json` in the workspace and start with
`--mcp .mcp.json` (or add the `mcp` row with `{"path": ".mcp.json"}`):

```json
{"mcpServers": {
  "github": {"command": "npx", "args": ["-y", "@modelcontextprotocol/server-github"],
             "env": {"GITHUB_TOKEN": "${GITHUB_TOKEN}"}},
  "docs":   {"url": "https://example.com/mcp", "headers": {"Authorization": "Bearer ${DOCS_TOKEN}"}}
}}
```

Tools appear as `mcp__github__create_issue` and so on. `${VAR}` is filled
from your environment for that server only; other secrets in your
environment are not passed to the server process.

### 8. More

- `AGENTS.md` (or `CLAUDE.md`) in the workspace root is added to the
  system prompt.
- An existing Claude Code `hooks.json` runs with the `claude-hooks` row
  (`{"path": "hooks.json"}`): PreToolUse, PostToolUse, UserPromptSubmit,
  Stop.
- Long commands: in `standard`, `bash` takes `run_in_background`, and a
  command that outlives its timeout keeps running as a job.
- Transient model errors (429, 5xx, dropped connections) are retried
  with backoff; each attempt is in the session log.

## Legacy plugin harness (v1.0-v1.22, 23 releases)

The `harness/` plugin context below predates hsh and is kept for its
existing callers; new agent work uses hsh (above).

All plugins register as `ctx.<name>.<op>` with per-call audit effects.
Install extras as needed: `pip install openpyxl python-docx python-pptx
pytesseract pyserial cryptography psycopg2 boto3 google-cloud-storage
azure-storage-blob pyttsx3 speech_recognition ultralytics transformers`

### Core (always available)
| Plugin | Namespace | Key ops |
|---|---|---|
| ModelPlugin | `ctx.model.*` | mid (360M HeliosLM), qwen (Qwen3-0.6B) |
| ToolPlugin | `ctx.tools.*` | registry, impls (calc/str/file_read/file_write/finish) |
| SessionPlugin | `ctx.session.*` | new (append-only, replay-verified) |
| DecisionPlugin | `ctx.decision.*` | grounding, trust, prefetcher |
| LoopPlugin | `ctx.loop.*` | make (AgentLoop factory) |
| StatePlugin | `ctx.state` | dict-backed, cross-plugin persistence |
| PresetPlugin | `ctx.preset` | minimal / full wiring |

### Agent (v1.8)
| Plugin | Namespace | Key ops |
|---|---|---|
| MemoryPlugin | `ctx.memory.*` | remember, recall (ExperienceStore) |
| TerminalPlugin | `ctx.terminal.*` | run (TrustGate-gated) |
| FetchPlugin | `ctx.fetch.*` | get (HTTP) |
| FilesystemPlugin | `ctx.fs.*` | read, write (path-escape blocked) |
| TimePlugin | `ctx.time.*` | now, utc |

### Common (v1.9)
| Plugin | Namespace | Key ops |
|---|---|---|
| SearchPlugin | `ctx.search.*` | query (local corpus) |
| WebSearchPlugin | `ctx.web.*` | search (Tavily/Exa stub) |
| PDFPlugin | `ctx.pdf.*` | extract (pypdf) |
| SQLitePlugin | `ctx.sqlite.*` | query (constrained db) |
| TemplatePlugin | `ctx.plugin.*` | scaffold (new-plugin template) |

### Services (v1.10)
| Plugin | Namespace | Key ops |
|---|---|---|
| GitHubPlugin | `ctx.github.*` | get_repo, list_issues, create_issue (TrustGate-gated) |
| PostgresPlugin | `ctx.postgres.*` | query (psycopg2/pg8000) |

### Office (v1.11)
| Plugin | Namespace | Key ops |
|---|---|---|
| ExcelPlugin | `ctx.excel.*` | read, write (openpyxl) |
| DocxPlugin | `ctx.docx.*` | read, write (python-docx) |
| PptxPlugin | `ctx.pptx.*` | write (python-pptx) |
| YouTubePlugin | `ctx.youtube.*` | transcript (youtube-transcript-api) |

### Niche (v1.12)
| Plugin | Namespace | Key ops |
|---|---|---|
| MCPWizardPlugin | `ctx.mcp.*` | wrap (MCP server) |
| TmuxPlugin | `ctx.tmux.*` | new, list, send |
| EverythingPlugin | `ctx.everything.*` | search (local file content) |

### Media (v1.13)
| Plugin | Namespace | Key ops |
|---|---|---|
| VideoPlugin | `ctx.video.*` | generate (Runway/Pika stub) |
| AudioPlugin | `ctx.audio.*` | tts (pyttsx3/external), stt |
| NanoVideoPlugin | `ctx.nanovideo.*` | from_images (ffmpeg slideshow) |

### Hardware (v1.14, v1.16, v1.18, v1.21)
| Plugin | Namespace | Key ops |
|---|---|---|
| BlenderPlugin | `ctx.blender.*` | gen, run (headless blender) |
| OmiPlugin | `ctx.omi.*` | connect, transcribe (BLE wearable) |
| RobotControlPlugin | `ctx.robot.*` | move_base, move_arm, gripper (mock/ROS2, TrustGate-gated) |
| MultiRobotPlugin | `ctx.fleet.*` | register, allocate, formation |
| HardwarePlugin | `ctx.hw.*` | arduino_write, gpio_write, i2c_write |

### Science (v1.15, v1.20-v1.22)
| Plugin | Namespace | Key ops |
|---|---|---|
| BenchmarkPlugin | `ctx.benchmark.*` | alignbench, api_probe |
| ReportPlugin | `ctx.report.*` | gen (markdown from benchmarks) |
| BenchmarkV2Plugin | `ctx.bench.*` | run_suite, trend |
| DNAPlugin | `ctx.dna.*` | validate, reverse_complement, gc, transcribe, translate, align, pcr_primers, restriction_sites |
| ProteinPlugin | `ctx.protein.*` | validate, mol_weight, hydrophobicity, fold_toy |
| ChemPlugin | `ctx.chem.*` | formula_weight, ph, bond_energy |
| MathV2Plugin | `ctx.linalg.*`, `ctx.stats.*`, `ctx.signal.*` | matmul, summary, fft_magnitudes |

### Collaboration (v1.17, v1.19)
| Plugin | Namespace | Key ops |
|---|---|---|
| AgentSwarmPlugin | `ctx.swarm.*` | spawn, delegate, aggregate, blackboard |
| VoiceDialogPlugin | `ctx.voice.*` | turn (STT->LLM->TTS with barge-in) |
| TerminalUIPlugin | `ctx.ui.*` | progress, menu |

### Autonomy (v1.18, v1.19)
| Plugin | Namespace | Key ops |
|---|---|---|
| SLAMPlugin | `ctx.slam.*` | create_map, scan, integrate, frontier |
| VisionPlugin | `ctx.vision.*` | detect, ocr, caption |
| AutoDrivePlugin | `ctx.autodrive.*` | simulate (lane keep + obstacle avoid) |

### Security / Cloud / RL (v1.21-v1.22)
| Plugin | Namespace | Key ops |
|---|---|---|
| CryptoPlugin | `ctx.crypto.*` | hash, gen_key, xor_encrypt/decrypt, hmac_sign/verify, fernet |
| RLPlugin | `ctx.rl.*` | train_grpo, rlcd_reward |
| CloudPlugin | `ctx.cloud.*` | aws_s3_list, gcp_storage_list, azure_blob_list |

## Paper Highlights (for HF Papers)

**One-liner:** The open-source reference for the RLCD "decision layer" --
calibrated confidence on acting models, measured across three scales
(8.5M / 360M / Qwen3-0.6B) and on the 7,193-instance RLCDAlignBench.

**Why this matters for the HF community:**
- Open-weight models now span the full price spectrum ($0 -> $0.14/M ->
  $0.20/M frontier tiers) but NO vendor publishes calibration-on-wrong-
  answers curves. This repo is the open, replay-verifiable measurement.
- The API probe (GPT-5.6 Luna + DeepSeek V4 Flash) found THREE distinct
  calibration failure modes across the price spectrum: stable
  hallucination (360M/Qwen), unstable hallucination (Luna), refusal
  (V4 Flash) -- none healthy. Probe cost: $0.02.
- On RLCDAlignBench our supervised TF-IDF readout reaches 0.726 median
  AUROC and beats the commercial Jev detector's zero-shot numbers on
  11/41 benchmarks. The open audit is 15x cheaper than the commercial
  detector it audits.
- HeliosLM's policy-in-code route (GroundingGate, CalibratedPrefetcher)
  is the engineering embodiment of the vendor's own finding: prompt
  engineering adds only +0.006 AUROC (p=0.055) -- retaining full
  probability and routing decisions in code is what works.

**Artifacts:** paper PDF + LaTeX in `docs/`; all benchmark JSONs in
`benchmarks/`; three-tier deployment (lite 8.5M / mid 360M / full V3-class)
with per-task replay verification and tokenizer-fingerprint pairing.

cc @osanseviero @philschmid for HF Papers consideration

## Paper

**Calibrated Agency: An Open-Source RLCD Stack, from Toy Scale to 360M, with a
Canonical-Benchmark Comparison** — Chien-Hsin Lin, working draft 2026-10-03.

**[📄 Read the PDF](docs/paper_draft_2026-10-03.pdf)** (7pp, figures included) ·
[LaTeX source](docs/paper_draft_2026-10-03.tex) (arXiv-ready) ·
[markdown](https://github.com/tonythetiger168/helioslm/blob/main/helioslm_v5/docs/paper_draft_2026-10-03.md) ·
[HF copy](https://huggingface.co/chienhsinlin/helioslm/tree/main/docs).
(Looking for `paper.pdf`? The PDF ships under a dated filename —
`docs/paper_draft_2026-10-03.pdf`; no file is literally named `paper.pdf`.)
Headline numbers: overconfidence on wrong answers grows with scale
(0.94 -> 0.9999 -> ~1.0 at 8.5M/360M/Qwen3-0.6B); deterministic grounding
cures a 360M model's copy-shaped agentic failure (0/12 -> 9/12 + 3
abstained); trust is f(state, intervention) with an 8.4x measured
contrast; on RLCDAlignBench our open readout reaches 0.726 median AUROC
and beats the commercial Jev detector's zero-shot numbers on 11/41
benchmarks. Every claim traces to a versioned artifact in `benchmarks/`.

## Citation

```bibtex
@article{lin2026calibrated,
  title={Calibrated Agency: An Open-Source {RLCD} Stack,
         from Toy Scale to 360M, with a Canonical-Benchmark Comparison},
  author={Lin, Chien-Hsin},
  year={2026},
  note={Working draft. arXiv ID: [pending endorsement]},
  url={https://github.com/tonythetiger168/helioslm}
}
```

Every numeric claim traces to a versioned artifact: code and tests in
this repository (CHANGELOG v5.30.2--v5.38j), weights/tokenizer/
fingerprints and benchmark data in
[huggingface.co/chienhsinlin/helioslm](https://huggingface.co/chienhsinlin/helioslm),
and the paper source + PDF in `docs/`.

## Plugins (legacy helios-harness)

The harness ships with 35+ plugins covering DeepSeek-Harness scenarios and beyond.
All plugins register as `ctx.<name>.<op>` with per-call audit effects.

| Category | Plugins | Version |
|---|---|---|
| Core | Model, Tool, Session, Decision (grounding/trust/prefetcher), Loop, State, Preset | v1.1–v1.3 |
| Agent | Memory, Terminal (TrustGate-gated), Fetch, Filesystem, Time | v1.8 |
| Common | Search (local), WebSearch (stub), PDF, SQLite, Template | v1.9 |
| Services | GitHub (gated), PostgreSQL | v1.10 |
| Office | Excel, Docx, Pptx, YouTube | v1.11 |
| Niche | MCP-wizard, Tmux, Everything | v1.12 |
| Media | Video, Audio (TTS/STT), NanoVideo | v1.13 |
| Hardware | Blender, Omi (BLE) | v1.14 |
| Robot | RobotControl (mock/ROS2), MultiRobot, SLAM, Vision | v1.16–v1.18 |
| Agentic | AgentWorkflow (ReAct), AgentSwarm | v1.2, v1.17 |
| Science | DNA, Protein, Chem, Math-v2 | v1.20–v1.22 |
| RL | GRPO training, RLCD reward shaping | v1.21 |
| Crypto | Hash/XOR/HMAC/Fernet | v1.22 |
| Cloud | AWS S3, GCP Storage, Azure Blob | v1.22 |
| Benchmark | AlignBench summary, Report gen, Trend | v1.15, v1.22 |
| Voice/UI | VoiceDialog, TerminalUI, AutoDrive, WebUI (v2: stdlib-only rewrite) | v1.19, v1.36/v2 |
| IC | RTL gen, UVM scaffold, Coverage | v1.7 |

**Usage:**
```python
from harness.harness_core import Context
from harness.harness_plugins import *
ctx = Context()
ctx.use(PresetPlugin("full", model_fn=my_fn))
ctx.use(RobotControlPlugin(mode="mock"))
ctx.robot.move_base(1.0, 0.0, 0.0)   # TrustGate-gated, effect-audited
```

**Gating policy.** Side-effecting ops (`terminal.run`, `github.create_issue`,
`robot.*`) fail **closed**: with no `decision.trust` registered they return a
`refused` error instead of running, and they execute only when the gate
returns `Route.DIRECT` — `CLARIFY` and `ESCALATE` both block. The `full`
preset wires a TrustGate; with a bare `Context`, register one yourself
(`ctx.register("decision.trust", gate)`), or pass `terminal.run(cmd,
gated=False)` to opt out explicitly. The legacy web UI (`helioslm_v5/harness/web_ui_v2.py`,
superseded by `python -m helioslm_v5.hsh.web`)
binds to 127.0.0.1 and rejects requests with a non-loopback `Host`
(DNS rebinding) or, on POST/PATCH/DELETE, a non-loopback `Origin`.

## Deployment tiers

Three rungs, each with one recorded reason to exist (no spectrum theater):

| Tier | Params | Vocab | Weights (bf16) | Deployment RAM | Why it exists |
|---|---|---|---|---|---|
| **lite** | 8.5M | 1,024 char-level | ~17 MB | **<500 MB, CPU, millisecond latency** | Protocol/audit research at $0: agent, chat, decision layer, replay verification |
| **mid** (v5.33) | 360M | 32,768 BPE (HeliosBPE, T31) | ~720 MB | ~1.5 GB, CPU-runnable inference | **Scale validation**: separate toy artifacts from scale-invariant findings (acceptance oracles T32; GPU training run pending) |
| **full** | DeepSeek-V3/Kimi-K3-class spec | 160,000 | fp8 still needs hundreds of GB HBM | H200/B200-class servers | Architecture decision reference (MLA + MoE + FP8 + MTP), not a local target |

Any intermediate size is constructible by explicit field overrides --
caller-provided non-None values always win over presets and are validated
in `__post_init__` (no silent truncation).

## Quick Start

```bash
pip install -r requirements-ci.txt        # the exact CPU torch / numpy / pytest versions CI uses
python -m helioslm_v5.tests.test_v5     # 92 unit tests (3 checkpoint-dependent tests self-skip without checkpoints/toy_v5.13.pt)
python integration_test_v51.py          # 9 end-to-end integration tests
python -m helioslm_v5.tests.test_agent  # 9 agent-layer oracles (v5.23)
python -m helioslm_v5.tests.test_v5_stage_a   # T15 real-model oracles (v5.26)
python -m helioslm_v5.tests.test_v5_stage_b   # T17 tool-tuned end-to-end (v5.27)
python -m helioslm_v5.tests.test_disagg_pareto  # T18 Pareto sweep oracles (v5.28)
python helioslm_v5/tests/test_chat.py            # T20 chat capability oracles (v5.30)
python helioslm_v5/tests/test_mhc.py             # T22 mHC oracles (v5.31)
python helioslm_v5/tests/test_kv_compress.py     # T23 compressed-attention causality (v5.31)
python helioslm_v5/tests/test_file_env.py        # T24 long-horizon env (v5.31)
python helioslm_v5/tests/test_think.py           # T25 think + experience reuse (v5.31)
python helioslm_v5/tests/test_async_grpo.py      # T26 async GRPO sync-parity (v5.31)
python helioslm_v5/tests/test_decision.py        # T27 typed decision primitives (v5.32)
python helioslm_v5/tests/test_decision_head.py   # T28 DecisionHead calibration (v5.32)
python helioslm_v5/tests/test_v110_github_pg.py    # T52 GitHub/PostgreSQL plugin oracles (offline, mocked HTTP)
python helioslm_v5/tests/test_web_ui_v2.py         # T75 harness web UI v2 contract (stdlib-only UI, Host/Origin checks)
python -m pytest helioslm_v5/tests/test_hsh_*.py   # hsh harness: kernel, session log, tools, loop, presets, skills, MCP, jobs, plan mode, streaming, web UI
python helioslm_v5/tests/test_trust_gate_clarify.py # T86 TrustGate CLARIFY three-state routing (v5.53)
python helioslm_v5/tests/test_review_fixes.py       # T87 review fixes: deferred validation, fail-closed gates, calc faults
python helioslm_v5/tests/test_calibration_board.py  # T88 calibration leaderboard: metric oracles + byte-exact board rebuild
python helioslm_v5/tests/test_std_bench.py          # T89 GSM8K/HumanEval harness: provenance, sandbox, reference gates, mock API run
python helioslm_v5/tests/test_term_bench.py         # T90 terminal sandbox tasks: references, trap rejection, sandbox limits, mock agent run
python helioslm_v5/tests/test_k3_block.py           # T91 K3 block preset: Gated MLA bitwise oracle, K3 layer order, decode == full
python helioslm_v5/tests/test_csa.py                # T92 CSA compress-then-select: dense reduction, slow-reference equality, V4 stack preset
python helioslm_v5/tests/test_csa_learned.py        # T93 CSA learned compressor: starts as mean pooling, masked softmax, stored-weight re-evaluation
python helioslm_v5/tests/test_csa_learned_h64.py    # T94 learned compressor round 2: 16x re-evaluation, oracle ceiling, artifact-sized --check
python helioslm_v5/tests/test_csa_learned_r3.py     # T95 learned compressor round 3: 4x re-evaluation beats round 2, relu2 option
python helioslm_v5/tests/test_mimo_hybrid.py        # T96 MiMo 5:1 sliding/global interleave + learnable sink: defaults, sink math, decode == full, recall artifact
python helioslm_v5/tests/test_swe_bench.py          # T97 mini SWE tasks: per-task fix/untouched/trap gates, grading guards, episode row
python helioslm_v5/tests/test_stop_calibration.py   # T98 知止 stop metrics: false DONE, tested-before-done, trap = false DONE
python helioslm_v5/tests/test_tool_recovery.py      # T99 tool-call recovery: 7 repairs, never-repaired classes, 3000-case fuzz
python helioslm_v5/tests/test_swe_agent_rl.py       # T100 agent RL on the SWE sandbox: GRPO action_mask, env semantics, rollouts
python helioslm_v5/tests/test_qwen_next.py          # T101 Qwen Q1-Q5: GDA short conv, zero-centered norm, global-batch balance, preset, MTP gate
python helioslm_v5/tests/test_grpo_u1.py           # T102 U1 GRPO options: defaults bit-identical, token-level loss, filtering, TIS, clip-higher
python helioslm_v5/tests/test_first_k_dense.py      # T103 B2-3 dense first layers: default unchanged, layout, decode == full, routing stats exclude dense
python helioslm_v5/tests/test_mhc_sinkhorn.py       # T104 B2-4 learnable mHC: identity at init, doubly stochastic always, HC branch-fold regression
python helioslm_v5/tests/test_mtp_light.py          # T105 B2-5 light MTP: default unchanged, window mask oracle, receptive field, A/B artifact
python helioslm_v5/tests/test_clamped_swiglu.py     # T106 U5 gpt-oss clamped SwiGLU: defaults unchanged, reference match, activation bound
python helioslm_v5/tests/test_nope_temperature.py   # T107 U3 Llama 4 NoPE temperature: formula, short-context identity, decode == full, A/B artifact
python helioslm_v5/tests/test_global_prope_kv.py    # T108 U4 Gemma 4 p-RoPE + global K=V: defaults identical, identity dims, only global layers, shared V cache
python helioslm_v5/tests/test_kv_share.py           # T109 U6 cross-layer KV sharing: defaults identical, manual-attention oracle, aliased cache (L-N)/L, decode == full, ppl A/B artifact
python helioslm_v5/tests/test_option_matrix.py      # T110 option matrix: every pair of 28 feature options builds with decode == full or is rejected by OPTION_RULES; core modules read config fields directly
python helioslm_v5/tests/test_smebu_sandwich.py     # T111 U7 Arcee Trinity: SMEBU oracle, bounded steps, sandwich-norm gain 1/sqrt(L) + layer oracle, A/B artifact
python helioslm_v5/tests/test_thinking_budget.py    # T112 U2 thinking budget: forced close at exactly b tokens (greedy + sampled), no reopen, truncation helpers, A/B artifact
python helioslm_v5/tests/test_repo_longctx.py       # T113 AC4 repo-scale probe: task generator, causality of every arm, all-SWA receptive-field oracle, A/B artifact
python helioslm_v5/tests/test_zh_tokenizer_coverage.py  # T114 TW4 zh-TW tokenizer coverage: held-out split and leak check, UTF-8 / char-toy oracles, report artifact
python helioslm_v5/tests/test_mopd.py               # T115 B2-6 MOPD: expected gradient = reverse-KL gradient, self-distillation zero, TROPD bound, routing, action mask, probe artifact
python helioslm_v5/tests/test_quantile_balancing.py  # T116 Stable LatentMoE: K3 Quantile Balancing — histogram quantile, fixed-batch convergence, centring, summed-rank histograms, A/B artifact
python helioslm_v5/tests/test_csa_learned_16x.py   # T117 16x CSA compressor by 4x->16x curriculum: state transfer, fine-tuned and zero-shot slices, ordering up to the oracle ceiling
python helioslm_v5/tests/test_zh_train_speed.py    # T118 Chinese training speed: Big5/GB2312 seed lists, seeded BPE single tokens, bits-per-char metric, speed artifact
python -m pytest helioslm_v5/tests/ -q --ignore=helioslm_v5/tests/test_v5.py  # full pytest suite (396 tests)
python -m pytest helioslm_v5/tests/ -q -n 4 --ignore=helioslm_v5/tests/test_v5.py  # what CI runs: pytest + test_v5.py's 93 cases (test_v5_cases.py) on 4 workers, about 4 min
```

```python
import torch
from helioslm_v5.configs.config_v5 import HeliosLMv5Config
from helioslm_v5.src.model_v5 import HeliosLMv5

model = HeliosLMv5(HeliosLMv5Config(size="lite"))   # CPU-friendly
out = model.generate(torch.tensor([[1, 2, 3]]), max_new_tokens=20, temperature=0)
print(out)
# tensor([[  1,   2,   3, 754, 754, 754, 754, 754, 754, 754, 666, 136, ...]])
# (random init -> gibberish ids; load checkpoints/toy_v5.13.pt for text)
```


## Capabilities at a glance

| Area | Implementation |
| --- | --- |
| **Attention** | MLA with **weight absorption** — latent-only KV cache, **−97.7% memory vs MHA** (full config), verified equivalent to the expanded path (<1e-4). Hybrid linear attention: Gated Delta Rule layers interleaved with MLA (3:1 default), fixed-size recurrent state cache, decode ≡ one-shot (<2e-7), optional per-channel (KDA-style) decay gate so state rows forget at independent rates (v5.21), NoPE option to drop RoPE entirely (v5.22). RoPE scaling: linear / NTK / YaRN. DSA-style sparse top-k decode over the latent cache (k≥L exactly dense), with a **learned lightning indexer** option (ReLU-scored low-dim heads reading the cached latents, DSA-style distill training helper; default head-mean "free" indexer bit-identical) and opt-in sparse prefill. Sliding-window attention with StreamingLLM sinks (O(W) decode), per-head QK-norm, Gemma-style logit soft-capping. |
| **MoE** | Sigmoid-gated fine-grained experts with **auxiliary-loss-free** load balancing (selection-only bias, heuristic or quantile updates). **LatentMoE**: routed experts in a shared latent space. **SiTU-GLU** tanh soft-capped activation. |
| **Cross-layer** | **Attention Residuals** — per-layer gated injection of accumulated lower-layer attention outputs, threaded through DualPipe (gradient-exact, bitwise-verified). |
| **Agent** | v5.23 agent layer: strict tool-call schema/parser (16 error classes), deterministic sandboxed tools, trajectory bitwise-replay oracle, ground-truth-by-construction toy envs, routing gates (v5.22 decision-audit discipline); v5.26 real-model oracles (T15); v5.27 **tool-tuned checkpoint** trained on agent-loop replays (data format == inference by construction) — T17 end-to-end baseline parse 0.15 / finish 1/9, `sparse_top_k=4` ~= dense; hardened by real-model findings (TOOL_ERROR recovery, ASCII-safe docs) ; v5.29 three-modes benchmark with strict-monotone tau routing curves and a recorded overconfidence finding (max conf 0.925-0.944 on wrong answers), artifact oracles T19 |
| **Chat** | v5.30: dual-mode protocol (plain text OR @@tool@@ block; text bypasses the gate, gate governs tools only), multi-turn ChatSession with transcript replay, chat SFT data with inference-identical prompts; v5.30.2 fixed a dataset filter bias that had hidden the text channel (mode-choice: text 11/20) — T20/T21 |
| **Frontier references** | v5.31: five readable toy-scale references distilled from the 2026 frontier — manifold-constrained hyper-connections (DS-V4), HCA/CSA-style compressed attention with exact causality, long-horizon file env (8-14 steps), think-mode + experience reuse (Qwen3-Max direction), async GRPO with bitwise sync-parity (GLM-5 direction) — T22-T26 |
| **Decision layer** | v5.32: typed decision primitives Choice/Score/Noul (System One / Jev direction) with schema enforcement, gate ask/ask_batch extension point, non-autoregressive DecisionHead answering K questions in one pass, outcome-targeted Brier calibration (RLCD direction; the label-targeted variant is provably redundant with CE — recorded) — T27/T28 ; v5.36 continuous Noul (P(yes), floor retired); v5.34 deterministic grounding (0/12 -> 9/12 + 3 abstained on a real 360M checkpoint, zero hallucination leakage); v5.37 TrustGate calibrated abstention + TherapyPair composition (trust is f(state, intervention) — 8.4x measured contrast); v5.38 RLCDAlignBench: our supervised TF-IDF readout median 0.726 AUROC, beats the commercial Jev detector's zero-shot numbers on 11/41 benchmarks (charts in benchmarks/charts/) — T27-T39 |
| **Benchmarks** | Top-10 LLM position paper with verified leaderboard data and the calibration/replay axes no vendor publishes (`helioslm_v5/docs/benchmark_top5_2026-09-27.md`); probe suite (toy + file-env + chat) with scripted oracles, runnable against any API |
| **Disaggregation** | v5.25 Mooncake-style prefill/decode module behind a monotonicity gate; v5.28 three-axis **Pareto sweep** (makespan / workers / worker-seconds) with latency-cost curves per workload — cache-aware anti-monotonicity recorded as a structural finding, not hidden |
| **Speculative decoding** | DeepSeek-style MTP with **strict verification** (residual (p−q)₊ resampling), batch support, O(1) cache-truncation rollback; hybrid recurrent-state rollback via restore+replay. |
| **Serving** | vLLM-style engine: paged KV accounting, copy-on-write forks, watermark-aligned continuous batching. |
| **Training** | FP8 trainer (native float8 + STE, E5M2 gradient hooks, AdamW master weights), DualPipe schedule simulation (recompute-based, gradient-exact), GRPO (real sampling, k3 KL, answer-extraction rewards), Muon optimizer (Newton–Schulz orthogonalized momentum, optional per-head blocks), QAT straight-through fake-quant training. |
| **Adaptation** | `AdaptationLoop`: transferred harnesses must be re-accepted under the target workload's gate or dropped; cold/prefix-free targets shed draft+pool, matching the v5.16 break-even data |
| **Harness evolution** | `inference/harness_evolver.py`: ModularRSI-style module-wise search over draft/pool/tier configs; the temp-0 bitwise gate makes 'latency evolves, answers never change' an enforced invariant (1.41x modeled speedup, 0 gate rejects) |
| **Prefix pool** | `inference/prefix_pool.py`: blake2b(token-block + config fingerprint) keyed KV snapshots, LRU, exact-length past; pooled greedy == from-scratch bitwise |
| **Stream scoring** | `eval/score_stream.py`: score any engine's (prompt, output) JSONL under a reference model; A/B compare with bootstrap CI — the audit-side complement to serving engines |
| **Spec telemetry** | `bench_spec_breakeven.py`: draft x cache-state sweep in the colibri-P3 schema; acceptance + expert hit-rate per decode context, `best_draft_per_cache_state()` picker |
| **Expert streaming** | `expert_store.py`: routed experts tiered to a memory-mapped file, LRU residency with hit/miss/eviction telemetry; streaming forward is bitwise-identical to dense (oracle-verified, roadmap #6) |
| **Toy checkpoints** | `checkpoints/toy_v5.13.pt` — 8.5M char-level model trained on the repo's own source in ~10 CPU-minutes (`examples/train_toy_checkpoint.py`); `checkpoints/tool_tuned_v5.27.pt` — tool-tuned on agent-loop replays (`examples/train_tool_tuned.py`, periodic save + resume); `generate()` / harness / MTP / agent loop run against trained weights |
| **Quantization** | **True GPTQ** (Hessian OBS with error compensation, optional act-order), AWQ with activation-aware grid search, native FP8, MXFP4 — all with `from_linear` real-weight packing. QAT fake-quant training (STE) targets: MXFP4, AWQ, and NVFP4 (E2M1×16 + E4M3 block scales, local-line v5.22-L). |
| **Eval** | Log-likelihood harness (`helioslm_v5/eval/harness.py`): `loglikelihood` / `multiple_choice` / `run_harness` + built-in synthetic tasks (v5.11), token-id based, lm-eval-harness spirit |
| **Multimodal** | NaViT vision encoder (row/col position decomposition, mixed-resolution packing), streaming audio encoder (causal, sliding-window memory, bit-equivalent to one-shot). |

## Why HeliosLM vs. alternatives?

| | HeliosLM | `transformers` | `vLLM` | nanoGPT-style |
|---|---|---|---|---|
| Purpose | Understand + hack the full stack | Run pre-trained models | Max serving throughput | Learn basics |
| Runs on CPU end-to-end | ✅ | partial | ❌ | ✅ |
| Training + serving + quantization in one repo | ✅ | ❌ | ❌ | ❌ |
| Bitwise/strict correctness checks on core paths | ✅ | — | — | — |
| Production throughput | ❌ (by design) | — | ✅ | ❌ |

## Benchmarks

![KV cache](https://raw.githubusercontent.com/tonythetiger168/helioslm/main/docs/benchmarks_kv_cache.png)

The full config holds **99.2% less KV-cache memory than MHA** at 128k context
(2.0 GB vs 257.7 GB): 36 of 48 layers are Gated-Delta linear attention with a
fixed ~1 MB recurrent state, and the 12 MLA layers store only the compressed
latent (512 + 64 values/token). Full numbers and methodology:
[docs/BENCHMARKS.md](https://github.com/tonythetiger168/helioslm/blob/main/docs/BENCHMARKS.md) — reproducible on CPU via
`python benchmarks/bench_cpu.py`.

**Sparse × MTP acceptance sweep** (`benchmarks/bench_sparse_mtp.py`):
extends the v5.16 break-even sweep with the missing sparse_top_k axis —
MTP acceptance, CPU tok/s, and output overlap vs dense under DSA-style
top-k decode; k ≥ L row is bitwise-equal to dense (test-enforced oracle).
`indexer="learned"` rows need a checkpoint with indexer weights —
`examples/train_indexer_distill.py` distills one from the toy checkpoint
(DSA recipe: teacher = the head-mean free indexer, trunk frozen
bit-exact).

**v5.28 serving Pareto** (`benchmarks/disagg_pareto_2026-09-25.json`, regenerate via
`python examples/disagg_pareto.py`): latency-cost curves for cache-heavy / cold / mixed
workloads — the cost-axis alignment artifact, see
[docs/benchmark_alignment.md](https://github.com/tonythetiger168/helioslm/blob/main/helioslm_v5/docs/benchmark_alignment.md).

## Repository layout

```
helioslm_v5/          # source (configs, src/{attention,moe,inference,training,vision,audio,quantization}, agent/, tests)
examples/             # train_toy_checkpoint.py, train_tool_tuned.py (v5.27), disagg_pareto.py (v5.28)
benchmarks/           # CPU bench results + disagg_pareto_2026-09-25.json (v5.28 artifact)
docs/                 # code review reports, benchmark_alignment.md, k3_alignment_targets(.md/.csv),
                      # competitive_intel_2026-09-25.md (+ raw claims CSV), helioslm_handoff.md
integration_test_v51.py
helioslm_v5/CHANGELOG.md  # full version history (v5.0 → latest)
```

## Roadmap

See the [GitHub Project board](https://github.com/tonythetiger168/helioslm/projects) for the live plan. Highlights:

- [x] Pre-trained toy checkpoints (v5.13 text, v5.27 tool-tuned) — `load` and `generate()` / agent-loop immediately
- [ ] Epoch-2 + scaled tool-tuning (T17 parse 0.15 → target 0.4+; trainer resume-ready)
- [x] Fused quantization kernels (v5.11; measured in v5.84)
- [x] CUDA end-to-end verification: `cuda_smoke.py` 23/23 on an RTX 6000 Ada (v5.91)
- [x] Hugging Face Hub: `chienhsinlin/helioslm-agent` hosts the agent layer + tool-tuned artifacts
- [ ] Example notebooks: "Train a tiny HeliosLM on your laptop" / "Add a new attention variant in 30 lines"

## Contributing

Contributions are very welcome — see [CONTRIBUTING.md](https://github.com/tonythetiger168/helioslm/blob/main/CONTRIBUTING.md). Issues labeled [`good first issue`](https://github.com/tonythetiger168/helioslm/labels/good%20first%20issue) are the best entry points.

## Known Limitations

As of 2026-10-26:

- **No trained product weights yet.** The checkpoints in this repo are toy
  and pilot models; the deployment series below are a plan until each
  generation is trained and passes its pre-registered evaluation.
- **kimi-k3 GGUF files need a recent runtime.** llama.cpp has the `kimi-k3`
  architecture since f5027fb; Ollama builds that bundle an older llama.cpp
  cannot load them. `deepseek2` files load everywhere.
- **The older HeliosBPE tokenizer stays token-id only** in llama.cpp (no
  equivalent pre-tokenizer); byte-level BPE `tokenizer.json` files embed fine.
- **Q4_K at non-256 widths falls back to Q8_0** for most kimi-k3 tensors (pilot
  width 640); real model widths are multiples of 256.
- **`moe.dispatch="grouped"` is not the default** until it is measured on a
  GPU; `batched` stays the default.

## Recent fixes (2026-10-23 to 2026-10-26)

- **Test gate after the rename (v5.153):** `python -m helioslm_v5.tests.test_v5`
  failed after the v5.134 rename because the alias loader had no `get_code`;
  it now serves the real `helioslm_v5` files, and the suite gains a `--list` probe.
- **Daily gate covers every suite (v5.155):** the zh-TW corpus, API
  accounting and MoE dispatch tests used to run only under pytest; the daily
  `test_v5` gate now runs them too and fails if a new test is not registered.
  `test_zhtw_corpus.py` reads its files as UTF-8 (it failed on cp950 Windows
  locales), and MoE balance state resumes bit for bit under batched and
  grouped dispatch.
- **Six review findings (#114):** `cpt_hf.py` never takes an optimizer step
  on a NaN/inf loss or gradient, on any rank; prefix-pool eviction no longer breaks a running
  generation; mixed KD keeps unaligned and masked tokens; pretraining without
  teacher targets equals plain cross-entropy; session-store updates are atomic
  under concurrency and listeners get detached events; the release zip keeps
  the `helioslm_v5` alias importable. Each has a regression test in `tests/test_review_regressions.py`.
- **Gated-conv models in the engine and MTP:** LFM2-style conv layers are
  kept off the batched pad-prefix path, and MTP speculative rollback never
  truncates conv state (clone-replay equals a fresh decode; MTP greedy equals
  plain decoding bit for bit, with a rejection taken).
- **Memory and speed (v5.139–v5.152):** MLA trains on the expanded SDPA path
  (2.1x tokens/s, -40% memory at seq 1024 on an RTX 4060); KDA chunks no
  longer build the pair-decay tensor (5.7x faster on CPU at T=1024); the output
  head and cross-entropy run a chunk at a time in `pretrain` and in `cpt_hf.py
  --ce-chunk`, so the full [tokens, vocab] logits never exist.
- **`cpt_hf.py` from the RTX 4060 smoke run (v5.138):** short runs keep the
  WSD decay, `dtype=` for transformers 5, and a `--dry-run` freeze check.
- **Pilot pod safety (v5.144, v5.146):** Hub access is checked before any GPU
  work, and the checkpoint is uploaded off the pod before it stops.
- **Offline API cost audit (#131):** `examples/analyze_api_usage.py` reports
  cache-aware token cost and evaluation provenance from saved responses, with
  **unknown** instead of a guess when a counter or rate is missing.

## Version history

Headlines (full details in [CHANGELOG.md](https://github.com/tonythetiger168/helioslm/blob/main/helioslm_v5/CHANGELOG.md)):

- **v5.194** — CVDP scoring: a test-command timeout now kills its simulators too, not only the shell — T144
- **v5.193** — The Stack SystemVerilog repositories go through the lint + synth gate too (an empty `projects` setting no longer skips it) — T162
- **v5.192** — `fetch_mixture.py` prints each HDL project's gate result as it comes — T162
- **v5.190** — Unified-memory tiers: new **L128** (NVIDIA DGX Spark, AMD Ryzen AI Max+ 395, Mac 128 GB) for v4.0 100B-A10B Q4_K or v3.0 30B-A3B Q8_0 / BF16; new **L256** / **L512** (Mac M4 / M5 series incl. Ultra, 256 / 512 GB unified memory) for v4.0 100B-A10B Q8_0 or v5.0 300B-A20B Q4_K, and 300B-A20B Q8_0; GGUF runs on CUDA, Vulkan/ROCm and Metal — T164
- **v5.189** — Pilot B0 on one GPU: the step-3 eval uses GPU 0 alone on a 1-GPU pod, `SKIP_STD=1` skips the step-4 lm-eval benchmarks
- **v5.188** — Cloudflare downloads per GPU-memory tier (`examples/publish_cloudflare.py`, `scripts/cloudflare_publish.sh`): one zip per tier and quantization on R2, a static download page on Cloudflare Pages whose links point at the R2 domain (GitHub hosts refused), page deployed by direct upload — T163
- **v5.187** — HDL track H5: synth gate measured with sv2v (24 of 38 OpenTitan IPs, Ibex, CV32E40P pass); include-path and `translate_off` fixes, `projects.tops`; CV32E40P replaces CVA6 — T162
- **v5.186** — HDL track H5: projects that synthesise. Each OpenTitan IP, core and GitHub repository must pass Verilator lint and a Yosys synth of its top before training; one document per project in dependency order, MinHash near-dedup, 13-gram benchmark decontamination, permissive-licence filter for GitHub rows, CVA6 added — T162
- **v5.185** — HDL track H4: simulation-equivalence reward with a gate-count bonus (`hdl_reward.py`) and a one-GPU RL loop (`examples/hdl_rl_hf.py`) — T161
- **v5.184** — HDL eval fixes: UVM answer extraction (`[END]` / cut-off answers), UVM 1.2 sequence-macro compatibility, yosys cell counts for `area_ratio` — T159
- **v5.183** — HDL track H5: `git:` mixture sources (pinned commit, LICENSE check, include/exclude globs) and the SystemVerilog / UVM mixture `hdl_h5_sv_uvm_v1.json` (OpenTitan, Ibex, core-v-verif) — T160
- **v5.182** — HDL track: UVM testbench benchmark (`uvm_bench.py`, Verilator + UVM 2017, golden design plus mutants), CVDP `pass@k_by_language`, yosys gate-count `area_ratio` — T159
- **v5.181** — hsh network egress audit: tool subprocesses go through a local proxy and a Python audit hook; hosts off the allow list are blocked (`deny`) or recorded (`audit`); `web --egress` / `--egress-allow`; optional `unshare -rn` isolation — T158
- **v5.180** — `cpt_hf.py --ce-chunk --frozen-bf16` runs its chunked-head check under bf16 autocast (it failed on the first H3 pod) — T145
- **v5.179** — HDL pod fixes: CVDP `env_file` lists, `hdl_eval_all.py` keeps generations before scoring, `cpt_hf.py` / `sft_hf.py --frozen-bf16` (frozen parameters in bf16, E4B fits 96 GB) — T144, T157
- **v5.178** — HDL track H1: `data/mixtures/hdl_h1_cpt_v1.json` (Verilog / SystemVerilog / VHDL code + English and maths replay, licences and columns in `docs/HDL_DATA.md`); mixture sources take `field`, `data_dir` / `data_files` and `"heuristics": false`; `hdl_eval_all.py` (all three HDL benchmarks in one vLLM batch) and `sft_hf.py` (response-only SFT of a Hugging Face model) — T156, T157
- **v5.177** — HDL track H0: CVDP v1.1 non-agentic code-generation harness without docker (`helioslm_v5/eval/cvdp_bench.py`): upstream cocotb + Icarus test commands run in a temp root, pass when every command exits 0, `pass@k` overall and per category — T144
- **v5.176** — HDL track H0: RTLLM v2.1 harness on Icarus (`helioslm_v5/eval/rtllm_bench.py`, `hdl_bench_run.py --bench rtllm`): pinned 50-design cache, `syntax@k` / `pass@k`, `self-check` lists designs Icarus cannot run — T143
- **v5.175** — pilot B0 script: `ALLOW_BLOCKED=1` trains without still-gated mixture sources, `SKIP_GAP=1` skips the already-run base-gap eval so smaller GPUs fit
- **v5.174** — KDA layer skips `as_subclass` on plain tensors, so `--compile` no longer breaks the graph there (k3 pilot: 14 -> 11 breaks) — T155
- **v5.173** — `tw_bench` Phase A gate: the loglik oracle follows task order, so a set with two identical questions that have different answers (C-Eval val) no longer fails the gate
- **v5.165** — no host-sync graph breaks under `--compile` in RoPE or the MoE load count (L4: MoE pilot compiled 20.4k vs 18.2k tokens/s eager, hit dynamo's recompile limit at RoPE) — T155
- **v5.164** — Name back to HeliosLM（羲和）: `helioslm_v5` is the real package again, `HeliosLMv5*` classes and `HELIOS_*` env vars are primary; `import evastar`, `EvastarV5*` and `EVASTAR_*` keep working as aliases
- **v5.163** — `pilot_100m.py --kda-kernel` defaults to `fla` (torch kernel off CUDA / without fla)
- **v5.162** — MoE shared-expert outputs cast to the routed dtype before the add (torch 2.8 inductor fused a mixed-dtype addmm under `--compile`); L4: k3 pilot `--kda-kernel fla --compile` 20.3k tokens/s (2.45x recompute + compile)
- **v5.161** — `get_adaptive_batch_size` / `fit_micro_batch` (pilot `--auto-batch`, halves the micro-batch on CUDA OOM at equal tokens per step); `safe_grouped_matmul` with an order-keeping fallback without `F.grouped_mm` — T154
- **v5.160** — `hybrid_attention.kernel="fla"` / pilot `--kda-kernel fla`: k3 KDA chunked forward via flash-linear-attention `chunk_kda` on CUDA (A5000: 20x forward, 28-48x forward + backward vs `chunked_delta_rule`); torch kernel off CUDA / with packed documents — T153
- **v5.159** — MoE layer re-enters the caller's autocast after the eager dispatch, so bf16 `--compile` works on CUDA; k3 pilot `--kda-recompute --compile` 13.3k tokens/s on an A5000 — T152
- **v5.158** — MoE dispatch excluded from torch.compile (`torch.compiler.disable`), so `--compile` works on the MoE pilot; L4: dense `--compile` +23% tokens/s — T152
- **v5.157** — `hybrid_attention.recompute` / pilot `--kda-recompute`: the chunked KDA recurrence is recomputed in backward instead of saving its fp32 intermediates (k3 pilot: 41% less saved, same gradients) — T151
- **v5.156** — `moe.dispatch="grouped"` runs the batched GEMMs on a torch without `F.grouped_mm` (< 2.10) instead of failing; first L4 MFU numbers in the CHANGELOG
- **v5.155** — upstream's v5.154 pytest-only sidecar suites (TW2 zh-TW corpus T148, offline API accounting) folded into the daily TESTS registry via wrappers that pin the exact test-function set (upstream additions fail loudly until registered); `test_api_accounting.py` runs without pytest via a two-feature shim (parametrize no-op + raises match); `test_zhtw_corpus.py` UTF-8 read repairs for non-UTF-8 Windows locales — T149, T150
- **v5.154** — Roadmap TW2: zh-TW pretraining corpus pipeline (`src/training/zhtw_corpus.py`, `examples/tw2_prepare_corpus.py`): zhwiki through OpenCC `s2twp`, FineWeb-2 `cmn_Hani` kept only when already Traditional, local data.gov.tw JSONL; exact + MinHash near-dup filter, per-source drop counts and Traditional share; attributed JSONL shards, sha256 manifest and `ATTRIBUTION.md` — T148
- **v5.153** — `python -m helioslm_v5.tests.test_v5` (the daily-mandated gate) repaired after the v5.134 rename broke it (`_AliasLoader` lacked `get_code`, required by runpy); alias loader now serves code/source from the real helioslm_v5 files, and the suite `main()` gains a `--list` probe (prints `TESTS=<count>`, exits 0) — T146, T147
- **v5.152** — `cpt_hf.py --ce-chunk N`: route-B continued pretraining without the full [tokens, vocab] logits (4.3 GB fp32 per 4096-token micro-batch at Gemma's 262k vocab): `logits_to_keep=1` + last hidden state through the DDP wrapper, chunked head + CE with the final soft cap; checked against the full loss on one window before training — T145
- **v5.149** — `moe.dispatch="grouped"` (`--moe-dispatch grouped`): routed experts as two `F.grouped_mm` calls over the expert-sorted rows, no capacity padding; same outputs and gradients as blocked / batched; CPU fp32 fwd+bwd 1.43x faster than batched (4096 tokens, 16 experts); default stays batched until a GPU measurement
- **v5.148** — Local deployment packages per tier (`package_tier`, `examples/package_tiers.py`): each of L-Edge / L8 / L16 / L24 / L48+ gets its GGUF files, an Ollama Modelfile, llama.cpp commands and a manifest (sizes, sha256, fits the tier's memory); pilot packages load and run in llama.cpp f5027fb — T142
- **v5.147** — HDL track H0: VerilogEval v2 spec-to-rtl harness (`eval/hdl_bench.py`, `examples/hdl_bench_run.py`): upstream prompt and extraction, Icarus Verilog simulation, upstream failure codes, unbiased pass@k, optional pyslang lint / yosys synthesis reported beside the pass rate; pinned, hash-checked problem cache and a 156/156 reference self-check — T141
- **v5.146** — Pilot B0 step 5: the final checkpoint uploads to a private Hugging Face model repo while the benchmarks run, so the pilot no longer lives only on a billed pod volume
- **v5.144** — Pilot B0 access preflight: `fetch_mixture.py` and `base_gap_eval.py --check-access` check Hub access for every dataset and model before any GPU work and stop the pod if one is missing; pod packages pinned to the dry run's versions
- **v5.143** — Pilot B0 step 4: common benchmarks via `lm-eval[hf,math]==0.4.13` (MMLU, ARC, HellaSwag, GSM8K, BBH, MATH-500, HumanEval, MBPP, medicine and science sets) for the base and the pilot checkpoint; recorded only, the pass rule is unchanged
- **v5.142** — Hybrid KDA + MLA and full K3-block models run in llama.cpp: `hybrid_attention.style="k3"`, block attention residuals, full-width LatentMoE router and `situ_k3` match llama.cpp's `kimi-k3` graph; `export_llamacpp_gguf` writes `kimi-k3` GGUF; `k3_block_config(llamacpp=True)` / pilot arch `k3_llamacpp`; llama.cpp f5027fb logits match to ~1e-4 (f32), prefill and token by token — T137, T138, T139
- **v5.141** — Fused chunked output head + cross-entropy (`pretrain(ce_chunk=...)`, `pilot_100m.py --ce-chunk`): loss and gradients computed a chunk of tokens at a time in one pass, so the [tokens, vocab] logits never exist at once; same loss and gradients as full logits — T140
- **v5.140** — KDA chunk form without the per-chunk fp32 [n, n, Dk] pair-decay tensor: state-free parts of all chunks computed at once, cross-sub-chunk pairs as plain matmuls (both decay factors <= 1); same results as the per-token loop (fp64 1e-10), CPU fwd+bwd 5.7x faster at T=1024 — T136
- **v5.139** — Training speed from an RTX 4060 MFU sweep: MLA trains on the expanded SDPA path (`attention.expanded_training`, same weights; 2.1x tok/s, -40% memory at seq 1024; best dense 14.8k -> 20.3k tok/s), pilot defaults `--moe-dispatch batched` (2.95x) and `--kda-chunk 64`, per-step tok/s and peak memory in `pretrain` logs — T135
- **v5.138** — `cpt_hf.py` fixes from an RTX 4060 smoke run (tiny Gemma 4, 20 steps, passed): short runs keep the WSD decay / source switch (`fit_warmup`), `dtype=` for transformers 5, `--dry-run` freeze check that `pilot_b0_pod.sh` runs before training — T134
- **v5.137** — Route B pilot B0: `examples/fetch_mixture.py` (licence-checked, filtered uint32 shards per mixture source), `examples/cpt_hf.py` (WSD continued pretraining of a Hugging Face base with DDP and a `--max-hours` stop), `examples/pilot_b0_pod.sh` (one pod: base-gap eval, Gemma 4 E2B pilot on 3B tokens, pilot eval; runbook `docs/PILOT_B0.md`) — T134
- **v5.135** — Text in for llama.cpp / Ollama: `export_llamacpp_gguf(..., tokenizer="tokenizer.json")` embeds the pretraining byte-level BPE as llama.cpp's `gpt2` vocabulary (`gpt-2` pre-tokenizer); llama.cpp f5027fb gives the same ids on 320 mixed zh-TW / code / emoji / whitespace samples and the same greedy continuations; tokenizers it would split differently are refused — T133
- **v5.132** — Windows `test_sandbox_limits` repair (Git Bash /c/... path form), CI "LF guard" step (no CRLF blobs in tracked text files), `model_name` never-ahead-of-changelog lock; 3 new locks in `tests/case_harness_repair.py`
- **v5.130** — llama.cpp-runnable GGUF: `export_llamacpp_gguf` writes llama.cpp's `deepseek2` architecture (MLA as MQA, sigmoid MoE with shared experts, RoPE rows reordered) and `llamacpp_incompatibilities` lists the features it cannot express (KDA, NoPE, cross-layer KV, LatentMoE, LB8 PLE, LB9 conv, ...); f32/f16 logits match HeliosLM in llama.cpp f5027fb; token-id input only, tokenizer is a follow-up — T131
- **v5.129** — Route B base-gap eval: `examples/base_gap_eval.py` scores `google/gemma-4-E4B` and four Taiwanese reference models on TMMLU+ / TW-JusticeBench / C-Eval / CMMLU with one scorer and reports paired-bootstrap gaps (`tw_bench.hf_score_fn`, `tw_bench_run.py --hf`; runbook `docs/BASE_GAP_EVAL.md`) — T132
- **v5.127** — Route B: continued-pretraining plan on `google/gemma-4-E4B` (Apache-2.0) on Taiwanese data (`docs/CPT_PLAN_B.md`; `cpt_plan.py` cost model, B1 ≈ US$3.8k / 1,360 H100-h, E2B pilot ≈ US$200; `prereg/b1.0.json`; CPT mixture; uint32 token shards) — T130
- **v5.126** — LB9: LFM2-style gated short-convolution layers (`attention/gated_conv.py` `GatedShortConv`, `gated_conv_layers` / `gated_conv_kernel`, fixed-size StateTensor cache; pilot arch `lfm_conv`) — T129
- **v5.125** — LB8: Gemma 3n-style per-layer embeddings (`per_layer_embed_dim`, `model_v5.PerLayerEmbedding`, zero-init so identical at init; `pilot_100m.py --ple-dim`) — T128
- **v5.124** — LB7: checkpoint merging (`src/training/merge.py`: linear soups, task arithmetic, TIES; `examples/merge_checkpoints.py`) and MiniCPM5-style top-k-union reverse KL for on-policy distillation (`MOPDTrainer(full_vocab=True, topk_union=k)`) — T127
- **v5.123** — Deployment series (proposed): local L-Edge / L8 / L16 / L24 / L48+ by GPU memory and cloud C1-Lite / C1 / C8 / CX by GPUs per replica, all cut from the six generations by quantization (README *Product roadmap*); docs only
- **v5.122** — GGUF block quantization: `export_gguf(dtype="q8_0" | "q4_k")` writes ggml Q8_0 / Q4_K blocks (`src/export/ggml_quants.py`, numpy), 1-D tensors and MoE routers stay F32, `read_gguf` dequantizes; Q4_K decoding checked against a scalar port of ggml's `dequantize_row_q4_K`. The `helioslm` architecture name still keeps llama.cpp from running the file
- **v5.121** — LB5: mid-training data for the WSD decay phase (`src/training/mid_train.py`: `render_trace` turns D1 records into chat-format documents with reasoning, `mix_docs` interleaves sources to token budgets; manifest `format: trace`; `examples/build_midtrain.py` -> `pilot_100m.py --decay-data`) — T126
- **v5.120** — LB4: cross-tokenizer logit KD for pretraining (`src/training/logit_kd.py` aligns teacher top-k to the BPE student on shared character boundaries; `pretrain.KDShard`, `sparse_kd_loss`, `kd_alpha`; `examples/teacher_topk.py` + `build_kd_targets.py`; `pilot_100m.py --kd-train`) — T125
- **v5.119** — E5: split pre-registration: v1.0 0.5B gated against Qwen3.5-0.8B, #41 3B against a 3B Taiwanese reference model (`docs/prereg/`), each beyond sampling noise (95% paired-bootstrap lower bound > 0); `prereg.py` gains relative gates, `paired_bootstrap` and PENDING revisions
- **v5.118** — LB6: hybrid-thinking soft switch for distillation SFT (`distill_sft` `mode_switch`: the last user turn ends in ` /think` or ` /no_think` to match the target; `distill_sft.py prepare --mode-switch`)
- **v5.117** — LB6: GRPO rollout length schedule (`grpo.length_stages`, `two_stage_lengths`) and DAPO overlong handling (`overlong_buffer` soft punishment, `overlong_filter`); `truncated_frac` / `mean_response_tokens` in the update stats — T124
- **v5.116** — LB3: pretraining data quality (`src/training/data_quality.py`: Gopher-style heuristics, Big5/GB2312 Traditional-vs-Simplified labels, hashed n-gram quality classifier) and mixture manifests with licence gates and main/decay phases (`src/training/data_mixture.py`, `data/mixtures/lb3_0p5b_v1.json`); `examples/data_quality.py`; `pilot_prepare_data.py --heuristics --zh-variants --quality-model` — T123
- **v5.115** — LB2: warmup-stable-decay LR schedule (`pretrain.lr_at(schedule="wsd")`, 1-sqrt / linear / cosine decay), decay-phase data switch (`decay_train`) and `schedule_total` for stop-and-resume on one plan; `pilot_100m.py --schedule wsd` — T122
- **v5.110** — Windows checkout repairs: root `.gitattributes` LF policy (pinned digest artifacts), sandbox `python3` PATH resolution, timeout process-tree kill (orphan grandchild fix), base64 patch transport for regex-bearing edits; `model_name` calibrated to the changelog (v5.49 -> v5.110); 4 new locks in `tests/case_harness_repair.py`
- **v5.108** — Plan E3 / TW3: Traditional-Chinese MCQ pack (`eval/tw_bench.py`): TMMLU+, TW-JusticeBench, C-Eval / CMMLU loaders, loglikelihood and generation scoring, Phase A gates (reference 100%, empty 0%), results without question text; `examples/tw_bench_run.py`
- **v5.107** — #42 prep: chunked-parallel KDA / gated delta rule (`hybrid_attention.chunk_size`): one triangular solve per chunk instead of a Python step per token, equal to the loop to 1e-10 in float64 (padding, packed documents, carried state, per-channel decay); 1.68x faster forward + backward per layer on CPU
- **v5.106** — #42 prep: `moe.dispatch="batched"` runs all routed experts in two `bmm` calls instead of one call per expert and 64-row block; equal to the default blocked path to float tolerance (outputs and gradients), 1.7–1.8x faster forward + backward on CPU; GPU speed via `cuda_smoke.py` / `examples/moe_dispatch_bench.py`
- **v5.105** — `cuda_smoke.py` `train_parity_fp32` gates on the 99.9th-percentile weight difference: AdamW moves near-zero-gradient weights by up to lr in the direction of their rounding noise, so the max alone failed (3.0e-3) on torch 2.8 with the loss in parity
- **v5.92** — Plan E1: lm-evaluation-harness adapter (`--model helioslm`); `examples/eval_harness.py` scores a pilot checkpoint on any harness task (ARC, HellaSwag, TMMLU+, ...); checked offline against lm-eval's own `simple_evaluate`
- **v5.91** — Issue #1 closed: `cuda_smoke.py` 23/23 on an RTX 6000 Ada (incl. CUDA-vs-CPU training parity, compile, MTP, GRPO, real-size memory); compiled 100M pilot 1.35 bits/byte at 67k tokens/s, 20.6% MFU
- **v5.90** — Issue #1, first CUDA run (RTX 6000 Ada): `examples/cuda_smoke.py` 11/13 → two real CUDA bugs fixed (FP8Trainer built its layers on the CPU; GPUMonitor crashed without NVML); first GPU pilot 2.77 → 1.79 bits/byte at 29.8k tokens/s — `docs/CUDA_VERIFICATION_2026-10-21.md`
- **v5.89** — Issue #5 measured: byte-level BPE toy model beats the char baseline on held-out NLL per byte in all 3 seeds (mean −0.48 nats/byte, same text, same token budget) — `docs/TOY_BPE.md`
- **v5.88** — Evaluation plan ([`docs/GPU_AND_EVAL_PLAN_2026-10.md`](docs/GPU_AND_EVAL_PLAN_2026-10.md): GPU sizing for #40–#45, leaderboard review, E1–E8); E2 train/eval 13-gram overlap check (`pilot_prepare_data.py --decontaminate`); E5 pre-registered evaluation plans locked by hash
- **v5.86** — Issue #55: `--seeds S1 S2 ...` on all eight A/B scripts (`eval/multiseed.py`): paired per-seed runs plus mean / std / min / max aggregates; without `--seeds` every script writes exactly the old payload
- **v5.85** — Issue #54: legacy `eval_mid_agent` v1–v4 moved to `examples/legacy/`, web UI v1 to `harness/legacy/` behind a shim; **security fix:** v1's `/video/` and `/audio/` routes served any readable file via `../` (now confined to the media dir, regression-tested)
- **v5.84** — Issue #2 measured: fused group-wise dequant x matmul vs unfused on a 4096 x 14336 layer (CPU) — AWQ 7.4x, GPTQ 11.9x, MXFP4 2.3x faster at batch 1, outputs equal to 1e-6 (`docs/BENCHMARKS.md`)
- **v5.83** — ~100M single-GPU pilot (Phase C gate): `examples/pilot_prepare_data.py` + `examples/pilot_100m.py` over `src/training/pretrain.py` — dense / MoE / K3-block / MiMo-hybrid at one width, held-out bits per byte, MFU, exact kill-and-resume — T122
- **v5.82** — Distillation from API teachers: sequence KD with a verifier, sparse top-k KD across tokenizers (`src/training/api_distill.py`), and a licence audit of ten open-weight teacher models; on a toy task verified sequence KD reaches 1.00 in every seed vs 0.54 unfiltered — T120, T121
- **v5.81** — Dataset licence audit (TW1): 45 Hugging Face dataset cards (Traditional / Simplified Chinese, Spanish, Japanese, Korean, eval sets) read live and classified commercial / non-commercial / no-derivatives / unknown — T119; plus [`docs/K3_V4_COMPARISON.md`](docs/K3_V4_COMPARISON.md), the Phase B K3 vs V4 comparison
- **v5.80** — Chinese training speed: Big5 + GB2312 common characters as BPE seeds (`train_toy_bpe.py --zh-seed both`); a tokenizer that never saw Chinese goes from 3.00 to 1.00 tokens per character, 1.82x the Chinese text per training step — T118
- **v5.78** — Phase B: 16x CSA compressor by curriculum (4x weights, fine-tuned at 16x): passkey recall 0.375/0.390 -> 0.790/0.740, 92%/88% of the oracle ceiling — T117
- **v5.77** — Stable LatentMoE: Kimi K3 Quantile Balancing router bias; fastest from a skewed start and after a drift, no bias drift; SMEBU stays lower in steady state — T116
- **v5.76** — B2-6: multi-teacher on-policy distillation (MOPD); routed teachers keep the student's inc skill (1.00) and halve its KL to the rev teacher, one teacher for everything erases inc — T115
- **v5.75** — TW4: zh-TW tokenizer coverage report; the toy char tokenizer loses every CJK character, a BPE without Chinese pays 3 tokens per character, the repo BPE 0.86 with 78% single-token coverage — T114
- **v5.74** — AC4: repo-scale long-context probe; the sliding/global hybrid finds definitions across files at 1.00 even at 4x the training size, dense needs 2x the steps and still drops to 0.38 at 4x — T113
- **v5.73** — U2: thinking-budget control (forced close after b thinking tokens) + truncated-trace training; closed one step early, accuracy 0.12 → 0.88 with truncation training — T112
- **v5.72** — U7 (Arcee Trinity): SMEBU soft-clamped momentum expert-bias updates (fewer bias reversals at every step size; MaxVio variance 0.0067 → 0.0041 at λ=1e-2) and depth-scaled sandwich norm (post-norm gain 1/√L), both off by default — T111
- **v5.71** — U6: cross-layer KV sharing (Gemma 3n/4); with the last 4 of 8 MLA layers sharing KV, the cache halves exactly and val ppl stays 14.71 vs 14.94 (toy char LM, single seed) — T109
- **v5.70** — U4: Gemma 4 p-RoPE (partial rotary) and K=V on global attention layers; `kv_b_proj` halves and the expanded cache shares one K/V tensor, defaults bit-identical — T108
- **v5.69** — U3: Llama 4 NoPE attention temperature; on one set of weights it lifts recall at 4x the training length from 0.267 to 0.317 (scale 0.3) and leaves the training length unchanged — T107
- **v5.68** — U5: gpt-oss clamped SwiGLU option for MoE experts (pre-w2 activation bounded to [-42, 56] with gpt-oss settings), defaults bit-identical — T106
- **v5.67** — B2-5: MiMo-style light MTP head (sliding-window attention + narrow FFN): -20% head parameters, -39% head step latency, acceptance 0.813 vs 0.822 — T105
- **v5.66** — B2-4: learnable doubly stochastic mHC read matrices (Sinkhorn), identity at init; fixes the v5.7 HC branch fold that scrambled sequence positions across branches (decode != full once B != 0) — T104
- **v5.65** — B2-3: `moe.first_k_dense` dense first layers (GLM-5.3 / MiMo), default unchanged, routing statistics exclude dense layers — T103
- **v5.64** — U1: DAPO / Olmo 3 options for GRPO (token-level loss, clip-higher, no-std advantages, zero-gradient group filtering, TIS), defaults bit-identical; on the agent task a KL anchor to the SFT model lets GRPO improve on SFT (0.367 → 0.733) — T102
- **v5.63** — Qwen3-Next ingredients: GDA short conv, zero-centered RMSNorm with decay groups, global-batch load-balancing loss (equals the full-batch loss over micro-batches), `qwen_next_config`, MTP greedy-equivalence gate (toy acceptance 0.824); roadmap Phase B4 (US models) and TW (zh-TW data survey) — T101
- **v5.62** — MiMo 5:1 sliding-window/global interleave (`mimo_hybrid_config`) keeps long-range recall at 39% of the attended keys (0.424 vs dense 0.428; all-SWA at chance); learnable attention sink (no effect at this scale); agentic coding: 20-task mini SWE set with hidden tests and trap gates, 知止 stop-decision metrics, fuzz-tested tool-call recovery; roadmap Phase B3 (Qwen) and AC — T96–T99
- **v5.61** — learned compressor round 3: 4× passkey recall 0.825/0.800, about 92% of the oracle ceiling (squared ReLU and longer 16× training recorded as negative results); README retitle + Chinese name 羲和; roadmap Phase B2 (GLM-5.3 / MiMo) — T95
- **v5.60** — learned compressor round 2 (hidden 64): 16× passkey recall off chance (0.13 → 0.39), 4× mixed (0.725 at L=512, 0.645 at L=1024); an oracle-pooling ceiling shows CSA can reach 0.86–0.90 (dense 0.91), so the gap is compressor detection — T94
- **v5.59** — CSA learned compressor (`compressor="learned"`): compressor-only training lifts passkey recall at 4× from 0.13 to 0.67 (dense 0.91), generalizing to 4–8× the training length; selecting every chunk gives the same accuracy, so the remaining gap is the compressed entry, not selection; 16× does not learn — T93
- **v5.58** — DeepSeek-V4-order CSA (`CompressedSparseAttention`: compress, then top-k select) and `v4_attention_stack` CSA/HCA preset with IndexShare; passkey A/B shows mean-pooled CSA at chance although its indexer finds the right chunk 86–100% of the time — the compressor is the bottleneck — T92
- **v5.57** — Kimi-K3-style block preset `k3_block_config()`: KDA/KDA/KDA/MLA order (`hybrid_attention.first_layer_full=False`), Gated MLA (`attention.output_gate`), per-channel decay, attention residuals, LatentMoE; defaults bit-identical — T91
- **v5.56** — 10 terminal-style sandbox tasks (`examples/std_bench_run.py --bench terminal`): one bash command per turn, final-state verifiers that reject the naive solutions, verbalized confidence for the calibration board — T90
- **v5.55** — GSM8K / HumanEval harness (`examples/std_bench_run.py`) on pinned upstream subsets with a sandboxed HumanEval executor and oracle-tested reference gates; any model behind an OpenAI-compatible API can be run and its output goes straight onto the calibration board; fixes `probe_api` retrying every HTTP 400 with 30 s of sleeps — T89
- **v5.54** — roadmap Phase A calibration leaderboard (`examples/calibration_board.py`): AUROC/Brier/ECE/overconfident-wrong over committed probe artifacts; finding: every wrong answer from the local mid model came at conf ≥ 0.9; milestone tracker `docs/MILESTONES.md`; 2026 Q4 roadmap `docs/ROADMAP_2026Q4.md` — T88
- **2026-10-15 review fixes (post v5.53)** — `execute()` validates args, so `defer_validation=True` with a non-clarify gate yields `TOOL_ERROR` instead of crashing (and a malformed `finish` no longer ends the run); missing+malformed calls ESCALATE rather than CLARIFY; `calc` overflow / `0**-1` are tool errors; harness terminal/GitHub/robot ops fail closed without a TrustGate and run only on `DIRECT`; web UI Host/Origin checks; `torch.load(weights_only=True)` for runtime checkpoint loads — T87
- **2026-10 review fix wave** — full-repo review follow-through: PrefixPool fingerprint hash-collision, agent bare imports, GitHubPlugin `import json` (caught by the new CI syntax gate), duplicate-plugin merges, metadata drift; new CI (`.github/workflows/ci.yml`): flake8 syntax gate (E9/F63/F7/F82) + py3.11/3.12 test matrix + release build; harness web UI v2 (stdlib-only rewrite, T75 contract tests)
- **v5.22-L** (local line) — NVFP4-format QAT fake-quant target (E2M1×16 + E4M3 block scales) + NoPE option for MLA (Kimi-K3 direction; default off, bit-identical)
- **v5.21** (local line) — KDA-style per-channel decay gate: linear-attention state rows forget at independent rates (Kimi Linear / GLM-5.3-Flash direction), bit-identical default, rollback-safe
- **v5.29** — Three-mode benchmark on the real checkpoint (direct/routed/oracle): tau-routing curve strictly monotone (v5.22 gate PASS on real confidence); headline finding = systematic overconfidence on wrong answers (conf 0.944) — the exact failure class the v5.22 audit toolkit measures
- **v5.28** — Cost-axis alignment: disagg three-axis Pareto sweep (latency-cost curves per workload; cache-aware anti-monotonicity recorded as structural finding)
- **v5.27** — Tool-tuned checkpoint: agent-loop-replay training, T17 end-to-end (parse 0.15/finish 1/9 baseline, regression-guard floors), sparse_top_k=4 ~= dense in agent inference; agent hardened (TOOL_ERROR recovery, ASCII-safe docs)
- **v5.26** — Stage A real-model oracles (T15): zero-gate attention residuals bitwise-verified on HeliosLMv5; sparse top-k decode oracle (k≥L bit-identical, selection validity + determinism); agent-loop smoke on the real toy checkpoint
- **v5.25** — Disagg evolver module: Mooncake-style prefill/decode separation as a HarnessEvolver search module, monotonicity gate, three-axis Pareto (makespan / workers / worker-seconds)
- **v5.24** — Attention variants with two-layer oracles: DSA sparse decode (fp32 certificate ⇒ fp64 gate), AttnRes mixing (zero-init ⇒ bitwise migration gate)
- **v5.23** — Agent layer: strict tool-call schema + parser, deterministic sandboxed tools, trajectory bitwise-replay oracle, ground-truth-by-construction envs, routing gates in the agent loop
- **v5.22** — Decision-layer audit toolkit: calibration metrics (ECE / Brier) against constructed ground truth — no reference LLM required
- **v5.20** — Pareto-aware integration (memory axis) + cross-workload adaptation loop — ModularRSI gaps 2/3 closed at the inference layer
- **v5.19** — Evolvable serving harness: ModularRSI-style module-wise search (draft/pool/tier) behind a deterministic bitwise oracle gate
- **v5.18** — Content-addressed KV prefix pool: cross-session prefix reuse, fingerprint-guarded, bit-exact oracle (roadmap #6 complete)
- **v5.17** — Standalone token-stream scorer: quality-gate any engine's output (A/B compare + bootstrap CI, colibri-container-style)
- **v5.16** — Speculation break-even sweep: colibri-P3-compatible JSONL schema, first honest data point (draft pays only when warm)
- **v5.15** — Disk-tier expert store: bit-exact streaming oracle, mmap + LRU, router/shared stay resident (roadmap #6)
- **v5.14** — Multi-process DualPipe (one stage per process, phased queue protocol, gradient-exact vs single-process)
- **v5.13** — CPU-trained toy char-level checkpoint (MTP aux loss, acceptance 1.00 on greedy) + train_toy_checkpoint.py
- **v5.12** — Batched equal-length prefill in the engine (hybrid recurrent-state models included)
- **v5.11** — Fused dequant×matmul kernels (AWQ/GPTQ/MXFP4), MXFP4 decode fix, eval loglikelihood harness
- **v5.10** — CPU benchmark suite: analytic KV-cache accounting + wall-clock generation, BENCHMARKS.md
- **v5.9** — Gemma-style attention + final logit soft-capping, per-head QK-norm, sliding-window attention with StreamingLLM sinks
- **v5.8** — YaRN RoPE scaling, DSA-style sparse top-k decode, per-head Muon, GPTQ act-order
- **v5.7** — RoPE scaling (linear/NTK), FP8 latent KV cache, Hyper-Connections, QAT training
- **v5.6** — Hybrid packed-sequence training, MXFP4, Muon optimizer
- **v5.5** — K3-aligned feature wave: hybrid Gated-Delta attention, LatentMoE, quantile balancing, attention residuals, SiTU-GLU
- **v5.0–v5.4** — Review-hardened core: MLA weight absorption, true GPTQ, strict speculative sampling, NaViT + audio encoders

## License

Apache License 2.0 — see [LICENSE](https://github.com/tonythetiger168/helioslm/blob/main/LICENSE).


<!-- HELIOSLM_PRODUCT_ROADMAP_START -->
## Product roadmap (proposed; not a release claim)

> **Planning baseline: October 2026.** This roadmap describes **future product generations**, not the existing `helioslm_v5` Python package, historical v5.x research iterations, or harness plugin releases. The current repository is primarily a correctness-focused reference implementation. None of the model sizes below should be interpreted as already-trained checkpoints or committed delivery dates. Promotion requires benchmark, reproducibility, security and budget gates.

### Vision and product portfolio

HeliosLM aims to evolve from a verifiable, modifiable LLM research stack into an independent model and agent platform. Commercial rollout priority: **HeliosLM Code → HeliosLM Cloud → HeliosLM Enterprise → Frontier research commercialization**.

| Product | Audience | Proposed offering |
| --- | --- | --- |
| **HeliosLM Models** | Researchers and developers | Open research components, independent model weights when trained and licensed, SDK and inference |
| **HeliosLM Code** | Software engineers and teams | Auditable GitHub coding agent, sandboxed tools, tests, patch review and rollback |
| **HeliosLM Cloud** | AI application builders | Model-serving API, routing, metering, monitoring and usage-based billing |
| **HeliosLM Enterprise** | Organizations | Private deployment, RBAC, audit trails, enterprise integrations and support |

### Six proposed product generations and parameter targets

**B = billion (10^9) parameters; T = trillion (10^12) parameters.** *Active parameters* denotes the approximate per-token compute path in an MoE model, **not** the amount of weight memory required to load the model. Exact counts depend on final shared layers, experts and routing configuration.

| Product generation | Target model | Total parameters | Active parameters / token | Architecture | Proposed deliverable |
| --- | --- | ---: | ---: | --- | --- |
| **v1.0 — Foundation** | HeliosLM Lite | **0.5B** | 0.5B | Dense | Reproducible training/inference, CLI, tests, developer preview |
| **v2.0 — Training & API** | HeliosLM Base | **3B** | 3B | Dense | Data pipeline, SFT/LoRA, checkpoints, evaluation, SDK/API |
| **v3.0 — Advanced MoE / Code** | HeliosLM MoE | **30B** | ~3B | MoE | Expert routing, quantized serving, HeliosLM Code beta |
| **v4.0 — Reasoning / Cloud** | HeliosLM Reasoner | **100B** | ~10B | MoE | Reasoning, coding agent, cloud API, metering and safety gates |
| **v5.0 — Pro / Enterprise** | HeliosLM Pro | **300B** | ~20B | MoE | Multimodal/long-context R&D, enterprise pilot, distributed serving |
| **v6.0 — Frontier** | HeliosLM Frontier | **1T (1,000B)** | ~40B | MoE | Frontier research candidate; competitive performance on defined evaluations |

These are **stretch targets, not mandatory training commitments**. A version may ship with a smaller verified model if quality, economics or safety gates are not met. The 1T target is contingent on major external funding, training data rights, infrastructure and distributed-systems validation; matching parameter count alone does not imply matching DeepSeek, Kimi, OpenAI or Anthropic capabilities.

### Deployment series (proposed, 2026-10-23)

Each generation above is trained once; the deployment series below are cut from those checkpoints by quantization and context length, so the product line adds no training runs. **All sizes and speeds are estimates** (weights = parameters × bytes per weight: Q4_K_M ≈ 0.61, Q8_0 ≈ 1.06, BF16 = 2; decode speed from memory bandwidth). None is measured, and no series is supported until it passes its own pre-registered evaluation (E5) on the quantized file it ships.

**Local series (by GPU memory)**

| Series | GPU memory | Example hardware | Model | Quantization | Weights (est.) | Default context | Source generation |
| --- | --- | --- | --- | --- | ---: | --- | --- |
| **L-Edge** | none, ≤ 8 GB RAM | laptop CPU, Raspberry Pi 5, phone NPU | 0.5B dense | Q4_K_M / Q8_0 | 0.3 / 0.5 GB | 8K | v1.0 |
| **L8** | 8 GB | RTX 4060, 3060 Ti | 3B dense | Q4_K_M (Q8_0 optional) | 1.8 / 3.2 GB | 32K | v2.0 |
| **L16** | 12–16 GB | RTX 3060 12GB, 4070, 4060 Ti / 5060 Ti 16GB | 3B dense, or 30B-A3B with routed experts in system RAM | BF16; Q4_K_M | 6.0 GB; 18 GB | 128K; 16K | v2.0, v3.0 |
| **L24** | 24–32 GB | RTX 3090, 4090, 5090 | 30B-A3B MoE | Q4_K_M (Q6_K on 32 GB) | 18 GB | 32K (128K on 32 GB) | v3.0 |
| **L48+** | 48–96 GB | RTX PRO 6000 96GB, 2×24 GB, 64–96 GB unified memory | 30B-A3B, or 100B-A10B | Q8_0 / BF16; Q4_K_M | 32 / 60 GB; 61 GB | 128K | v3.0, v4.0 |
| **L128** | 128 GB unified memory | NVIDIA DGX Spark, AMD Ryzen AI Max+ 395, Mac (M4 / M5 series incl. Max / Ultra) 128 GB | 100B-A10B, or 30B-A3B | Q4_K_M; Q8_0 / BF16 | 61 GB; 32 / 60 GB | 128K | v4.0, v3.0 |
| **L256** | 256 GB unified memory | Mac (M4 / M5 series incl. Ultra; Mac Studio M3 Ultra) 256 GB | 100B-A10B, or 300B-A20B | Q8_0; Q4_K_M | 106 GB; 183 GB | 128K; 32K | v4.0, v5.0 |
| **L512** | 512 GB unified memory | Mac (M4 / M5 series incl. Ultra; Mac Studio M3 Ultra) 512 GB | 300B-A20B MoE | Q8_0; Q4_K_M | 318 GB; 183 GB | 128K | v5.0 |

**Cloud series (by GPUs per replica)**

| Series | One replica | Model | Precision | Weights (est.) | Role | Source generation |
| --- | --- | --- | --- | ---: | --- | --- |
| **C1-Lite** | 1× RTX 4090 / 5090 | 0.5B, 3B | BF16 | 1 / 6 GB | Low-cost API, decision-head serving, speculative-decoding draft | v1.0, v2.0 |
| **C1** | 1× H100 80GB | 30B-A3B | FP8 | 31 GB | Main API tier; ~45 GB left for KV cache | v3.0 |
| **C8** | one 8-GPU NVLink node | 100B-A10B; 300B-A20B | FP8 | 100 GB; 300 GB | Reasoning and code agents, private deployment; expert parallelism inside the node | v4.0, v5.0 |
| **CX** | 8× B200 and up, multi-node or NVL72 | 1T-A40B | FP8 or MXFP4 | 1,000 GB; 530 GB | Flagship; prefill/decode disaggregation, cross-node experts | v6.0 |

Context sizes assume the default MLA cache (latent 512 + RoPE 64, about 37 KB per token for a 3B model); plain 8-head GQA needs about 3.5x more. The local series need checkpoints that llama.cpp-family runtimes can load: `src/export/gguf.py` writes Q8_0 / Q4_K blocks since v5.122, and since v5.130 `src/export/llamacpp.py` writes a model as llama.cpp's `deepseek2` architecture when `llamacpp_incompatibilities(config)` is empty (no KDA, NoPE, cross-layer KV, LatentMoE, per-layer embeddings or conv layers). Since v5.135, passing the pretraining `tokenizer.json` embeds the vocabulary, so llama.cpp and Ollama take text; the older HeliosBPE (space-only splitting) has no llama.cpp equivalent and stays id-only. Since v5.142, hybrid KDA + MLA and K3-block configs export as llama.cpp's `kimi-k3` architecture instead of being refused (`llamacpp_arch(config)` picks `deepseek2` or `kimi-k3`). A series whose source generation is still gated (v4.0 and later) inherits that gate.

**Status of the local series (2026-10-26)**

| Series | Package tooling | Runtime check | Weights |
| --- | --- | --- | --- |
| **L-Edge** | `package_tier(..., "L-Edge")`: Q4_K + Q8_0, CPU only, 8K | pilot files load and run in llama.cpp f5027fb | v1.0 0.5B not trained yet |
| **L8** | Q4_K + Q8_0, `-ngl 99`, 32K | same | v2.0 3B not trained yet |
| **L16** | F16 + Q4_K, 32K | same | v2.0 / v3.0 not trained yet |
| **L24** | Q4_K, 32K | same | v3.0 not trained yet |
| **L48+** | Q8_0 + F16 + Q4_K, 128K | same | v3.0 / v4.0 not trained yet |
| **L128** | Q4_K + Q8_0 + F16, 128K (v5.190) | same (GGUF on CUDA / Vulkan / ROCm / Metal) | v3.0 / v4.0 not trained yet |
| **L256** | Q8_0 + Q4_K, 128K (v5.190) | same (GGUF on Metal; MLX export not yet) | v4.0 / v5.0 not trained yet |
| **L512** | Q8_0 + Q4_K, 128K (v5.190) | same (GGUF on Metal; MLX export not yet) | v5.0 not trained yet |

`examples/package_tiers.py --run <pilot run> --out dist/` (v5.148) writes each tier's GGUF files, an Ollama `Modelfile` (when a tokenizer is embedded), the `llama-cli` / `llama-server` commands and a `manifest.json` with sizes, sha256 and whether the weights fit in 75% of the tier's memory. F16 stands in for the table's BF16 and Q4_K for Q4_K_M / Q6_K. On the pilots, llama.cpp logits match PyTorch at cosine 1.0000 (F16), ≥ 0.989 (Q8_0) and ≥ 0.983 (Q4_K). The cloud series has no packaging step yet; it serves the same checkpoints.

**Downloads (v5.188).** `examples/publish_cloudflare.py --packages dist/ --out cf/ --base-url https://<R2 download domain> --version <v>` zips each tier package once per quantization (GGUF + Modelfile + README + manifest) and writes a static page listing every local tier by GPU memory; `scripts/cloudflare_publish.sh cf/` uploads the zips to R2 (S3 API, multipart) and the page with `wrangler pages deploy` (direct upload, no Git integration). Credentials come only from environment variables. Download links never point at GitHub.

### Tentative product windows (subject to gates)

| Window | Product milestone | Exit criteria |
| --- | --- | --- |
| **2026 Q4** | Foundation developer preview | Windows/Linux CI, reproducible test suite, documented local inference |
| **2027 Q1** | Training & SDK/API | End-to-end train → save → restore → evaluate reproducibility |
| **2027 Q2** | HeliosLM Code beta | Auditable tool execution, human approval for writes, benchmarked real-repo tasks |
| **2027 H2** | HeliosLM Cloud pilot | API reliability, authentication, tenant isolation, billing reconciliation |
| **2028 (tentative)** | Enterprise pilots and Pro research | Private deployment security review and paid pilot validation |
| **2029+ (conditional)** | Frontier challenger research | Pre-registered, reproducible domain-specific evaluation against current baselines |

### Engineering priorities and acceptance gates

- **P0 — Correctness and security:** fix outstanding pytest failures; verify Windows RTX 4060 and Linux CUDA paths; reproducible seeds/checkpoints; sandbox tools, approval gates, rollback and secrets handling.
- **P1 — Usable product:** training/data pipeline, model registry, OpenAI-compatible API, agent task execution, observability, token/GPU cost reporting and user documentation.
- **P2 — Scale research:** MLA/KDA/DSA ablations, expert parallelism, long-context evaluations, distributed training, quantization and multimodal experiments.
- **Evaluation:** publish methodology and comparable baselines for coding (e.g. SWE-bench Verified), reasoning, tool-use, latency, VRAM, reliability and cost; report limitations and negative results.
- **Go/no-go:** do not scale parameter count or launch paid services until quality, security, repeatability, licensing and unit economics meet predeclared thresholds.

### Compute and model strategy

- **Local RTX 4060:** development, CPU/CUDA oracles, tiny-model training and CI reproduction.
- **On-demand RTX PRO 6000 Blackwell 96GB:** prototyping, fine-tuning/quantized inference for suitably sized models, memory and performance measurements; **not** a single-GPU path to training 300B–1T models.
- **Multi-GPU clusters:** only after measured throughput, interconnect, memory, data, reliability and cost justify the scale.
- **Independent model + optional external APIs:** research open architectures (including DeepSeek/Kimi ideas) while using properly licensed OpenAI/Anthropic or other APIs for routing/benchmarks where permitted. Do not present external model output as an independently trained HeliosLM model or assume API outputs may be used for competitive training.

### Roadmap governance

This section tracks **intent**, not verified implementation status. Track actual completion in GitHub Issues/Milestones and link benchmark artifacts, commit SHAs, reproducible commands and release notes before marking an item complete. Keep the historical `helioslm_v5` code/release numbering distinct from these forward-looking product-generation labels.
<!-- HELIOSLM_PRODUCT_ROADMAP_END -->

## Offline API cost and comparison audit

`examples/analyze_api_usage.py` analyzes saved response JSON without network
access or API credentials. It separates ordinary input, cached reads, cache
writes by lifetime, and output tokens for OpenAI and Anthropic usage formats.
Supply a model-specific rate card with its source and date; missing counters,
missing rates, or mismatched model/tier/context scope yield **unknown** cost.
Estimates cover token charges only, not tool/search fees or an actual invoice.

```bash
python examples/analyze_api_usage.py --records saved_records.json \
  --corpus tasks.jsonl --harness-revision <evaluation-commit> \
  --tokenizer <name-and-revision> --settings evaluation_settings.json \
  --rates rates.json --out api_cost_audit.json
```

The report preserves requested/reported model names and fingerprints, the
corpus SHA-256 and UTF-8 byte count, declared tokenizer and evaluation settings,
and supplied rate assumptions. Labels are not proof of an immutable snapshot
or verified tokenizer. Optional `--outcomes` counts successful **tasks**, so
multiple calls in one task do not inflate the denominator. See `--help` for
input schemas. Legacy aggregate probe summaries cannot recover cache details;
the existing live probe client and historical presets remain unchanged.
