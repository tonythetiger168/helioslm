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

# HeliosLM（羲和）

DeepSeek-V3/V4 and Kimi-K3-style LLM architectures in pure PyTorch, where every mechanism ships with an oracle test and everything runs on CPU.

> 中文名「羲和」：中國神話中駕日車的太陽神，對應希臘的 Helios。標語「知止」（《大學》「知止而后有定」）：可驗證的機制，加上知道自己何時可能出錯、該停下來的決策層。

[![CI](https://github.com/tonythetiger168/helioslm/actions/workflows/ci.yml/badge.svg)](https://github.com/tonythetiger168/helioslm/actions)
![Tests](https://img.shields.io/badge/tests-unit%20%2B%20integration%20%2B%20oracle-brightgreen)
![License](https://img.shields.io/badge/license-Apache%202.0-blue)
![PyTorch](https://img.shields.io/badge/framework-PyTorch%20(pure)-ee4c2c)

## What's here

| Part | What it is | Entry point |
|---|---|---|
| **Model stack** | MLA with weight absorption, sigmoid-gated MoE with auxiliary-loss-free balancing, hybrid Gated-Delta / KDA linear attention, CSA / DSA sparse attention, MTP; `lite` / `mid` / `full` presets | `helioslm_v5/src/model_v5.py`, `helioslm_v5/configs/config_v5.py` |
| **Training** | Pretraining, FP8, DualPipe simulation, GRPO, Muon, distillation; continued pretraining and SFT of open Hugging Face bases | `helioslm_v5/src/training/`, `examples/cpt_hf.py`, `examples/sft_hf.py` |
| **Serving + export** | vLLM-style paged engine, speculative decoding, quantization; GGUF export and per-VRAM-tier packages for llama.cpp / Ollama | `helioslm_v5/src/inference/`, `helioslm_v5/src/export/`, `examples/package_hf.py` |
| **Agent harness (hsh)** | Runs a model in a loop with tools: append-only session log, presets, skills, MCP, web UI | `helioslm_v5/hsh/`, `python -m helioslm_v5.hsh.web --open` |
| **Decision layer** | Calibrated confidence for acting models (RLCD): DecisionHead, TrustGate, deterministic grounding | `helioslm_v5/agent/`, `helioslm_v5/src/decision_head.py` |
| **HDL track** | Verilog / SystemVerilog benchmark, data and RL for chip-design models | `helioslm_v5/eval/hdl_bench.py`, `examples/hdl_bench_run.py`, [docs/HDL_BENCH.md](https://github.com/tonythetiger168/helioslm/blob/main/docs/HDL_BENCH.md) |

**Status:** a correctness-focused reference implementation with no trained product weights yet (only toy and pilot checkpoints); see [Known Limitations](#known-limitations).

## Who is this for?

| You are... | What HeliosLM gives you |
|---|---|
| **A learner** who wants to understand MLA, MoE routing, DualPipe, speculative decoding | Annotated, review-hardened PyTorch whose oracle tests act as executable documentation (`python -m helioslm_v5.tests.test_v5 --list` shows the current set) |
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

The pre-hsh plugin harness (`harness/`, v1.0–v1.22) is documented in [docs/LEGACY_HARNESS.md](https://github.com/tonythetiger168/helioslm/blob/main/docs/LEGACY_HARNESS.md).

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
python -m helioslm_v5.tests.test_v5 --list   # prints TESTS=<count>, the registered unit tests, and the version pair
python -m helioslm_v5.tests.test_v5     # the mandated unit suite (checkpoint-dependent tests self-skip without checkpoints/toy_v5.13.pt)
python integration_test_v51.py          # end-to-end integration tests
python -m helioslm_v5.tests.test_agent  # agent-layer oracles (v5.23)
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
python -m pytest helioslm_v5/tests/ -q --ignore=helioslm_v5/tests/test_v5.py  # full pytest suite
python -m pytest helioslm_v5/tests/ -q -n 4 --ignore=helioslm_v5/tests/test_v5.py  # what CI runs: pytest + test_v5.py's cases (test_v5_cases.py) on 4 workers, about 4 min
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

As of 2026-10-31:

- **`helioslm_v5` is the only package name.** v5.199 removed the last
  compatibility aliases of the v5.134 Evastar rename (the package itself
  was renamed back in v5.164), so `evastar` imports no longer work.
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

The last five releases; every earlier release is in [CHANGELOG.md](https://github.com/tonythetiger168/helioslm/blob/main/helioslm_v5/CHANGELOG.md).

- **v5.215** — PHI and licence guard for medical data (`src/training/data_guard.py`): `cpt_hf.py` / `sft_hf.py` / `distill_generate.py` take `--data-manifest` / `--medical`; medical sources must be de-identified with licence and origin, are PHI-scanned, never go to an external API teacher, and research-only data cannot train releasable weights
- **v5.214** — rule-based JSON → report grounding grader (`helioslm_v5/eval/grounded_report.py`): unverified facts, omissions and contradictions in zh-TW / English nodule reports; `examples/grounded_report_run.py` grades any OpenAI-compatible models on the same prompt
- **v5.213** — download page and tier table list 32 GB unified memory under L24 and 64–96 GB under L48+ (RTX Spark laptops, Ryzen AI Max+ 395, Mac)
- **v5.212** — README restructured: short opening with a "What's here" table, no hard-coded test counts (use `python -m helioslm_v5.tests.test_v5 --list`), legacy plugin harness moved to [docs/LEGACY_HARNESS.md](https://github.com/tonythetiger168/helioslm/blob/main/docs/LEGACY_HARNESS.md)
- **v5.211** — MoE packages (GGUF `expert_count`) tell how to keep experts in system RAM with llama.cpp `--cpu-moe` / `--n-cpu-moe` when the file does not fit the GPU

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
| **L24** | 24–32 GB | RTX 3090, 4090, 5090; 32 GB unified memory (RTX Spark laptops, Ryzen AI Max+ 395, Mac) | 30B-A3B MoE | Q4_K_M (Q6_K on 32 GB) | 18 GB | 32K (128K on 32 GB) | v3.0 |
| **L48+** | 48–96 GB | RTX PRO 6000 96GB, 2×24 GB, 64–96 GB unified memory (RTX Spark laptops, Ryzen AI Max+ 395, Mac) | 30B-A3B, or 100B-A10B | Q8_0 / BF16; Q4_K_M | 32 / 60 GB; 61 GB | 128K | v3.0, v4.0 |
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

**Status of the local series (2026-10-29)**

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

**Download packages by version (2026-10-29).** Every version is listed on the download page as 開發中 (in development) until it passes its gate; nothing below is downloadable yet.

| Version | Packages (tiers) | Quantizations | Status |
| --- | --- | --- | --- |
| **Route-B pilot** (Taiwanese continued pretraining of Gemma 4 E2B) | L8, L16 | Q4_K_M + Q8_0 (L8); F16 + Q4_K_M (L16) | pilot B0 training on public data; packaged with `examples/package_hf.py` once it passes (TMMLU+ above its base, C-Eval / CMMLU within 2 points) |
| **v1.0** Lite 0.5B | L-Edge | Q4_K + Q8_0 | 開發中, not trained |
| **v2.0** Base 3B | L8, L16 | Q4_K + Q8_0; F16 + Q4_K | 開發中, not trained |
| **v3.0** MoE 30B-A3B | L16 (experts in RAM), L24, L48+, L128 | Q4_K; Q8_0 / F16 | 開發中, not trained |
| **v4.0** Reasoner 100B-A10B | L48+, L128, L256 | Q4_K; Q8_0 | 開發中, not trained |
| **v5.0** Pro 300B-A20B | L256, L512 | Q4_K; Q8_0 | 開發中, not trained |
| **v6.0** Frontier 1T-A40B | cloud only (CX) | FP8 / MXFP4 | 開發中, not trained |
| **HDL** (Verilog / SystemVerilog / UVM) | follows its source generation | same | 開發中: eval harness (VerilogEval, RTLLM, CVDP, UVM) and data tools done; no model yet |

`examples/package_tiers.py --run <pilot run> --out dist/` (v5.148) writes each tier's GGUF files, an Ollama `Modelfile` (when a tokenizer is embedded), the `llama-cli` / `llama-server` commands and a `manifest.json` with sizes, sha256 and whether the weights fit in 75% of the tier's memory. F16 stands in for the table's BF16 and Q4_K for Q4_K_M / Q6_K. On the pilots, llama.cpp logits match PyTorch at cosine 1.0000 (F16), ≥ 0.989 (Q8_0) and ≥ 0.983 (Q4_K). The cloud series has no packaging step yet; it serves the same checkpoints.

**Downloads (v5.188).** `examples/publish_cloudflare.py --packages dist/ --out cf/ --base-url https://<R2 download domain> --version <v>` zips each tier package once per quantization (GGUF + Modelfile + README + manifest) and writes a static page listing every local tier by GPU memory; `scripts/cloudflare_publish.sh cf/` uploads the zips to R2 (S3 API, multipart) and the page with `wrangler pages deploy` (direct upload, no Git integration). Credentials come only from environment variables (kept as RunPod secrets, since the upload runs on a pod). Download links never point at GitHub; they use the R2 bucket's own custom download domain.

**Packages from a Hugging Face checkpoint (v5.196).** Route-B checkpoints are not HeliosLM models, so `examples/package_hf.py --hf <checkpoint> --llama-cpp <checkout> --out dist/ --tiers L8 L16 --name <name> --arch gemma4 --base google/gemma-4-E2B` converts them with llama.cpp's `convert_hf_to_gguf.py` (F16) and `llama-quantize` (Q8_0, Q4_K_M), then writes the same tier layout (`package_gguf`), so `publish_cloudflare.py` bundles them unchanged. With `--imatrix-data calib.txt` (v5.208) it first runs `llama-imatrix` on the F16 file and makes Q4_K_M with that importance matrix (Q8_0 / F16 are unchanged); the manifest records the matrix's sha256. `examples/imatrix_calib.py --src <txt/jsonl ...> --out calib.txt` interleaves calibration sources (default 600k characters). With `--imatrix-data calib.txt` (v5.208) it first runs `llama-imatrix` on the F16 file and makes Q4_K_M with that importance matrix (Q8_0 / F16 are unchanged); the manifest records the matrix's sha256. `examples/imatrix_calib.py --src <txt/jsonl ...> --out calib.txt` interleaves calibration sources (default 600k characters).

**Speculative decoding (v5.209).** A cheap drafter proposes tokens that the main model checks in one batch, so generation is faster and the output stays the main model's own. `package_hf.py --draft L24=<drafter GGUF> --draft-base <id>` copies the drafter into the tier package and its zips and writes the `llama-server --spec-type ... -md <drafter>` command into the package README (llama.cpp only; Ollama has no draft models). The default kind, `--spec-type draft-mtp`, is the main model's multi-token-prediction head: Gemma 4 ships one per size (`google/gemma-4-*-it-assistant`, GGUF architecture `gemma4-assistant`), it reuses the main model's KV cache; measured on an A40, 26B-A4B Q4_K_M goes from 95.9 to 152.9 tok/s (1.60x, acceptance 0.59). `draft-simple` takes a smaller model with the same vocabulary, checked the way llama.cpp does (`speculative.vocab_mismatch`: same tokenizer and special tokens, vocab sizes within 128, token texts equal from id 5). Gemma 4 E2B as such a drafter for 26B-A4B was slower (68.1 tok/s, 0.71x), so it is not shipped. `--spec-type ngram-mod` needs no drafter (147.8 tok/s, 1.54x) and suits code and HDL. `examples/spec_bench.py --server <llama-server> --target <main> --draft <drafter> --spec-type draft-mtp` reports tokens/s with and without speculation, the acceptance rate and whether greedy outputs match.

**KV-cache quantization (v5.210).** At long context the KV cache can outgrow the weights. Every package README now gives the llama.cpp flags `-fa on -ctk q8_0 -ctv q8_0` (half the KV memory, near-lossless; the recommended setting, also in the manifest's `kv_cache`) and `q4_0` (a quarter, some loss), plus Ollama's `OLLAMA_FLASH_ATTENTION=1 OLLAMA_KV_CACHE_TYPE=q8_0`. `examples/quant_bench.py --bin <llama.cpp bin> --gguf q4=<a> q4_imatrix=<b> --text <eval text> --kv f16 q8_0 q4_0` runs `llama-perplexity` over each GGUF and KV type and reports perplexity and allocated KV size, so imatrix and KV choices are measured, not assumed.

**Expert offload for MoE models (v5.211).** When a package's GGUF is a mixture-of-experts model (its `<arch>.expert_count` metadata), `package_gguf` records the expert count in the manifest and the package README explains llama.cpp's `--cpu-moe` (all expert weights in system RAM, attention and shared weights on the GPU) and `--n-cpu-moe N` (only the first N layers' experts), e.g. Gemma 4 26B-A4B Q4_K_M on a 16 GB card. Unified-memory machines need neither.

**Packages from a Hugging Face checkpoint (v5.196).** Route-B checkpoints are not HeliosLM models, so `examples/package_hf.py --hf <checkpoint> --llama-cpp <checkout> --out dist/ --tiers L8 L16 --name <name> --arch gemma4 --base google/gemma-4-E2B` converts them with llama.cpp's `convert_hf_to_gguf.py` (F16) and `llama-quantize` (Q8_0, Q4_K_M), then writes the same tier layout (`package_gguf`), so `publish_cloudflare.py` bundles them unchanged.

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
