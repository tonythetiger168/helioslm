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

> 中文名「羲和」：中國神話中駕日車的太陽神，對應希臘的 Helios。標語「知止」（《大學》「知止而后有定」）：可驗證的機制，加上知道自己何時可能出錯、該停下來的決策層。

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

## Harness Plugins (v1.0-v1.22, 23 releases)

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

## Plugins (helios-harness)

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
gated=False)` to opt out explicitly. The web UI (`helioslm_v5/harness/web_ui_v2.py`)
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
CHANGELOG.md          # full version history (v5.0 → v5.28)
```

## Roadmap

See the [GitHub Project board](https://github.com/tonythetiger168/helioslm/projects) for the live plan. Highlights:

- [x] Pre-trained toy checkpoints (v5.13 text, v5.27 tool-tuned) — `load` and `generate()` / agent-loop immediately
- [ ] Epoch-2 + scaled tool-tuning (T17 parse 0.15 → target 0.4+; trainer resume-ready)
- [ ] Fused quantization kernels
- [ ] CUDA end-to-end verification (paths are currently static-checked; CPU-verified)
- [x] Hugging Face Hub: `chienhsinlin/helioslm-agent` hosts the agent layer + tool-tuned artifacts
- [ ] Example notebooks: "Train a tiny HeliosLM on your laptop" / "Add a new attention variant in 30 lines"

## Contributing

Contributions are very welcome — see [CONTRIBUTING.md](https://github.com/tonythetiger168/helioslm/blob/main/CONTRIBUTING.md). Issues labeled [`good first issue`](https://github.com/tonythetiger168/helioslm/labels/good%20first%20issue) are the best entry points.

## Known Limitations

All five previously known limitations are resolved as of v5.14. Remaining
hardware-dependent item: CUDA end-to-end verification (CPU-verified paths
are static-checked) — tracked as a community issue.

## Version history

Headlines (full details in [CHANGELOG.md](https://github.com/tonythetiger168/helioslm/blob/main/helioslm_v5/CHANGELOG.md)):

- **v5.69** — U3: Llama 4 NoPE attention temperature; on one set of weights it lifts recall at 4x the training length from 0.267 to 0.317 (scale 0.3) and leaves the training length unchanged — T107
- **v5.70** — U4: Gemma 4 p-RoPE (partial rotary) and K=V on global attention layers; `kv_b_proj` halves and the expanded cache shares one K/V tensor, defaults bit-identical — T108
- **v5.71** — U6: cross-layer KV sharing (Gemma 3n/4); with the last 4 of 8 MLA layers sharing KV, the cache halves exactly and val ppl stays 14.71 vs 14.94 (toy char LM, single seed) — T109
- **v5.72** — U7 (Arcee Trinity): SMEBU soft-clamped momentum expert-bias updates (fewer bias reversals at every step size; MaxVio variance 0.0067 → 0.0041 at λ=1e-2) and depth-scaled sandwich norm (post-norm gain 1/√L), both off by default — T111
- **v5.73** — U2: thinking-budget control (forced close after b thinking tokens) + truncated-trace training; closed one step early, accuracy 0.12 → 0.88 with truncation training — T112
- **v5.74** — AC4: repo-scale long-context probe; the sliding/global hybrid finds definitions across files at 1.00 even at 4x the training size, dense needs 2x the steps and still drops to 0.38 at 4x — T113
- **v5.75** — TW4: zh-TW tokenizer coverage report; the toy char tokenizer loses every CJK character, a BPE without Chinese pays 3 tokens per character, the repo BPE 0.86 with 78% single-token coverage — T114
- **v5.76** — B2-6: multi-teacher on-policy distillation (MOPD); routed teachers keep the student's inc skill (1.00) and halve its KL to the rev teacher, one teacher for everything erases inc — T115
- **v5.77** — Stable LatentMoE: Kimi K3 Quantile Balancing router bias; fastest from a skewed start and after a drift, no bias drift; SMEBU stays lower in steady state — T116
- **v5.78** — Phase B: 16x CSA compressor by curriculum (4x weights, fine-tuned at 16x): passkey recall 0.375/0.390 -> 0.790/0.740, 92%/88% of the oracle ceiling — T117
- **v5.84** — Issue #2 measured: fused group-wise dequant x matmul vs unfused on a 4096 x 14336 layer (CPU) — AWQ 7.4x, GPTQ 11.9x, MXFP4 2.3x faster at batch 1, outputs equal to 1e-6 (`docs/BENCHMARKS.md`)
- **v5.83** — ~100M single-GPU pilot (Phase C gate): `examples/pilot_prepare_data.py` + `examples/pilot_100m.py` over `src/training/pretrain.py` — dense / MoE / K3-block / MiMo-hybrid at one width, held-out bits per byte, MFU, exact kill-and-resume — T122
- **v5.82** — Distillation from API teachers: sequence KD with a verifier, sparse top-k KD across tokenizers (`src/training/api_distill.py`), and a licence audit of ten open-weight teacher models; on a toy task verified sequence KD reaches 1.00 in every seed vs 0.54 unfiltered — T120, T121
- **v5.81** — Dataset licence audit (TW1): 45 Hugging Face dataset cards (Traditional / Simplified Chinese, Spanish, Japanese, Korean, eval sets) read live and classified commercial / non-commercial / no-derivatives / unknown — T119; plus [`docs/K3_V4_COMPARISON.md`](docs/K3_V4_COMPARISON.md), the Phase B K3 vs V4 comparison
- **v5.80** — Chinese training speed: Big5 + GB2312 common characters as BPE seeds (`train_toy_bpe.py --zh-seed both`); a tokenizer that never saw Chinese goes from 3.00 to 1.00 tokens per character, 1.82x the Chinese text per training step — T118
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

HeliosLM aims to evolve from a verifiable, modifiable LLM research stack into an independent model and agent platform. Commercial rollout priority: **Helios Code → Helios Cloud → Helios Enterprise → Frontier research commercialization**.

| Product | Audience | Proposed offering |
| --- | --- | --- |
| **HeliosLM Models** | Researchers and developers | Open research components, independent model weights when trained and licensed, SDK and inference |
| **Helios Code** | Software engineers and teams | Auditable GitHub coding agent, sandboxed tools, tests, patch review and rollback |
| **Helios Cloud** | AI application builders | Model-serving API, routing, metering, monitoring and usage-based billing |
| **Helios Enterprise** | Organizations | Private deployment, RBAC, audit trails, enterprise integrations and support |

### Six proposed product generations and parameter targets

**B = billion (10^9) parameters; T = trillion (10^12) parameters.** *Active parameters* denotes the approximate per-token compute path in an MoE model, **not** the amount of weight memory required to load the model. Exact counts depend on final shared layers, experts and routing configuration.

| Product generation | Target model | Total parameters | Active parameters / token | Architecture | Proposed deliverable |
| --- | --- | ---: | ---: | --- | --- |
| **v1.0 — Foundation** | HeliosLM Lite | **0.5B** | 0.5B | Dense | Reproducible training/inference, CLI, tests, developer preview |
| **v2.0 — Training & API** | HeliosLM Base | **3B** | 3B | Dense | Data pipeline, SFT/LoRA, checkpoints, evaluation, SDK/API |
| **v3.0 — Advanced MoE / Code** | HeliosLM MoE | **30B** | ~3B | MoE | Expert routing, quantized serving, Helios Code beta |
| **v4.0 — Reasoning / Cloud** | HeliosLM Reasoner | **100B** | ~10B | MoE | Reasoning, coding agent, cloud API, metering and safety gates |
| **v5.0 — Pro / Enterprise** | HeliosLM Pro | **300B** | ~20B | MoE | Multimodal/long-context R&D, enterprise pilot, distributed serving |
| **v6.0 — Frontier** | HeliosLM Frontier | **1T (1,000B)** | ~40B | MoE | Frontier research candidate; competitive performance on defined evaluations |

These are **stretch targets, not mandatory training commitments**. A version may ship with a smaller verified model if quality, economics or safety gates are not met. The 1T target is contingent on major external funding, training data rights, infrastructure and distributed-systems validation; matching parameter count alone does not imply matching DeepSeek, Kimi, OpenAI or Anthropic capabilities.

### Tentative product windows (subject to gates)

| Window | Product milestone | Exit criteria |
| --- | --- | --- |
| **2026 Q4** | Foundation developer preview | Windows/Linux CI, reproducible test suite, documented local inference |
| **2027 Q1** | Training & SDK/API | End-to-end train → save → restore → evaluate reproducibility |
| **2027 Q2** | Helios Code beta | Auditable tool execution, human approval for writes, benchmarked real-repo tasks |
| **2027 H2** | Helios Cloud pilot | API reliability, authentication, tenant isolation, billing reconciliation |
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
