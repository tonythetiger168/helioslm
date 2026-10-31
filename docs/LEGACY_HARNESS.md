# Legacy plugin harness (helios-harness, v1.0–v1.22)

Moved from the root README in v5.212. The `harness/` plugin context
predates the hsh agent harness (`helioslm_v5/hsh/`, see the root README's
"Using the harness (hsh)") and is kept for its existing callers; new agent
work uses hsh.

## Plugins by release (v1.0-v1.22, 23 releases)

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
| SpatialPlugin (v1.23) | `ctx.spatial.*` | generate, scene, query, describe (3D worlds and spatial questions via [pano2world](https://github.com/tonythetiger168/pano2world); `PANO2WORLD_URL`) |

### Security / Cloud / RL (v1.21-v1.22)
| Plugin | Namespace | Key ops |
|---|---|---|
| CryptoPlugin | `ctx.crypto.*` | hash, gen_key, xor_encrypt/decrypt, hmac_sign/verify, fernet |
| RLPlugin | `ctx.rl.*` | train_grpo, rlcd_reward |
| CloudPlugin | `ctx.cloud.*` | aws_s3_list, gcp_storage_list, azure_blob_list |

## Plugin overview and gating policy

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
