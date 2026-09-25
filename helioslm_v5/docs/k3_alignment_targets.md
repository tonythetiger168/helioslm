# K3 Alignment Targets — Frontier Benchmark Reference (2026-09-25)

> 對齊標的： frontier model benchmark scorecard（使用者提供原始 CSV）。
> 用途： HeliosLM v5.x roadmap「K3 對齊實驗」階段的量化目標參照。
> 說明： 本表為**外部標的**，非 HeliosLM 實測數據；HeliosLM 現況為 toy-scale
> 驗證（v5.23–v5.27），兩者差距即 roadmap 的收斂空間。

## Benchmark scores

| Benchmark | Score | Notes |
|---|---|---|
| GPQA Diamond | 92.3 | Thinking High, no tools |
| Terminal-Bench 2.1 | 85.4 | 當時開源最高分，約追平 GPT-5.6 Sol |
| SWE-bench Multilingual | 82.9 | repo-level |
| SWE-bench Pro | 65.7 | |
| DeepSWE | 64.3 | |
| MCP-Atlas | 83.7 | tool use |
| SkillsBench V1.1 | 62.9 | |
| Apex Agents | 37.1 | |
| HLE (w/ tools) | 55.4 | |

## 對 HeliosLM roadmap 的意義

| 維度 | 對應条目 | 對齊切入點 |
|---|---|---|
| Tool use | MCP-Atlas 83.7 | v5.23 agent layer + v5.27 tool-tuned checkpoint 的 protocol 遵從率（T17）是同一能力軸的最小可驗證版本 |
| Agentic coding | Terminal-Bench 85.4 / SWE-bench 系列 | 對應 roadmap F–G（env 擴張 → agentic RL）：需要真實 repo 環境，toy env 之後的第二梯隊 |
| 推理（無工具） | GPQA 92.3 | v5.22 decision-audit 校準框架可延伸到推理置信度，但模型規模是主要差距 |
| 長程任務 | Apex Agents 37.1（最低分） | 業界共同瓶頸；HeliosLM 的 trajectory bitwise-replay oracle 是此軸的驗證基礎設施 |

## 誠實聲明

HeliosLM v5.27 是 correctness-first **參考實作**（toy scale），與上表 frontier
模型的差距包含規模、資料、訓練算力三個數量級以上的因素。本表的作用是给
「K3 對齊實驗」提供**規格參照點**（架構、serving、tool protocol），而非
短期分數目標。
