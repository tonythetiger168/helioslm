# HeliosLM v5.23–v5.25 專案交接文件

> 更新時間：2026-09-24 16:50｜狀態：**已交付並驗證，PR #6 待 merge，main 已推進至 09-24 基線**
> 本文件自成一體，新 session / 新協作者只需要這一份即可接手。

---

## 1. 背景與目標（v5.23–25 目標，09-24 更新版）

- **起點（09-22）**：HeliosLM v5.22（tonythetiger168/helioslm，Apache-2.0，DeepSeek-V3/K3 風格的 correctness-first 參考實作）被確認**完全不支援 agent**——v5.22 的 "Decision audit" 是審計 agent 決策模型的工具，而非 agent 能力本身。
- **v5.23–25 目標（本版交付完成 ✅）**：
  1. **P0 agent layer**——嚴格 tool-call schema、確定性工具、bitwise trajectory replay、v5.22 決策稽核 gate 接入 agent loop → **完成，9 測試**
  2. **P1 注意力變體**——DSA sparse decode（兩層 oracle）、AttnRes mixing（零初始化 ⇒ bitwise 遷移 gate）→ **完成，2 測試**
  3. **P2 disagg evolver 模組**——Mooncake-style prefill/decode 分析模型納入 HarnessEvolver 搜尋空間 → **完成，4 測試**
  4. 全程維持 HeliosLM「每個能力附可執行 oracle」哲學，對齊 DeepSeek V4 / Kimi K3 的架構與 serving 思路
- **驗證狀態**：**15/15 gated 測試全綠**（Windows 11 / Python 3.14 實機，經 5 輪真實 debug）
- **v5.23–25 後續目標（下一階段，見 §7）**：merge PR #6 → A 階段 model_v5.py 真整合 → tool-tuned checkpoint → 端到端 demo

## 2. 交付物位置（三件，互相獨立可復原）

| 平台 | 位置 | 內容 |
|---|---|---|
| GitHub | `tonythetiger168/helioslm` branch `v5.23-agent-layer`，**PR #6**（未 merge） | 21 檔案歸位至 `helioslm_v5/agent/`、`helioslm_v5/tests/`、`helioslm_v5/docs/`，三個 commit（v5.23 / v5.24 / v5.25） |
| Hugging Face | `chienhsinlin/helioslm-agent` | 同一批 21 檔（`agent/`、`tests/`、`docs/`）+ 本交接文件；`test_v5.py` 為誤傳可刪 |
| 本地（chien 的機器） | `C:\Users\chien\helioslm_tools\` + `helioslm\` working clone | 工具腳本與訊息檔 |

**復原方式**：HF 或 GitHub branch 任一處拉下 21 檔即完整；測試純 stdlib，`python3 tests/test_agent.py` 等 7 個檔即驗證。

**main 現狀（09-24）**：CHANGELOG 停在 v5.22；`v5.23-agent-layer` 分支未 merge——本文件與 CHANGELOG v5.23–25 條目已備好（見同步腳本），隨 PR #6 merge 進 main。

## 3. 交付內容總覽

### P0 — Agent layer（v5.23，9 測試）
| 模組 | 職責 | Oracle / Gate |
|---|---|---|
| `agent/schema.py` | 嚴格 tool-call wire format + 解析（16 類錯誤、深度/大小上限、並行介面預留） | T1 往返、T2 十六錯誤案例 |
| `agent/tools.py` | 確定性工具：AST 白名單 calc、str_op、sandbox 檔案 I/O、finish | T3 等價性 |
| `agent/trajectory.py` | Trajectory serde + `verify_replay` | **T5 硬 oracle**：同 ids ⇒ 同 observations，bitwise；竄改必抓 |
| `agent/envs/toy_envs.py` | calc / str / compose 環境 | T9：ground truth by construction |
| `agent/gate.py` | Fixed / Oracle / ThresholdGate(τ) + routing monotonicity gate | **T6 硬 oracle**：τ-grid 單調，竄改必抓 |
| `agent/loop.py` | plan→act→observe，PARSE_ERROR 恢復 | T4 / T7（direct ≤ routed ≤ oracle）/ T8 |
| `agent/benchmark.py` | 三模式同題集對比 + score_stream JSONL 匯出 | — |
| `agent/finetune_data.py` | tool-tuning 資料管線（teacher = scripted policy） | — |

### P1 — 注意力變體（v5.24，2 測試）
- `agent/dsa.py`：DSA-style sparse top-k decode，**兩層 oracle**——fp32 證書（pseudo-max gap ⇒ dropped mass < 2⁻⁴⁰）⇒ fp64 gate（≤1e-9 vs dense）。明確不宣稱 fp32 bitwise。T11。
- `agent/attn_res.py`：AttnRes-style 層輸出混合，**α 零初始化 ⇒ 與 vanilla residual bitwise 相等（遷移 gate）**；訓練後 ⇒ 確定性 gate。T12。命名 `attn_res_mixing`，待 K3 報告對齊。

### P2 — Disagg evolver 模組（v5.25，4 測試含工具）
- `agent/disagg.py`：Mooncake-style prefill/decode 分離的分析模型（greedy cache-aware 路由 + sojourn 延遲），`DisaggConfig` duck-type 加入 HarnessEvolver 搜尋空間；monotonicity gate（固定 workload 指紋）；三軸 Pareto（makespan / n_workers / worker_seconds）。T13。
- `agent/longctx.py`：needle/RULER 探針（planted ground truth、seeded corpus 跨變體可比）。T16。

## 4. 驗證與修復歷程（證明這套東西真的跑過）

15/15 ALL PASS，經歷 **5 輪真機 debug**——全部失敗都是交付側 bug，被 Windows/Python 3.14 真實執行抓住：

| 輪 | Bug | 類型 |
|---|---|---|
| R1 | `//`/`%` 零除數產生器（×2）、AttnRes 第 0 層 IndexError、行數算錯、tuple seed、DC_B 過高 | 實作疏漏 |
| R3 | `SYSTEM.format` 大括號陷阱（`{"calls"}` 被當佔位符）；StrEnv 尾端空白 | 潛伏 bug，被前置錯誤遮蔽 |
| R4 | **disagg workload 缺陷**：`prefix=i%8` 與 `worker=i%2` 完全相關，round_robin 免費滿命中率 | 測試設計教訓：合成資料要檢查與系統結構的隱含相關 |
| R5 | T7 每次呼叫交替 vs 每 task 交替（finish 首呼叫終止 episode）；benchmark verify 單側 strip | 對執行語義的假設錯誤 |

工程教訓：fix 腳本要冪等；**不要用 PowerShell `Set-Content` 改程式檔**（Windows ANSI 編碼會毀 UTF-8 中文）；`git apply` 的 patch 要 LF 行尾。

## 5. 關鍵設計決策（ADR 摘要）

1. 每個能力附可執行 oracle，agent 層最優先（最難驗證）
2. Toy 環境 ground truth by construction，不用 LLM judge
3. Wire format 預留並行呼叫，行為漸進到位
4. DSA 兩層 oracle：宣稱範圍縮小但為真
5. AttnRes α 零初始化 ⇒ 最強的遷移 gate（bitwise == vanilla）
6. Disagg 效率 claim 用「固定 workload 指紋上的經驗單調性」
7. Tool-tuning 一個 step 一個樣本（先學局部映射，loop 負責組合）
8. 層正確性用 scripted policy 證明，真模型是後續 milestone（誠實 caveat）

## 6. 已知限制與未解阻塞

| 項目 | 狀態 |
|---|---|
| 真 `model_v5.py` 整合（DSA decode 掛點、AttnRes 掛 forward、T15） | **未做**——A 階段文件已備好，需要 repo 實際 forward 程式碼逐行對位 |
| Tool-tuned checkpoint（PR-8） | 未做——`train_tool_tuned.py`（自包含 torch 版）已交付 |
| K3 報告對齊 | 未做——5 規格點萃取表 + instrumentation 已備好 |
| PR #6 merge | **待處理**——本文件 + CHANGELOG v5.23–25 條目已備好隨 merge 進 main |

## 7. 下一步路線圖（優先序，09-24 更新版）

```
✅ v5.23–25 交付（15/15 測試全綠）──→ ① merge PR #6 到 main（含 CHANGELOG + 本文件）
                                        │
② A 階段：model_v5.py 真整合（需 forward 原始碼）──┐
③ B 階段：tool-tuned HeliosLMv5（train_tool_tuned）─┼─→ ④ C 階段：端到端 demo（真模型跑 benchmark 三模式）
⑤ D 階段：K3 對齊實驗 ──→ ⑥ E 階段：AttnRes 定稿
⑦ F：env 擴張（3 → N）→ ⑧ G：agentic RL（rlvr_toy 接 trainer）
```
②③④ 約一週，完成後 HeliosLM 具備「K3 風格架構 + agent 能力 + 可驗證 serving」完整最小閉環。

## 8. 交接操作指引

- **新 session 接手**：說「接手 HeliosLM，讀 PR #6 與 `chienhsinlin/helioslm-agent`」即可取得全部上下文
- **跑測試**：任一目錄下 `python3 tests/test_agent.py`（7 個測試檔，共 15 測試，零依賴）
- **繼續開發**：以 GitHub PR #6 為 base；commit message 範本在 `helioslm_tools/commit_v5*.txt`
- **同步 main**：依 `sync_v523_to_main.ps1` 操作（docs/CHANGELOG 更新 + push）
