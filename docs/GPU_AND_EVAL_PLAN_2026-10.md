# GPU 需求（#40–#45）與 LLM 排行榜檢視、改進計劃（2026-10-21）

**資料可信度：** 本文的規格、價格與排行榜數字都來自 2026-10-21 的網路搜尋摘要，研究過程無法直接開啟原始頁面。下單租機或對外宣稱名次前，必須回到原始頁面再核對一次。標「估」的數字是本文自行推算的，推算方法寫在 §1.2。

## 1. 每一代需要哪種 GPU

### 1.1 GPU 規格與租價（每張 GPU，dense 算力）

| GPU | 記憶體 | BF16 dense | FP8 dense | 隨需租價（USD/小時，2026-09～10） |
|---|---|---|---|---|
| RTX 4090 / 5090 | 24 / 32 GB | 5090 約 0.42 PF（估） | — | 0.34–0.89 / 0.87–1.19 |
| H100 SXM | 80 GB | 0.99 PF | 1.98 PF | 2.9–4.5（Vast、台灣 GMI 約 1.5–2.3；AWS / Azure / GCP 7–12） |
| H200 | 141 GB | 0.99 PF | 1.98 PF | 5.3–6.0（reserved 4–5） |
| B200 | 180 GB | 2.25 PF | 4.5 PF | 6.7–8.6（spot 4.3–5.1） |
| B300 / GB300 NVL72 | 約 288 GB | 2.25 PF | 4.5–5 PF（來源不一） | 7.1–9.5（AWS 17.8） |
| MI300X / MI355X | 192 / 288 GB | 1.3 / 2.5 PF | 2.6 / 5.0 PF | 1.7–3.5 / 3–6 |

- 2026 年租價一路上漲：H100 一年約的價格指數 2025-10 → 2026-03 漲了 38%。
- 以每美元買到的 BF16 算力計：B200（2.25 PF ÷ 約 6.7 美元）比 H100（0.99 PF ÷ 約 3.5 美元）便宜約 20%，而且記憶體多 2.25 倍。
- **AMD 暫不建議用於訓練。** 本 repo 連 CUDA 端到端都還沒驗證（#1），ROCm 路徑完全沒測過。

### 1.2 估算方法

GPU 小時是用兩個公開的實際訓練紀錄等比推算的：

- **Dense 模型：** SmolLM3-3B 訓練 11.2T token，用了約 22 萬 H100 小時（MFU 29%）。換算約 6,500 H100 小時／每 1B 參數／每 1T token。
- **MoE 模型：** DeepSeek-V3（37B active）預訓練 14.8T token，用了 266 萬 H800 小時。換算約 4,900 小時／每 1B active 參數／每 1T token。
- **調整：** active 參數小的 MoE 通訊佔比較高，所以 30B-A3B 那一列再乘 1.5。

### 1.3 各代需求

| Issue | 目標 | Token 數（建議） | H100 小時（估） | 成本（估，每 H100 小時 2–3.5 美元） | 最少 GPU（只看記憶體） | 建議配置 |
|---|---|---|---|---|---|---|
| #40 v1.0 | 0.5B dense | 100B | 約 330 | 700–1,200 美元 | 1 張 24 GB 卡 | 租 1 台 8×H100 跑約 2 天。RTX 4090/5090 只適合開發和 100M 試跑 |
| #41 v2.0 | 3B dense | 1T（最低 60B） | 約 2 萬（60B token 約 1,200） | 4–7 萬美元 | 1 張 80 GB 卡（實務上 8 張） | 32–64 張 H100 或 H200，約 2 週 |
| #42 v3.0 | 30B-A3B MoE | 2T | 3–4.5 萬 | 6–16 萬美元 | 8 張 H200 或 4–8 張 B200 | 32–64 張 H200 或 B200；expert parallel 必須留在同一個 NVLink 節點內 |
| #43 v4.0 | 100B-A10B | 4T | 約 20 萬 | 40–70 萬美元 | 16 張 H200 或 16 張 B200 | 128–256 張 B200 或 GB200 |
| #44 v5.0 | 300B-A20B | 8T | 約 78 萬 | 160–270 萬美元 | 48–64 張 H200 | 256–512 張 B200 或 B300，或 GB300 NVL72 |
| #45 v6.0 | 1T-A40B | 15T | 約 290 萬 | 580–1,000 萬美元 | 約 72 張 288 GB 卡（一個 GB300 NVL72） | 1,000–2,000 張 B200 或 GB300，訓練約 2 個月 |

補充說明：

- 記憶體下限按每個參數 16 bytes 計（bf16 權重、梯度，加上 fp32 Adam 狀態）。MoE 要用總參數算，不是只算 active 參數。activation 另外要再加。
- 若改用 Blackwell 的 FP8 訓練，同樣的 token 數大約只需上表一半的 GPU 小時。但本 repo 的 FP8 trainer 還沒在 GPU 上驗證過。
- **#43–#45 的成本是七到八位數美元。** 在 #42 的 30B-A3B 證明「每單位品質的成本」以前，不應該承諾這三代。這和 #45 issue 本身寫的 gate 一致：沒有證明過訓練經濟性，就不做 1T。

### 1.4 建議順序

1. **v5.83 100M 試跑。** 用 1 張 RTX 4090 或 5090（自有或租，約 0.5–1.2 美元／小時），或 1 張 H100。目的是驗證 loss 曲線、MFU，以及中斷後能接續。
2. **#1 CUDA 驗證。** 同一台機器就能做，是後面所有工作的前提。
3. **#40 0.5B。** 租 1 台 8×H100，約 1–2 天。
4. **#41 3B。** 先跑 60B token（約 4,000 美元）作為 gate，評測通過後才加碼到 1T。
5. **#42 30B-A3B。** 用 H200 或 B200 節點，前提是 #41 的評測證明 pipeline 可靠。

## 2. 全球排行榜與 benchmark 現況（2026-10）

| 類型 | 代表 | 現況 |
|---|---|---|
| 人類偏好 | LMArena（text / webdev / vision） | 前段差距只有數十分。有「Leaderboard Illusion」爭議：私下測試多個變體、取樣不均 |
| 綜合指數 | Artificial Analysis Intelligence Index v4.x | 2026 年改版多次，不同版本的分數不可直接比較；有依參數量分級的 tiny（≤4B）和 small 榜 |
| 開源舊榜 | HF Open LLM Leaderboard | **已退役**（v2 約 2025-03 宣布退役，理由是題目飽和） |
| 程式 / agent | SWE-bench Verified → **SWE-bench Pro**、Terminal-Bench 2.0、OSWorld | Verified 有污染問題，OpenAI 已在 2026-02 停用；目前以 Pro（部分私有）和 Terminal-Bench 為主 |
| 推理 / 知識 | HLE、ARC-AGI-3、MathArena AIME 2026、GPQA | 用新題或私有題來對抗污染 |
| 工具呼叫 | BFCL v4、tau2-bench | Epoch AI 評 BFCL v4 為「Flawed」（抽查 50 題有 24 題有問題） |
| 中文 | OpenCompass、C-Eval、CMMLU | MCQ 類在前沿已飽和 |
| **繁體中文** | **TMMLU+**（2026-09 推出 v1.1，22,690 題）、TMLU、Open TW LLM Leaderboard、TCEval | 繁中專屬榜的維護多半停滯，HeliosLM 在這裡最有機會做出差異 |

**小模型的比較對象：**

- ≤4B：SmolLM3-3B（完全開放、可重現，最適合當 baseline）、Gemma 4 E2B / E4B（Apache 2.0）、Qwen3.5 0.8B / 2B。
- 30B-A3B 級：Qwen3.5-35B-A3B（256 experts，8 個 routed 加 1 個 shared）。

**哪些可信：**

- 前沿模型：MMLU、GSM8K、HellaSwag、C-Eval 已飽和，不能拿來宣稱名次。
- **0.5B–3B 模型：** 這些題目仍然有鑑別力。
- 可信度較高：題目在模型訓練截止後才出的評測（LiveCodeBench、MathArena），以及私有測試集（SWE-bench Pro、ARC）。

## 3. HeliosLM 現況差距

1. **沒有任何模型能上榜。** 所有證據都是 CPU 玩具規模，repo 裡也沒有標準評測 harness 的介接。
2. **沒有污染檢查。** `pilot_prepare_data.py` 還不會比對訓練資料和評測集。
3. **沒有同條件 baseline。** 「比誰好」必須在同一個 harness、同樣 few-shot 設定下比較。
4. **TW3（繁中評測放進 harness）尚未開始。**
5. **多數結果是單一 seed。** #55 已提供 `--seeds`，但還沒有多 seed 的 artifact。

## 4. 改進計劃

| 編號 | 項目 | 需要的資源 | 驗收標準 |
|---|---|---|---|
| **E1** ✅ | lm-evaluation-harness 介接（`--model helioslm`） | CPU | 已完成（v5.92）：`eval/lm_harness.py`、`examples/eval_harness.py`；離線測試以 lm-eval 自己的 `simple_evaluate` 驗證 |
| **E2** ✅ | 污染檢查：13-gram 比對訓練 shard 與評測集，報告重疊比例 | CPU | 已完成（v5.88）：`eval/decontam.py`、`pilot_prepare_data.py --decontaminate`；人造洩漏（英、中）能被抓到 |
| **E3** | 繁中評測包（TW3）：TMMLU+ v1.1、TMLU、TW-JusticeBench，加 C-Eval / CMMLU 對照 | CPU 或 GPU | 授權照 TW1 審核結果；只在本地評測，不再散布題目 |
| **E4** | 同條件 baseline 表：SmolLM3-3B、Qwen3.5-0.8B / 2B、Gemma 4 E2B 用同一個 harness 重跑 | 1 張 GPU，約 10–30 美元 | `benchmarks/baseline_eval_*.json` 附 harness 版本與 few-shot 設定 |
| **E5** ✅ | 預先登記評測計劃：每一代訓練前先 commit 評測設定、題目版本與 gate 門檻 | — | 工具已完成（v5.88）：`eval/prereg.py` 鎖定計劃 hash，改過門檻後舊結果失效。v1.0 的實際門檻待負責人決定 |
| **E6** | 新題評測：LiveCodeBench（只取訓練截止後的題目）、MathArena AIME 2026 | GPU | 只在 3B 以上才有意義 |
| **E7** | Agent 評測：Terminal-Bench 2.0、SWE-bench Pro（Docker sandbox），沿用 AC1 / AC5 的知止指標 | GPU 節點 | 只在 30B-A3B（#42）以上才有意義 |
| **E8** | 對外發表：模型卡、可重現的資料 manifest（SmolLM3 式）、多 seed 結果；投 Open TW LLM Leaderboard 和 Artificial Analysis tiny 榜 | — | 3B 通過 E1–E5 之後才做 |

**差異化定位（不靠參數量）：**

1. **繁中第一：** 有 Big5 seed tokenizer（v5.80，每個中文字 3 → 1 個 token），以及 TMMLU+ / TMLU 本地評測。
2. **可驗證：** 每項宣稱都附 bit-exact oracle 和 artifact hash，結果可重現。
3. **小而可重現：** 3B 等級做到 SmolLM3 式的完全開放，並附多 seed 結果。

**下一步：** E1、E2、E5 已完成。下次開 GPU pod 時，同時跑 E4（同條件 baseline）與 pilot checkpoint 的評測；E3 接在 TW3 之後。
