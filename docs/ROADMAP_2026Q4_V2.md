# HeliosLM v2 改善計劃 — 2026-11 全球技術對齊

- **日期：** 2026-11-24
- **版本：** v2（取代 [`ROADMAP_2026Q4.md`](ROADMAP_2026Q4.md) v1 的技術細節；策略層「可驗證參考實作＋校準決策層」不變）
- **基準：** `main` @ `0272cebc`（PR #302，mid candidate 五-track 正式化）
- **驗證層級：** `unverified`（計劃文件）。外部數字一律標「廠商自報」或「第三方」；本 repo 自測數字標「自測」並附 artifact 位置。

---

## 一、2026-11 技術版圖（調查摘要）

### 1. Agentic 評測是主戰場，我們已有可重現的 trained-candidate 基線

- **PR #302（2026-11-23 合併，squash `0272cebc`，自測）**：mid candidate（`mid_sft_v5.33`，359,992,352 params，sha256 `fb0cc813…b554d6`）五-track 正式結果——terminal 0/50、swe 0/20、long_horizon 0/5、tool_recovery 0/70、hdl 0/156，**Agentic Index 0.000000**（`is_model_score: true`，`--check` 位元級通過；artifacts 備份於 `mid_run_backup/results_mid-v5.33/`）。失敗模式：五個 harness protocol 對 chat-SFT mid 全是 OOD。
- **Grounded chat 對照組（自測，與 `checkpoints/mid_agent_eval_v5.json` 位元級一致）**：FixedGate 0/12（幻覺工具參數）；GroundingGate 9/12 correct + 3/12 abstained（premature-finish refusal，無幻覺外洩）。啟示：問題是未驗證的自信，不是純能力缺口。

### 2. K3.1 Pro 當老師／對標（計劃已落地，見 `docs/K31_DISTILL_PLAN.md`）

- K3.1 Pro：開放權重 2.78T total／104B active（MXFP4 checkpoint 約 1.56 TB），API 價格 $2／1M input、$10／1M output（廠商自報）。
- 硬約束已定：不做 logit-level KD（vocab 不合、1.56 TB 不可行）；teacher output 是 untrusted data（過 `data_quality.py`＋`decontam.py`）；license 先查；廠商自報數字分列。
- Route A（trajectory 蒸餾）、Route B（preference pairs）、Route C（external reference row，先做、<$1、約 50 episodes）。Route C 目前只缺一個 API key。

### 3. Gemma 4 家族 — HDL track 底模定案：31B dense

- **Gemma 4 31B dense（2026-04-02 發佈，廠商自報數字）**：Apache 2.0（無 MAU cap，相對 Gemma 3 的 custom terms 是關鍵改善）；256K context；MMLU-Pro 85.2、GPQA Diamond 84.3、LiveCodeBench v6 80.0、Codeforces ELO 2150。家族另有 26B-A4B MoE（3.8B active；MMLU-Pro 82.6、LCB 77.1）、12B、E4B、E2B；官方提供 Apache 2.0 的 MTP drafter 變體（第三方實測稱可約 2× 程式 agent 生成吞吐）。
- **硬體現實**：Q4 約 18GB（單張 4090/5090 級可推論）；BF16 約 64GB（單張 80GB 級）。本開發沙箱（2 CPU／3GB）跑不動，SFT 與 KD teacher 角色皆需 pod／GPU，屬訓練類動作、到時需明確授權。
- **同家族同 tokenizer**：K31 計劃的 logit-KD 禁令只針對 K3.1（vocab 不合）。Gemma 4 家族內（31B teacher → 較小 Gemma 4 學生）可做 logit-level KD——repo 現成的 `examples/prune_kd_gemma4.py`（prune：block-influence 層重要性＋channel 重要性＋KV-shared＋PLE；KD：forward KL、chunked logits、WSD schedule、`--hours` sizing）直接適用，只需把規模從 E2B/E4B pilot 換到 31B teacher。

### 4. Verilog／RTL 評測版圖 — 舊榜飽和、新榜有空間

- **VerilogEval 已接近飽和（第三方）**：frontier 模型約 88–89.7% pass@1；繼續刷它的邊際價值低。
- **領域 SFT 仍然值錢（第三方）**：RTL-Coder 6.7B 專門 fine-tune 後 VerilogEval 35.9%，勝過多款更大的通用模型；Gemma-7B 原版在 VerilogEval-Human 僅 17.3% pass@1。通用 code 能力決定起點，Verilog-specific SFT 決定終點——這正是「31B 底模＋領域 SFT」的依據。
- **RTL-BenchLS（arXiv 2606.08976，2026-06-08，學術）**：10,028 個 formally verified Verilog designs（平均 63 LoC、最大 479），三種 self-supervised reasoning task（round-trip／masked-content／repository-issue），全部用 formal equivalence checking 驗證、不需人工 testbench。最好模型僅約 23%（round-trip）／28%（masked）／12%（repo-issue）——頭部空間大，且與本 repo 的 oracle 文化合拍。
- 啟示：HDL track 的 eval 應從「只有自家 H4」擴成 H4＋VerilogEval＋RTL-BenchLS 三層。

### 5. Dependency 上游（2026-11 查證）

- torch 2.15.0 stable（現 pin：2.14.1 for py<3.15／2.15.0 for py≥3.15）；numpy 2.5.5 patch（現 pin 2.5.4）。`requirements-ci.txt` 檔頭原則不變：bump 走獨立 PR、重跑完整套件。

---

## 二、差距分析（`main` @ `0272cebc` vs 前沿）

| 前沿 | 現況（自測） | 差距 | 對策 |
|---|---|---|---|
| agentic 五 protocol 能力 | Agentic Index 0.000000，五 protocol 全 OOD | 🔴 | Route C 對標＋Route A 蒸四 protocol（改善一、三） |
| 外部參照行（可比較數字） | 無；排行榜只有自家 candidate | 🔴 | Route C external reference row（改善一） |
| dependency pins 時效 | torch 2.14.1 vs 上游 2.15.0；numpy 2.5.4 vs 2.5.5 | 🟡 | 獨立 bump PR（改善二） |
| Verilog 領域能力 | hdl 0/156（121 syntax／35 unknown-module） | 🔴 | Gemma 4 31B 底模＋領域 SFT＋家族內 logit KD（改善四） |
| MTP 訓練參數與推論策略耦合 | `use_mtp`×`thinking_budget` 互斥 raise 在同一層 | 🟡 | 分層重構（改善五） |
| 上游 Terminal-Bench 可比分數 | Phase 2 只有計劃，無 adapter、無 pod run | 🟡 | 可行性評估→adapter（改善六） |
| HDL eval 覆蓋 | 只有自家 H4（156 題） | 🟡 | 加 VerilogEval＋RTL-BenchLS（改善四一部分） |

**核心洞察：** 這一輪的勝出公式不是「更大的底模」，而是「對的 teacher＋可驗證的 protocol 資料＋領域 SFT」——K3.1 管 agentic protocol、Gemma 4 31B 管 Verilog 領域、两者產出的資料全部過既有的 quality/decontam gate，能力缺口用資料分佈補，不用參數補。

---

## 三、改善計劃 v2

### 改善一：Route C — external reference row（M1，最高優先，等 API key）

- 依 `docs/K31_DISTILL_PLAN.md` Route C：K3.1 Pro 跑約 50 個 agentic episodes（五-track 子集），費用 <$1，結果作為 external reference row 寫入 `docs/AGENTIC_BENCH.md`，與自家 candidate 分列、廠商自報與自測分列。
- **先行項（不需 key）**：把 Route C 腳本寫好（endpoint 抽象、episode 記錄格式、成本記錄、decontam 掛鉤），key 一到即跑。
- 驗收：reference row 進文件，每個 episode 附 hash 與成本；總費用 <$1；CI 全綠。

### 改善二：Dependency pins bump（M1，獨立 PR，可與改善一並行）

- torch 2.14.1→2.15.0（py<3.15 側）、numpy 2.5.4→2.5.5；逐 pin 檢查 changelog 的 breaking change；GPU pod 側 `torch 2.14.1+cu130` 屬 DEVICE_MATRIX 層級，**不在此 PR 內動**。
- 驗收：雙矩陣（py3.11＋py3.12）CI 全綠、228/228、integration_test_v51 與 CVDP step 通過；bump 單獨成 PR，不夾帶其他改動。

### 改善三：Route A — 蒸四個 agentic protocol（M2–M3，權重 0.90）

- 對象：terminal、swe、long_horizon、tool_recovery 四個 protocol（合佔 Agentic Index 權重 0.90）。task 來源：`term_bench.py` Phase-1 任務、swe 迷你集、long_horizon 任務、tool_recovery 修復格式；teacher 軌跡過 verifier 過濾＋`data_quality.py`＋`decontam.py` 後進 `sft_hf.py`。
- 產出 SFT 候選後重跑五-track 正式化（沿用 PR #302 的 runner 與 `--check` 流程）。
- 驗收：新候選在 ≥3/4 protocol 上 Agentic Index 分量 > 0（現況全 0）；grounded-chat standing regression gate 不退步（維持 9/12 correct + 3/12 abstained、0 幻覺外洩）；結果文件比照 `AGENTIC_MID_CANDIDATE.md` 格式。

### 改善四：HDL track — Gemma 4 31B 底模（M3 起，殿後，權重 0.10）

順序上殿後不變，但底模自此定案為 **Gemma 4 31B（Apache 2.0）**。四個子項：

1. **HDL4-1 資料**：現有 H4 task set＋31B 生成的 synthetic problem-code pairs（VerilogEval 已證明 self-instruct 路線有效）；全部過 `decontam.py`（對 H4／VerilogEval／RTL-BenchLS 三個 eval set）；provenance 記進 `docs/HDL_DATA.md`（license 已確認 Apache 2.0，仍建檔）。
2. **HDL4-2 SFT**：`sft_hf.py` 對 31B 做 Verilog SFT（pod／GPU，需授權；BF16 約 64GB，Q4 推論約 18GB）。
3. **HDL4-3 家族內 KD**：`examples/prune_kd_gemma4.py` 以 31B 為 teacher 蒸到較小 Gemma 4 學生（同 tokenizer，logit KD 解禁）；目標是讓進 serving 層的是小學生，不是 31B 本身。
4. **HDL4-4 eval 三層化**：自家 hdl track（iverilog，156 題）＋ VerilogEval ＋ RTL-BenchLS；iverilog/vvp 只在 pod／container 跑（untrusted code 紀律不變）。
- 驗收（分段）：第一階段 hdl track syntax-error 率從 121/156 降到 <50%；第二階段 hdl pass@1 > 0；第三階段 VerilogEval 達 RTL-Coder 6.7B 級（約 35.9%，第三方數字）或以上；RTL-BenchLS 只報三 task 自測值、不設硬門檻（新榜無可比基線）。

### 改善五：MTP 分層重構（M2–M3，可與改善三並行）

- 把「MTP 訓練／結構參數」（`config_v5.py`、`model_v5.py` 的 `use_mtp` 路徑）與「推論策略」（`inference/thinking_budget.py` 的 thinking budget）分層；現行 `use_mtp`×`thinking_budget` 互斥 raise 是切入點，重構後互斥規則移到策略層並附明確錯誤訊息。
- 參照：B2-5（MiMo 輕量 MTP head）、Q5（MTP 推測解碼接受率）、Gemma 4 官方 MTP drafter（Apache 2.0，第三方稱程式 agent 吞吐約 2×）——先用 `bench_spec_breakeven` 量自家 acceptance length，再決定要不要跟進 drafter 式設計。
- 驗收：預設行為逐位元不變（現有 228 測試全過，含 `test_thinking_budget.py`、`case_model_mtp.py`）；新增互斥規則的 oracle 測試（修正前會刻意失敗）。

### 改善六：term_bench Phase 2 — 上游 Terminal-Bench（M3–M4，pod）

- 三個暫停項重啟：上游 adapter（`tbench_adapter.py`，task.toml→term_bench protocol）、contamination control（對上游 task text）、pod run 可行性（上游任務假設 docker，pod 無 docker，需替代執行層）。
- 先做可行性評估報告，過了才寫 adapter。
- 驗收：可行性報告進 `docs/`；adapter 完成後至少一個上游 task 端到端跑通並標驗證層級；結果寫入 `AGENTIC_BENCH.md`。

---

## 四、風險與對策

| 風險 | 對策 |
|---|---|
| Route C 缺 API key，時程不可控 | 先行項腳本先寫（無沉沒成本）；key 長期不到則 Route C 降級為「文件占位」，不阻塞改善三 |
| Gemma 4 31B 數字全是廠商自報，實際 Verilog 起點未知 | 任何規模承諾前先用 Q4 在 pod 跑 RTL-BenchLS 小切片自測（一個下午級別）；自測與自報分列 |
| 31B SFT／KD 需要 pod／GPU，費用與授權 | 屬訓練類動作，到時提具體 sizing（`--hours` 估）與費用再請授權；沙箱內只做文件、管線、資料格式、CI 可跑的 toy-scale 測試 |
| Synthetic Verilog 資料污染 eval set | `decontam.py` 對三個 eval set 全跑；generated code 只在 pod／container 經 iverilog 執行 |
| Dependency bump 引入上游 breaking change | 獨立 PR、完整套件重跑；失敗即退 pin，不夾帶其他改動、無沉沒成本 |
| MTP 重構碰到既有行為 | 預設逐位元不變為硬驗收；互斥規則變更附刻意失敗的 oracle 測試 |
| 排行榜／第三方數字彼此矛盾 | 一律標來源源與「廠商自報／第三方／自測」；不自動採信任何單一來源 |

---

## 五、建議執行順序

1. **改善二（dependency bump PR）**——前置條件已全滿（雙矩陣 CI、228/228 可重現基線、main 乾淨），成本最低、立即清掉時效差距。
2. **改善一先行項（Route C 腳本）**——不需 key，與改善二並行；key 到即跑。
3. **改善五（MTP 分層）**——純重構、無外部依賴，與改善三並行。
4. **改善三（Route A 四 protocol）**——權重 0.90 的主菜，等 1–2 就位後全力做。
5. **改善六（term_bench Phase 2 可行性）**——報告先行，過了才寫 adapter。
6. **改善四（HDL／Gemma 4 31B）殿後**——權重 0.10、雙重 OOD、工具鏈最重；底模已定案，文件層先落地，pod 授權後才動 31B。

前瞻項（Gemma 4 MTP drafter 式設計、RTL-BenchLS 硬門檻）留到改善三／四的自測數字出來後再評估。

---

## 來源

- PR #302（squash `0272cebc`）、`docs/AGENTIC_MID_CANDIDATE.md`、`checkpoints/mid_agent_eval_v5.json`（自測）
- [`docs/K31_DISTILL_PLAN.md`](K31_DISTILL_PLAN.md)、`examples/prune_kd_gemma4.py`、`examples/agentic_candidate_run.py`
- [Gemma 4 Architecture and Benchmark Notes（S. Raschka，2026-04-02）](https://sebastianraschka.com/blog/2026/gemma-4-release-notes.html)（第三方彙整廠商自報數字）
- [Gemma 4: All 5 Sizes Benchmarked（Auriga IT，2026-04-03）](https://aurigait.com/blog/gemma-4-features-benchmarks-guide/)（第三方彙整廠商自報數字）
- [RTL-BenchLS（arXiv 2606.08976，2026-06-08）](https://arxiv.org/html/2606.08976v1)（學術）
- [Revisiting VerilogEval（ACM TODAES，10.1145/3718088）](https://dl.acm.org/doi/10.1145/3718088)（學術）
- [A Multi-Expert LLM Architecture for Verilog Code Generation（arXiv 2404.08029）](https://arxiv.org/html/2404.08029v1)（學術；Gemma-2B/7B VerilogEval 數字）
- [RTLLM benchmark（github.com/hkust-zhiyao/RTLLM）](https://github.com/hkust-zhiyao/RTLLM)
- 上游版本查證（2026-11）：torch 2.15.0 stable、numpy 2.5.5
