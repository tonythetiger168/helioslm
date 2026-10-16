# HeliosLM 改進計劃與 Roadmap（2026 Q4 – 2027 Q2）

- **日期：** 2026-10-15
- **基準：** `main` @ `1fd3358`（v5.53）+ PR #21（審查修正）
- **驗證層級：** `unverified`（計劃文件；第 1 節的排行榜數字來自第三方彙整網站，未經本 repo 重現）
- **相關文件：** [`ROADMAP.md`](ROADMAP.md)（Local Decision Engine 方向）、[`PHASE0_3_PLAN.md`](PHASE0_3_PLAN.md)

---

## 1. 2026 年 10 月 LLM 排行榜概況

資料取自 2026-10-15 的彙整網站。各站的第一名說法不一，請把這裡當作方向參考，不是精確數字。

| 面向 | 領先者 | 備註 |
|---|---|---|
| 綜合智力（Artificial Analysis / BenchLM / LLM Stats） | Claude Opus 5.5 三站皆第一；其後為 Claude Sonnet 5.5、GPT-6.1 Sol／GPT-6 Astra | GPT-6 Astra 在 GPQA Diamond 領先 |
| Arena（人類偏好） | 10/2 時 gemini-4-argon-high 約 1525；多個 Anthropic 模型在 1501–1505 | 來源互相矛盾，以 arena.ai 為準 |
| 開放權重 | MiMo-V2.6-Pro、Qwen3.8 Max、GLM-5.3、Kimi K3、DeepSeek V4.1 Flash | Kimi K3：總 2.8T／啟用 104B 參數，Terminal-Bench 2.1 為 88.3 |
| 實際用量（OpenRouter token 量） | DeepSeek V4.1 Flash 第一；Opus 5.5 第 8 | 實際採用由成本決定 |

### 對 HeliosLM 有意義的四個趨勢

1. **Agent 與程式能力已是主要評量**（Terminal-Bench、SWE-bench Pro、FrontierSWE），傳統聊天型 benchmark 的比重下降。
2. **「推理強度」設定已是分數的一部分。** Artificial Analysis 依強度把同一模型分開排名；「每完成一項任務的成本」正成為標準指標，與 HeliosLM 的 M6 方向一致。
3. **開放權重的架構配方已收斂：**
   - Kimi K3：每 3 層 Kimi Delta Attention（KDA，線性注意力）配 1 層 Gated MLA，加上 Attention Residuals 與 Stable LatentMoE。
   - DeepSeek-V4：Compressed Sparse Attention（CSA），先壓縮 KV cache 再做稀疏注意力。
4. **排行榜彼此矛盾，且許多分數由廠商自報。** 可驗證、可重現的評測仍然稀缺，這正是 HeliosLM 驗證層級（verification tiers）與測試 oracle 文化能補上的缺口。

## 2. HeliosLM 現況

**架構跟得上前沿。** repo 已有：
- KDA 風格的逐通道衰減閘（v5.21）；
- NoPE MLA 與 NVFP4 QAT（v5.22-L）；
- Attention Residual 累加器（v5.5）；
- HCA 128× 壓縮加稀疏 top-k 補救（B1-2），接近 V4 的 CSA；
- IndexShare：跨層組共用的學習式稀疏索引器（B1-1）；
- 1M context 的 YaRN 壓力測試（v5.49）。

**缺口：**

| 缺口 | 現況 |
|---|---|
| 規模 | 最大模型為 mid（360M），GPU 訓練尚未執行；CUDA 從未端到端驗證（issue #1） |
| 評測 | agent 評測只有 12 個合成任務（12/12），沒有標準 benchmark harness，結果無法與他人比較 |
| 生態系 | 真正的 tokenizer（#5）、Hugging Face Hub（#4）、fused 量化 benchmark（#2）都還沒做 |
| 決策引擎 | Level 1 達 0.796 AUROC；Level 2 失敗（約 0.50）；多環境 GRPO 尚未開始 |
| 維護 | `helioslm_v5/CHANGELOG.md` 缺 v5.52／v5.53 條目 |

## 3. 定位

HeliosLM 不應追排行榜，它的優勢在別處：

1. **前沿配方的可驗證參考實作。** 唯一能在 CPU 上閱讀、執行、並逐位元驗證 K3／V4 配方的地方。
2. **校準的決策層**（TrustGate／RLCD）。問題是「模型知不知道自己錯了」——這正是排行榜不量測的。

## 4. Roadmap

### Phase A：基礎與評測（2026 年 10–11 月，v5.54–v5.56）

- **維護：**
  - 合併 PR #21；
  - 補上 v5.52／v5.53 的 CHANGELOG 條目；
  - 在 repo 內建立 milestone 文件。
- **標準評測 harness**，每個結果都標上驗證層級：
  - GSM8K、HumanEval 的小型子集；
  - 以現有工具層建構的終端型沙盒任務（迷你版 Terminal-Bench）。
- **校準排行榜（旗艦成果）：**
  - 使用 `examples/probe_api.py`（已有 opus-5.5、sonnet-5.5、astra、sol、argon 預設）；
  - 讓前沿模型跑 RLCDAlignBench（7,193 題）；
  - 量測「模型知不知道自己錯了」的 AUROC、ECE、Brier，以及每個正確答案的成本；
  - 目前沒有其他排行榜涵蓋這一項。
- **完成標準：** 一張可重現的表格（每個 artifact 都記錄 hash），比較至少 5 個前沿模型。

### Phase B：架構對齊（2026 年 11–12 月，v5.57–v5.60）

- **`k3_block` preset：**
  - 3 層 KDA 配 1 層 Gated MLA，MLA 加上輸出閘；
  - 全層 Attention Residuals；
  - 與既有模組做逐位元 A/B oracle。
- **`v4_csa` preset：** 將 HCA + IndexShare 對齊 V4 CSA「先壓縮再稀疏注意力」的順序，並與 dense attention 比較記憶體與準確度。
- **Stable LatentMoE：** 研究在潛在空間做 expert routing，附負載平衡測試。
- **完成標準：** 每個 preset 都附有「修正前會刻意失敗」的測試，既有 1M context 壓力測試通過，並發佈 K3／V4 比較文件。

### Phase B2：GLM-5.3 與 MiMo 對照（2026-10-15 新增）

對照智譜 GLM-5.3 與小米 MiMo（V2-Flash／V2.6-Pro）的公開資料後，找出 HeliosLM 尚缺的設計。依「每個機制都附 oracle 測試」原則，每項都有可驗證的完成標準。

| ID | 項目 | 來源 | 完成標準 |
|---|---|---|---|
| **B2-1** | 逐層交錯的 sliding window／全域注意力（MiMo 式 5:1，窗口 128，第一層全域）＋ `mimo_hybrid_config()` preset | MiMo-V2-Flash | 預設輸出逐位元不變；用 passkey 長 context 探針比較 dense、純 sliding window、混合三者的回憶率，並記錄 KV cache 節省倍數 |
| **B2-2** | 可學習的 attention sink bias（每個 head 一個 sink logit） | MiMo-V2-Flash | 預設關閉、關閉時逐位元不變（開啟時 sink logit 初始化為 0，即多一個 exp(0) 的分母項，不是恆等）；在小窗口下量測有無 sink bias 的長 context 回憶率差異 |
| B2-3 | 前 k 層用 dense FFN（`moe.first_k_dense`） | GLM-5.3（前 3 層）、MiMo（第一個 block） | 預設 0 時逐位元一致；檢查 checkpoint 相容與路由統計排除 dense 層 |
| B2-4 | 真正的 mHC：Sinkhorn 投影成雙隨機矩陣的可學習混合矩陣 | GLM-5.3-Flash | 混合矩陣始終滿足雙隨機約束；單位矩陣初始化時逐位元一致 |
| B2-5 | MiMo 式輕量 MTP head（sliding window ＋ dense FFN） | MiMo-V2-Flash | 用既有 MTP 損益兩平工具比較 acceptance length 與加速比 |
| B2-6 | On-policy 蒸餾訓練器（學生自生成、老師 reverse-KL） | MiMo（MOPD） | 動工前先讀原報告確認 MOPD 定義；接上既有 GRPO／環境框架 |

- **已具備（不需再做）：** lightning indexer 稀疏注意力、IndexShare、3:1 KDA:MLA 交錯（v5.57 `k3_block`）。
- **暫不做：IndexPool**（GLM-5.3-Flash）。公開資料只有名稱，照做等於猜測；等官方技術報告。
- **執行順序：** B2-1 ＋ B2-2 先做（延續長 context 探針，直接檢驗 MiMo 的核心主張），接著 B2-3，再來 B2-4、B2-5、B2-6。
- **資料可信度：** GLM-5.3-Flash 的「3:1 KDA:MLA」只見於第三方分析；MiMo-V2.6-Pro 的 MTP 細節各來源不一致。表中只採官方或技術報告可佐證的部分。

### Phase B3：Qwen 對照（2026-10-15 新增）

對照 Qwen3 / Qwen3-Next / Qwen3.5 的公開架構。Gated attention（`output_gate`）、線性與全注意力混合、QK-Norm、MTP、細粒度 MoE 加共享專家都已具備，以下只列缺口。

| ID | 項目 | 來源 | 完成標準 |
|---|---|---|---|
| Q1 | 線性注意力層（GDA）的 q/k/v 加 causal depthwise short conv（kernel 4） | Qwen3-Next Gated DeltaNet | 預設逐位元不變；用 B2 的 recall 任務 A/B 比較混合架構 |
| Q2 | Zero-centered RMSNorm（gain = 1 + w，w 從 0 初始化並參與 weight decay） | Qwen3-Next | 預設逐位元不變；toy 訓練比較 norm 權重最大值與 loss |
| Q3 | Global-batch load-balancing loss | Qwen3 MoE | 與現有 bias-update 平衡法比較專家使用熵與 loss |
| Q4 | `qwen_next_config()` preset（超稀疏 MoE 512 選 10 加 1 共享、3:1 混合、gated attention、MTP） | Qwen3-Next | 結構 preset 加 smoke test，比照 `k3_block_config` |
| Q5（可選） | MTP 推測解碼接受率量測（接上 `bench_spec_breakeven`） | Qwen3-Next | 報告 acceptance length 與加速比 |

- **不跟：** 規模與資料策略（例如 36T token 預訓練），與「可驗證、純 PyTorch」的定位無關。

### Phase B4：美國模型對照（2026-10-16 新增）

對照 OpenAI（gpt-oss）、Google（Gemma 3／4）、Meta（Llama 4）、Microsoft（Phi-4-mini-flash／SambaY）、NVIDIA（Nemotron-H／Nemotron 3）、AI2（Olmo 3）與 Arcee（Trinity）的官方報告、model card 與程式碼。已具備：attention sink（gpt-oss）、5:1 local:global 加 QK-norm（Gemma 3）、MXFP4 QAT、LatentMoE、MTP、多環境 GRPO。Anthropic 與 GPT-5 的架構未公開，只當概念參考。

| ID | 項目 | 來源 | 完成標準 |
|---|---|---|---|
| **U1** | GRPO 改成 DAPO／OlmoRL 版本：token-level loss、clip-higher、零梯度組過濾與補樣、可關閉的 KL 與 std 正規化、truncated importance sampling | Olmo 3 | 全部旗標關閉時 loss 與梯度逐位元不變；oracle：全同分組貢獻 0、新舊 logprob 相同時 TIS 權重全為 1 |
| **U2** | 思考預算控制（訓練時隨機截斷思考軌跡；推論到預算強制插入 `</think>`） | Nemotron 3 Nano；gpt-oss low/med/high | oracle：思考 token 數 ≤ B 且到 B 必插入結束標記；小型數學集上準確率對預算單調 |
| U3 | NoPE 層的注意力溫度縮放 `log(floor(pos/8192)+1)*0.1+1` | Llama 4（vLLM 原始碼） | 關閉時逐位元不變；pos < 8192 時縮放恰為 1；4 倍訓練長度 passkey A/B |
| U4 | p-RoPE（partial rotary 0.25）與 global 層 K=V | Gemma 4 | partial=1.0 時與現有 RoPE 逐位元一致；K=V 時 global 層 KV bytes 精確減半 |
| U5 | Clamped SwiGLU（含 up+1） | gpt-oss | limit=∞ 且不加 1 時逐位元一致；輸出受 limit 限制 |
| U6 | 跨層 KV 共享（末 N 層共用）或 SambaY 的 Gated Memory Unit | Gemma 4、Phi-4-mini-flash | KV cache 精確降為 (L−N)/L；小模型 ppl A/B |
| U7 | SMEBU（軟截斷 momentum expert bias）加 depth-scaled sandwich norm | Arcee Trinity | 關閉時逐位元不變；專家負載最大違反量方差下降 |

- **執行順序：** U1 → U2 → U5 → U3 → U4 → U6 → U7。U1 直接改善 AC2 的 agent RL（目前用的是序列級 ratio 加 k3 KL）。
- **資料可信度：** 調查時沙箱連不到 arxiv／huggingface，主要來源以搜尋摘錄核對；Gemma 4 的 PLE 適用範圍、Llama 4 部分訓練細節為第三方說法。

### Phase TW：台灣繁體中文資料（2026-10-16 新增）

詳見 [`TW_DATASETS_2026Q4.md`](TW_DATASETS_2026Q4.md)。

| ID | 項目 | 完成標準 |
|---|---|---|
| TW1 | 授權稽核工具：逐一讀取 HF 卡片的 `license:` 與 gated 狀態，輸出可商用／非商用／未知的清單 | 清單進 `benchmarks/`，每列附來源 hash |
| TW2 | zh-TW 預訓練資料管線：zh Wikipedia（OpenCC `s2twp`）＋ FineWeb-2 中文子集繁體過濾＋ data.gov.tw，約 1 億 token | 去重與繁體比例統計可重現；每份資料保留授權與標示 |
| TW3 | 繁中評測接入 harness：TMMLU+、TMLU（本地，不再散布）、TW-JusticeBench | 參考答案 gate 100%、空答案 0%，比照 Phase A |
| TW4 | 繁中 tokenizer 覆蓋率量測（與現有 tokenizer 比較每字 token 數） | 報告可重現 |

- **原則：** 用到 NC 資料時，權重與衍生資料標為非商用；不再散布任何非 CC BY／OGDL 的原始資料。

### Phase AC：Agentic coding（2026-10-15 新增）

排行榜的主戰場已轉向 agent 與程式能力（Terminal-Bench、SWE-bench Pro）。Qwen3-Coder 的公開做法是大規模 agent RL（多輪、以執行結果為獎勵）、「難解但容易驗證」的程式 RL、repo 級長 context 與穩定的 tool-call 格式。HeliosLM 已有 10 題終端沙盒（T90）、HumanEval 子集（T89）、GRPO 設定、`train_tool_tuned.py` 與 TrustGate。

| ID | 項目 | 完成標準 |
|---|---|---|
| **AC1** | Mini SWE 任務集：約 20 個小 repo，各含 bug 與隱藏測試；agent 讀檔、產生 patch、跑測試 | 沙盒執行可重現；參考解答 gate（正解必過、空 patch 必敗）；結果標驗證層級 |
| **AC2** | Agent RL 迴圈：GRPO 接上沙盒，多輪 rollout，以測試通過率為獎勵 | toy 模型在 AC1 簡單子集上通過率明顯高於訓練前；固定 seed 可重現 |
| **AC3** | Tool-call 格式與解析器加固（JSON schema 驗證、截斷或錯格式恢復） | fuzz 測試：錯格式輸入不 crash，量測恢復率 |
| **AC4** | Repo 級長 context 探針：在大量檔案中找到定義並正確引用 | 沿用 B2 的 SWA/global 與 CSA，比較 dense、混合、壓縮三種架構 |
| **AC5** | 知止：校準「我修好了嗎／該提交了嗎」的信心 | 在 AC1 上報告自報完成與真實結果的 AUROC/ECE，連同每完成任務成本放上校準排行榜 |

- **執行順序：** 先收尾 B2-1／B2-2，接著 AC1 → AC5 → AC3 → AC2 → AC4；Q1 可與 AC1 平行（它直接沿用 B2 的 recall 任務）。
- **差異化：** AC5 是本 repo 的獨特賣點——排行榜不量「agent 知不知道自己真的做完了」，正好呼應「知止」。

### Phase C：規模與生態系（2026 年 12 月 – 2027 年 2 月）

- CUDA 端到端驗證（#1），接著在 GPU 上做 fused vs unfused 量化 benchmark（#2）。
- 訓練真正的 BPE tokenizer（#5），再以 `k3_block` 架構做約 1B 參數的 GPU 訓練。
- Hugging Face Hub 的 checkpoint 與 config 上傳／下載（#4）。
- **完成標準：** 1B 模型能在 Phase A 的 harness 上重現結果，且結果包含 M6 每完成任務成本。

### Phase D：決策引擎（2027 年 Q1–Q2）

- 在 1B 模型上以 RLCD 加 Brier loss 訓練 DecisionHead，目標 AUROC 高於 0.85（超越 Level 1 的 0.796）。
- **多環境 GRPO**：數學、程式、對齊稽核三種環境（即尚待進行的 Phase 2）。
- 在真實任務上校準 TrustGate CLARIFY 門檻，目標同時降低錯誤執行與不必要的升級。
- 把校準排行榜的結果寫入論文（`docs/paper_draft_2026-10-03.tex`）。

### 追蹤指標

| 指標 | 現在 | Phase D 結束時目標 |
|---|---|---|
| 可比較的標準 benchmark 數 | 0 | ≥4 |
| 最大訓練模型 | 360M | 約 1B |
| 決策 AUROC | 0.796 | ≥0.85 |
| oracle-tested 以上的 artifact 比例 | 部分 | ≥60% |

**建議起點：** Phase A 的校準排行榜——這是其他專案都沒有提供的項目，而且不需要 GPU。

## 來源

- [Artificial Analysis LLM Leaderboard](https://artificialanalysis.ai/leaderboards/models)
- [BenchLM — October 2026](https://benchlm.ai/)
- [BenchLM Open-Source Leaderboard](https://benchlm.ai/best/open-source)
- [BenchLM Coding](https://benchlm.ai/coding)
- [BenchLM Leaderboard History](https://benchlm.ai/llm-leaderboard-history)
- [LLM Stats](https://llm-stats.com/leaderboards/llm-leaderboard)
- [OpenRouter Rankings](https://openrouter.ai/rankings)
- [Manifold: Arena #1 on Oct 31, 2026](https://manifold.markets/Terminator2/which-company-will-have-the-1-model)
- [modelgrep Leaderboard](https://modelgrep.com/leaderboard)
- [Kimi K3 Technical Report (arXiv 2607.24653)](https://arxiv.org/pdf/2607.24653)
- [DeepSeek-V4 Report (arXiv 2606.19348)](https://arxiv.org/pdf/2606.19348)
- [LongCat Sparse Attention (arXiv 2608.01662)](https://arxiv.org/pdf/2608.01662)
- [Morph: Best Open Source LLMs 2026](https://www.morphllm.com/best-open-source-llm)
- [geotoolbox: Best Open-Source LLMs October 2026](https://geotoolbox.ai/blog/best-open-source-llms)

Phase B2（GLM／MiMo 對照）：

- [NVIDIA GLM-5.3 model card](https://build.nvidia.com/z-ai/glm-5-3/modelcard)
- [Z.ai GLM-5.3-Flash 文件](https://docs.z.ai/guides/vlm/glm-5.3-flash)
- [SGLang GLM-5.3](https://docs.sglang.io/cookbook/autoregressive/GLM/GLM-5.3)
- [SemiAnalysis GLM-5.2/5.3](https://inferencex.semianalysis.com/model/glm-5-3)
- [daily.dev GLM-5.3-Flash notes](https://daily.dev/posts/glm-5-3-flash-architecture-notes-4pnix3fdg)
- [MiMo-V2-Flash Technical Report (arXiv 2601.02780)](https://arxiv.org/abs/2601.02780)
- [MiMo-V2-Flash GitHub](https://github.com/xiaomimimo/MiMo-V2-Flash)
- [MiMo-V2.6-Pro-MOPD (Hugging Face)](https://huggingface.co/XiaomiMiMo/MiMo-V2.6-Pro-MOPD)
- [MiMo-V2.6-Pro-RL (Hugging Face)](https://huggingface.co/XiaomiMiMo/MiMo-V2.6-Pro-RL)
- [MiMo-V2.5 inference optimization (arXiv 2607.13095)](https://arxiv.org/pdf/2607.13095)
