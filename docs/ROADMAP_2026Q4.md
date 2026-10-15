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
| **B2-2** | 可學習的 attention sink bias（每個 head 一個 sink logit） | MiMo-V2-Flash | 初始值設定成與現在逐位元一致；在小窗口下量測有無 sink bias 的長 context 回憶率差異 |
| B2-3 | 前 k 層用 dense FFN（`moe.first_k_dense`） | GLM-5.3（前 3 層）、MiMo（第一個 block） | 預設 0 時逐位元一致；檢查 checkpoint 相容與路由統計排除 dense 層 |
| B2-4 | 真正的 mHC：Sinkhorn 投影成雙隨機矩陣的可學習混合矩陣 | GLM-5.3-Flash | 混合矩陣始終滿足雙隨機約束；單位矩陣初始化時逐位元一致 |
| B2-5 | MiMo 式輕量 MTP head（sliding window ＋ dense FFN） | MiMo-V2-Flash | 用既有 MTP 損益兩平工具比較 acceptance length 與加速比 |
| B2-6 | On-policy 蒸餾訓練器（學生自生成、老師 reverse-KL） | MiMo（MOPD） | 動工前先讀原報告確認 MOPD 定義；接上既有 GRPO／環境框架 |

- **已具備（不需再做）：** lightning indexer 稀疏注意力、IndexShare、3:1 KDA:MLA 交錯（v5.57 `k3_block`）。
- **暫不做：IndexPool**（GLM-5.3-Flash）。公開資料只有名稱，照做等於猜測；等官方技術報告。
- **執行順序：** B2-1 ＋ B2-2 先做（延續長 context 探針，直接檢驗 MiMo 的核心主張），接著 B2-3，再來 B2-4、B2-5、B2-6。
- **資料可信度：** GLM-5.3-Flash 的「3:1 KDA:MLA」只見於第三方分析；MiMo-V2.6-Pro 的 MTP 細節各來源不一致。表中只採官方或技術報告可佐證的部分。

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
