# HeliosLM 交接文件 — 2026-09-25 全日工作封存

> 狀態： v5.23–v5.29 全部在 GitHub main（11 commits 今日）。
> 下一個工作日從 §6「下次開工接點」開始。本文件取代舊版 handoff（v5.26 版）。

## 1. 版本線總表

| 版本 | 內容 | 驗證 |
|---|---|---|
| v5.23 | Agent layer（schema/tools/trajectory/envs/gate/loop） | 9 測試 |
| v5.24 | DSA sparse decode（兩層 oracle）+ AttnRes（零閘 bitwise 遷移） | T11/T12 |
| v5.25 | Disagg evolver 模組 + longctx 探針 | T13/T16 |
| v5.26 | T15 真模型 oracle（AttnRes 零閘 bitwise、k≥L sparse bitwise、agent loop 真權重煙霧） | 3/3 |
| v5.27 | Tool-tuned checkpoint + T17 end-to-end + **TOOL_ERROR recovery** + replay mirror | T17 PASS |
| v5.27.1 | **ep2 訓練**：parse 0.15→**0.67**、finish 1/9→**4/6**、loss 0.48→0.25 | T17 重測 |
| v5.28 | Disagg 三軸 Pareto 搜尋 + **cache-aware 反單調發現**（記錄不隱藏） | T18 3/3 |
| v5.29 | **三模式 benchmark**（真實信心 τ 曲線嚴格單調、gate PASS）+ **錯答案信心 0.944** 頭條發現 | T19 1/1 |

## 2. 關鍵數字（全部實測、seeded artifact 可復現）

| 指標 | 值 | 出處 |
|---|---|---|
| tool protocol parse（ep2） | 0.67（12/18） | T17 dense |
| sparse K=4 vs dense parse | 0.147 vs 0.15 | T17 sparse_k4 |
| τ 曲線（0.5→0.95 correctness） | 0.000→0.083→0.250→0.333→1.000，嚴格單調 | three_modes_2026-09-25.json |
| 過度自信（錯答案最大信心） | 0.944 | 同上 |
| KV cache 節省（MLA vs MHA） | 71.9% | benchmarks_kv_cache.png（v5.28 重跑） |
| disagg Pareto | cache_heavy 16.2s@2w → 9.4s@12w | disagg_pareto_2026-09-25.json |
| cache-aware 反單調 | 4P4D→6P6D makespan +4.3% | 同上 findings |

## 3. 四大發現（品牌敘事素材，已在 blog 草稿）

1. **錯答案信心 0.944**——toy 模型系統性過度自信；routing 有效只因信心有區分度（str 0.65–0.86 vs calc 0.92–0.94）。v5.22 decision-audit 框架的存在意義被自證。
2. **cache-aware routing 的 worker 反單調**——同 key 序列化在 holder；gate 收斂到 RR ladder，發現記錄不隱藏。
3. **sparse decode agent 場景不塌縮**——K=4 ≈ dense，成本軸可用。
4. **parse 4.4× 證明協議學習是資料/步數問題；correct=0 是 8.5M 骨架的複製上限**——下一步是資料擴量或更大模型，不是更多 epoch。

## 4. 產品線狀態

- **HeliosLM**：main @ v5.29；26 項測試全綠；文件 9 份（docs/：product_roadmap、benchmark_alignment、blog_correctness_first、product_line_kimi_vs_deepseek、competitive_intel_2026-09-25、glm53_raw_claims、k3_alignment_targets+CSV、helioslm_handoff、p0_agent_design）
- **genforge**：定價工程 P0 三項已定案（cache 階梯價 ≤1/10、錯位峰谷 UTC 01–04/06–10 off-peak 5 折、走量檔 ≤$0.30/$1.20），待實作；對標 Seedance 差距已收斂至 30 秒單次生成，待 LTX 2.5 上限
- **情報資產**：Kimi vs DeepSeek 比對（K3 $3/$15 vs V4 Pro $0.66/$1.98；走量檔 DeepSeek 便宜 6×）；AA v4.3 全榜 + GDPVal + LMArena + OpenRouter；GLM-5.3 原始 claim 對照（1769→1655 修正）
- **自動化**：cron 每日 09:00 追蹤 AA 收錄 Hy4（task id 1a0d77ed，state 在 /mnt/automation/）

## 5. 環境與操作知識（重要）

- **沙箱**：clone 在 `/mnt/agents/output/hlwork/main`（跨 session 持久）；ep2 checkpoint 在 `checkpoints/tool_tuned_v5.27.pt`（**已是 ep2 權重，v5.27 原始權重已被覆蓋**）
- **GitHub push**：沙箱 git push 上傳通道會卡死——一律用 REST API（contents PUT + pulls merge），token 目前用 09-18 那支（`ghp_rmch…`，**已出現在對話中，待 rotate**）
- **HF**：沙箱連不上 huggingface.co——checkpoint 快照需本機 push（token `hf_WosE…`，**待 rotate**；命令見 2026-09-25 16:38 訊息）
- **長訓練**：孤兒行程會被隨機回收——trainer 已內建每 50 步存檔 + resume（本日驗證可跨多次 kill 無縫續跑）
- **背景任務**：`nohup python3 -u X > log 2>&1 &` + 輪詢 log 檔；pgrep 用 `[t]rain_xxx` 括號技巧防自我匹配

## 6. 下次開工接點（依 product_roadmap.md）

1. **里程碑 13**：env 擴張（3→N）+ agentic RL（rlvr_toy 接 trainer）
2. **genforge P0**：定價工程三項實作（cache 階梯/錯位峰谷/走量檔）
3. **blog 發布**：`docs/blog_correctness_first.md` 潤稿發 Medium
4. **Phase 3**：correctness oracle → agent 審計 API 設計
5. **ep2 checkpoint → HF 快照**（本機，優先——目前權重只有沙箱一份）

## 7. 待使用者處理（阻塞項）

| 事項 | 狀態 |
|---|---|
| GitHub token rotate（ghp_rmch… 已暴露） | 待辦 |
| HF token rotate（hf_WosE… 已暴露） | 待辦 |
| checkpoint HF 快照（本機 push，命令在 16:38 訊息） | 待辦 |
