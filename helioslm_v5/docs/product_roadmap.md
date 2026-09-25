# 產品 Roadmap — genforge × HeliosLM（2026-09-25）

> 雙線定位：**genforge** = 收入線（閉源生成式 API，對標 Seedance/Seedream）；
> **HeliosLM** = 品牌/方法論線（開源 correctness-first LLM，Apache-2.0）。
> 交會點： HeliosLM 的架構資產（DSA sparse decode、disagg Pareto、correctness oracle）
> 是 genforge 的成本軸與差異化故事。基準日： 2026-09-25。

## 現況盤點（已交付 ✅）

| 資產 | 狀態 |
|---|---|
| HeliosLM v5.23–v5.28（agent layer / attention 變體 / disagg / T15 真模型 oracle / tool-tuned checkpoint / Pareto 成本軸） | ✅ GitHub main |
| 驗證基建：15/15 + T15 3/3 + T17（sparse≈dense）+ T18 3/3 | ✅ 全綠 |
| 情報資產：K3 對齊標的、AA/GDPVal/LMArena/OpenRouter、Kimi vs DeepSeek 產品線比對 | ✅ docs/ |
| genforge 對齊 Seedance：30 秒單次生成差距已收斂（前次 session 交付） | ✅ 待 LTX 2.5 上限解除 |
| AA 收錄 Hy4 追蹤 | ✅ cron 每日 09:00 |

## Phase 1 — 現在～2026-10（定價工程 + 收口）

### genforge（P0，直接收入影響）
1. **cache 階梯價**：hit ≤1/10 牌價 + TTL/keying 文件（對照 DeepSeek 98% off / Kimi 無文件）
2. **峰谷定價**：off-peak 5 折，窗口錯位 UTC 01–04 / 06–10（吸 DeepSeek 溢價流量）
3. **走量檔上線**：≤$0.30/$1.20，錨定 DeepSeek off-peak 價（OpenRouter 證據：用量冠軍全是便宜檔）
4. 里程碑：**API 定價頁 v2 上線**；觀測指標 = OpenRouter 路由量、cache hit 率

### HeliosLM（收口 + 快照）
5. ep2 tool-tuning 完成 → T17 重測（目標 parse 0.15→0.3+）→ checkpoint 推 HF 快照
6. demo.gif + KV 圖上 main（本週已就緒/重錄）
7. 里程碑：**v5.28 tag + HF 雙 artifact（agent 層 + checkpoint）**

## Phase 2 — 2026-11～12（產品線結構）

### genforge
8. code/agentic 專用端點（K2.7 Code 前例：同價雙軌）
9. OpenRouter 上架 + Fireworks 託管評估
10. 繁中渲染品質差異化 campaign（Seedance 弱點軸）

### HeliosLM
11. **C 階段**：三模式 benchmark（direct/routed/oracle）→ HeliosLM 版 Intelligence Index 方法論最小實作 → 技術 blog
12. correctness-first 品牌內容：T5 bitwise-replay / monotonicity gate 寫成「長程 agent benchmark 審計層」敘事（對手沒人講）
13. roadmap F–G：env 擴張（3→N）+ agentic RL 接 rlvr_toy

## Phase 3 — 2027 Q1（方法論變現）

### 交會（HeliosLM → genforge）
14. disagg Pareto 格式對齊 AA serving 報告 → genforge 技術行銷素材（成本軸故事）
15. correctness oracle 模組化為可售的「agent 審計 API」（企業客戶：金融/醫療長程 agent 需要 trajectory 審計）

### genforge
16. 視訊生成上限解除後（LTX 2.5）的完整版發布 + 定價 v3

## 量化目標（對齊表）

| 指標 | 現況 | Phase 1 目標 | Phase 2 目標 |
|---|---|---|---|
| T17 tool parse 率 | 0.15 | 0.30（ep2） | 0.50（資料擴量） |
| sparse decode 協議穩定 | K=4 ≈ dense | K=4 生產就緒評估 | genforge 推理端採用評估 |
| genforge 定價機制 | 牌價單一 | cache+峰谷+走量檔 | 雙軌端點 |
| 情報覆蓋 | 3 份 docs | +AA Hy4 追蹤自動化 | +季度競品 refresh |

## 風險與對沖

| 風險 | 對沖 |
|---|---|
| DeepSeek 再降價（90 天 -60% 前科） | 錯位峰谷 + cache 工程（唯一非虧本應對）；不跟價，打品質軸 |
| Kimi K3 降價/Batch | genforge 旗艦不跟價；HeliosLM 不參與價格戰（開源線無收入壓力） |
| ep2 提升有限 | 據實記錄；瓶頸是複製能力 → Phase 2 資料擴量或換更大骨架 |
| 工具鏈/沙箱限制（本環境教訓） | 長訓練一律週期存檔+resume；推送走 REST API |
