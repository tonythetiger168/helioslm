# 產品線比對：Kimi（Moonshot）vs DeepSeek + 改善計畫（2026-09-25）

> 資料： 官方定價頁 + AA v4.3 / GDPVal-AA / OpenRouter（2026-09 即時抓取）。
> 用途： genforge / HeliosLM 產品定位參照。

## 一、產品線比對總表

| 維度 | Kimi（Moonshot） | DeepSeek | 差距解讀 |
|---|---|---|---|
| **消費者端** | Kimi app/web（免費，K3） | DeepSeek app/web（免費、無訂閱制） | 平手——都不向 C 端收費 |
| **API 旗艦** | K3：$3/$15，1M ctx，cache $0.30 | V4 Pro：$0.66/$1.98（off-peak），1M ctx，cache $0.022 | **K3 輸出價是 V4 Pro 的 7.6×**；K3 貴但 AA 44 vs 36 領先 8 分 |
| **API 走量檔** | K2.6 / K2.7 Code：$0.95/$4，256K | V4.1 Flash：$0.15/$0.60（off-peak），1M ctx + vision | **DeepSeek 便宜 6× 且 ctx 大 4×**——走量檔被碾壓 |
| **定價機制** | cache 折扣（無 TTL/keying 文件）；Batch 60%（K3 不適用）；無峰谷 | cache 98% off（$0.003）；**峰谷定價 2×**（peak=UTC 01–04、06–10 工作日） | DeepSeek 定價工程精細兩個量級 |
| **開源權重** | K3 2.8T/104B（**Kimi K3 License**，非 MIT）；K2.6/K2.7 Modified MIT | V4 Pro 權重 **MIT**；V4.1 Pro 未出 | 許可證是 Kimi 的軟肋（商用需讀條款）；DeepSeek MIT 最開放 |
| **第三方分銷** | OpenRouter、GitHub Copilot（Fireworks）、Perplexity | OpenRouter（#2，18.9T/月 +60%）、自架 | 平手——都上 router；DeepSeek 自架 MIT 生態更大 |
| **效能定位（AA v4.3）** | K3 = 44（#6 群） | V4 Pro = 36、V4.1 Flash = 39 | **K3 品質贏 5–8 分**；GDPVal：K3 1551(#26) vs GLM-5.3 1655(#8) |
| **供給穩定性** | K3 上線初期頻繁 429（容量受限） | V4 Pro 曾公告要路由到 Flash（後撤回）——定價/路由策略反覆 | 雙方容量管理都曾踩坑；DeepSeek 公開 changelog 較透明 |
| **已知軟肋** | K3 License、$15 輸出價、無峰谷、K3 無 Batch | 資料存中國的合規疑慮、thinking token 計費輸出價、並發檔差（Pro 500 vs Flash 2500） | 各自軟肋即對方機會 |

## 二、量化差距框架

**世代差（quality）**：K3 44 vs V4 Pro 36 = +8 分（AA Index），K3 是「貴而強」；V4.1 Flash 39 以 1/25 價格拿到接近旗艦的分數——**性價比軸 DeepSeek 全勝**。

**版本差（pricing mechanics）**：DeepSeek 一年內 V4 Pro $1.74→$0.435→$0.66 三次調價 + 峰谷 + 98% cache 折扣，是「定價工程」；Kimi 是「定價階梯」（K2.5→K2.6→K3），機制簡單。

**OpenRouter 真實用量（9/24）證言**：GLM-5.3 Flash 19T、DeepSeek V4.1 Flash 18.9T、Hy4 preview 12.3T——**用量冠軍全是便宜檔**，印證走量檔決定市場份額，旗艦決定聲量。

## 三、改善計畫（genforge / HeliosLM）

### P0 — 立即（定價與 cache 工程，複製 DeepSeek 機制）
1. **cache 階梯價**：cache-hit 輸入 ≤1/10 牌價，並寫明 TTL/keying 規則（Kimi 沒寫文件的坑直接跳過）
2. **峰谷定價**：off-peak 5 折，peak 窗口選 UTC 01–04 / 06–10（與 DeepSeek 錯位競爭：同窗口 = 正面剛，錯開 = 吸收對方溢價流量）
3. **走量檔優先**：旗艦打聲量、Flash 檔打收入。定價錨：走量檔 ≤ $0.30/$1.20（peak 價壓到 DeepSeek off-peak 價）

### P1 — 本季（產品線結構）
4. **雙檔模型線**：genforge 對齊 Seedance 已有基礎；補一個 code/agentic 專用端點（對照 K2.7 Code 定位：同價 general+coding 雙軌）
5. **開源權重策略**：HeliosLM 維持 Apache-2.0（比 Kimi K3 License 和 DeepSeek MIT 都更商用友善），作為「許可證差異化」賣點寫進 README
6. **第三方分銷**：上 OpenRouter（最低成本獲客）；評估 Fireworks 託管（Copilot 通路的前例）

### P2 — 方法論資產（HeliosLM 已就緒，對外輸出）
7. **驗證方法論商品化**：T5 bitwise-replay / monotonicity gate / sparse oracle 這套基建 = SWE-bench、Apex Agents 類長程 benchmark 的審計層——寫成技術 blog + 開源模組，建立「correctness-first」品牌（對手沒有人講這個故事）
8. **成本軸報表**：v5.28 Pareto artifact（`benchmarks/disagg_pareto_2026-09-25.json`）格式對齊 AA serving 報告，作為 genforge 的技術行銷素材

### 風險對沖
- DeepSeek 定價再砍（90 天內已降 60%）→ genforge 走量檔毛利承壓：P0-3 的錯位峰谷 + cache 工程是唯一非虧本應對
- Kimi K3 降價/開 Batch → 旗艦帶壓力：genforge 不跟價，轉打繁中渲染品質差異化（既定策略）
