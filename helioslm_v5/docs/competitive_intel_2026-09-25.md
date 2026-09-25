# Competitive Intelligence — Frontier Model Landscape (2026-09-25)

> 來源： Artificial Analysis 即時抓取（v4.3）、GDPVal-AA v2（9/15 快照）、LMArena、OpenRouter（9/24）。
> 用途： HeliosLM K3 對齊的競品座標系；與 `k3_alignment_targets.md` 搭配閱讀。
> 已交叉驗證，含三處對二手轉載的修正（見文末）。

## Artificial Analysis Intelligence Index（v4.3）

| Model | Score | Notes |
|---|---|---|
| Claude Opus 5.5 (max) | 58 | 全場最高 |
| Claude Fable 5.1 (max) / GPT-6 Astra (max) | 53 | |
| Qwen3.8 Max (0902) | 45 | 與 GLM-5.3 並列開源第一 |
| GLM-5.3 (max) | 45 | $2.01 blended，60 t/s |
| Kimi K3 (max) | 44 | |
| GLM-5.3-Flash | 42 | $0.25，43 t/s |
| DeepSeek V4.1 Flash (max) | 39 | |
| DeepSeek V4 Pro 0813 (max) | 36 | |
| MiniMax-M3 | 29 | |
| Hy3 | 25 | 前代 |
| Hy4 preview | — 未收錄 | 頁面只寫 "independent evaluation forthcoming" |

## GDPVal-AA v2（9/15 快照，245 參賽者）

- GLM-5.3 max 1655±15，第 8 名；GLM-5.3-Flash 同分第 9
- Kimi K3 max 1551（#26）
- Hy3 1136（#102）；Hy4 preview 缺席

## LMArena（即時）

- Hy4 preview：Arena WebDev 1626±17，#9（初步，僅 1,472 票，誤差大）；$0.16/task
- GLM-5.3：Text Arena Elo 1474（9/10，95.7 percentile）——task benchmark 水準 vs 偏好票偏低，reasoning 系模型老毛病

## OpenRouter 真實使用量（9/24，月 token）

| 排名 | Model | Volume |
|---|---|---|
| 1 | GLM-5.3 Flash | 19T（+67%） |
| 2 | DeepSeek V4.1 Flash | 18.9T（+60%） |
| 3 | Hy4 preview | 12.3T（+6%） |

**訊號：benchmark 旗艦之爭被 Flash 化趨勢蓋過——真實世界用腳投票都用便宜檔。**

## 交叉驗證後的三個修正

1. RankLLMs「GDPVal-AA 1769 Elo 第一」是錯的——AA 官方榜 GLM-5.3 實際 1655、第 8（前面還有 Fable 5.1 1764、Opus 5 1735）。1769 疑為舊快照或轉載失真：**第三方聚合站數據鏈條會長歪的活例**。
2. 「GLM-5.3 與 K3 並列 60 分」是舊版 v4.1.1/v4.2 指數；9 月改版（退役 GPQA Diamond、私有 held-out 權重加倍到 40%）後全榜通縮（Fable 5.1 從 66→53）。現版 GLM-5.3 45 vs K3 44——**排名不變，跨版本絕對分不可比**。
3. Hy4 的 GPQA 92.3 永遠無法被 AA 驗證——v4.2 正好把 GPQA Diamond 退役了。騰訊自報對 GLM-5.3 的 62.5 vs 2.92 盲測懸殊差距，**無任何中立評測能背書或推翻**。

## 結論（驗證後誠實版圖）

- GLM-5.3 宣稱基本經得起 AA 檢驗（開源並列第一、GDPVal 前十），但非「總榜第一」
- Hy4 所有能力主張至今**無第三方複現**；AA 掛了近一個月未出分——本身即訊號（未送測或管線滿）
- 值得盯：AA 會否在 Hy4 正式版（非 preview）出來時一次補測——已設每日 cron 追蹤

## HeliosLM 啟示

- 對齊錨點現況：K3 (max) 44 分居第 6 群；開源並列第一檔（GLM-5.3/Qwen3.8）45
- 成本軸同樣重要：GLM-5.3-Flash 以 $0.25/43tps 拿下使用量冠軍——HeliosLM 的 DSA sparse decode + disagg serving（v5.24/v5.25）正是對應這條軸
