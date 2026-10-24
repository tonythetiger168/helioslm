# 我們不只展示分數：一個 correctness-first LLM 的驗證方法論

> HeliosLM 技術 blog 素材（v5.29，2026-09-25）
> 所有數據可復現：`examples/` 腳本 + `benchmarks/` artifact + `tests/` oracle。
> 一句話： 當所有人都在貼 benchmark 分數，我們展示為什麼不能信分數——以及怎麼驗證。

## 背景：分數通縮的時代

2026-09 的 benchmark 版圖充滿噪音：AA Intelligence Index v4.2 改版讓全榜通縮
（Fable 5.1 從 66 掉到 53）；GPQA Diamond 被退役，Hy4 自報的 92.3 永遠無法被
中立驗證；二手聚合站把 GDPVal-AA 的 1655（第 8 名）轉載成 1769（第 1 名）。
（來源：`docs/competitive_intel_2026-09-25.md`，含原始 claim 對照表。）

在這個環境裡，**驗證方法論本身就是產品**。HeliosLM 是 toy-scale 的
correctness-first 參考實作，但它的價值不在分數，在於每個能力都掛著
**可執行的 oracle**。本文用四個真實模型實驗展示這套方法論如何運作。

## 實驗一：trajectory 的 bitwise 審計（T5）

Agent 的軌跡（prompt → response → observation 序列）應該是**可重放的證據**。
`verify_replay` 斷言：同樣的 token ids 重放，必須得到 bitwise 相同的
observations；任何竄改（改一個參數、改一個觀察值）都會被抓出。

這不是理論宣示——Stage B 用它抓到了真 bug：真模型會產生語法錯誤的 tool
call（如 `calc("1+")`），原先 `execute` 直接拋例外崩潰 loop，`verify_replay`
也在同一處崩潰。修復（`TOOL_ERROR` 回饋為 observation、replay 鏡像同樣行為）
讓 oracle 與 loop 重新一致，9/9 agent 測試保持全綠。

**啟示**：長程 agent benchmark（Apex Agents、SWE-bench 類）缺的不是更多任務，
是軌跡審計層。誰的 trajectory 能 bitwise 重放，誰的分數才可信。

## 實驗二：sparse attention 的誠實邊界（T15b + T17）

v5.8 的 DSA sparse decode（`sparse_top_k`）我們只宣稱可證的部分：

- `k >= kv_len` 時與 dense **bitwise 相同**（config 契約，實測 PASS）
- prefill 永遠 bitwise 不變（sparse 只在 `seq==1 且 kv_len>k` 觸發，實測驗證）
- `k < kv_len` 是近似：**greedy 翻轉不斷言、只報告**（隨機權重下 7/8 翻轉）

端到端驗證（T17）：tool-tuned checkpoint 跑 agent loop，sparse_k4 vs dense 的
protocol parse 率 **0.147 vs 0.15**——sparse decode 在真實 agent 推理路徑
不塌縮。這是架構級證據：成本軸（OpenRouter 用量冠軍全是便宜檔）可以用
sparse decode 追，不需要等更大模型。

## 實驗三：serving 的 Pareto 語言（T18）

v5.28 把 v5.25 的 disagg 模組跑成三軸 Pareto 搜尋（makespan / workers /
worker_seconds），三種 workload 的 latency-cost 曲線全部種子化可復現。
過程中 monotonicity gate 觸發——但我們沒有悄悄放寬它，而是做了實驗區分：

- round_robin 下 gate **成立**（無 cache 親和時 worker 單調）
- cache_aware 下 **結構性違反**（同 key 請求序列化在 holder，加 worker 反增
  makespan ~4%）——記錄為發現，不隱藏

這就是生產系統的真實物理（Mooncake 類全會遇到），而我們用一個 200 行的
分析模型就把它抓了出來。**方法論可輸出**：這套 gate + Pareto 格式可以直接
對齊 AA 的 serving 報告，做規格級成本對照。

## 實驗四：routing gate 遇上真實信心（T19，頭條）

三模式 benchmark（direct / routed(τ) / oracle）在真 checkpoint 上實測：

```
tau    direct   correctness
0.50   1.00     0.000
0.70   0.92     0.083
0.80   0.75     0.250
0.90   0.67     0.333
0.95   0.00     1.000     ← 全 escalate = oracle
```

τ 曲線**嚴格單調，routing gate PASS**——v5.22 的決策稽核紀律第一次在真實模型
+ 真實信心（幾何平均 token 機率）下端到端成立。

但真正的頭條是：**這個模型對錯誤答案給出 0.944 的信心**。系統性過度自信。
Routing 仍然有效，只因為信心有**區分度**（str 任務 0.65–0.86 vs calc
0.92–0.94），絕對校準則完全不可信。

這正是 v5.22 decision-audit 工具包的設計目標（ECE/Brier 對 constructed
ground truth 校準）。換句話說：**我們的 toy 模型親身示範了為什麼業界的
agent benchmark 分數不能直接信**——連 0.94 的信心都能掛在錯答案上。

## 結論：方法論即產品

四個實驗的共同點：**每個能力掛 oracle，每個違反被記錄而非隱藏，每個數字
可復現**（seeded artifact + 測試斷言）。這套基建（bitwise replay、
monotonicity gate、routing gate、Pareto 報告）對 toy 規模是工程紀律，
對真實規模就是競爭壁壘——因為長程 agent 的企業客戶（金融、醫療）需要的
不是更高的 leaderboard 分數，是**可審計的 trajectory**。

Roadmap 的 Phase 3 計畫把這套模組化為 agent 審計 API。如果你想看我們
對你的 agent 軌跡做 bitwise 審計，聯繫方式在 repo。

## 附：復現索引

| 實驗 | 腳本 | 測試 | artifact |
|---|---|---|---|
| T5 replay | — | `tests/test_agent.py` | — |
| T15 sparse oracle | — | `tests/test_v5_stage_a.py` | — |
| T17 sparse≈dense | `examples/train_tool_tuned.py` | `tests/test_v5_stage_b.py` | `checkpoints/tool_tuned_v5.27.*` |
| T18 Pareto | `examples/disagg_pareto.py` | `tests/test_disagg_pareto.py` | `benchmarks/disagg_pareto_2026-09-25.json` |
| T19 三模式 | `examples/benchmark_three_modes.py` | `tests/test_three_modes.py` | `benchmarks/three_modes_2026-09-25.json` |
