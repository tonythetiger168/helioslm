# Benchmark Alignment — HeliosLM v5.x 對齊表（2026-09-25）

> 目的： 把 frontier benchmark（K3/GLM-5.3 標的，見 `k3_alignment_targets.md`、
> `competitive_intel_2026-09-25.md`）逐項映射到 HeliosLM **今天可測的軸**，
> 量化差距、定義對齊路徑。框架：三層分級（T0 今天可測 / T1 方法論採用 /
> T2 規格參照），差距分「規格差」與「規模差」兩類，不混為一谈。

## 對齊總表

| Frontier benchmark | 測什麼 | HeliosLM 對應軸 | 現況（v5.27 實測） | 差距類型 | 對齊路徑 |
|---|---|---|---|---|---|
| MCP-Atlas 83.7 | tool use 綜合 | agent protocol 遵從（T17） | parse 0.15 / finish 1/9 / correct 0/9 | 規模差為主 | T0→C 階段三模式 benchmark；ep2 + 資料擴量後重測 |
| Terminal-Bench 2.1 85.4 | 終端 agentic 操作 | env 擴張（roadmap F） | 無（僅 3 個 toy env） | 規格差 | T1：採用其 sandbox 任務格式為 env 介面規格 |
| SWE-bench Multilingual 82.9 / Pro 65.7 / DeepSWE 64.3 | repo 級修 bug | trajectory oracle + replay 驗證基礎設施 | T5 bitwise replay ✅ | 規格差 | T1：T5 機制可直接做 SWE 軌跡審計層（方法論輸出） |
| GPQA Diamond 92.3（Hy4 自報；AA 已退役此項） | 無工具推理 | v5.22 decision-audit 校準框架 | ECE/Brier 工具就緒，無對應模型能力 | 規模差 | T2：僅規格參照；AA v4.2 已退役，對齊意義下降 |
| SkillsBench 62.9 | 技能執行 | tool-tuning 資料管線（finetune_data.py） | teacher=replay 管線 ✅ | 混合 | T1：採用其技能分類法組織 env 庫 |
| Apex Agents 37.1（全榜最低） | 長程多步 agent | trajectory 長度 + replay 完整性 | max_steps=8 toy；oracle 可擴 | 規格差 | T1：長程 = replay 優勢軸，OracleGate 審計天然適配 |
| HLE (w/ tools) 55.4 | 極難推理+工具 | 同上（推理軸） | — | 規模差 | T2 |
| AA Intelligence Index v4.3（K3=44, GLM-5.3=45, Opus 5.5=58） | 複合智能分 | HeliosLM 無 chat 能力，**不入榜** | 不適用 | — | T2：方法論參照（私有 held-out 40% 權重設計可借鑑） |
| GDPVal-AA v2（GLM-5.3 1655 #8 / K3 1551 #26） | 真實工作任務 Elo | 可參賽但 toy 規模無意義 | 未參賽 | 規模差 | T1：採用 GDPVal 任務分佈做 eval 集設計參照 |
| LMArena（偏好 Elo） | 人類偏好 | 不適用（無 chat） | — | — | T2 |
| OpenRouter 使用量（GLM Flash 19T） | 成本×採用 | **DSA sparse decode + disagg serving** | T15b：k≥L bitwise ✅；T17：sparse K=4 ≈ dense ✅ | 規格差（非規模差！） | **T0→v5.28：serving 成本軸是 HeliosLM 相對優勢軸，優先對齊** |

## 量化差距框架

**規模差**（GPQA / HLE / Arena 類）：差 3 個數量級以上（參數、資料、算力），
短期不可收斂，對齊方式 = 方法論與規格參照（T2），不設分數目標。

**規格差**（Terminal-Bench / SWE-bench / Apex 類）：任務格式與環境基礎設施
差距，**與模型規模無關**——HeliosLM 的 oracle/replay/gate 基建可直接對齊
這類 benchmark 的**驗證方法論**（T1）。這是 toy 規模也能輸出的價值。

**HeliosLM 相對優勢軸**（成本/serving）：OpenRouter 數據顯示真實世界用腳投票
給 Flash 化（$0.25/43tps 冠軍）——與規模無關、與架構有關。HeliosLM 的
v5.24 DSA sparse decode（T15b oracle 驗證）+ v5.25 disagg evolver 正是這條軸
的架構答案，**v5.28 優先級最高的對齊方向**。

## 當前 HeliosLM 數字（v5.27 baseline，供未來對照）

| 軸 | ep1 (v5.27) | ep2 (09-25) | 出處 |
|---|---|---|---|
| tool protocol parse | 0.15 | **0.67**（12/18） | T17 dense |
| tool protocol finish | 1/9 | **4/6** | T17 dense |
| correct | 0/9 | 0/6（複製瓶頸未變） | T17 |
| sparse(K=4) vs dense | parse 0.147 vs 0.15 | — | T17 sparse_k4 |
| oracle 完整性 | 15/15 + T15 3/3 + T17 + T18 3/3 | | v5.23–v5.28 測試 |
| 訓練 loss | 0.48 | 0.25（ep2 中途） | trainer log |

ep2 = 1.2 epoch 額外訓練（resume 週期存檔跨 kill window 完成）。parse 4.4× 提升證明
協議學習是資料/步數問題；correct=0 確認內容複製是模型骨架上限，需資料擴量或更大模型。

## 下一個可對齊動作

1. **v5.28（成本軸）**：disagg evolver 跑 Pareto 搜尋 → 產出 (makespan, workers, cost) 三軸報告，與 GLM-5.3-Flash 的 $0.25/43tps 做**規格級**對照（非分數對照）
2. **C 階段**：三模式 benchmark（direct/routed/oracle）產出 HeliosLM 版「Intelligence Index 方法論」最小實作
3. **ep2 重訓後**：T17 三率重測，更新本表「現況」欄
