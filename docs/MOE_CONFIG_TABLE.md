# #42 30B-A3B 的 MoE 配置決策表（OS6）

來源：2026-10-22 開源模型評析的 OS6。各模型數值讀自各自 Hugging Face repo 的
`config.json`（2026-10-22）。計算在 `helioslm_v5/eval/moe_config_table.py`，
測試是 `tests/test_moe_config_table.py`。

## 五個模型的 MoE 欄位

| 模型 | 層數 | hidden | routed experts | shared | top-k | 前 k 層 dense | expert 寬度 | 路由 |
|---|---|---|---|---|---|---|---|---|
| Hy3 | 80 | 4096 | 192 | 1 | 8 | 1 | 1536 | sigmoid + bias，scale 2.826 |
| MiMo-V2.5 | 48 | 4096 | 256 | 0 | 8 | 1 | 2048 | sigmoid |
| MiMo-V2.5-Pro | 70 | 6144 | 384 | 0 | 8 | 1 | 2048 | sigmoid |
| DeepSeek V4 Flash | 43 | 4096 | 256 | 1 | 6 | 0（前 3 層 hash 路由） | 2048 | sqrtsoftplus，scale 1.5 |
| MiniMax M3 | 60 | 6144 | 128 | 1 | 4 | 3 | 3072 | sigmoid + bias，scale 2.0 |
| GLM 5.2 | 78 | 6144 | 256 | 1 | 8 | 3 | 2048 | sigmoid，noaux_tc，scale 2.5 |

**核對：** 用「(層數 − dense 層) × (routed + shared) × 3 × hidden × expert 寬度」
算出的 expert 權重，落在各廠商公布總參數的 96–100%，例如 Hy3 是 287.8B 對
295B，GLM 5.2 是 727.6B 對 753B。所以上表欄位讀得正確，剩下的是 attention 和
embedding。

## 共同模式

- **啟用比例：** top-k ÷ experts 在 2.1%（MiMo-Pro）到 4.2%（Hy3）之間，五家
  都是細粒度 MoE。
- **路由：** 六個裡五個用 sigmoid 分數，加上無輔助損失的 bias 平衡。Evastar
  已有這套（aux-free bias，加上 quantile、SMEBU、quantile_balancing 策略）。
- **expert 寬度：** 是 hidden 的 1/3 到 1/2。
- **shared expert：** 四家用 1 個，小米兩個都不用。
- **dense 層：** 前 0–3 層 dense；DeepSeek 改成前 3 層 hash 路由。
- **top-k：** 只有 MiniMax M3 走粗粒度路線（128 選 4、expert 寬 3072）。

Evastar 的 `MoEConfig`（num_experts、num_shared_experts、
num_activated_experts、expert_hidden_size、first_k_dense）可以直接表達表中
每一列，不需要新程式碼。sqrtsoftplus 分數和 hash 路由層例外，目前沒有實作。

## 30B-A3B 候選

`candidates()` 在小網格上列出總參數和啟用參數都在 30B／3B ±12% 內的配置
（vocab 128K、head_dim 128、GQA 8:1、前 1 層 dense、不 tie embedding）。前幾名：

| 層數 | hidden | experts | shared | top-k | expert 寬度 | 總參數 | 啟用 |
|---|---|---|---|---|---|---|---|
| 48 | 2048 | 128 | 1 | 8 | 768 | 29.6B | 3.02B |
| 40 | 2560 | 128 | 0 | 8 | 768 | 30.8B | 3.15B |
| 40 | 2560 | 64 | 0 | 4 | 1536 | 30.8B | 3.15B |
| 48 | 2048 | 64 | 1 | 4 | 1536 | 29.9B | 3.25B |
| 32 | 3072 | 128 | 0 | 8 | 768 | 29.7B | 3.33B |

這只是規模估算，不代表哪個訓練得最好。256 選 8 的 Qwen3.5-35B-A3B 型配置在
hidden 2048 下總參數約 39B、啟用約 2.4B。要用 256 個 experts，就得把 expert
寬度降到 384 左右，或減少層數。

## 建議的小規模掃描（100M，GPU）

固定總參數和啟用參數，一次只改一個變數，各跑同一份資料與 token 數：

1. **粒度：** 64 選 4（寬 1536）、128 選 8（寬 768）、256 選 16（寬 384）。
   這是 M3 的粗粒度路線對上 GLM／MiMo 細粒度路線的直接比較。
2. **shared expert：** 0 或 1（小米 vs 其他四家）。
3. **前 k 層 dense：** 0、1、3。
4. **平衡策略：** quantile vs SMEBU vs quantile_balancing，同時記錄每個 expert
   的負載分佈。

**驗收：** 驗證 loss，加上每 token 的 FLOPs 與實測吞吐。吞吐要等批次化
dispatch（#72）合併後再量，否則逐 expert 迴圈會讓細粒度配置吃虧。

腳本：`examples/moe_sweep_100m.py`（v5.109），由 `examples/gpu_batch_run.py`
一起跑，見 `docs/GPU_BATCH_RUNBOOK.md`。

## 與其他工作的關係

- #72（MoE 批次化 dispatch）與 #73（chunked KDA）還沒合併。本表只新增文件和
  獨立的計算模組，不碰 `src/moe/`，不會衝突。
- 掃描需要 GPU，跟 OS5、OS7 一起排進下次開 GPU。
