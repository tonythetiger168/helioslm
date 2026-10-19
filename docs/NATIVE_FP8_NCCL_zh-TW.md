# 原生 FP8 訓練與多 GPU

原生模式為明確選項，不會更動原有模擬 FP8 的預設行為。
`FP8Trainer(model, config, backend="native")` 將模型 Linear 轉為原生模式：
前向 E4M3×E4M3，輸入梯度與權重梯度 E5M2×E4M3，均透過
`torch._scaled_mm`，FP32 累加。FP32 主權重與 AdamW 狀態保留，
正規化、softmax、embedding 等仍為較高精度；這是完整訓練步驟的 FP8
混合精度路徑，不是所有運算及 optimizer 都使用 FP8。

任意 Linear 尺寸會補齊到 16 再裁切；空的 MoE 路由不呼叫 GEMM。
目前採每個張量即時計算 amax 的縮放，不是 block scaling 或 fused kernel。
不支援的 CUDA、FP8 dtype 或算子會報錯，沒有模擬 fallback。
使用 PyTorch 私有算子，升級 PyTorch 後需重跑驗證；不保證速度提升。

在專案根目錄執行單卡完整模型合成資料驗收：

```powershell
python -u scripts/train_native_fp8.py --steps 10 --size lite --out extended_results/native_fp8_training_latest.json
```

`--size mid` 需更多主記憶體／顯示記憶體，尚未在本次驗證。
`--tokens` 預設 32。這是訓練管線驗收，不是資料集收斂測試。
原本 verify_cuda_extended 也加入 3 步 lite 原生訓練檢查。

Linux、NCCL-enabled PyTorch、至少兩張支援 FP8 的 GPU：

```bash
torchrun --standalone --nproc_per_node=2 scripts/train_native_fp8.py --steps 10 --out extended_results/native_fp8_nccl.json
```

每個程序綁定 LOCAL_RANK 對應 GPU；初始參數廣播，FP32 梯度平均後裁切、AdamW
更新，並檢查所有副本參數一致。使用同步資料平行 all-reduce，
啟動器將 MoE device_group_size 設為 1，讓每個完整副本都能選取所有專家；
不會把 mid 的四組專家配置誤當成雙卡資料平行配置。
不含 DualPipe 排程、tensor parallel、expert parallel 或通訊運算重疊。
多節點可使用標準 torchrun rendezvous 參數，但本次未實測。

用於真實資料時，先初始化 NCCL process group、設置 local CUDA device，再建構
模型與 `FP8Trainer(model, config, backend="native", distributed=True)`。
自行建構完整資料平行副本時，須在建立模型前設 `config.moe.device_group_size = 1`；
trainer 不會替既有模型重寫路由配置。
每個 rank 呼叫 `train_step(batch, labels=labels)`，以 DistributedSampler 分片資料。
所有 rank 必須相同步數、相同 local batch 與有效 target token 數；不等長資料
須自行均衡，否則等權平均不等於全域 token 平均。不要再加一層 DDP，避免重複同步。
模型結構及參數順序必須一致。任一 rank 非有限 loss／梯度時全體跳過該步。

Windows 的 PyTorch 不提供 NCCL；這台單 RTX4060 的 NCCL 結果仍為 blocked。
CPU Gloo 雙程序測試只驗證同步邏輯，不能替代多 GPU NCCL 驗收。

參考：
- https://docs.pytorch.org/docs/stable/distributed.html
- https://github.com/pytorch/pytorch/blob/main/aten/src/ATen/native/cuda/ScaledBlas.cpp
