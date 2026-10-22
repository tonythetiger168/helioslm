# GPU 批次執行手冊（`examples/gpu_batch_run.py`，v5.109）

下次開 GPU 時，一個指令把所有等 GPU 的工作跑完，結果收進一個 JSON 檔。

## 會跑什麼

| 順序 | 工作 | 腳本 | 何時跳過 | 時間上限 |
|---|---|---|---|---|
| 1 | `cuda_smoke` | `examples/cuda_smoke.py`（#1，全部 CUDA 與訓練檢查） | 不跳過 | 45 分 |
| 2 | `os5_mtp` | `examples/mtp_rollout_bench.py`（OS5 MTP 推測式 rollout） | 不跳過 | 30 分 |
| 3 | `os7_longctx` | `examples/longctx_arch_probe.py`（OS7 RULER-lite，四種注意力佈局） | 不跳過 | 120 分 |
| 4 | `pilot_data` | `examples/pilot_prepare_data.py`（FineWeb-Edu 10 萬篇，約 1 億 token） | OS6 不跑，或資料已在 `--pilot-data` | 60 分 |
| 5 | `os6_sweep` | `examples/moe_sweep_100m.py`（OS6，8 組 MoE 配置） | #72（`moe.dispatch`）還沒進這份 checkout | 360 分（`--os6-timeout`） |
| 6 | `tw_fetch` | 從 Hub 下載 TMMLU+ CSV 到 `data/tw_bench/`（只在本地，不提交） | #86 未合併、沒有 checkpoint，或檔案已在 | 30 分 |
| 7 | `tw_bench` | `examples/tw_bench_run.py`（E3/TW3 TMMLU+ 5-shot 真實分數） | #86 未合併，或 `--tw-run` 沒有 `run.json` + `ckpt.pt` | 180 分 |

每個工作各自計時；失敗或逾時會記下來，下一個照跑。

## 在 RunPod 上

1. 開 pod（上次用 RTX 6000 Ada 48 GB，PyTorch 範本）。掛同一個 network volume
   的話，上次的 `/workspace/data/pilot100k` 與 `/workspace/results/dense_compile`
   還在，可以省掉資料準備，也直接拿 dense pilot 跑 TMMLU+。
2. 在 pod 裡：

   ```bash
   cd /workspace && git clone https://github.com/tonythetiger168/helioslm && cd helioslm
   pip install tokenizers==0.22.1 datasets huggingface_hub   # pod 上需要的套件
   python examples/gpu_batch_run.py --plan                    # 先看哪些會跑、哪些跳過
   nohup python examples/gpu_batch_run.py --peak-tflops 182 --stop-pod \
       > /workspace/batch.log 2>&1 &
   ```

   `--stop-pod` 會在最後執行 `runpodctl stop pod $RUNPOD_POD_ID`（上次 pod 內建
   憑證就夠，不用 API key）。不加就不停 pod。
3. 跑完後把 `/workspace/results/gpu_batch_<時間>.json` 和 `logs/` 拿回來
   （下次 session 把它提交到 `benchmarks/`）。

## 結果檔

`<out-dir>/gpu_batch_<UTC 時間>.json`（schema `gpu-batch/1`）：

- `environment`：git commit 與是否有未提交修改、`nvidia-smi` 的 GPU、torch 與 CUDA 版本；
- `jobs`：每個工作一列，`status` 是 PASS / FAIL / TIMEOUT / SKIP，附秒數、指令、
  log 路徑、log 最後 15 行；工作自己的 JSON 輸出整份放在 `result`；
- `summary`：各狀態的數量。

每跑完一個工作就重寫一次，pod 中途掛掉也留得下已完成的部分。OS6 每跑完一組也會
更新自己的 `moe_sweep.json`，逾時時已完成的組別仍會收進結果檔。

## 常用選項

- `--jobs os5_mtp os7_longctx`：只跑其中幾個（`pilot_data`、`tw_fetch` 會跟著需要它們的工作）。
- `--os6-steps 600`：OS6 每組步數（預設 600 步 × 32k token ≈ 2 千萬 token）。
  上次 MoE pilot 用逐 expert 迴圈只有約 7.5k token/s；#72 的 batched dispatch
  會快多少還不知道。先看第一組的 `tokens_per_s` 再決定要不要加步數。
- `--compile`：OS6 用 `torch.compile`。
- `--tw-run DIR`：拿哪個 `pilot_100m.py` 的輸出目錄跑 TMMLU+；預設
  `/workspace/results/dense_compile`。`--tw-limit N` 只跑前 N 題（冒煙用，分數不可比）。
- `--smoke --device cpu`：每個工作都用玩具大小跑一次，只驗證串接與結果檔，不證明 CUDA。
  Tony 的 RTX 4060 也可以用 `--device cuda` 跑正式版，但 OS6 的 batch 16 × 1024
  在 8 GB 上可能不夠，需要時改 `moe_sweep_100m.py --batch 8 --grad-accum 4`。

## 和其他工作串接

蒸餾 teacher 的 pod（4×H200）停掉 teacher 之後可以接著跑：

```bash
python examples/distill_generate.py ... ; \
python examples/gpu_batch_run.py --stop-pod
```

這些工作都只用一張卡（`cuda:0`）。`--stop-pod` 只加在串接的最後一個指令。

## OS6 的 8 組（`examples/moe_sweep_100m.py`）

全部是 `pilot_config("moe")`（hidden 640、8 層、intermediate 1792），一次只改一個變數：

| 組別 | 改了什麼 | 總參數 / 啟用參數 |
|---|---|---|
| `base` | 4 選 2，寬 I/2，1 個 shared，SMEBU | 基準 |
| `gran_x2` | 8 選 4，寬 I/4，2 個 shared | 同 base（只差 router） |
| `gran_x4` | 16 選 8，寬 I/8，4 個 shared | 同 base |
| `shared0` | 5 選 3，寬 I/2，無 shared | 同 base |
| `fkd1` / `fkd3` | 前 1 / 3 層 dense（寬 I） | 不同，數字在結果裡 |
| `bal_quantile` / `bal_qbal` | 平衡策略 quantile / quantile_balancing | 同 base |

shared expert 的寬度跟 routed expert 一樣，所以細粒度組別把 shared 數量一起放大，
總參數和啟用參數才會相等（CI 測試檢查誤差在 1% 內）。每組記錄最後的驗證 bits/byte、
tokens/s、MFU、參數數，以及訓練完在驗證資料上每層每個 expert 的負載比例與
`max_vio`（最大負載 / 平均 − 1）。
