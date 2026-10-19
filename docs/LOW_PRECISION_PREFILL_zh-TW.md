# FP16／BF16 prefill 一致性

`stable_low_precision_prefill=True` 為預設值。模型在 `eval()` 且停用梯度的
FP16／BF16 一般文字推論中，依序使用單 token 路徑處理 prefill。
這讓相同 batch 與位置配置的整段／分段 prefill 與 KV cache decode
使用相同計算形狀，避免低精度誤差在 MoE top-k 邊界改變專家選擇。

權重、輸出與自動 KV cache 保留指定的 FP16／BF16 dtype；沒有將模型
改成 FP32，也沒有放寬檢查門檻。呼叫端不要求 cache 時仍回傳 `None`。
遮罩會逐步擴展，已存在的 cache 可繼續接收多個新 tokens。

速度改善保留上述逐 token 計算：同一層的新 query/key 共用一次 RoPE 表讀取，
內部已知的連續位置以 cache 長度計算索引，避免每層、每 token 重複讀回 GPU
位置上界。外部自訂位置仍經原有驗證，沒有取消負位置或 packed 位置的檢查。
這減少同步成本，但仍不是原本的平行 prefill，長提示詞仍有速度代價。

代價是低精度 prefill 的平行性降低，長提示詞會較慢。單 token decode
仍使用既有運算；FP32、訓練、開啟梯度的執行、非預設／packed 位置與
多模態輸入使用原路徑。此修正不保證不同 batch 分組之間 bit-identical。

需要原平行 prefill 時可明確關閉，但該路徑的低精度 mid 快取一致性
尚未通過本機測試，不能把關閉設定當成已修正：

```python
cfg = HeliosLMv5Config(size="mid", stable_low_precision_prefill=False)
```

驗證修正（包含 10 分鐘 FP32 推論／訓練穩定性）：

```powershell
python -u scripts/verify_cuda_extended.py --soak-seconds 600 --out extended_results/cuda_extended_latest.json
```

量測低精度 128-token prefill（可把 dtype 改為 `bfloat16`）：

```powershell
python -u scripts/bench_cuda_mid.py --dtype float16 --lengths 128 --warmup 1 --repeats 3 --skip-generation --out extended_results/prefill_fp16.json
```

未指定 dtype 時仍量測 FP32。移除 `--skip-generation` 會另測 128+16 生成。

其中 `blocked` 的多 GPU／NCCL 另列，不算通過。原生 FP8 混合精度訓練
另有 lite 合成資料驗收；不代表 mid 訓練或真實資料收斂已驗證。
