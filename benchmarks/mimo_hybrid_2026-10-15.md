# B2-1 / B2-2 — MiMo 式 5:1 sliding window／全域交錯與可學習 sink（2026-10-15）

訓練式 key-value 回憶任務（MQAR 風格），比較四種注意力配置。機器可讀結果：
[`mimo_hybrid_2026-10-15.json`](mimo_hybrid_2026-10-15.json)，
`results_sha256 = 22146344542c1f23f0bbaab4ecb1b7ebf09e91bcd940c078fb0080e08e04fc7d`。

## 任務

每條序列為 `[k1 v1 … k8 v8] [64 個填充 token] [kq vq] × 8`：8 組綁定、填充，再以隨機順序查詢 8 個 key。
只在 8 個查詢答案位置計算 loss 與準確率（chance = 1/16 = 0.0625）。查詢距離綁定至少 64 個 token。
6 層、寬度 128、窗口 8；訓練串流 seed 1000，評估串流 seed 0（1024 個查詢）。

## 結果（2000 步，batch 32，lr 1e-3，seed 0，CPU）

| 配置 | 層 | 參數 | 查詢準確率 | 最後一個查詢可讀的 key 數（6 層合計） |
| --- | --- | --- | --- | --- |
| dense | GGGGGG | 1,933,080 | **0.428** | 576 |
| all_swa | SSSSSS | 1,933,080 | 0.067（≈ chance） | 48 |
| hybrid_5to1 | GSSSSG | 1,933,080 | **0.424** | 224 |
| hybrid_5to1_sink | GSSSSG | 1,933,104 | 0.424 | 224 |

## 結論

1. **B2-1：5:1 交錯保住長距離回憶。** 混合配置以 dense 39% 的 key 讀取量（少 2.6 倍）達到 0.424 vs 0.428。
   純 sliding window 疊 6 層最多只能往回看約 6 × 7 = 42 個 token（< 64），停在 chance——
   這是結構上的負對照，與預測一致。
2. **B2-2：可學習 sink 在此設定下沒有效果**（0.424 vs 0.424）。訓練後的 sink logit 很小
   （各層平均 −0.20 到 +0.16），模型幾乎沒用它。MiMo 的主張是 sink 在窗口很小、長序列時穩定訓練；
   本任務序列只有 96 個 token，可能測不到。記為「無效果」，不是失敗。

## 誠實邊界（引用前必讀）

1. **單一 seed、toy 尺度**；dense 本身只到 0.43（任務在 2000 步內未飽和），結論是「混合 ≈ dense ≫ 純窗口」，
   不是精確差距。
2. **key 讀取數是存取量，不是記憶體**：目前的 KV cache 不會裁掉窗口外的 token，窗口層的實際記憶體並未減少。
   真正省記憶體需要環形 cache（後續工作）。
3. 窗口 8／序列 96 是為了 CPU 可跑而縮小；MiMo 實際設定是窗口 128。

## 重現

```bash
# 每個配置一個進程（約 55 分鐘／4 核），再合併：
python examples/mimo_hybrid_recall.py --arms dense --threads 1 --output d.json   # 其餘三臂同理
python examples/mimo_hybrid_recall.py --merge d.json s.json h.json hs.json --output benchmarks/mimo_hybrid_2026-10-15.json
```

測試：`python evastar/tests/test_mimo_hybrid.py`（T96）。
