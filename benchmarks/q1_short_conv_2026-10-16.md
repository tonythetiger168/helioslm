# Q1 — GDA short conv 的 recall A/B（2026-10-16）

與 B2 相同的訓練式 key-value 回憶任務（MQAR 風格；查詢距離 ≥ 64，chance = 0.0625），
比較 KDA:MLA 混合架構有無 Qwen3-Next 式 causal short conv（kernel 4）。機器可讀結果：
[`q1_short_conv_2026-10-16.json`](q1_short_conv_2026-10-16.json)，
`results_sha256 = 2d4a338abcc6fad33f070f3b662c340c72ba110a75c4178505c8fb53db7db2a7`。

## 結果（1000 步，batch 32，lr 1e-3，seed 0，CPU）

| 配置 | 層（G = MLA，L = GDA） | 參數 | 查詢準確率 | 中途評估（250／500／750／1000 步） |
| --- | --- | --- | --- | --- |
| gda_hybrid | GLGLLG | 1,912,548 | **0.426** | 0.412／0.414／0.433／0.415 |
| gda_hybrid_conv | GLGLLG | 1,917,156 | 0.420 | 0.415／0.393／0.429／0.422 |

## 結論

**在這個規模下 short conv 沒有幫助**（0.420 vs 0.426）。兩者的中途評估在 ±0.02 內交錯，差異落在雜訊範圍。
記為「無效果」，不是失敗。

可能原因：這個任務的關鍵是跨 64 個以上 token 的精確檢索，由 MLA 層（0、2、5）負責；
short conv 強化的是 GDA 層的局部模式，對本任務不是瓶頸。Qwen3-Next 的收益是在大規模語言建模上報告的。

## 誠實邊界

1. 單一 seed、toy 尺度，1000 步（原定 2000 步，容器兩次重啟中斷後縮短；B2 的 dense 在 2000 步為 0.428，
   本表兩組在 1000 步已到同一水準）。
2. 參考：B2 的 dense（全 MLA）0.428、MiMo 5:1 混合 0.424、純 sliding window 0.067（同一任務，2000 步）。

## 重現

```bash
python examples/mimo_hybrid_recall.py --arms gda_hybrid --threads 2 --steps 1000 --output a.json
python examples/mimo_hybrid_recall.py --arms gda_hybrid_conv --threads 2 --steps 1000 --output b.json
python examples/mimo_hybrid_recall.py --merge a.json b.json --output benchmarks/q1_short_conv_2026-10-16.json
```

測試：`python helioslm_v5/tests/test_qwen_next.py`（T101）。
