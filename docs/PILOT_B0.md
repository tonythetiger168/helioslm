# B 路線先導 B0：Gemma 4 E2B 續訓（`examples/pilot_b0_pod.sh`，v5.137）

擁有者 2026-10-24 批准：底模差距評測＋E2B 先導，總上限 US$320。
正式 B1（E4B，約 US$3.8k）不跑。

先導要回答兩件事：

1. 我們的資料管線（抓語料 → 篩選 → 切 token → 多卡續訓 → 存檔 → 評測）在真的 pod 上能不能一次跑完；
2. 用 `data/mixtures/b1_gemma4_e4b_cpt_v1.json` 這份配方續訓，台灣題目有沒有往上走，而一般中文能力沒有掉。

## 一個 pod 跑完

8×H200 SXM（Breexe-8x7B 用 bf16 要一張 141 GB 的卡），PyTorch 映像，
`/workspace` 掛 300 GB 以上的 volume，HF_TOKEN 放在 RunPod secret。

```bash
cd /workspace && git clone https://github.com/tonythetiger168/helioslm
PER_HOUR=<pod 實際每小時美金> bash helioslm/examples/pilot_b0_pod.sh 2>&1 | tee /workspace/pilot_b0.log
```

| 步驟 | 內容 | 大約時間 |
|---|---|---|
| 1a | `base_gap_eval.py`：E4B 對聯發科四個模型（`docs/BASE_GAP_EVAL.md`），用全部 GPU | 1.5–2.5 h |
| 1b | 同時 `fetch_mixture.py`：串流下載配方內的 `hf:` 來源，篩選後用 E2B tokenizer 切成 uint32 shard，共 3B token | 1–2 h |
| 2 | `torchrun cpt_hf.py`：E2B 續訓 3B token，WSD，最後 20% 換 decay 來源 | 約 4.8 h |
| 3 | `base_gap_eval.py --base <續訓後> --rivals google/gemma-4-E2B`：先導對它自己的底模 | 約 0.5 h |

花費上限有兩道：腳本開頭的 watchdog 在 `MAX_COST`（預設 US$300）時直接停 pod；
續訓用 `--max-hours` 預留一小時給存檔和評測，時間不夠會提早停下並照常存檔。
腳本結束時也會停 pod。

## 資料（`fetch_mixture.py`）

- 每個來源先讀 HF dataset card 的授權，和配方上寫的比對（同一授權家族即可，例如 Wikipedia card 的 cc-by-sa-3.0 對 CC-BY-SA-4.0）。
  不一致、配方標 `review`、或不是 `hf:` 來源，一律跳過並寫進 manifest，不會拿來訓練。
- 跳過的來源，份額依權重分給同一階段的其他來源。
- 每份文件經過 LB3 heuristics 和該來源的 `zh_variants` 篩選，包成 BOS + ids + EOS。
- 還沒有訓練好的品質分類器，所以 `quality_threshold` 這次不套用。
- `manifest.json` 記錄每個來源的授權比對、dataset revision、token 數、保留/看過的文件數、篩選統計。

## 續訓（`cpt_hf.py`）

- fp32 主權重＋bf16 autocast，AdamW(0.9, 0.95)，gradient checkpointing，DDP。
- 每步約 1M token（`--batch-tokens`），lr 5e-5，warmup 100 步，WSD 最後 20% 衰減，同時換到 decay 來源。
- 預設凍結 per-layer embedding 表和視覺/音訊塔（純文字訓練用不到）。
- 輸出：`run.json`、`log.jsonl`（每步 loss / lr / 階段 / token/s）、`final/`（bf16 權重＋tokenizer）、`result.json`。

## 過關規則

先導對 `google/gemma-4-E2B` 原版（同一評分器、同樣 shots）：

- TMMLU+ 和 TW-JusticeBench 都要上升（差距 > 0）；
- C-Eval 和 CMMLU 掉不超過 2 分。

過關代表配方和管線可用；不過關就先改配方，不往更大的模型走。
B1 不跑，這次只把結果和建議帶回來。

## 結果放哪裡

| 路徑 | 內容 |
|---|---|
| `/workspace/results/base_gap_<時間>.json` | E4B 對聯發科四個模型的差距 |
| `/workspace/results/pilot_b0/base_gap_<時間>.json` | 先導對 E2B 原版的差距 |
| `/workspace/cpt_data/manifest.json` | 實際用到的資料 |
| `/workspace/runs/b0/` | 續訓紀錄與權重 |
| `/workspace/*.log` | 各步驟 log |
