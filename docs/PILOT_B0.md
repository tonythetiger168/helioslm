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
| 0 | 預檢（約 1 分鐘，只用 CPU）：`base_gap_eval.py --check-access` 檢查評測集與模型、`fetch_mixture.py --check-only` 檢查配方每個來源的授權和下載權限；任何一個沒權限就直接停 pod | 約 1 min |
| 1a | `base_gap_eval.py`：E4B 對四個台灣參考模型（`docs/BASE_GAP_EVAL.md`），用全部 GPU | 1.5–2.5 h |
| 1b | 同時 `fetch_mixture.py`：串流下載配方內的 `hf:` 來源，篩選後用 E2B tokenizer 切成 uint32 shard，共 3B token | 1–2 h |
| 2 | `torchrun cpt_hf.py`：E2B 續訓 3B token，WSD，最後 20% 換 decay 來源 | 約 4.8 h |
| 3 | `base_gap_eval.py --base <續訓後> --refs google/gemma-4-E2B`：先導對它自己的底模 | 約 0.5 h |
| 4 | lm-evaluation-harness 0.4.13 常用 benchmark，E2B 原版與先導各用一半 GPU 同時跑（見下） | 約 1 h（估計） |

花費上限有兩道：腳本開頭的 watchdog 在 `MAX_COST`（預設 US$300）時直接停 pod；
續訓用 `--max-hours` 預留 `EVAL_RESERVE_H`（預設 2 小時）給存檔和步驟 3–4，時間不夠會提早停下並照常存檔。
腳本結束時也會停 pod。

## 資料（`fetch_mixture.py`）

- 每個來源先讀 HF dataset card 的授權，和配方上寫的比對（同一授權家族即可，例如 Wikipedia card 的 cc-by-sa-3.0 對 CC-BY-SA-4.0）。
  不一致、配方標 `review`、或不是 `hf:` 來源，一律跳過並寫進 manifest，不會拿來訓練。
- 跳過的來源，份額依權重分給同一階段的其他來源。
- 也會用 pod 的 HF 帳號檢查每個來源能不能下載（gated 資料集要先在 Hub 申請）。有來源沒權限時預設直接停下，不會用少掉的資料默默訓練；確定要照樣訓練才加 `--allow-blocked`（pod 腳本用 `ALLOW_BLOCKED=1`，v5.175）。底模差距評測已另外跑過時，`SKIP_GAP=1` 會跳過第 1 步，pod 就不需要 141 GB 的卡；第 3 步自己下載評測集，`BENCHES` 可限定只跑有權限的幾個。
- pod 上的套件版本固定（transformers 5.20.2、accelerate 1.15.0、datasets 5.1.0、huggingface_hub 2.3.0、tokenizers 0.23.3）。
- 每份文件經過 LB3 heuristics 和該來源的 `zh_variants` 篩選，包成 BOS + ids + EOS。
- 還沒有訓練好的品質分類器，所以 `quality_threshold` 這次不套用。
- `manifest.json` 記錄每個來源的授權比對、dataset revision、token 數、保留/看過的文件數、篩選統計。

## 續訓（`cpt_hf.py`）

- fp32 主權重＋bf16 autocast，AdamW(0.9, 0.95)，gradient checkpointing，DDP。
- 每步約 1M token（`--batch-tokens`），lr 5e-5，warmup 100 步，WSD 最後 20% 衰減，同時換到 decay 來源。
- 預設凍結 per-layer embedding 表和視覺/音訊塔（純文字訓練用不到）。
  正式續訓前，pod 腳本先跑一次 `cpt_hf.py --dry-run`：只載入模型、套用凍結規則，把模型類別、可訓練/凍結參數量和被凍結的參數群組寫進 `runs/b0/dry_run.json`，確認多模態的 E2B 能當 causal LM 載入、凍結規則有對到正確的塔（v5.138）。
- 總步數少於 warmup 時（例如冒煙測試），warmup 會自動縮短，decay 和換來源仍然在最後 20% 發生（v5.138）。
- 輸出：`run.json`、`log.jsonl`（每步 loss / lr / 階段 / token/s）、`final/`（bf16 權重＋tokenizer）、`result.json`。

## 常用 benchmark（步驟 4）

模型卡和排行榜常報的分數，讓先導的數字能和其他模型直接比。shots 用 base 模型的慣例：

| Benchmark | lm-eval task | shots |
|---|---|---|
| MMLU | `mmlu` | 5 |
| MMMLU 簡中 | `mmmlu_zh_cn` | 5 |
| Winogrande | `winogrande` | 5 |
| ARC-Challenge | `arc_challenge` | 25 |
| HellaSwag | `hellaswag` | 10 |
| TruthfulQA MC2 | `truthfulqa_mc2` | 0 |
| GSM8K | `gsm8k` | 5 |
| BBH（CoT） | `bbh_cot_fewshot` | 3 |
| MATH-500 | `minerva_math500` | 4 |
| HumanEval | `humaneval` | 0 |
| MBPP | `mbpp` | 3 |
| MedQA（USMLE 4 選 1） | `medqa_4options` | 0 |
| MedMCQA | `medmcqa` | 0 |
| SciQ | `sciq` | 0 |
| AGIEval 高考物理 / 化學 / 生物 / 數學 | `agieval_gaokao_physics` / `_chemistry` / `_biology` / `_mathqa` | 0 |
| AGIEval SAT 數學、MATH | `agieval_sat_math`、`agieval_math` | 0 |

HumanEval / MBPP 會在 pod 上執行模型寫的程式（`HF_ALLOW_CODE_EVAL=1`、`--confirm_run_unsafe_code`）。
PubMedQA 的資料集（bigbio/pubmed_qa）是載入腳本，datasets 4 以後不能跑，先不放。MMLU-Pro、GPQA、AIME、IFEval 等要等 instruct 模型再跑。這一步只記錄分數，不列入過關規則。

## 過關規則

先導對 `google/gemma-4-E2B` 原版（同一評分器、同樣 shots）：

- TMMLU+ 和 TW-JusticeBench 都要上升（差距 > 0）；
- C-Eval 和 CMMLU 掉不超過 2 分。

過關代表配方和管線可用；不過關就先改配方，不往更大的模型走。
B1 不跑，這次只把結果和建議帶回來。

## 結果放哪裡

| 路徑 | 內容 |
|---|---|
| `/workspace/results/base_gap_<時間>.json` | E4B 對四個台灣參考模型的差距 |
| `/workspace/results/pilot_b0/base_gap_<時間>.json` | 先導對 E2B 原版的差距 |
| `/workspace/results/std_bench/{e2b_base,pilot_b0}/` | 常用 benchmark 的 lm-eval 結果 |
| `/workspace/cpt_data/manifest.json` | 實際用到的資料 |
| `/workspace/runs/b0/` | 續訓紀錄與權重 |
| `/workspace/*.log` | 各步驟 log |
| HF 私人 repo `<帳號>/helioslm-pilot-b0`（`HF_REPO` 可改） | 最終權重和 `results/`，pod 關掉後也留得住（v5.146） |

權重上傳在步驟 4 跑 benchmark 的同時進行，只用網路，不多佔 GPU 時間。轉 GGUF 和 llama.cpp 生成檢查之後在本機顯卡上做，不在 pod 上花錢。

過關後的打包（v5.195）：在裝好 llama.cpp 的機器上跑 `python examples/package_hf.py --hf <checkpoint> --llama-cpp <llama.cpp> --out dist/ --tiers L8 L16 --name helioslm-tw-e2b-pilot --arch gemma4 --base google/gemma-4-E2B`，再用 `examples/publish_cloudflare.py` 上架。
