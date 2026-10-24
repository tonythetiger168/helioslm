# B 路線底模差距評測手冊（`examples/base_gap_eval.py`，v5.129）

續訓前先量清楚：原版 `google/gemma-4-E4B` 和聯發科四個文字模型差多少。
擁有者 2026-10-23 批准這一步；2026-10-24 再批准 E2B 先導（`docs/PILOT_B0.md`，兩者合計上限 US$320）。正式 B1 不跑。

## 會跑什麼

| 模型 | 大小（bf16） |
|---|---|
| google/gemma-4-E4B（底模） | 約 16 GB |
| MediaTek-Research/Breexe-8x7B-Instruct-v0_1 | 約 94 GB |
| MediaTek-Research/Llama-Breeze2-8B-Instruct | 約 18 GB |
| MediaTek-Research/Llama-Breeze2-3B-Instruct | 約 8 GB |
| MediaTek-Research/Breeze-7B-Instruct-v1_0 | 約 15 GB |

每個模型都跑以下四項，評分方式都是答案字母的 log-likelihood：

| 評測 | 題組 | 示範題數 |
|---|---|---|
| TMMLU+ | test | 5-shot |
| TW-JusticeBench | test | 0-shot |
| C-Eval | val | 5-shot |
| CMMLU | test | 5-shot |

評分器是 `helioslm_v5/eval/tw_bench.py`。所有模型用同一個評分器、同樣的 shots，
分數可以互相比較，但不能和論文上的數字比。

結果會寫出底模減去每個對手的差距，附 95% paired-bootstrap 信賴區間。

## 在蒸餾 pod 上（預設做法）

`examples/distill_pod.sh` 的步驟 7b 會自動跑這支腳本。時間點在教師（vLLM）關掉、
資料打包之後，最後的 `gpu_batch_run.py --stop-pod` 之前。

- 用 `--gpus 0,1,2,3`，四張 H200 各跑一個 worker，Breexe 自己佔一張。
  約 1 小時，4×H200 每小時約 21 美元，所以約 21–30 美元。
- 不需要另外開 pod。
- pod 上要有 `HF_TOKEN`，並且已經在 Hugging Face 接受各模型的授權（見下面第 2 步）。

## 單獨開 pod

1. **開 pod。**
   - GPU 選 **1 張 H200（141 GB）**，因為 Breexe 用 bf16 需要約 94 GB。範本用 PyTorch。
   - Volume 開 150 GB 以上。
   - 在 pod 的環境變數加 `HF_TOKEN`，用 RunPod 的 secret，不要貼到對話裡。
2. **到 Hugging Face 接受授權。** 先用同一個帳號打開上面五個模型的頁面。
   如果頁面要求同意授權（gated），先按同意。
3. **在 pod 裡執行：**

   ```bash
   cd /workspace && git clone https://github.com/tonythetiger168/helioslm && cd helioslm
   pip install -U transformers accelerate datasets huggingface_hub
   export HF_HOME=/workspace/hf
   # 冒煙：每項前 20 題、只跑底模，約 5 分鐘，確認下載和評分都正常
   python examples/base_gap_eval.py --fetch --limit 20 --models google/gemma-4-E4B
   # 正式：五個模型全跑，跑完自動刪模型快取並停 pod
   nohup python examples/base_gap_eval.py --fetch --cleanup --stop-pod \
       > /workspace/gap.log 2>&1 &
   ```

   - 正式跑約 3–4 小時。H200 每小時約 3.6–4 美元，所以約 15–20 美元。
   - 有多張 GPU 時加 `--gpus 0,1,...`，每張 GPU 跑一個 worker。
   - `--stop-pod` 會在最後執行 `runpodctl stop pod $RUNPOD_POD_ID`。
4. **把結果拿回來。** 把 `results/base_gap_<時間>.json` 傳回 thread。
   檔案只有分數和題號，沒有題目原文，下一次 session 會把它提交到 `benchmarks/`。

## 結果檔

`results/base_gap_<UTC 時間>.json`（schema `base-gap/1`）：

- **`fetch`**：每個評測的下載狀態。
- **`task_errors`**：載入失敗的評測。TW-JusticeBench 的欄位格式還沒有對著實檔確認過，
  載入失敗就記在這裡，其他三項照跑。
- **`models`**：每個模型的內容包括：
  - 實際評測的 revision（commit sha），
  - 各評測的題數、準確率和秒數，
  - 失敗時的錯誤訊息。
- **`gaps`**：每個評測、每個對手都有一組數字：
  - `diff`：底模減對手的準確率差，
  - `ci95_low` / `ci95_high`：95% 信賴區間，
  - `n`：雙方都有作答的題數。

每跑完一個模型就重寫一次結果檔。pod 中途掛掉，已完成的部分也留得下來。

## 怎麼讀

- **四項都贏**：`ci95_low > 0` 而且對手是 Breexe。這代表底模本身已經過了 MTK 這一關，
  續訓的重點會變成台灣用語和繁中流暢度，以及「贏過自己底模」那一道門檻。
- **TMMLU+ 或 TW-JusticeBench 輸很多**：例如輸 5 分以上。這時要在 B1 之前把續訓 token
  拉高，例如提高到 80B，並把預算重算一次。
