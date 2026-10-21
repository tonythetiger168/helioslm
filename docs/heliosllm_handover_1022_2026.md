# HeliosLM 交接文件（2026-10-22）

這份文件給下一個工作階段（人或 Claude）接手用：目前做到哪裡、哪些事還在進行、哪些需要負責人決定、下一步怎麼做。

- **Repo：** `tonythetiger168/helioslm`
- **工作分支：** `claude/fixed-issues-checkin-iludjn`（每項工作一個 PR，CI 綠燈後 squash merge）
- **Claude session：** https://claude.ai/code/session_01E2rbmg5RXFga5Ly3wh78xQ （每個 commit 的 `Claude-Session:` trailer 都指向它）

## 1. 本次完成並合併的 PR（2026-10-21）

| PR | 版本 | 內容 | 關閉的 issue |
|---|---|---|---|
| #56 | v5.80 | 中文訓練加速：Big5 + GB2312 常用字 BPE seed（3.00 → 1.00 token／中文字）；修好 main 的紅燈（test_v5 案例數 guard） | — |
| #57 | v5.81 | TW1 資料集授權稽核（45 張 HF 卡片：21 可商用、11 非商用、1 禁改作、12 未標示）；K3 / V4 比較文件 | — |
| #58 | v5.82 | API 教師蒸餾：sequence KD + 驗證器、跨 tokenizer 的 top-k KD；10 個開源權重教師的授權稽核 | — |
| #59 | v5.83 | 約 100M 單 GPU 試跑訓練迴圈：4 種架構、held-out bits/byte、MFU、中斷後逐位元續跑 | — |
| #60 | v5.84 | fused vs unfused 量化實測（CPU，4096×14336）：AWQ 7.4×、GPTQ 11.9×、MXFP4 2.3× | #2 |
| #61 | v5.85 | 封存舊版 eval_mid_agent v1–v4 與 web UI v1；**修補 v1 web UI 路徑穿越漏洞**（`/video/../..` 可讀任意檔案，且 v1 綁 0.0.0.0） | #54 |
| #62 | v5.86 | 所有 A/B 腳本支援 `--seeds`（配對多 seed + 彙總統計，預設輸出不變） | #55 |
| #63 | v5.87 | 確認 #3、#4 已達驗收標準；補上所有 preset 的 Hub config round-trip 測試 | — |

另外手動關閉了 #3（custom attention notebook）、#4（HF Hub），驗證說明留在各 issue。

## 2. 後續合併（2026-10-21 晚）與進行中

| PR | 版本 | 內容 | 關閉的 issue |
|---|---|---|---|
| #64 | — | `examples/cuda_smoke.py`（13 項真正上 GPU 的檢查）與本交接文件 | — |
| #65 | v5.88 | 評測計劃（GPU 需求、排行榜檢視、E1–E8）；E2 訓練／評測 13-gram 重疊檢查；E5 預先登記評測 | — |
| #66 | v5.89 | #5 實測：BPE 在 3 個 seed 的每 byte NLL 都優於 char（1.635／1.645／1.534 vs 1.884／2.013／2.359） | #5 |
| #67 | v5.90 | #1 首次 CUDA 執行：cuda_smoke 11/13，修好兩個真正的 CUDA bug（FP8Trainer 把層建在 CPU、GPUMonitor 缺 NVML 就崩潰）；新增 10 項 CUDA 訓練檢查與 `gpu_overnight.sh` | — |
| （下一個 PR） | v5.91 | 夜間 GPU 執行：cuda_smoke **23/23**；`--compile` 的 100M 試跑達 1.35 bits/byte、67.2k tokens/s、MFU 20.6%；腳本自動停止 pod | #1 |

**#1 CUDA 驗證**（詳見 `docs/CUDA_VERIFICATION_2026-10-21.md`）：
- RTX 6000 Ada（torch 2.14.1+cu130）上完整測試 574 passed、integration 9/9。但測試全部釘在 CPU，不算 CUDA 驗證。
- `cuda_smoke.py` 第一次跑是 11/13，兩個失敗都已在 v5.90 修好，並加了能在 CPU 上重現的測試。
- 夜間重跑（v5.91）：**23/23 PASS**，#1 完成。

**首次 GPU 100M 試跑**（93M dense，FineWeb-Edu 16.4M tokens，bf16）：
- val bits/byte 2.77 → 1.79，每秒約 29.8k tokens，MFU 9.1%。
- 整段都在 warmup：`--warmup` 預設 500，下次請設為總步數的約 5%。
- 在小寬度下，KDA block 比 dense 慢約 10 倍，MoE 慢約 2.5 倍。擴大規模前要先做吞吐量優化（`--compile`、fused kernel）。

## 3. 需要負責人決定

1. **E1（接 lm-evaluation-harness）**：需要安裝 `lm-eval` 與 `transformers`。依負責人規定，未經同意不裝套件。
2. **v1.0（#40，0.5B）的評測門檻**：例如 ARC-Easy、TMMLU+ 要多少分才算過。E5 工具已就緒，門檻由負責人決定。
3. **RunPod MCP／agent 設定**：要讓 Claude 直接操作 RunPod，需要兩件事：
   - 在環境設定的 Allowed domains 加入 `agents.runpod.io`、`runpod.io`、`api.runpod.io`、`rest.runpod.io`；
   - 把 API key 存成環境 secret `RUNPOD_API_KEY`，**不要貼在聊天裡**，並開新 session。
4. **#41–#45 的預算**：估計 3B 約 2 萬 H100 小時（4–7 萬美元），1T-A40B 約 290 萬 H100 小時（580–1,000 萬美元）。建議 #42 證明「每單位品質成本」之前，不承諾 #43 以後的代數。

## 4. RunPod 現況與指令

- **Pod：** RTX 6000 Ada 48 GB，On-Demand 每小時 0.99 美元，PyTorch 2.8 模板，已升級到 torch 2.14.1+cu130。
- **餘額：** 約 165 美元。
- **注意：** 不要跑 `pip install -r requirements-ci.txt`，它會把 torch 換成 CPU 版。
- **pod 上的指令**（在 `/workspace/helioslm` 執行）：
```bash
git fetch origin claude/fixed-issues-checkin-iludjn
git checkout FETCH_HEAD -- examples/cuda_smoke.py
python examples/cuda_smoke.py --out gpu_smoke.json 2>&1 | grep -v Warning | tee gpu_smoke.log
python examples/pilot_prepare_data.py --hf HuggingFaceFW/fineweb-edu:sample-10BT --max-docs 20000 --vocab 32768 --out data/pilot
python examples/pilot_100m.py --data data/pilot --out runs/dense --arch dense --steps 500 --batch 16 --seq 1024 --grad-accum 2 --eval-every 50 --ckpt-every 100 --peak-tflops 182
```
- **需要回傳：**
  - `CUDA SMOKE: x/13` 和每一行 PASS／FAIL；
  - `runs/dense/log.jsonl` 的最後幾行：`val_bits_per_byte`、`tokens_per_s`、`mfu`。
- **用完就 Terminate。** Stop 仍然會收儲存費，`/workspace` 會隨 pod 一起刪除。

## 5. 下一步（建議順序）

1. v5.90 合併後，在 pod 上重跑：
   `cd /workspace/helioslm && git pull origin main && python examples/cuda_smoke.py --out gpu_smoke.json`。
   得到 13/13 就關閉 #1。
2. 吞吐量：同一張卡用 `--compile`、較大 batch、`--warmup 50` 再跑一次 pilot，比較 tokens/s 與 MFU。
3. 若負責人同意安裝套件，做 E1，接著做 E4（同條件 baseline：SmolLM3-3B、Qwen3.5-2B、Gemma 4 E2B）。
4. 決定 v1.0 的評測門檻（E5），再考慮租 8×H100 跑 #40。
5. #42 之前：把 KDA 與 MoE 的 Python 迴圈改成向量化或 fused 實作。

## 6. 規則與注意事項（沿用）

- **憑證：** 負責人曾在聊天中貼出 HF 密碼與 token，**已建議更換，從未使用**，也不得寫入任何檔案。API key 一律放環境 secret。
- **套件：** 未經同意不安裝。曾兩次拒絕 `opencc`，不要再試。
- **蒸餾教師：** 不使用封閉模型（Claude、GPT、Gemini），其條款禁止拿輸出訓練競品。
- **固定 held-out 文件：** `docs/TW_DATASETS_2026Q4.md` 與 `docs/V5.4_FIXES_REPORT.md` 是 TW4／TW5 的 held-out 文字，**不可修改**，改了會讓 T114／T118 失敗。資料集更新寫在 `docs/TW_DATASETS_2026-10-20.md`。
- **CI：**
  - `python -m pytest helioslm_v5/tests/ -q -n 4 --ignore=helioslm_v5/tests/test_v5.py` 加 `python integration_test_v51.py`，本機約 11 分鐘；
  - 新的 benchmark JSON 必須登記到 `benchmarks/verification_tiers.json`。
- **網路：**
  - 沙箱可連 huggingface.co 的 metadata API 與 PyPI；
  - HF 檔案主機（`cas-server.xethub.hf.co`、`*.cdn.hf.co`）和 RunPod 被擋。
  - 所以西班牙文、日文、韓文的 tokenizer 成本量測仍卡在網路權限。
