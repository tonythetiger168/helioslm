# DeepSeek 教師蒸餾（D 系列）

負責人 2026-10-22 決定：從 DeepSeek 蒸餾，教師在 RunPod 上自架，與其他工作平行進行。
本文件是計畫與 pod 操作手冊。程式在 `helioslm_v5/src/training/distill_gen.py`（D1）以及後續的 D2–D4。

## 版本選擇

| 角色 | 選擇 | 理由 |
|---|---|---|
| 教師 | **DeepSeek-V4.1-Flash** | MIT；`teacher_license_audit.py` 已釘雜湊（`usable()` 為真）。V4-Flash 的雜湊仍是 PENDING，而且已被 4.1 取代。V4-Pro 的 API 從 2026-09-14 起改導向 4.1-Flash。 |
| 學生 1 | 100M pilot | 先驗證管線，看蒸餾資料是否讓 bits/byte 與 TMMLU+ 進步。 |
| 學生 2 | 0.5B（#40，v1.0） | 主要目標：比同尺寸台灣模型分數高。思考鏈上限約 4k tokens。 |
| 之後 | 3B（#41）、30B-A3B（#42） | 重用同一份資料，再補較難的題目。需要預算。 |

授權：自架權重只受 MIT 約束（保留著作權與授權聲明）。若改走官方 API，DeepSeek 開放平台服務條款（2026-04-29 生效）第 4.2 條允許將輸出用於「training other models (such as model distillation)」；第 8.1 條要求向終端使用者揭露內容由 AI 生成。閉源模型（Claude、GPT、Gemini）不在審查表內，`check_teacher` 會拒絕。

## 管線

1. **D1 生成**（`distill_gen.py`、`examples/distill_generate.py`）
   - 題目由教師依 `data/distill/seeds_zhtw.json` 的主題自己出（不開 thinking），不使用非商用資料集的文字。
   - 教師再開 thinking 回答。每筆紀錄保留 reasoning、content、finish_reason、usage、教師與取樣參數。
   - 可續跑：已完成的 id 會跳過；失敗的請求寫到 `.errors.jsonl`，下次重跑。
   - 上限：可設 `--max-cost`、`--max-output-tokens`、`--max-hours`，到達後停止送出新請求，已送出的仍會寫入。
   - `stats` 子命令提供 OS1 的 token 統計：總量、p50/p95、思考占比與截斷率，整體與分領域各一份。
2. **D2 過濾**：截斷、重複、長度、繁中檢查、評測集去污染（`eval/decontam.py`），數學與程式題另加驗證器。
3. **D3 學生 SFT**：keep_thinking 決定學生看到多少思考內容；先跑 100M，再跑 0.5B。
4. **D4 代理軌跡**：教師在 hsh harness 裡使用工具，session log 中的軌跡當成 SFT 資料。

## Pod 規格與成本

- **4×H200 SXM（約 564 GB VRAM），TP4，vLLM ≥ 0.31.0。** 權重約 511 GB（專家層為 MXFP4）。vLLM recipe 驗證過的 H200 設定是 TP4，並把 Engram 表卸載到 CPU 記憶體。H100 沒有驗證過的設定。
- 100M pilot 共 2 萬題，平均輸出約 2,000 tokens（含思考），總計約 4,000 萬個輸出 tokens。
- 以每小時 US$15、吞吐 3,000 output tok/s 估算，加上 1 小時準備時間，總計約 4.7 小時、**約 US$70**。可用下面的指令重算：
  `python examples/distill_generate.py estimate --prompts 20000 --per-hour 15 --tok-per-s 3000`
- **吞吐量是推估值。** V4.1-Flash 的批次吞吐還沒有公開數字，所以實際跑時先用 `--limit 500` 量出 `output_tokens_per_s`，再用它重算成本。
- 代理軌跡（D4）：3,000 題、每題約 4 步，再加約 1 小時，**約 US$15**。
- 拿來對照的 API 價格（離峰）：輸入每百萬 tokens RMB 1，輸出 RMB 4；尖峰時段加倍。同樣的量約 RMB 170–340。

## Pod 操作步驟（複製貼上）

在 RunPod 建一個 pod：4×H200 SXM，image 用 `vllm/vllm-openai`（0.31 以上版本），掛 1 TB 的 volume 在 `/workspace`。

```bash
# 1. 權重（約 511 GB）與程式
export HF_HOME=/workspace/hf
cd /workspace && git clone https://github.com/tonythetiger168/helioslm && cd helioslm
huggingface-cli download deepseek-ai/DeepSeek-V4.1-Flash --local-dir /workspace/ds41
python examples/teacher_license_audit.py --pending   # 順便釘 V4-Flash 等列的雜湊（pod 連得到 HF）

# 2. 啟動教師（背景執行）
nohup vllm serve /workspace/ds41 --served-model-name deepseek-ai/DeepSeek-V4.1-Flash \
  --tensor-parallel-size 4 --enable-expert-parallel \
  --tokenizer-mode deepseek_v41 --reasoning-parser deepseek_v41 \
  --enable-auto-tool-choice --tool-call-parser deepseek_v41 \
  --engram-config '{"cpu_offload":true}' --kv-cache-dtype fp8 \
  --max-model-len 32768 --max-num-seqs 128 --gpu-memory-utilization 0.90 \
  --host 127.0.0.1 --port 8000 > /workspace/vllm.log 2>&1 &
until curl -s localhost:8000/v1/models >/dev/null; do sleep 30; done

# 3. 出題（約 2,100 次短請求）
R=/workspace/distill
python examples/distill_generate.py seed --per-topic 350 --out $R/meta.jsonl
python examples/distill_generate.py generate --prompts $R/meta.jsonl --out $R/meta_answers.jsonl \
  --no-thinking --max-tokens 2048 --temperature 0.9
python examples/distill_generate.py extract --answers $R/meta_answers.jsonl --out $R/prompts.jsonl

# 4. 先量 500 題，看吞吐與思考長度
python examples/distill_generate.py generate --prompts $R/prompts.jsonl --out $R/answers.jsonl \
  --shuffle-seed 0 --limit 500 --per-hour 15
python examples/distill_generate.py stats --answers $R/answers.jsonl

# 5. 正式跑（會接著 500 題繼續；設花費上限）
python examples/distill_generate.py generate --prompts $R/prompts.jsonl --out $R/answers.jsonl \
  --shuffle-seed 0 --limit 20000 --per-hour 15 --max-cost 120

# 6. 代理軌跡（D4，PR #91 合併後才有）：教師在 hsh 裡用工具解 3,000 個可驗證任務
python examples/distill_agent.py tasks --n 3000 --out $R/agent_tasks.jsonl
python examples/distill_agent.py run --tasks $R/agent_tasks.jsonl --out $R/trajectories.jsonl --workers 32
python examples/distill_agent.py to-sft --trajectories $R/trajectories.jsonl --out $R/agent_sft.jsonl

# 7. 關掉教師，先把資料備份出去（見下方），再用同一個 pod 跑 GPU 批次待辦，最後自動停 pod
pkill -f "vllm serve"; sleep 20
python examples/gpu_batch_run.py --out-dir /workspace/results --stop-pod
```

把資料拿回來：在 pod 上執行 `runpodctl send /workspace/distill`，再於本機執行 `runpodctl receive <code>`。也可以由負責人自己在 pod 上設定 `HF_TOKEN`，再上傳到私有 HF dataset。資料**不要**留在只有 container disk 的 pod 上。

`--per-hour 15` 只用來估算成本；實際的每小時價格以 RunPod 頁面為準，請改成實際數字。

## 學生訓練（D3，100M pilot）

蒸餾資料拿回來之後，可以在 4060 上跑，也可以在便宜的單卡 pod 上跑。`data/pilot/tokenizer.json` 和 `runs/dense` 是 v5.91 pilot 的輸出。

```bash
cat $R/kept.jsonl $R/agent_sft.jsonl > $R/sft_all.jsonl
python examples/distill_sft.py prepare --records $R/sft_all.jsonl \
  --tokenizer data/pilot/tokenizer.json --preset pilot_100m --out data/distill_sft
python examples/distill_sft.py train --data data/distill_sft --base-run runs/dense \
  --out runs/distill_sft --preset pilot_100m
```
