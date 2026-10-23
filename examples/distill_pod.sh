#!/usr/bin/env bash
# HeliosLM DeepSeek-V4.1-Flash 教師蒸餾：在 RunPod pod 上一次跑完（docs/DISTILL_DEEPSEEK.md 步驟 1–7）
# Pod：4×H200 SXM，image vllm/vllm-openai (>=0.31)，volume 1 TB 掛在 /workspace。
# 用法（pod 的 web terminal）：
#   bash distill_pod.sh 2>&1 | tee /workspace/distill_pod.log
# 可調：PER_HOUR（實際每小時價格，US$）、MAX_COST（生成階段花費上限，US$）、N_PROMPTS、N_AGENT
set -euo pipefail
AVAIL_GB=$(df -BG --output=avail /workspace | tail -1 | tr -dc 0-9)
if [ ! -d /workspace/ds41 ] && [ "$AVAIL_GB" -lt 700 ]; then echo "/workspace 只剩 ${AVAIL_GB} GB，權重要約 511 GB，加上資料至少要 700 GB"; exit 1; fi
PER_HOUR=${PER_HOUR:-15}
MAX_COST=${MAX_COST:-120}
N_PROMPTS=${N_PROMPTS:-20000}
N_AGENT=${N_AGENT:-3000}
R=/workspace/distill
export HF_HOME=/workspace/hf
mkdir -p $R

# 1. 程式與權重（約 511 GB，可續傳）
cd /workspace
[ -d helioslm ] || git clone https://github.com/tonythetiger168/helioslm
cd helioslm && git pull -q
if command -v hf >/dev/null; then HFDL="hf download"; else HFDL="huggingface-cli download"; fi
$HFDL deepseek-ai/DeepSeek-V4.1-Flash --local-dir /workspace/ds41
python examples/teacher_license_audit.py --pending || true

# 2. 啟動教師
if ! curl -s localhost:8000/v1/models >/dev/null; then
  nohup vllm serve /workspace/ds41 --served-model-name deepseek-ai/DeepSeek-V4.1-Flash \
    --tensor-parallel-size 4 --enable-expert-parallel \
    --tokenizer-mode deepseek_v41 --reasoning-parser deepseek_v41 \
    --enable-auto-tool-choice --tool-call-parser deepseek_v41 \
    --engram-config '{"cpu_offload":true}' --kv-cache-dtype fp8 \
    --max-model-len 32768 --max-num-seqs 128 --gpu-memory-utilization 0.90 \
    --host 127.0.0.1 --port 8000 > /workspace/vllm.log 2>&1 &
fi
until curl -s localhost:8000/v1/models >/dev/null; do sleep 30; echo "等 vLLM 啟動…"; done

# 3. 出題（已有就略過）
[ -s $R/meta.jsonl ] || python examples/distill_generate.py seed --per-topic 350 --out $R/meta.jsonl
python examples/distill_generate.py generate --prompts $R/meta.jsonl --out $R/meta_answers.jsonl \
  --no-thinking --max-tokens 2048 --temperature 0.9
python examples/distill_generate.py extract --answers $R/meta_answers.jsonl --out $R/prompts.jsonl

# 4. 先量 500 題
python examples/distill_generate.py generate --prompts $R/prompts.jsonl --out $R/answers.jsonl \
  --shuffle-seed 0 --limit 500 --per-hour $PER_HOUR
python examples/distill_generate.py stats --answers $R/answers.jsonl | tee $R/stats_500.json

# 5. 正式跑（從 500 題接著跑，有花費上限）
python examples/distill_generate.py generate --prompts $R/prompts.jsonl --out $R/answers.jsonl \
  --shuffle-seed 0 --limit $N_PROMPTS --per-hour $PER_HOUR --max-cost $MAX_COST
python examples/distill_generate.py stats --answers $R/answers.jsonl | tee $R/stats.json

# 5b. 過濾
python examples/distill_filter.py --answers $R/answers.jsonl --out $R/kept.jsonl \
  --eval data/std_bench/gsm8k_test_first50.jsonl data/std_bench/humaneval_first20.jsonl

# 6. 代理軌跡（D4）
[ -s $R/agent_tasks.jsonl ] || python examples/distill_agent.py tasks --n $N_AGENT --out $R/agent_tasks.jsonl
python examples/distill_agent.py run --tasks $R/agent_tasks.jsonl --out $R/trajectories.jsonl --workers 32
python examples/distill_agent.py to-sft --trajectories $R/trajectories.jsonl --out $R/agent_sft.jsonl

# 7. 關教師，打包資料（留在 /workspace volume），再跑 GPU 批次待辦並停 pod
pkill -f "vllm serve" || true; sleep 20
tar czf /workspace/distill_$(date +%Y%m%d).tgz -C /workspace distill
echo "資料已打包：/workspace/distill_*.tgz。可用 runpodctl send 傳回本機。"
python examples/gpu_batch_run.py --out-dir /workspace/results --stop-pod
