#!/bin/bash
# Route B pilot (v5.137): base-gap eval + Gemma 4 E2B continued-pretraining
# pilot on one RunPod pod, then stop the pod. docs/PILOT_B0.md has the
# steps and the pass rule. Approved spend: US$320 in total (2026-10-24).
#
# Pod: 8x H200 SXM (Breexe-8x7B needs one 141 GB GPU in bf16), PyTorch
# image, volume >= 300 GB at /workspace, HF_TOKEN as a RunPod secret.
#   bash examples/pilot_b0_pod.sh 2>&1 | tee /workspace/pilot_b0.log
# Tunables: PER_HOUR (the pod's real US$/h), MAX_COST (hard ceiling for
# this script, US$), TOKENS (pilot training tokens).
set -uo pipefail
PER_HOUR=${PER_HOUR:-30}
MAX_COST=${MAX_COST:-300}
TOKENS=${TOKENS:-3e9}
export HF_HOME=/workspace/hf
cd /workspace
[ -d helioslm ] || git clone https://github.com/tonythetiger168/helioslm
cd helioslm && git pull -q

# Spend ceiling: whatever happens below, the pod stops at MAX_COST.
LIMIT_S=$(python3 -c "print(int(3600 * $MAX_COST / $PER_HOUR))")
echo "spend ceiling: US\$$MAX_COST at US\$$PER_HOUR/h -> pod stops after ${LIMIT_S}s"
( sleep "$LIMIT_S"; echo "spend ceiling reached"; runpodctl stop pod "$RUNPOD_POD_ID" ) &
WATCHDOG=$!
T0=$(date +%s)
left_h() { python3 -c "print(max(0.1, ($LIMIT_S - ($(date +%s) - $T0)) / 3600 - 1.0))"; }

pip install -q -U transformers accelerate datasets huggingface_hub tokenizers
NGPU=$(nvidia-smi -L | wc -l)
GPUS=$(seq -s, 0 $((NGPU - 1)))
mkdir -p /workspace/results /workspace/runs

# 1. In parallel: base-gap eval on the GPUs (E4B vs the 4 MediaTek models)
#    and the pilot corpus on CPU / network.
timeout 3h python examples/base_gap_eval.py --fetch --cleanup --gpus "$GPUS" \
    --data /workspace/tw_bench --out-dir /workspace/results \
    > /workspace/base_gap.log 2>&1 &
GAP=$!
timeout 3h python examples/fetch_mixture.py \
    --mixture data/mixtures/b1_gemma4_e4b_cpt_v1.json \
    --tokenizer google/gemma-4-E2B --tokens "$TOKENS" --out /workspace/cpt_data \
    > /workspace/fetch_mixture.log 2>&1
echo "fetch_mixture exit $?"; tail -5 /workspace/fetch_mixture.log
wait $GAP; echo "base_gap_eval exit $?"; tail -3 /workspace/base_gap.log

# 2. Pilot continued pretraining, stopped early if the ceiling is near
#    (one hour is kept for saving and the pilot eval).
MAXH=$(left_h)
echo "training for at most ${MAXH} h"
torchrun --nproc_per_node "$NGPU" examples/cpt_hf.py --model google/gemma-4-E2B \
    --data /workspace/cpt_data --tokens "$TOKENS" --max-hours "$MAXH" \
    --out /workspace/runs/b0 2>&1 | tee /workspace/cpt_b0.log
echo "cpt_hf exit ${PIPESTATUS[0]}"

# 3. Pilot vs its own base on the same four tasks.
if [ -f /workspace/runs/b0/final/config.json ]; then
  timeout 1h python examples/base_gap_eval.py --gpus 0,1 \
      --models /workspace/runs/b0/final google/gemma-4-E2B \
      --base /workspace/runs/b0/final --rivals google/gemma-4-E2B \
      --data /workspace/tw_bench --out-dir /workspace/results/pilot_b0 \
      > /workspace/pilot_eval.log 2>&1
  echo "pilot eval exit $?"
fi

echo "done: results in /workspace/results, run in /workspace/runs/b0"
kill $WATCHDOG 2>/dev/null
runpodctl stop pod "$RUNPOD_POD_ID"
