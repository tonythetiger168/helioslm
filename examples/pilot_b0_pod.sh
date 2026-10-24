#!/bin/bash
# Route B pilot (v5.137, common benchmarks v5.141): base-gap eval + Gemma 4 E2B continued-pretraining
# pilot on one RunPod pod, then stop the pod. docs/PILOT_B0.md has the
# steps and the pass rule. Approved spend: US$320 in total (2026-10-24).
#
# Pod: 8x H200 SXM (Breexe-8x7B needs one 141 GB GPU in bf16), PyTorch
# image, volume >= 300 GB at /workspace, HF_TOKEN as a RunPod secret.
#   bash examples/pilot_b0_pod.sh 2>&1 | tee /workspace/pilot_b0.log
# Tunables: PER_HOUR (the pod's real US$/h), MAX_COST (hard ceiling for
# this script, US$), TOKENS (pilot training tokens), EVAL_RESERVE_H (hours
# kept after training for saving and steps 3-4).
set -uo pipefail
PER_HOUR=${PER_HOUR:-30}
MAX_COST=${MAX_COST:-300}
TOKENS=${TOKENS:-3e9}
EVAL_RESERVE_H=${EVAL_RESERVE_H:-2.0}
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
left_h() { python3 -c "print(max(0.1, ($LIMIT_S - ($(date +%s) - $T0)) / 3600 - $EVAL_RESERVE_H))"; }

pip install -q -U transformers accelerate datasets huggingface_hub tokenizers
# step 4 (common benchmarks); pinned so the task definitions stay fixed
pip install -q "lm-eval[hf,math]==0.4.13"
NGPU=$(nvidia-smi -L | wc -l)
GPUS=$(seq -s, 0 $((NGPU - 1)))
mkdir -p /workspace/results /workspace/runs

# 1. In parallel: base-gap eval on the GPUs (E4B vs the 4 reference models)
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
#    (EVAL_RESERVE_H is kept for saving and the evals in steps 3-4).
# Check first that the multimodal E2B loads for causal LM and that the
# default --freeze hits its per-layer tables and vision / audio towers.
python examples/cpt_hf.py --model google/gemma-4-E2B --data /workspace/cpt_data \
    --tokens "$TOKENS" --out /workspace/runs/b0 --dry-run \
    > /workspace/cpt_dry_run.log 2>&1
echo "cpt_hf dry run exit $?"; tail -20 /workspace/cpt_dry_run.log
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
      --base /workspace/runs/b0/final --refs google/gemma-4-E2B \
      --data /workspace/tw_bench --out-dir /workspace/results/pilot_b0 \
      > /workspace/pilot_eval.log 2>&1
  echo "pilot eval exit $?"
fi

# 4. Common benchmarks (lm-evaluation-harness, the numbers model cards
#    report), E2B base and pilot side by side, half the GPUs each. Shots
#    follow the usual base-model settings. humaneval / mbpp execute the
#    model's code, which lm-eval only does with HF_ALLOW_CODE_EVAL=1 and
#    --confirm_run_unsafe_code.
std_bench() {  # $1 model, $2 GPU list, $3 output dir
  local n; n=$(echo "$2" | tr , '\n' | wc -l)
  local common=(--model hf --model_args "pretrained=$1,dtype=bfloat16"
                --batch_size auto --output_path "$3")
  CUDA_VISIBLE_DEVICES=$2 accelerate launch --num_processes "$n" -m lm_eval \
      "${common[@]}" --tasks mmlu,mmmlu_zh_cn,winogrande --num_fewshot 5
  CUDA_VISIBLE_DEVICES=$2 accelerate launch --num_processes "$n" -m lm_eval \
      "${common[@]}" --tasks arc_challenge --num_fewshot 25
  CUDA_VISIBLE_DEVICES=$2 accelerate launch --num_processes "$n" -m lm_eval \
      "${common[@]}" --tasks hellaswag --num_fewshot 10
  # each task's own default shots: truthfulqa_mc2 0, gsm8k 5,
  # bbh_cot_fewshot 3, minerva_math500 4, humaneval 0, mbpp 3
  HF_ALLOW_CODE_EVAL=1 CUDA_VISIBLE_DEVICES=$2 accelerate launch \
      --num_processes "$n" -m lm_eval "${common[@]}" \
      --tasks truthfulqa_mc2,gsm8k,bbh_cot_fewshot,minerva_math500,humaneval,mbpp \
      --confirm_run_unsafe_code
}
HALF=$((NGPU / 2)); [ "$HALF" -ge 1 ] || HALF=1
GA=$(seq -s, 0 $((HALF - 1)))
GB=$(seq -s, "$HALF" $((NGPU - 1))); [ -n "$GB" ] || GB=$GA
STD_S=$(python3 -c "print(int(3600 * max(0.1, $(left_h) + $EVAL_RESERVE_H - 0.3)))")
mkdir -p /workspace/results/std_bench
( export -f std_bench; timeout "$STD_S" bash -c \
    "std_bench google/gemma-4-E2B $GA /workspace/results/std_bench/e2b_base" \
    > /workspace/std_bench_base.log 2>&1 ) &
SB=$!
if [ -f /workspace/runs/b0/final/config.json ]; then
  ( export -f std_bench; timeout "$STD_S" bash -c \
      "std_bench /workspace/runs/b0/final $GB /workspace/results/std_bench/pilot_b0" \
      > /workspace/std_bench_pilot.log 2>&1 )
  echo "std bench (pilot) exit $?"
fi
wait $SB; echo "std bench (base) exit $?"

echo "done: results in /workspace/results, run in /workspace/runs/b0"
kill $WATCHDOG 2>/dev/null
runpodctl stop pod "$RUNPOD_POD_ID"
