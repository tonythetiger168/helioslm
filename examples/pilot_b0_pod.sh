#!/bin/bash
# Route B pilot (v5.137, common benchmarks v5.143, preflight v5.144, upload v5.146): base-gap eval + Gemma 4 E2B continued-pretraining
# pilot on one RunPod pod, then stop the pod. docs/PILOT_B0.md has the
# steps and the pass rule. Approved spend: US$320 in total (2026-10-24).
#
# Pod: 8x H200 SXM (Breexe-8x7B needs one 141 GB GPU in bf16), PyTorch
# image, volume >= 300 GB at /workspace, HF_TOKEN as a RunPod secret.
#   bash examples/pilot_b0_pod.sh 2>&1 | tee /workspace/pilot_b0.log
# Tunables: PER_HOUR (the pod's real US$/h), MAX_COST (hard ceiling for
# this script, US$), TOKENS (pilot training tokens), EVAL_RESERVE_H (hours
# kept after training for saving and steps 3-4).
# v5.175: ALLOW_BLOCKED=1 trains without mixture sources this account
# cannot download yet (their share goes to the rest of the phase; the
# skipped sources are listed in fetch_mixture.log and the manifest).
# SKIP_GAP=1 skips the step-1 base-gap eval (already run on its own pod),
# so the pod no longer needs a 141 GB GPU for the 8x7B reference; step 3
# then fetches only BENCHES (default: every bench this account can read).
set -uo pipefail
PER_HOUR=${PER_HOUR:-30}
MAX_COST=${MAX_COST:-300}
TOKENS=${TOKENS:-3e9}
EVAL_RESERVE_H=${EVAL_RESERVE_H:-2.0}
ALLOW_BLOCKED=${ALLOW_BLOCKED:-0}
SKIP_GAP=${SKIP_GAP:-0}
BENCHES=${BENCHES:-}
MIXARGS=(); [ "$ALLOW_BLOCKED" = 1 ] && MIXARGS=(--allow-blocked)
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

# pinned (versions the 2026-10-24 cloud dry run used), so a new release
# cannot change the run between pods; lm-eval for step 4 (common
# benchmarks) in the same resolve so it keeps these pins
pip install -q transformers==5.20.2 accelerate==1.15.0 datasets==5.1.0 \
    huggingface_hub==2.3.0 tokenizers==0.23.3 "lm-eval[hf,math]==0.4.13"

# 0. Preflight (about a minute, CPU only): stop before any GPU work if this
#    HF account cannot download an eval set or a mixture source.
GAPCHECK=(python examples/base_gap_eval.py --check-access)
[ "$SKIP_GAP" = 1 ] && GAPCHECK=(true)
if ! "${GAPCHECK[@]}" \
     || ! python examples/fetch_mixture.py --check-only "${MIXARGS[@]}" \
          --mixture data/mixtures/b1_gemma4_e4b_cpt_v1.json \
          --tokenizer google/gemma-4-E2B --tokens "$TOKENS" \
          --out /workspace/cpt_data; then
  echo "preflight failed: see above; stopping the pod"
  kill $WATCHDOG 2>/dev/null
  runpodctl stop pod "$RUNPOD_POD_ID"
  exit 1
fi
NGPU=$(nvidia-smi -L | wc -l)
GPUS=$(seq -s, 0 $((NGPU - 1)))
mkdir -p /workspace/results /workspace/runs

# 1. In parallel: base-gap eval on the GPUs (E4B vs the 4 reference models)
#    and the pilot corpus on CPU / network.
if [ "$SKIP_GAP" = 1 ]; then
  echo "base-gap eval skipped (SKIP_GAP=1)"; true & GAP=$!
else
  timeout 3h python examples/base_gap_eval.py --fetch --cleanup --gpus "$GPUS" \
      --data /workspace/tw_bench --out-dir /workspace/results \
      > /workspace/base_gap.log 2>&1 &
  GAP=$!
fi
timeout 3h python examples/fetch_mixture.py "${MIXARGS[@]}" \
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
  FETCH=(); [ "$SKIP_GAP" = 1 ] && FETCH=(--fetch)
  [ -n "$BENCHES" ] && FETCH+=(--benches $BENCHES)
  timeout 1h python examples/base_gap_eval.py "${FETCH[@]}" --gpus 0,1 \
      --models /workspace/runs/b0/final google/gemma-4-E2B \
      --base /workspace/runs/b0/final --refs google/gemma-4-E2B \
      --data /workspace/tw_bench --out-dir /workspace/results/pilot_b0 \
      > /workspace/pilot_eval.log 2>&1
  echo "pilot eval exit $?"
fi

# 5 (runs during step 4, network only). Keep the pilot after the pod is
#    gone: upload the final checkpoint to a private HF model repo
#    (HF_REPO, default <account>/helioslm-pilot-b0); results follow after
#    step 4. GGUF
#    conversion and the llama.cpp check run later on a local GPU, not here.
UP=
if [ -f /workspace/runs/b0/final/config.json ]; then
  timeout 1h python - > /workspace/upload.log 2>&1 <<'PY' &
import os
from huggingface_hub import HfApi
api = HfApi()
repo = os.environ.get("HF_REPO") or f"{api.whoami()['name']}/helioslm-pilot-b0"
api.create_repo(repo, private=True, exist_ok=True)
api.upload_folder(repo_id=repo, folder_path="/workspace/runs/b0/final",
                  commit_message="pilot B0 final checkpoint")
print("uploaded to", repo)
PY
  UP=$!
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
  # medicine and science, 0-shot (their defaults). PubMedQA is left out:
  # its dataset (bigbio/pubmed_qa) is a loading script, which datasets>=4
  # no longer runs.
  CUDA_VISIBLE_DEVICES=$2 accelerate launch --num_processes "$n" -m lm_eval \
      "${common[@]}" --tasks medqa_4options,medmcqa,sciq,agieval_gaokao_physics,agieval_gaokao_chemistry,agieval_gaokao_biology,agieval_gaokao_mathqa,agieval_sat_math,agieval_math
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
if [ -n "$UP" ]; then
  wait $UP; echo "checkpoint upload exit $?"; tail -2 /workspace/upload.log
  timeout 10m python - <<'PY'
import os
from huggingface_hub import HfApi
api = HfApi()
repo = os.environ.get("HF_REPO") or f"{api.whoami()['name']}/helioslm-pilot-b0"
api.upload_folder(repo_id=repo, folder_path="/workspace/results",
                  path_in_repo="results", commit_message="pilot B0 results")
print("results uploaded to", repo)
PY
  echo "results upload exit $?"
fi

echo "done: results in /workspace/results, run in /workspace/runs/b0"
kill $WATCHDOG 2>/dev/null
runpodctl stop pod "$RUNPOD_POD_ID"
