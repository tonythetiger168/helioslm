#!/usr/bin/env bash
# v1_prune_ladder_pod.sh — v1.0 plan, stages 0 and 1, on one GPU pod; then
# the results go to the Hub and the pod is removed so billing ends.
#
# The 2026-11-06 prune + KD pilot (0.58B effective, 238M KD tokens) scored at
# chance on every bench (TMMLU+ 24.8, tw_justice 21.7) and its pruned start
# was already at chance. Before paying for a longer KD run, this measures:
#   stage 0  the untouched google/gemma-4-E2B and gemma-4-E4B, i.e. the
#            ceiling of the prune route;
#   stage 1  E2B pruned to four sizes with no training (LADDER below), to find
#            the smallest size that stays clearly above chance.
# Every model is scored by examples/base_gap_eval.py (the scorer of the
# pilot): TMMLU+ and C-Eval 5-shot, and the 2026 held-out tw_justice.
# Qwen3.5-0.8B was scored by the same scorer on 2026-11-06 (TMMLU+ 35.68,
# tw_justice 38.54); set EXTRA_MODELS="Qwen/Qwen3.5-0.8B" to re-score it.
#
# Run from the repo root inside the pod (one A100 80 GB is enough; about
# 3-4 h), with HF_TOKEN set (private data: the mixture's helioslm-twdata /
# rephrase sources and the held-out set; results are uploaded with it):
#
#   nohup bash examples/v1_prune_ladder_pod.sh > /workspace/ladder.log 2>&1 &
#
# Env: OUT (/workspace/ladder), CALIB_TOKENS (2e7, calibration text only),
#      EXTRA_MODELS (""), REMOVE_POD (1: runpodctl remove pod at the end;
#      0: stop it and keep /workspace), LOG_REPO (<hf user>/helios-pod-logs).
set -u
OUT=${OUT:-/workspace/ladder}
CALIB_TOKENS=${CALIB_TOKENS:-2e7}
EXTRA_MODELS=${EXTRA_MODELS:-}
REMOVE_POD=${REMOVE_POD:-1}
SRC=google/gemma-4-E2B
# name  layers-kv  layers-shared  hidden  ffn   (E2B: 15 + 20 layers, 1536, 6144)
LADDER="A 8 8 1024 4096
B 10 12 1280 5120
C 12 16 1536 6144
D 15 20 1024 4096"
export HF_HOME=${HF_HOME:-/workspace/hf} PYTORCH_ALLOC_CONF=expandable_segments:True
mkdir -p "$OUT/results"
step() { echo "=== $(date -u +%H:%M:%S) $*"; }

push() {
    python3 - "$OUT" <<'EOF' || true
import glob, os, sys
from huggingface_hub import HfApi
out = sys.argv[1]
api = HfApi(token=os.environ.get("HF_TOKEN"))
repo = os.environ.get("LOG_REPO") or api.whoami()["name"] + "/helios-pod-logs"
pod = os.environ.get("RUNPOD_POD_ID", "pod")
for f in glob.glob(f"{out}/*.log") + glob.glob(f"{out}/results/**/*.json", recursive=True):
    if os.path.getsize(f) < 50_000_000:
        api.upload_file(path_or_fileobj=f, repo_id=repo, repo_type="dataset",
                        path_in_repo=f"{pod}/{os.path.relpath(f, out)}",
                        commit_message=f"{pod} v1 prune ladder")
print("pushed", repo)
EOF
}
finish() {
    step "upload results"
    push
    if [ -n "${RUNPOD_POD_ID:-}" ]; then
        if [ "$REMOVE_POD" = 1 ]; then
            step "removing the pod"
            runpodctl remove pod "$RUNPOD_POD_ID" || runpodctl stop pod "$RUNPOD_POD_ID"
        else
            step "stopping the pod"
            runpodctl stop pod "$RUNPOD_POD_ID"
        fi
    fi
}
trap finish EXIT

[ -n "${HF_TOKEN:-}" ] || { echo "HF_TOKEN is not set"; exit 1; }
python3 -c "import torch; torch.zeros(1).cuda(); print(torch.__version__, 'cuda ok')" \
    || { echo "GPU unusable"; exit 1; }
nvidia-smi --query-gpu=name,memory.total --format=csv
git rev-parse HEAD > "$OUT/results/helioslm_commit.txt"
pip install -q transformers==5.20.2 accelerate==1.15.0 datasets==5.1.0 \
    huggingface_hub==2.3.0 tokenizers==0.23.3 safetensors

step "held-out tw_justice"
python3 - <<'EOF'
import os
from huggingface_hub import HfApi, snapshot_download
repo = os.environ.get("LOG_REPO") or HfApi().whoami()["name"] + "/helios-pod-logs"
snapshot_download(repo, repo_type="dataset", local_dir="/workspace/dl",
                  allow_patterns=["twexam/tw2026_bench/tw_justice/*"])
EOF
mkdir -p /workspace/heldout
[ -d /workspace/heldout/tw_justice ] || mv /workspace/dl/twexam/tw2026_bench/tw_justice /workspace/heldout/

step "calibration text ($CALIB_TOKENS tokens)"
[ -f "$OUT/data/manifest.json" ] || timeout 60m python examples/fetch_mixture.py --allow-blocked \
    --mixture data/mixtures/v1_prune_kd_pilot.json --tokenizer "$SRC" \
    --tokens "$CALIB_TOKENS" --workers "$(nproc)" --out "$OUT/data" > "$OUT/fetch.log" 2>&1
tail -3 "$OUT/fetch.log"
[ -f "$OUT/data/manifest.json" ] || { echo "no calibration data"; exit 1; }

step "stage 1: prune ladder (no training)"
MODELS="$SRC google/gemma-4-E4B $EXTRA_MODELS"
while read -r name kv sh hidden ffn; do
    d="$OUT/prune_$name"
    timeout 30m python examples/prune_kd_gemma4.py prune --model "$SRC" --data "$OUT/data" \
        --out "$d" --layers-kv "$kv" --layers-shared "$sh" --hidden "$hidden" --ffn "$ffn" \
        --calib 256 --seq 1024 > "$OUT/prune_$name.log" 2>&1
    echo "prune $name exit $?"
    if [ -f "$d/final/config.json" ]; then
        mkdir -p "$OUT/results/prune_$name"
        cp "$d/prune.json" "$OUT/results/prune_$name/" 2>/dev/null
        MODELS="$MODELS $d/final"
    fi
done <<< "$LADDER"
push

step "stage 0 + 1: TMMLU+ and C-Eval (5-shot)"
timeout 240m python examples/base_gap_eval.py --fetch --benches tmmluplus ceval \
    --base "$SRC" --refs --data /workspace/tw_bench --models $MODELS --gpus 0 \
    --out-dir "$OUT/results/twbench" > "$OUT/eval_twbench.log" 2>&1
echo "twbench exit $?"
push

step "stage 0 + 1: held-out tw_justice"
timeout 90m python examples/base_gap_eval.py --benches tw_justice \
    --base "$SRC" --refs --data /workspace/heldout --models $MODELS --gpus 0 \
    --out-dir "$OUT/results/heldout" > "$OUT/eval_heldout.log" 2>&1
echo "heldout exit $?"

step "summary"
python3 - "$OUT/results" <<'EOF' | tee "$OUT/summary.log"
import glob, json, sys
rows = {}
for f in sorted(glob.glob(f"{sys.argv[1]}/*/base_gap_*.json")):
    for m, r in json.load(open(f))["models"].items():
        for b, v in (r.get("benches") or {}).items():
            rows.setdefault(m, {})[b] = v["accuracy"]
for f in sorted(glob.glob(f"{sys.argv[1]}/prune_*/prune.json")):
    p = json.load(open(f))
    print(f.split("/")[-2], "effective params", round(p["params"]["effective"] / 1e9, 3), "B")
print("chance = 0.25; Qwen3.5-0.8B (2026-11-06): tmmluplus 0.3568, tw_justice 0.3854")
for m, r in rows.items():
    print(m, {b: round(a, 4) for b, a in sorted(r.items())})
EOF
step "done"
