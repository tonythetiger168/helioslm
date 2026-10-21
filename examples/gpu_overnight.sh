#!/usr/bin/env bash
# gpu_overnight.sh — unattended GPU session on a RunPod pod, then stop the
# pod so GPU billing ends. Run from the repo root inside the pod:
#
#   nohup bash examples/gpu_overnight.sh > /workspace/overnight.log 2>&1 &
#
# 1. examples/cuda_smoke.py (all CUDA + training checks, ~5-10 min)
# 2. pilot data: 100k FineWeb-Edu documents (~100M tokens, ~15 min)
# 3. 100M dense pilot with torch.compile, 3000 steps x 32k tokens
#    (~98M tokens, ~1 h at ~30k tokens/s)
# 4. runpodctl stop pod: GPU billing stops; /workspace (results) is kept and
#    only volume storage is billed until the pod is terminated.
#
# Each step has a time limit; a failed step does not block the next ones,
# and the pod is stopped whatever happens. Results: /workspace/results/.
set -u
OUT=/workspace/results
mkdir -p "$OUT"
step() { echo "=== $(date -u +%H:%M:%S) $*"; }

step "cuda_smoke"
timeout 45m python examples/cuda_smoke.py --out "$OUT/gpu_smoke.json" \
    > "$OUT/gpu_smoke.log" 2>&1
grep -E "^\[(PASS|FAIL)\]|CUDA SMOKE" "$OUT/gpu_smoke.log"

step "pilot data"
[ -f /workspace/data/pilot100k/meta.json ] || timeout 60m \
    python examples/pilot_prepare_data.py --hf HuggingFaceFW/fineweb-edu:sample-10BT \
    --max-docs 100000 --vocab 32768 --out /workspace/data/pilot100k \
    > "$OUT/prepare.log" 2>&1
tail -2 "$OUT/prepare.log" 2>/dev/null

step "pilot dense (compile)"
timeout 150m python examples/pilot_100m.py --data /workspace/data/pilot100k \
    --out "$OUT/dense_compile" --arch dense --steps 3000 --warmup 150 \
    --batch 16 --seq 1024 --grad-accum 2 --eval-every 100 --ckpt-every 250 \
    --peak-tflops 182 --compile > "$OUT/pilot.log" 2>&1
tail -3 "$OUT/pilot.log"

step "done; stopping the pod"
if command -v runpodctl >/dev/null && [ -n "${RUNPOD_POD_ID:-}" ]; then
    runpodctl stop pod "$RUNPOD_POD_ID"
else
    echo "runpodctl or RUNPOD_POD_ID missing: stop the pod by hand"
fi
