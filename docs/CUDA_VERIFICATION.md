# CUDA verification — Issue #1

Status on 2026-10-18: **14/14 explicit CUDA area checks PASSED** on the
user's Windows RTX 4060 8 GB host. Python 3.14.8, PyTorch 2.14.1+cu126,
CUDA build 12.6, driver 596.49, compute capability 8.9. The returned launcher
summary has exit code 0 and `cuda_checks_passed=true`; its console log and JSON
agree. Report timestamp: 2026-10-18T07:00:13Z (15:00 Taiwan time).
Evidence: `benchmarks/cuda_rtx4060_windows_2026-10-18.json`.
The report identifies base commit a0f0933; the distributed snapshot additionally
contains the FP8 placement fix and verification harness described below.

The separate development environment is CPU-only. Its earlier blocked-CUDA
report is retained as a negative-path check, not the current hardware result.
The initial 15:00 Windows run did not request regression. The later 16:40
extended run verified 93 isolated cases, 425 pytest cases and integration 9/9;
see [extended Windows evidence](CUDA_WINDOWS_EXTENDED.md) for the exact process
split, benchmark slowdown findings and provenance limitations.

Base source: a0f09332f1fc6633615d48397dda77e629ac6cf6 (v5.72 plus balancing-state tests).
The historical `docs/V5.5_FEATURES_REPORT.md` referenced in the issue is not
present in this checkout; the scope below is grounded in current modules.

## Concrete fix

`FP8Trainer._convert_to_fp8` previously constructed each replacement FP8Linear
on default CPU/float32, even if the source Linear was already on CUDA or a
different dtype. It now moves the replacement to the source weight's device
and dtype BEFORE copying values and creating fp32 optimizer masters.

The pre-fix tests fail on CPU float64 (dtype is lost) and meta-device weights
(copying a meta tensor into a CPU allocation is impossible). Post-fix tests
cover CPU float64, CPU bfloat16 and meta placement, preserving values,
registered-buffer device, master fp32 dtype/device and finite CPU gradients.
Meta is a placement oracle, not a GPU execution substitute. The Windows CUDA run passed the two-step FP8 training check with losses
6.997920 and 4.575287, including parameter and master placement checks.

## Run on an actual GPU host

Use an existing CUDA-enabled PyTorch environment with a supported NVIDIA
driver. Do not install this repo's `requirements-ci.txt` on that environment:
it deliberately selects CPU wheels for ordinary GitHub CI.

```sh
python -c 'import torch; assert torch.cuda.is_available(); print(torch.__version__, torch.version.cuda, torch.cuda.get_device_name(0))'
python scripts/verify_cuda.py --out benchmarks/cuda_actual.json --run-regression
```

The script runs explicit-device checks and then, when requested, the unchanged
monolithic test runner, pytest suite and `integration_test_v51.py`, storing their
logs next to the JSON report. Those legacy fixtures are primarily CPU tests,
even on a GPU host; this is labeled in the report. Do not turn their pass counts
into CUDA evidence by changing the global default device.

Use `--device-index N`, `--dtypes float32 float16 bfloat16` and `--timeout SECONDS`
to select the device, requested cache dtype matrix and each regression command's
timeout. Defaults request all three dtypes, index 0, timeout 1800 seconds.
BF16-unsupported hardware is recorded as unsupported (not passed), and makes
the requested area check incomplete. A deliberately smaller dtype selection
must be disclosed with the resulting scope. TF32 matmul/cuDNN paths are disabled
for CUDA checks so numeric tolerances target the requested dtype.

## What the explicit-device checks exercise

| Area | Oracle / scope |
|---|---|
| Full-model MLA cache | Cached token-by-token vs full-forward logits; requested dtype matrix |
| Hybrid cache | Interleaved GDA/MLA cached/full equivalence in float32 |
| MTP | Speculative vs plain greedy token IDs exact |
| Engine + paged cache | Unequal-length serving vs greedy prefix, no block leak, actual paged tensor placement |
| FP8 training | Real lite model, two labeled CE steps, finite loss/gradients, all parameters/masters on requested device |
| GRPO | Real lite-model train_step and finite device-resident gradients |
| DualPipe | Single-process schedule, real model layers, finite nonzero attention gradients |
| Audio | Three streaming chunks vs one-shot output |
| Vision | NaViT small-image forward, expected shape, finite output/device |
| AWQ | Calibrated quantized Linear; fused/unfused relative max error <1e-4 |
| GPTQ | Calibrated act-order ragged Linear; fused/unfused relative max error <1e-4 |
| MXFP4 | Ragged fixed-format Linear; fused/unfused relative max error <1e-4 |

Cache normalized max-error thresholds are 1e-4 (fp32), 2e-2 (fp16), 5e-2
(bf16); audio uses max absolute error <1e-3. Errors/tolerances are written to
JSON rather than inferred from a successful process exit. The predeclared tolerances were unchanged for the Windows hardware run.
Passing on this device does not establish a universal hardware tolerance.

This does NOT verify multi-GPU/NCCL, CUDA multi-process DualPipe, every possible
architecture-option combination, native FP8 tensor-core GEMMs, or throughput.
The FP8 implementation uses value quantization + STE and ordinary matmul; no
native tensor-core speed claim is made. Quantized Linear checks are not a
whole-model quantization performance claim. Review the remaining scope before
closing the issue; the successful single-GPU run does not cover these paths.

## Honest report and exit semantics

- No requested CUDA device: JSON `status=blocked_no_cuda`, no executed areas,
  `cuda_checks_passed=false`, exit 2. No implicit CPU fallback.
- Explicit `--cpu-smoke`: 12 float32 CPU areas; successful status is
  `cpu_smoke_passed`, but `cuda_checks_passed` stays false. This validates
  harness wiring only.
- Requested CUDA areas all pass: `cuda_area_checks_passed`; any failure or
  unsupported requested dtype gives `failed_or_partial` and exit 1.
- CUDA initialization error: `failed_cuda_initialization`, exit 1, traceback
  retained before any area runs.
- Optional regression failure/timeout: `regression_failed`, exit 1, separate
  command logs retained. A failed check never becomes a skip/pass.

These reports deliberately do not include an automatic "Issue #1 complete"
claim. Logs, device/model identity and uncovered paths require human review.

## Local evidence

```sh
OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 python -m pytest \
  helioslm_v5/tests/test_fp8_conversion_device.py \
  helioslm_v5/tests/test_cuda_verifier.py helioslm_v5/tests/case_trainers.py -q
python scripts/verify_cuda.py --cpu-smoke --out benchmarks/cuda_cpu_smoke_2026-10-18.json
python scripts/verify_cuda.py --out benchmarks/cuda_verification_2026-10-18.json
```

- 12 targeted CPU tests passed (including portable source-revision fallback); one existing no-labels FP8 warning is expected.
- 12/12 explicit CPU area checks passed; their JSON says CUDA is unverified.
- No-CUDA preflight returned exit 2 as designed.
- Integration suite: 9/9 PASS on CPU.
- The entire pytest collection was not rerun for this change. The targeted
  trainer tests and the monolithic supported-suite check cover the changed path.

The changes and returned Windows GPU evidence are prepared on branch
`codex/issue1-cuda-readiness` for review. The remaining uncovered scope above
still applies; the hardware result does not automatically close Issue #1.

Supported monolithic CPU regression: **91/91 PASS**, excluding exactly
`test_multi_process_dualpipe` and
`test_multi_process_dualpipe_attention_residuals` because their local socket
resource-sharing paths were blocked in this environment earlier in the task.
No multiprocess CUDA claim is made. Reproduction of that local supported subset:

```sh
OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 python - <<'PY'
from helioslm_v5.tests import test_v5
excluded = {"test_multi_process_dualpipe",
            "test_multi_process_dualpipe_attention_residuals"}
test_v5.TESTS = [t for t in test_v5.TESTS if t.__name__ not in excluded]
test_v5.main()
PY
```

The GPU host's `--run-regression` uses the unmodified full runner instead;
this local exclusion is not built into the verification command.

## Returned Windows GPU evidence

The 14 checks all executed on `cuda:0`. Cache coverage includes float32,
float16 and bfloat16; the remaining area checks use the harness defaults.
Cache max scaled errors were 5.2812e-7, 0.00122951 and 0 respectively, below
thresholds 1e-4, 0.02 and 0.05. A zero BF16 error is the observed result for
this particular input, not a general exactness guarantee. Hybrid cache passed;
MTP token IDs and engine greedy prefixes matched exactly. Engine paged-cache
placement was `cuda:0`. Audio max absolute error was 1.3113e-6 (<1e-3), and
vision output shape was [1, 16, 64]. AWQ/GPTQ/MXFP4 max scaled errors were
1.9499e-7 / 2.8486e-7 / 2.0791e-7 (<1e-4).

GRPO returned finite loss 0.0 and 55 gradient tensors. This check does not
assert nonzero GRPO update magnitude or learning improvement. Single-process
DualPipe passed with loss 0.950670. Neither its result nor FP8's two decreasing
loss values proves convergence, multi-GPU operation, throughput or tensor-core
acceleration. No additional GPU fix was needed after this returned run.

## Extended Windows run

See [CUDA_WINDOWS_EXTENDED.md](CUDA_WINDOWS_EXTENDED.md) and the three
`benchmarks/cuda_windows_extended_2026-10-18_*.json` evidence files. The later
run covers the two previously blocked CPU multiprocess tests on Windows.
Fused quantized Linear execution was slower for all three methods on the
measured CUDA shape; successful benchmark status is not a speedup claim.
