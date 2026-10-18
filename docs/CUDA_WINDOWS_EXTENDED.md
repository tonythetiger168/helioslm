# Windows RTX 4060 extended verification — 2026-10-18

Returned archive: HeliosLM_Extended_Results_20261018_164030_416582.zip.
Summary, individual case logs, pytest/JUnit records and benchmark JSON were
cross-checked. 98 runner jobs passed, including 93 isolated original test
functions, two pytest commands, integration, CUDA areas and the benchmark.

## Results and scope

- Explicit CUDA checks: 14/14, including FP32/FP16/BF16 cache; all cuda:0.
- Original monolithic functions: 93/93, each unchanged in a fresh process.
  This includes the two CPU multiprocess DualPipe numerical equivalence tests;
  they do not establish CUDA multiprocessing, multi-GPU or NCCL support.
- Pytest: 421/421 plus 4/4 mid acceptance tests in a fresh process, zero
  failures/errors/skips in both JUnit XML files (425 total). Existing fixtures
  mainly use CPU. These are not 425 CUDA tests. Separate process execution
  does not verify shared interpreter state or the original all-in-one runner.
- Integration: 9/9. Its existing unlabeled FP8 step warning remains expected;
  those raw-logit losses do not establish supervised learning convergence.
- Benchmark: all seven numerical/execution checks passed. No speed threshold.

Hardware: Windows, Python 3.14.8, torch 2.14.1+cu126, RTX 4060 8 GB, compute
capability 8.9. Base marker: a0f09332f1fc6633615d48397dda77e629ac6cf6.
The tested bundle also includes the FP8 placement fix and local runners.
The returned commands split test_v533_mid.py out of pytest into its own
process and enable verbose/JUnit output; the originally delivered launcher
used one pytest command. The modified runner source was not returned, so
exact launcher/tree provenance cannot be cryptographically verified from this
ZIP. The original uploaded ZIP SHA256 is retained in the summary evidence.
No secrets or user-specific Windows paths are included in public summaries.

## Model latency

Lite random-weight model, batch 1, 64 prompt tokens, 16 new tokens actually
produced in every sample. Two warmups and seven synchronized wall-clock
measurements. Generation includes prefill. No trained quality claim.

| dtype | Prefill median ms | Generate median ms | Tokens/s | Generate peak allocated MiB |
|---|---:|---:|---:|---:|
| FP32 | 6.143 | 86.875 | 184.17 | 41.88 |
| FP16 | 5.779 | 78.645 | 203.45 | 25.01 |
| BF16 | 5.176 | 75.398 | 212.21 | 25.01 |

Memory is PyTorch allocator memory, not total VRAM usage from nvidia-smi.
Sequential measurements share the process/allocator; peak reserved memory
can retain earlier allocations. These numbers are a small-model snapshot,
not a capacity or throughput guarantee for larger checkpoints.

## Quantized Linear: correctness passed, fused paths slower

Float32 input [16, 512], weight [1024, 512]. Preparation excluded. Two
warmups/seven samples per arm in one process, unfused measured first.
No randomized arm ordering or independent process trials. Max scaled errors
AWQ 1.919e-7, GPTQ 5.633e-7, MXFP4 1.845e-7, all below 1e-4.

| Method | Unfused ms | Fused ms | Fused/unfused latency | Unfused/fused peak allocated MiB |
|---|---:|---:|---:|---:|
| AWQ | 0.1982 | 0.8582 | 4.33x slower | 18.51 / 11.98 |
| GPTQ | 0.2548 | 66.0700 | 259.30x slower | 18.51 / 10.86 |
| MXFP4 | 0.2350 | 3.6786 | 15.65x slower | 23.04 / 11.76 |

The fused implementations trade allocation for tiled work and Python/operator
overhead. The timings establish a limitation for this measured shape/device;
they do not identify the exact bottleneck or prove a universal slowdown.
No dispatch policy or production kernel was changed based on this one run.
The successful status means the benchmark ran and numerical checks passed;
it does not mean fused execution accelerated inference. The CPU benchmark
from Issue #2 cannot be used to infer CUDA speedups.

## Remaining unverified scope

Multi-GPU/NCCL, CUDA multiprocessing, exhaustive architecture combinations,
large-model capacity/throughput and native FP8 tensor-core product kernels.
The FP8 operator profiler records float32 tensors and aten::matmul/aten::mm;
it traces CPU dispatch of CUDA work, not device kernel events. Current
FP8Linear simulates the FP8 value grid with STE and ordinary matmul.
An absence of native FP8 GEMM implementation is not a passing hardware test.
