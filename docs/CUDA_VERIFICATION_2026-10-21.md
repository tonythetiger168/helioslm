# CUDA verification on an RTX 6000 Ada (Issue #1, 2026-10-21)

- **Machine:** RunPod, NVIDIA RTX 6000 Ada Generation (48 GB, compute
  capability 8.9). Driver 595.91, torch 2.14.1+cu130 (CUDA 13.0),
  Python 3.12.3.
- **Source:** numbers transcribed from the terminal output (screenshot), not
  from a committed JSON file.

## 1. Test suite on the GPU machine

`pytest helioslm_v5/tests -n 4` gave 574 passed, and `integration_test_v51.py`
gave 9/9 PASS.

**This is not a CUDA test:** no test in the suite moves a model to the GPU.
It shows that the suite runs on that machine and torch build.

## 2. `examples/cuda_smoke.py`: the code on the GPU

| Check | Result | Detail |
|---|---|---|
| forward_fp32 | PASS | logits vs CPU max abs diff 9.3e-7 (TF32 off) |
| greedy_generate | PASS | 16 greedy tokens with the KV cache identical to CPU |
| bf16_train_step | PASS | losses 6.97, 5.02, 3.87 |
| fp8_kv_cache | PASS | float8_e4m3fn latent cache generates (2 x 20) |
| pilot_dense | PASS | val loss 3.38 -> 0.64; ~19.6k tokens/s |
| pilot_moe | PASS | val loss 3.92 -> 1.47; ~7.5k tokens/s; router biases move |
| pilot_k3_block | PASS | val loss 0.47 -> 0.015; ~1.9k tokens/s |
| pilot_mimo_hybrid | PASS | val loss 3.72 -> 0.56; ~8.2k tokens/s |
| quant_awq | PASS | fused vs unfused 3.6e-7 |
| quant_gptq | PASS | fused vs unfused 4.5e-7 |
| quant_mxfp4 | PASS | fused vs unfused 4.8e-7 |
| fp8_trainer | **FAIL -> fixed** | "mat2 is on cpu, different from cuda:0" |
| gpu_monitor | **FAIL -> fixed** | "nvidia-ml-py does not seem to be installed" |

Each pilot row is the v5.83 loop at a tiny width (hidden 128, 4 layers) in
bf16, run for 40 steps, killed, then resumed to 80.

**The two failures were real CUDA bugs. Both are fixed in v5.90:**

- **`FP8Trainer`** built its `FP8Linear` layers on the default device, so
  a model already on the GPU got CPU layers. The layers are now created on
  the replaced layer's device.
- **`GPUMonitor.collect`** called `torch.cuda.utilization`, which needs
  NVML (`nvidia-ml-py`); without it the whole monitor raised. It now
  records memory and skips utilization, and the replica recommendation
  uses memory alone.
- `tests/test_cuda_device_paths.py` reproduces both on the CPU. Both
  tests fail on the old code.

**Performance observation, not a correctness issue:** at the same tiny
width, `k3_block` (KDA linear attention) trains about 10x slower than
dense, and the MoE arms 2.4-2.6x slower. That is consistent with
per-token / per-expert Python loops rather than fused kernels. It matters
for the cost of #42; see section 4.

## 3. 100M dense pilot (v5.83 loop, first GPU run)

Setup:
- Data: `pilot_prepare_data.py` on FineWeb-Edu `sample-10BT`, first 20,000
  documents. That gives 20.7M training tokens and 97k validation tokens,
  with a 32,768-token BPE.
- Model: dense, 93.0M parameters (42M of them embeddings).
- Training: bf16, batch 16 x 1024 tokens with 2 accumulation steps (32k
  tokens per step), 500 steps, 16.4M tokens in total.

| step | val loss | val bits / byte | tokens / s | MFU |
|---|---|---|---|---|
| 50 | 8.664 | 2.768 | 29,674 | 9.1% |
| 100 | 7.398 | 2.364 | 29,921 | 9.2% |
| 200 | 6.538 | 2.089 | 29,823 | 9.1% |
| 300 | 6.128 | 1.958 | 29,810 | 9.1% |
| 400 | 5.845 | 1.868 | 29,809 | 9.1% |
| 500 | 5.615 | **1.794** | 29,797 | 9.1% |

MFU is computed as 6 x 92.98M x tokens/s / 182 TFLOPS (dense BF16 peak).

- **The curve is sane.** Loss falls monotonically, grad norm stays
  0.5-0.8, and throughput is flat.
- **The whole run was learning-rate warmup.** `--warmup` defaults to
  500, so with `--steps 500` the learning rate only reached its peak at
  the last step and the cosine decay never ran. Pass `--warmup` of about
  5% of the steps next time.
- **MFU of 9% is low.** Likely causes:
  - no `torch.compile` (`--compile` exists but was not used);
  - small per-step matmuls at hidden 640;
  - the 32k-vocabulary output layer;
  - the formula counts the input-embedding parameters, which do no
    matmul work, so MFU on the true FLOPs is somewhat higher.
- **Not checked:** the kill-and-resume step of the gate. It is covered
  by the `pilot_*` smoke checks above at a small width.

## 4. What this means for the Phase C gate

- **Correctness on CUDA is established** for the paths above once v5.90
  is merged. Re-running `cuda_smoke.py` should give 13/13; that is the
  last step for #1.
- **Before renting a multi-GPU node, improve throughput.** Rough cost of
  the next pilot on the same card at about 30k tokens/s:
  - 1B tokens takes about 9.3 hours, or about $9 at $0.99 per hour.
  - Try `--compile` and a larger batch first.
  - At 9% MFU, the #40 estimate (about 330 H100-hours, which assumed
    roughly 30% MFU) would roughly triple.
- **Fix the KDA and MoE paths before #42 (30B-A3B).** At the measured
  10x and 2.5x slowdowns, MoE training cost would be several times the
  plan's estimate.
