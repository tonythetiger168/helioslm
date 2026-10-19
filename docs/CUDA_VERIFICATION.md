# Windows RTX 4060 validation: evidence and limits

The executable checks use explicit CUDA devices. Missing CUDA produces a
blocked report and exit 2; `--cpu-smoke` is a separate harness check and
does not count as CUDA execution. No script installs packages automatically.

## Reproduce

```powershell
python -u run_extended.py
python -u scripts/verify_cuda_extended.py --soak-seconds 600 --out extended_results/cuda_extended_latest.json
python -u scripts/train_native_fp8.py --steps 10 --out extended_results/native_fp8_training.json
python -u scripts/bench_cuda_mid.py --dtype float16 --lengths 128 --warmup 1 --repeats 3 --skip-generation --out extended_results/prefill_fp16.json
```

`Run-Extended.cmd` invokes the first command. The extended runner discovers
the monolithic case inventory in a short-lived child, executes each case
in a fresh process, runs general pytest and mid acceptance separately,
and records timeouts, exit codes, JUnit, integration and CUDA results.
Windows access-violation diagnostic messages are counted separately from
exit status; faulthandler is not disabled to hide them.

## Recorded single-machine scope

Hardware: NVIDIA GeForce RTX 4060, 8 GB; Windows, Python 3.14,
PyTorch 2.14.1+cu126. Other desktop applications remained running.

The Windows package derived from source `a0f0933` completed a 99-job run
on 2026-10-19 before the latest upstream port: 93 isolated original cases,
explicit CUDA areas, native FP8 lite training, general/mid pytest,
integration and CUDA microbenchmarks. Pytest had 434 passes and one
hardware-dependent NCCL skip; integration had 9/9 passes. This is historical
evidence for that package, not the test count for every later revision.

The stable-prefill optimization was separately checked with 55 related
regression tests, six mid-model GPU cases (FP16/BF16, batch 2, left padding,
17/33/65 tokens), chunked cache continuation and explicit contiguous
positions. Logits, hidden states and caches matched the prior token
schedule exactly; both dtypes produced the same bounded greedy sequences.
A four-token profiler comparison reduced `aten::_local_scalar_dense`
calls from 184 to 32. A 120-second mixed FP32 inference/training soak
returned to the same live allocation baseline; this is not a multi-hour
or low-precision training stability claim.

The reference schedule and tolerances are preserved. Internally generated
contiguous positions can use a table slice; explicit invalid position dtypes,
negative/custom positions and inconsistent layer-cache prefixes retain
their validation path. Training and general FP32 parallel scheduling are
not changed by the stable-prefill wrapper.

## Interpret performance cautiously

Timing is synchronized wall clock on an untrained mid 360M model, batch 1,
128 tokens. Baseline and optimized modes run interleaved in one process.
There is no hardware-isolation or production-throughput claim. Main-memory
pressure changed substantially during this session; cross-run absolute
times must not be compared as if they were a controlled experiment.

The MoE candidate that evaluated all experts to avoid routing readbacks was
rejected: it matched the bounded numerical oracles but regressed BF16
latency. It is not enabled in the model. The accepted optimization still
uses the conservative token schedule; the original roughly 20 ms parallel
prefill is not restored.

`scripts/bench_stable_prefill_ab.py` reconstructs the prior stable token
schedule with separate query/key RoPE lookups, while retaining the same
weights and compute dtypes. It interleaves both modes, synchronizes CUDA,
and rejects any changed logits, hidden states or cache tensor. The committed
[sample artifact](../benchmarks/windows_rtx4060_prefill_2026-10-19.json)
records raw samples, warmups, repeats and source hashes; its tier is measured,
not a training-quality or universal speed claim.

## Remaining scope

- Linux CUDA, two real GPUs and NCCL collectives have not been tested here.
  CPU/Gloo multiprocess tests cover gradient averaging and collective
  non-finite loss/gradient skipping, not GPU transport.
- Native FP8 uses actual E4M3 forward and E5M2 gradient GEMMs, with FP32
  accumulation/optimizer masters. Norms, embeddings, softmax and optimizer
  state are not all FP8. The operator is PyTorch's private `_scaled_mm`;
  revalidate when changing the runtime.
- Native training acceptance uses synthetic lite batches, not a convergence
  result or a mid-model training memory/quality certification.
- Stable low-precision prefill does not promise invariance under changing
  batch grouping. Nondefault/packed and multimodal execution retain their
  ordinary paths.
- The root cause of earlier Windows access-violation diagnostics has not
  been established. Successful runs without those messages do not prove a
  runtime-wide fix.
