# Bounded CUDA dequantization tiles

Follow-up to the Windows RTX 4060 findings in PR #38. The old fused GPTQ
path enumerates maximal runs of g_idx, calls nonzero/tolist and int(CUDA
scalar), then launches one tiny matmul per run. Act-order can interleave group
IDs in original input-column order, creating hundreds of runs and repeated
host synchronization. AWQ and MXFP4 also launch many small group/block ops.
These are source-level mechanisms; no device kernel profile establishes the
exact fraction of the observed slowdown attributable to each.

CUDA fused=True now uses contiguous column tiles and device-side per-column
group gathers. No nonzero, tolist, item or scalar conversion is needed in the
tiled forward. CPU fused execution and fused=False dense references stay on
the original paths. No quantization/calibration formula, packed layout,
state_dict key, constructor argument or legacy .pt loader changes.

Tiles contain at most 1024 columns and approximately 1M decoded elements
(width=max(1,min(1024,1048576//out_features))). This bounds each decoded
weight tensor for ordinary layer widths; for >1M outputs even one column
exceeds that element budget. It does not cap total intermediate buffers,
matmul workspace or autograd saves. Small matrices can fit in a single tile
and thus materialize their whole weight. CUDA temporaries can exceed the
old group-wise path's allocation; this is an explicit latency/memory tradeoff.
No dense-weight cache is retained. Buffer mutations and strict checkpoint
reloads therefore take effect immediately.

Validation on CPU: targeted quantization tests exercise the same tile
arithmetic directly, including noncontiguous 3D input, odd shapes, width 1031
crossing a tile boundary, arbitrary/nonmonotonic GPTQ g_idx, FP32/FP16/BF16
inputs, input gradients, converted scale buffers, bias/no-bias, strict tensor
checkpoint round-trip and buffer mutation. Actual CUDA dispatch tests are
separate and must run on GPU; a CPU skip is not a pass. Existing integration
passed 9/9 on CPU. CPU benchmark wiring checks all three comparison arms.

## Windows retest

The standalone bundle contains original a0f0933 standard_quant.py as
baseline_standard_quant.py. From its extracted directory run:

```powershell
.\Run-Quant-Retest.cmd
```

Or use `python scripts/run_quant_cuda_retest.py`. No dependency installation
or upload is automatic. Existing CUDA PyTorch is used, with pytest required
for the targeted regression command. The result ZIP retains exact tested
sources and SHA256 hashes, numerical errors, timing samples and device name.
Return the ZIP even after failure. No --self-check option is provided.

Benchmark compares original fused, new tiled and dense-reference execution
with identical quantized tensors. Calibration is performed once on CPU and
excluded from timing; this differs from the earlier CUDA-prepared one-shot
benchmark and comparisons should be made between arms in this new run.
Each of three trials randomizes arm order and uses two warmups plus seven
synchronized wall-clock samples per arm. All arms share one process/allocator;
peak allocated bytes include both copies of the quantized module and other
live tensors, and are not total VRAM. No minimum speedup is asserted.
Shapes are [16,512] x [1024,512] and ragged [16,1031] x [65,1031].

There are no CUDA timing results for this patch yet. It is a PyTorch-vectorized
implementation, not a native fused CUDA/Triton kernel. It may still be slower
than dense dequantization. Do not replace the earlier measured numbers with
predicted speedups or claim this patch resolves the measured regression until
the new RTX 4060 report is reviewed.

## Build verification

55 targeted CPU tests passed; 9 actual-CUDA dispatch cases skipped because
the build host has CPU-only torch. Integration 9/9 passed. Benchmark CPU
wiring passed all three methods; lint and git diff checks passed. The skips
are expected to execute on the RTX 4060 retest, not be counted as GPU passes.
