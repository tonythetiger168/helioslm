# Device capability matrix and per-token cost (#175, #135)

`helioslm_v5/eval/device_matrix.py` records what each model path does on
each torch backend, checks numbers against the CPU float32 reference, and
writes timings with the context they were measured in.

## Backends

| Backend | How it is detected | Status |
|---|---|---|
| cpu | always | CI runs every test |
| cuda | `torch.cuda.is_available()` | runs recorded in `docs/CUDA_VERIFICATION_2026-10-21.md` |
| mps | `torch.backends.mps` | untested |
| xpu | `torch.xpu` | untested |
| xla (TPU) | `torch_xla` installed + `xla_device()` | untested, no TPU access |
| npu | `torch_npu` installed + `torch.npu` | untested, no NPU access |

`python -c "from helioslm_v5.eval.device_matrix import probe_backends as p; print(p())"`
prints what this machine has, with a reason for each missing backend.

## Feature statuses

`FEATURES` gives each feature one of `native`, `fallback`, `unsupported`,
`untested` per backend. `native` must name the run that showed it (the
tests reject it otherwise), and only CPU and CUDA have such runs today.
The FLA KDA kernel is `fallback` everywhere except CUDA: off CUDA the
model uses `chunked_delta_rule`. `resolve(feature, backend, strict=True)`
raises on a fallback, so a strict benchmark cannot report the generic path
as the accelerated one.

## Oracle

`forward_oracle(model, device, dtype)` compares a copy on the target with
CPU float32: full-sequence logits, and prefill followed by token-by-token
decode with the KV cache. The bound is max|got − ref| / max|ref|, fixed in
`TOLERANCES` before any run: float32 1e-4, float16 2e-2, bfloat16 6e-2.

The oracle runs on a dense lite config (`oracle_config()`). With MoE
layers, near-tied router scores pick different experts in bf16 and the
logits jump: on lite, seed 0, bf16 error is 0.33 with MoE and 0.011
dense. That is a routing difference, not a kernel error, so MoE is
checked in float32 only (`--moe`). CPU runs on lite measured bf16
0.011–0.016 and float16 0.0013–0.0016 over seeds 0–2.

## Timing and cost

```bash
python examples/device_bench.py --device cpu --dtype float32
python examples/device_bench.py --device cuda --dtype bfloat16 --usd-per-hour 0.59 --out bench.json
```

One `device-bench/1` record: backend, device name, dtype, batch, prompt
length, new tokens, model, torch version, commit, the oracle result, and
TTFT / decode tokens/s (p50 and tail) / peak memory (CUDA only; `None`
with a reason elsewhere). A failed oracle exits 1 and writes no timing.
USD per million decode tokens is filled only when `--usd-per-hour` is
given. The numbers are for the `lite` config and say nothing about a
packaged GGUF in llama.cpp.

## Adding a TPU or NPU result

Run `pytest helioslm_v5/tests/test_device_matrix.py` and
`examples/device_bench.py` on the machine. If the oracle passes, change
that backend's entries to `native` with the run as evidence (hardware,
driver, torch / plugin versions, date) and commit the JSON. Until then
the rows stay `untested`; the skipped tests are not a pass.
