# Mistral Large 4 (1T-A52B): licence gate and L512 fit (#185)

Status: **unverified**, licence **pending_review**. Checked 2026-11-01
without downloading weights, renting GPUs or calling paid APIs. Code:
`helioslm_v5/eval/external_model_gate.py`; tests:
`helioslm_v5/tests/test_external_model_gate.py`.

This is an external baseline, not a HeliosLM result, and it does not
count toward the v6.0 1T target (#45).

## Source pin

| | |
|---|---|
| Model | `mistralai/Mistral-Large-4-1T-A52B` @ `f12dabdc0c1e99e7f7924461a6f308859b05af4d` |
| Card licence | `other` |
| LICENSE | Mistral Ecosystem Open-Weight License (modified MIT), sha256 `d242ebed…8353c78` |
| GGUF | `ggml-org/Mistral-Large-4-1T-A52B-GGUF` @ `7d4f13b81e55e1223b3c125a58af4c19bb1ddf6a` |

`check_source` refuses the model if any of the four values changes.

## Licence reading per use

| Use | Reading |
|---|---|
| internal research | allowed (s.3(a): no third-party access) |
| redistribute weights | allowed with conditions (keep the notice; MaaS operators need their own licence) |
| commercial MaaS | needs a separate commercial licence (s.2; no revenue threshold) |
| official API outputs | pending review (API terms, not read) |
| training on outputs | pending review (licence is silent on outputs; a student as a derivative would inherit s.2) |

The model is **not** in `teacher_license_audit.REVIEWS`, so
`distill_gen` cannot use it as a teacher. Moving it there takes a
person's review of the last two rows.

## Memory fit (weights + vision projector 3.2 GiB + MLA latent KV + 4 GiB runtime buffer, 128k context)

Budget = 75 % of tier memory (`package.FIT_SHARE`), the rest left for
the OS and desktop. The KV cache is 82,944 bytes per token (72 layers ×
576 × fp16), 10.1 GiB at 128k, assuming the runtime caches the MLA latent.

| Quant | Weights (GiB) | L128 | L256 | L512 (384 GiB budget) |
|---|---|---|---|---|
| IQ2_XXS | 277.2 | no | no | fits (294.6) |
| Q2_K | 354.6 | no | no | fits (371.9) |
| Q4_K_M | 591.2 | no | no | no |
| Q8_0 | 1038.6 | no | no | no |
| BF16 | 1954.7 | no | no | no |

At the full 1M context the latent cache alone is about 81 GiB and Q2_K
no longer fits L512. Fitting says nothing about speed, and a 2-bit
quant's quality is unmeasured.

## Runtime support

| Runtime | State |
|---|---|
| llama.cpp | ggml-org publishes GGUFs, so upstream converts the architecture; not run here |
| vLLM / SGLang | not checked |
| reasoning-effort switch, tool-call parser, long context | not checked |

## What a real test needs

A 512 GB unified-memory machine (none available), then
`examples/device_bench.py`-style records (#175) for tokens/s, TTFT and
peak memory, plus the Traditional Chinese / code / agent benchmarks.
Vendor benchmark numbers are not reported as ours.
