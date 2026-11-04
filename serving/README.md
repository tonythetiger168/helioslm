# Serving

Launch settings for a serverless vLLM worker that exposes an
OpenAI-compatible server. These files do not contact a host, create an
endpoint, or upload weights. Tokens stay in the platform secret store.

## Files

- `worker.env.example` — container environment. Every assigned name is a
  worker value flag or a historical alias (`src/args_builder.py`).
  `HF_TOKEN` is named only in that file's header; leave it unset here.
- `openai_stream_smoke.py` — one streaming chat completion. It reads
  `SERVING_API_KEY` and does nothing until you pass `--base-url`.

Build the worker image with `VLLM_VERSION=v0.33.0`. The upstream image
default is `v0.23.0`, which is too old for this checkpoint. On the
endpoint, attach one 80 GB GPU and scale to zero when idle. Tensor
parallel size is 1.

## GPU

BF16 weights for this 26B-A4B checkpoint are about 52 GB. An 80 GB card
(A100 80GB, otherwise H100 80GB) loads them and still has KV room at
context 16384 and 16 sequences. A 48 GB card does not fit BF16. FP8 on
Ampere is weight-only and is not faster, so this env sets no quantization.

## API

Clients use only `POST /chat/completions` with `stream: true`. The base
URL must end with `/openai/v1`. The served model name is `helioslm`.

## MTP

`SPECULATIVE_CONFIG` pins `"method":"mtp"` and four speculative tokens.
That is this checkpoint's native multi-token head (the assistant repo in
the `model` field), which shares KV with the target. DFlash is a separate
block drafter with its own weights and non-causal attention. On Ampere
that attention has no fast backend. The figures below were measured for
MTP.

Measured on A100 80GB, BF16, vLLM 0.33.0, 512 greedy tokens. One request
went from 128.5 to 182 tok/s (1.42x). Sixteen concurrent requests went
from 583 to 882 tok/s (1.51x). MTP acceptance was 0.377 on this
checkpoint and 0.391 on the stock instruct model.

## Smoke client

From the repository root, with `SERVING_API_KEY` already set:

```bash
python serving/openai_stream_smoke.py \
  --base-url "${BASE_URL}/openai/v1" \
  --timeout 600
```

Scale-to-zero stops the worker when it is idle. The next request is a
cold start: weights load before the first token. The client's default
timeout is 30 seconds, which is too short for that load, so pass a
longer `--timeout` on the first call.
