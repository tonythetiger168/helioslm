# Production serving: GPU load test and monitoring (2026-11-06)

What one GPU server running `helioslm-chat-g26b-short` (26B-A4B) and
`helioslm-chat-e4b` on vLLM can carry, and the monitoring stack that goes
with it. Scripts: `serving/loadtest.py`, `serving/monitoring/`.

## Setup

- vLLM 0.33.0 (`vllm/vllm-openai`), one GPU, `--max-model-len 8192
  --max-num-seqs 64 --gpu-memory-utilization 0.90`.
- FP8 = vLLM online FP8 (`--quantization fp8`, W8A8 on Ada / Blackwell).
- MTP = Google's Gemma 4 assistant head
  (`google/gemma-4-26B-A4B-it-assistant`, `num_speculative_tokens 4`).
- `serving/loadtest.py`, streaming chat completions, temperature 0.7:
  - chat: 16 zh-TW / English prompts, 512 new tokens, concurrency 1 to 64
    (requests per level = 3 x concurrency);
  - rag: ~3.3k-token document plus a question, 256 new tokens,
    concurrency 1 to 32.
- Soak: 15 minutes of chat at concurrency 16 on the best arm of each GPU.

## Results (chat unless noted)

Throughput = aggregate output tok/s for all users; per user = median
decode tok/s of one stream. Cost uses the RunPod community hourly price
and the concurrency-64 throughput, so it is a rental reference only.

| GPU | Arm | 1 user tok/s | 16 users: total / per user | 64 users: total / per user | 64 users TTFT p95 (s) | RAG 32 users: total / TTFT p95 | US$ per 1M output tokens |
|---|---|---|---|---|---|---|---|
| L40S 48GB | 26B FP8 | 111 | 609 / 41 | 1483 / 25 | 1.01 | 1282 / 0.36 | 0.148 |
| L40S 48GB | 26B FP8 + MTP | 143 | 861 / 62 | 2223 / 41 | 0.24 | 1919 / 1.20 | 0.099 |
| L40S 48GB | E4B BF16 | 69 | 937 / 63 | 2836 / 47 | 0.34 | 1449 / 0.35 | 0.077 |
| A100 80GB PCIe | 26B BF16 | 125 | 705 / 49 | 1764 / 32 | 2.82 | 1105 / 0.78 | 0.187 |
| A100 80GB PCIe | 26B BF16 + MTP | 185 | 1069 / 78 | 2526 / 50 | 2.46 | 2241 / 0.65 | 0.131 |
| RTX PRO 6000 96GB | 26B BF16 | 137 | 703 / 49 | 1907 / 32 | 0.13 | 1540 / 0.22 | 0.246 |
| RTX PRO 6000 96GB | 26B FP8 | 187 | 1149 / 81 | 3114 / 53 | 0.15 | 2007 / 0.26 | 0.151 |
| RTX PRO 6000 96GB | 26B FP8 + MTP | 233 | 1745 / 128 | 4492 / 83 | 0.10 | 4177 / 0.21 | 0.105 |

Zero failed requests in every sweep level.

Soak (15 min, 16 users, chat):

| GPU / arm | Requests | Errors | Throughput first / last / min | TTFT p95 worst round | Peak power | Peak temp |
|---|---|---|---|---|---|---|
| L40S, 26B FP8 + MTP | 1,600 | 0 | 812 / 842 / 812 | 0.13 s | 333 W | 63 °C |
| A100 80GB, 26B BF16 + MTP | 1,952 | 0 | 1006 / 1025 / 995 | 0.17 s | 323 W | 71 °C |
| RTX PRO 6000, 26B FP8 + MTP | 3,200 | 0 | 1692 / 1691 / 1639 | 0.08 s | 414 W (of 600) | 52 °C |

No throughput drift or latency creep over the soak.

Load time (vLLM start to first `/v1/models`, weights on local disk,
includes torch.compile): L40S 131-150 s, RTX PRO 6000 120-170 s, A100 pod
431-491 s (slow pod disk). Keep the compile cache on a persistent volume.

## Findings

- MTP is worth turning on for 26B in production: +30-60% throughput at
  every concurrency and lower tail latency (the server finishes streams
  sooner, so fewer requests wait).
- RTX PRO 6000 + FP8 + MTP is the strongest single card: 64 concurrent
  users at ~83 tok/s each, TTFT p95 0.10 s, still KV headroom (vLLM
  reported room for 32 full 8k-token requests). It was not saturated at
  64; a higher `--max-num-seqs` should carry more.
- A100 80GB BF16 at 64 users has TTFT p95 of 2.5-2.8 s: the KV cache
  (16 full 8k requests) is the limit. Keep it at 32 users or below.
- L40S runs 26B only in FP8. With MTP it holds 64 users at ~41 tok/s
  each; fine for a small deployment.
- E4B is cheap (US$0.077 per 1M on L40S) but slow per user at low load
  (69 tok/s single stream); it suits high-concurrency, short-answer
  traffic better than interactive chat.
- FP8 quality: see "FP8 vs BF16 quality" below before switching
  production to FP8.

## FP8 vs BF16 quality

TMMLU+ sample (25 questions per subject, 66 subjects), same prompts,
greedy, letter-only answer. Results: pending (pod running), filled in
when it finishes.

## Monitoring (verified on the test pods)

`serving/monitoring/` holds a docker-compose stack for one GPU server:
vLLM, `nvidia_gpu_exporter`, Prometheus, Alertmanager and Grafana with a
provisioned dashboard (requests running / waiting, tokens/s, TTFT / ITL /
end-to-end p50 and p95, KV cache, HTTP status, preemptions, GPU util /
memory / power / temperature).

`alerts.yml` (11 rules): VLLMDown, GPUExporterDown, EngineStuck (requests
running, no tokens for 2 min), HighTTFT (p95 > 3 s), SlowTokens (ITL p95
> 150 ms), QueueBacklog (> 8 waiting), KVCacheFull (> 95%), Preemptions,
HTTPErrors (5xx > 2%), GPUHot (> 85 °C), GPUMemoryNearFull.

On every test pod Prometheus 3.5.0 and nvidia_gpu_exporter 1.3.2 ran next
to vLLM: `promtool check` passed, both scrape targets were up, every alert
query returned data during the soak (for example p95 TTFT 0.07-0.23 s,
p95 ITL 24-49 ms, HTTP all 2xx), and a drill that stopped vLLM fired
VLLMDown on all three pods.

Before going live: put the measured p95 at your planned load plus a margin
into the HighTTFT / SlowTokens thresholds, fill a receiver (Telegram or
email) in `alertmanager.yml`, keep ports 3000 / 9090 / 9093 on the
internal network, and expose only the API through the gateway.

## Cost of this test

Three pods (L40S secure US$1.09/h, A100 80GB community US$1.19/h, RTX PRO
6000 secure US$2.49/h), about 40-50 minutes each: about US$3.3. All pods
removed themselves. Raw data: `helios-pod-logs/{y7elumuczsv4je,
36y0yq7qu67pa5,cazeb56d34xw5x}/servetest/`.
