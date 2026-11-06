# Load test and monitoring (production sizing)

`serving/loadtest.py` runs a streaming concurrency sweep (chat and ~3k-token RAG)
against any OpenAI-compatible server and prints one JSON line per level
(TTFT, inter-token and end-to-end p50/p95/p99, per-user and total tok/s,
errors). `--soak SECONDS` repeats the first level for a stability check.

```bash
python serving/loadtest.py --base-url http://127.0.0.1:8000/v1 \
  --model helioslm --scenario chat --conc 1,4,8,16,32,64
```

This folder (`serving/monitoring/`) is a docker-compose stack for one GPU server: vLLM,
nvidia_gpu_exporter, Prometheus (`alerts.yml`, 11 rules), Alertmanager
(fill a receiver first) and Grafana with the `HeliosLM serving`
dashboard. Measured GPU numbers and the verification run are in
`docs/SERVING_GPU_TEST.md`.
