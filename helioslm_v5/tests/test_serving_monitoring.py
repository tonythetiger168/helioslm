"""Serving load test + monitoring stack (serving/loadtest.py, serving/monitoring/).

In-process only: no socket, no sleep, no network. Pins that

* the load-test summary math (percentiles, aggregate tok/s, error count),
* every vLLM metric used by alerts.yml and the Grafana dashboard exists in
  vLLM 0.33.0's /metrics (names captured from a live server on 2026-11-06),
* scrape job names, the Grafana datasource uid and the compose mounts agree.
"""

import json
import re
import sys
from pathlib import Path

_REPO = Path(__file__).resolve().parents[2]
if str(_REPO) not in sys.path:
    sys.path.insert(0, str(_REPO))

from serving import loadtest as L  # noqa: E402

MON = _REPO / "serving" / "monitoring"

# Base names (histograms without _bucket) exported by vLLM 0.33.0 and
# nvidia_gpu_exporter 1.3.2, captured from the servetest pods.
KNOWN = {
    "vllm:num_requests_running", "vllm:num_requests_waiting",
    "vllm:kv_cache_usage_perc", "vllm:num_preemptions_total",
    "vllm:prompt_tokens_total", "vllm:generation_tokens_total",
    "vllm:request_success_total", "vllm:time_to_first_token_seconds",
    "vllm:inter_token_latency_seconds", "vllm:e2e_request_latency_seconds",
    "vllm:request_queue_time_seconds", "http_requests_total",
    "nvidia_smi_utilization_gpu_ratio", "nvidia_smi_memory_used_bytes",
    "nvidia_smi_memory_total_bytes", "nvidia_smi_power_draw_watts",
    "nvidia_smi_temperature_gpu", "up",
}
_METRIC = re.compile(r"\b((?:vllm:)?[a-z_][a-z0-9_:]*)(?=\s*[\[{)]|\s*(?:==|>|<|/|$|\s+and))")
_FUNCS = {"sum", "max", "min", "rate", "histogram_quantile", "clamp_min", "by", "le", "and"}


def _metrics(expr):
    out = set()
    expr = re.sub(r"\bby\s*\([^)]*\)", "", expr)  # grouping labels are not metrics
    for name in _METRIC.findall(expr):
        if name in _FUNCS or name.startswith("$"):
            continue
        out.add(name[:-len("_bucket")] if name.endswith("_bucket") else name)
    return out


def _alert_exprs():
    return re.findall(r"^\s*expr:\s*(.+)$", (MON / "alerts.yml").read_text(), re.M)


def test_summarize_math():
    ok = {"ttft": 0.1, "e2e": 1.0, "itl": [0.01, 0.03], "tok": 10, "pin": 5, "tps": 9.0}
    s = L.summarize([ok, dict(ok, ttft=0.3, tok=30), {"err": "http 500"}], 2.0, "chat", 3)
    assert (s["n"], s["ok"], s["errors"]) == (3, 2, 1)
    assert s["agg_out_tps"] == 20.0 and s["req_per_s"] == 1.0
    assert s["ttft_s"][50] in (0.1, 0.3) and s["ttft_s"][99] == 0.3
    assert s["itl_ms"][99] == 30.0 and s["err_samples"] == ["http 500"]
    assert L.pct([], 50) is None and L.pct([4, 1, 3], 0) == 1
    assert "HX-103" in L.messages(3, "rag")[0]["content"] and len(L.DOC) > 5000


def test_alert_and_dashboard_metrics_exist():
    exprs = _alert_exprs()
    assert len(exprs) >= 10
    dash = json.loads((MON / "grafana/dashboards/helioslm-serving.json").read_text())
    exprs += [t["expr"] for p in dash["panels"] for t in p["targets"]]
    used = set().union(*(_metrics(e) for e in exprs))
    assert {"vllm:time_to_first_token_seconds", "vllm:kv_cache_usage_perc"} <= used
    assert used <= KNOWN, sorted(used - KNOWN)
    assert {p["datasource"]["uid"] for p in dash["panels"]} == {"prom"}


def test_stack_wiring():
    prom = (MON / "prometheus.yml").read_text()
    jobs = set(re.findall(r"job_name:\s*(\S+)", prom))
    assert jobs == {"vllm", "gpu"}
    for job in re.findall(r'job="(\w+)"', (MON / "alerts.yml").read_text()):
        assert job in jobs
    assert "uid: prom" in (MON / "grafana/provisioning/datasources/prometheus.yml").read_text()
    compose = (MON / "docker-compose.yml").read_text()
    for f in ("prometheus.yml", "alerts.yml", "alertmanager.yml", "grafana/dashboards"):
        assert f"./{f}" in compose
    assert "vllm:8000" in prom and "gpu-exporter:9835" in prom
    assert "environment: [HF_TOKEN]" in compose  # token passed through from the host, never written here


def test_serving_monitoring():
    test_summarize_math()
    test_alert_and_dashboard_metrics_exist()
    test_stack_wiring()
