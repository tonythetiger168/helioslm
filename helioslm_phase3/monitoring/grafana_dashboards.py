"""Grafana dashboard definitions for HeliosLM monitoring."""
import json


def generate_dashboard_json() -> dict:
    """Generate Grafana dashboard JSON."""

    dashboard = {
        "dashboard": {
            "title": "HeliosLM Inference Dashboard",
            "tags": ["helioslm", "inference"],
            "timezone": "UTC",
            "panels": [
                {
                    "id": 1,
                    "title": "Request Rate (req/s)",
                    "type": "graph",
                    "targets": [{
                        "expr": "rate(helioslm_requests_total[1m])",
                        "legendFormat": "Requests/sec",
                    }],
                    "gridPos": {"h": 8, "w": 12, "x": 0, "y": 0},
                },
                {
                    "id": 2,
                    "title": "Token Throughput (tok/s)",
                    "type": "graph",
                    "targets": [{
                        "expr": "rate(helioslm_tokens_generated_total[1m])",
                        "legendFormat": "Tokens/sec",
                    }],
                    "gridPos": {"h": 8, "w": 12, "x": 12, "y": 0},
                },
                {
                    "id": 3,
                    "title": "P50/P99 Latency (ms)",
                    "type": "graph",
                    "targets": [
                        {"expr": "histogram_quantile(0.50, rate(helioslm_request_duration_seconds_bucket[5m])) * 1000", "legendFormat": "P50"},
                        {"expr": "histogram_quantile(0.99, rate(helioslm_request_duration_seconds_bucket[5m])) * 1000", "legendFormat": "P99"},
                    ],
                    "gridPos": {"h": 8, "w": 12, "x": 0, "y": 8},
                },
                {
                    "id": 4,
                    "title": "GPU Utilization (%)",
                    "type": "graph",
                    "targets": [{
                        "expr": "helioslm_gpu_utilization * 100",
                        "legendFormat": "GPU {{gpu}}",
                    }],
                    "gridPos": {"h": 8, "w": 12, "x": 12, "y": 8},
                },
                {
                    "id": 5,
                    "title": "Queue Depth",
                    "type": "graph",
                    "targets": [{
                        "expr": "helioslm_queue_depth",
                        "legendFormat": "Queue Depth",
                    }],
                    "gridPos": {"h": 8, "w": 12, "x": 0, "y": 16},
                },
                {
                    "id": 6,
                    "title": "Error Rate (%)",
                    "type": "graph",
                    "targets": [{
                        "expr": "rate(helioslm_errors_total[1m]) / rate(helioslm_requests_total[1m]) * 100",
                        "legendFormat": "Error Rate",
                    }],
                    "gridPos": {"h": 8, "w": 12, "x": 12, "y": 16},
                },
                {
                    "id": 7,
                    "title": "Batch Size Distribution",
                    "type": "graph",
                    "targets": [{
                        "expr": "helioslm_batch_size",
                        "legendFormat": "Batch Size",
                    }],
                    "gridPos": {"h": 8, "w": 24, "x": 0, "y": 24},
                },
            ],
            "time": {"from": "now-1h", "to": "now"},
            "refresh": "5s",
        },
        "overwrite": True,
    }

    return dashboard


def save_dashboard(path: str = "monitoring/grafana_dashboard.json"):
    """Save dashboard JSON to file."""
    dashboard = generate_dashboard_json()
    with open(path, "w") as f:
        json.dump(dashboard, f, indent=2)
    print(f"💾 Dashboard saved: {path}")
