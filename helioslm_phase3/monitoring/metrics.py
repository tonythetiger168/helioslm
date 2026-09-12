"""Prometheus metrics collection for HeliosLM inference."""
import time
from typing import Dict, Optional, List
from dataclasses import dataclass, field
from collections import deque


@dataclass
class InferenceMetrics:
    """Metrics for a single inference request."""
    request_id: str
    prompt_tokens: int = 0
    completion_tokens: int = 0
    total_tokens: int = 0
    prefill_time_ms: float = 0.0
    decode_time_ms: float = 0.0
    total_time_ms: float = 0.0
    tokens_per_second: float = 0.0
    queue_wait_time_ms: float = 0.0
    batch_size: int = 1
    model_name: str = "helioslm-ultra"
    timestamp: float = field(default_factory=time.time)
    error: Optional[str] = None


class MetricsCollector:
    """
    Collect and expose Prometheus-compatible metrics.

    Metrics exposed:
      - helioslm_requests_total (counter)
      - helioslm_request_duration_seconds (histogram)
      - helioslm_tokens_generated_total (counter)
      - helioslm_tokens_per_second (gauge)
      - helioslm_gpu_utilization (gauge)
      - helioslm_batch_size (gauge)
      - helioslm_queue_depth (gauge)
      - helioslm_errors_total (counter)
    """

    def __init__(self, window_size: int = 10000):
        self.window_size = window_size
        self.metrics_history: deque = deque(maxlen=window_size)

        # Counters
        self.requests_total = 0
        self.tokens_generated_total = 0
        self.errors_total = 0

        # Current values
        self.current_gpu_util = 0.0
        self.current_batch_size = 0
        self.current_queue_depth = 0

        # Histogram buckets (ms)
        self.latency_buckets = [10, 50, 100, 200, 500, 1000, 2000, 5000, 10000, 30000]
        self.latency_histogram = {b: 0 for b in self.latency_buckets}

    def record_request(self, metrics: InferenceMetrics):
        """Record metrics from a completed request."""
        self.metrics_history.append(metrics)

        self.requests_total += 1
        self.tokens_generated_total += metrics.completion_tokens

        if metrics.error:
            self.errors_total += 1

        # Update latency histogram
        latency_ms = metrics.total_time_ms
        for bucket in self.latency_buckets:
            if latency_ms <= bucket:
                self.latency_histogram[bucket] += 1
                break

    def update_gauges(self, gpu_util: float, batch_size: int, queue_depth: int):
        """Update current gauge values."""
        self.current_gpu_util = gpu_util
        self.current_batch_size = batch_size
        self.current_queue_depth = queue_depth

    def get_prometheus_format(self) -> str:
        """Export metrics in Prometheus text format."""
        lines = []

        # Counters
        lines.append(f"# HELP helioslm_requests_total Total requests")
        lines.append(f"# TYPE helioslm_requests_total counter")
        lines.append(f"helioslm_requests_total {self.requests_total}")

        lines.append(f"# HELP helioslm_tokens_generated_total Total tokens generated")
        lines.append(f"# TYPE helioslm_tokens_generated_total counter")
        lines.append(f"helioslm_tokens_generated_total {self.tokens_generated_total}")

        lines.append(f"# HELP helioslm_errors_total Total errors")
        lines.append(f"# TYPE helioslm_errors_total counter")
        lines.append(f"helioslm_errors_total {self.errors_total}")

        # Gauges
        lines.append(f"# HELP helioslm_gpu_utilization Current GPU utilization")
        lines.append(f"# TYPE helioslm_gpu_utilization gauge")
        lines.append(f"helioslm_gpu_utilization {self.current_gpu_util}")

        lines.append(f"# HELP helioslm_batch_size Current batch size")
        lines.append(f"# TYPE helioslm_batch_size gauge")
        lines.append(f"helioslm_batch_size {self.current_batch_size}")

        lines.append(f"# HELP helioslm_queue_depth Current request queue depth")
        lines.append(f"# TYPE helioslm_queue_depth gauge")
        lines.append(f"helioslm_queue_depth {self.current_queue_depth}")

        # Histogram
        lines.append(f"# HELP helioslm_request_duration_seconds Request duration")
        lines.append(f"# TYPE helioslm_request_duration_seconds histogram")
        cumulative = 0
        for bucket in self.latency_buckets:
            cumulative += self.latency_histogram[bucket]
            lines.append(f'helioslm_request_duration_seconds_bucket{{le="{bucket/1000}"}} {cumulative}')
        lines.append(f'helioslm_request_duration_seconds_bucket{{le="+Inf"}} {self.requests_total}')
        lines.append(f"helioslm_request_duration_seconds_count {self.requests_total}")

        return "\n".join(lines)

    def get_stats(self) -> Dict:
        """Get summary statistics."""
        if not self.metrics_history:
            return {}

        recent = list(self.metrics_history)[-1000:]
        latencies = [m.total_time_ms for m in recent]
        throughputs = [m.tokens_per_second for m in recent]

        return {
            "total_requests": self.requests_total,
            "total_tokens": self.tokens_generated_total,
            "error_rate": self.errors_total / max(self.requests_total, 1),
            "p50_latency_ms": sorted(latencies)[len(latencies)//2] if latencies else 0,
            "p99_latency_ms": sorted(latencies)[int(len(latencies)*0.99)] if latencies else 0,
            "avg_throughput": sum(throughputs) / max(len(throughputs), 1),
            "current_gpu_util": self.current_gpu_util,
            "current_queue_depth": self.current_queue_depth,
        }
