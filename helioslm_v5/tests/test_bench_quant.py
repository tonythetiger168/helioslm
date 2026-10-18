"""Smoke tests for the subprocess-isolated quantization benchmark (#2)."""
import json
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]


def test_quant_benchmark_awq_smoke(tmp_path):
    output = tmp_path / "quant.json"
    cmd = [sys.executable, str(ROOT / "benchmarks" / "bench_quant.py"),
           "--methods", "awq", "--in-features", "16",
           "--out-features", "24", "--group-size", "8",
           "--batch", "1", "--warmup", "0", "--repeats", "2",
           "--out", str(output)]
    subprocess.run(cmd, cwd=ROOT, capture_output=True, text=True, check=True, timeout=120)
    data = json.loads(output.read_text(encoding="utf-8"))
    assert data["shape"]["in"] == 16
    assert len(data["results"]) == 2
    assert {r["fused"] for r in data["results"]} == {False, True}
    assert all(r["median_ms"] > 0 for r in data["results"])
    assert data["summary"]["awq"]["speedup_x"] > 0
    assert all(r["max_abs_error_vs_reference"] <= 1e-4 for r in data["results"])
