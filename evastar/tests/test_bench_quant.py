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


def test_quant_benchmark_mxfp4_smoke(tmp_path):
    output = tmp_path / "quant.json"
    cmd = [sys.executable, str(ROOT / "benchmarks" / "bench_quant.py"),
           "--methods", "mxfp4", "--in-features", "64", "--out-features", "24",
           "--batch", "1", "--warmup", "0", "--repeats", "2", "--out", str(output)]
    subprocess.run(cmd, cwd=ROOT, capture_output=True, text=True, check=True, timeout=120)
    data = json.loads(output.read_text(encoding="utf-8"))
    assert [r["method"] for r in data["results"]] == ["mxfp4", "mxfp4"]
    assert all(r["max_abs_error_vs_reference"] <= 1e-4 for r in data["results"])


def test_committed_measurements():
    """The issue #2 runs (4096 x 14336, group 128, batch 1 and 16): each
    summary row follows from its samples, fused matches unfused on the same
    quantized weights, and fused is faster for every method at both batch
    sizes."""
    import statistics
    for batch in (1, 16):
        art = json.loads((ROOT / "benchmarks" / f"quant_fused_b{batch}_2026-10-21.json")
                         .read_text(encoding="utf-8"))
        assert art["shape"] == {"in": 4096, "out": 14336, "batch": batch, "group_size": 128}
        assert set(art["summary"]) == {"awq", "gptq", "mxfp4"}
        for r in art["results"]:
            assert len(r["samples_ms"]) == art["repeats"]
            assert abs(statistics.median(r["samples_ms"]) - r["median_ms"]) < 1e-9
            assert r["max_abs_error_vs_reference"] <= 1e-4
        for method, s in art["summary"].items():
            assert abs(s["speedup_x"] - s["unfused_median_ms"] / s["fused_median_ms"]) < 1e-9
            assert s["speedup_x"] > 1.5, (batch, method)
