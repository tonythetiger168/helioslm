"""T57 - v1.15: benchmark + report plugins."""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from harness.harness_core import Context
from harness.harness_plugins import (BenchmarkPlugin, PresetPlugin,
                                     ReportPlugin)


def test_benchmark_summary():
    ctx = Context()
    ctx.use(PresetPlugin("full", model_fn=None))
    ctx.use(BenchmarkPlugin())
    r = ctx.benchmark.alignbench()
    assert isinstance(r, dict)
    print("PASS test_benchmark_summary", r)


def test_report_gen():
    ctx = Context()
    ctx.use(PresetPlugin("full", model_fn=None))
    ctx.use(BenchmarkPlugin())
    ctx.use(ReportPlugin())
    md = ctx.report.gen()
    assert "# HeliosLM Benchmark Report" in md
    assert "Median AUROC" in md
    print("PASS test_report_gen")


if __name__ == "__main__":
    test_benchmark_summary()
    test_report_gen()
    print("\nbenchmark/report plugin tests done")
