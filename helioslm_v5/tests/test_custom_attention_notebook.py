"""Offline oracle for examples/notebooks/custom_attention.ipynb (Issue #3)."""
import json
from pathlib import Path

NOTEBOOK = Path(__file__).resolve().parents[2] / "examples" / "notebooks" / "custom_attention.ipynb"


def test_notebook_has_executable_oracles():
    notebook = json.loads(NOTEBOOK.read_text(encoding="utf-8"))
    assert notebook["nbformat"] == 4
    source = "\n".join("".join(c["source"]) for c in notebook["cells"]
                       if c["cell_type"] == "code")
    compile(source, str(NOTEBOOK), "exec")
    assert "torch.testing.assert_close" in source
    assert "use_cache=True" in source
    assert "custom_attention_cpu_results.json" in source


def test_notebook_runs_top_to_bottom_cpu(tmp_path, monkeypatch):
    """Execute every code cell as a single notebook kernel would."""
    monkeypatch.chdir(NOTEBOOK.parents[2])
    # write the result file into the test's tmp dir, never benchmarks/ (a
    # stray unregistered artifact there fails test_artifact_tiers when the
    # suite runs in parallel)
    monkeypatch.setenv("HELIOS_NOTEBOOK_OUT", str(tmp_path))
    notebook = json.loads(NOTEBOOK.read_text(encoding="utf-8"))
    namespace = {"__name__": "__main__"}
    for index, cell in enumerate(notebook["cells"]):
        if cell["cell_type"] == "code":
            source = "".join(cell["source"])
            exec(compile(source, f"{NOTEBOOK}:cell-{index}", "exec"), namespace)
    metrics = namespace["result"]
    written = json.loads((tmp_path / "custom_attention_cpu_results.json")
                         .read_text(encoding="utf-8"))
    assert written == metrics
    assert metrics["mla_median_ms"] > 0
    assert metrics["gain_mla_median_ms"] > 0
    assert metrics["ratio_variant_to_mla"] > 0
