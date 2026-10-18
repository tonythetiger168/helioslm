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
