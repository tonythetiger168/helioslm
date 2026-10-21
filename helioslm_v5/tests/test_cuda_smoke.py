"""examples/cuda_smoke.py (#1) stays runnable between GPU sessions.

CI has no GPU, so this runs the script's fast checks with --device cpu:
it proves the script's own logic (APIs it calls, comparisons, JSON), not
CUDA. The GPU run is recorded separately.
"""
import json
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
FAST = ["forward_fp32", "greedy_generate", "bf16_train_step", "fp8_kv_cache",
        "quant_awq", "quant_gptq", "quant_mxfp4", "fp8_trainer"]


def test_cuda_smoke_logic_on_cpu(tmp_path):
    out = tmp_path / "smoke.json"
    r = subprocess.run([sys.executable, str(ROOT / "examples" / "cuda_smoke.py"),
                        "--device", "cpu", "--only", *FAST, "--out", str(out)],
                       capture_output=True, text=True, timeout=300, cwd=ROOT)
    assert r.returncode == 0, r.stdout[-2000:] + r.stderr[-2000:]
    res = json.loads(out.read_text())
    assert res["environment"]["device"] == "cpu"
    assert [x["check"] for x in res["results"]] == FAST
    assert all(x["status"] == "PASS" for x in res["results"])
