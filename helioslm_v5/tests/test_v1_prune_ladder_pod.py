"""examples/v1_prune_ladder_pod.sh (v1.0 plan, stages 0 and 1).

The script runs on a GPU pod, so it is checked statically here. Pins:
1. bash parses it;
2. every flag it passes to prune_kd_gemma4.py prune, fetch_mixture.py and
   base_gap_eval.py exists in that tool;
3. each LADDER row fits google/gemma-4-E2B (15 non-shared + 20 KV-shared
   layers, hidden 1536, FFN 6144) and keeps at least one non-shared layer
   per attention type; the rows are distinct and include the pilot's size;
4. the mixture it reads parses and each phase's weights sum to 1;
5. the pod is removed (or stopped) on every exit, results are uploaded
   first, and no token is written into the script.
"""
import json
import re
import shutil
import subprocess
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
SH = ROOT / "examples" / "v1_prune_ladder_pod.sh"
TEXT = SH.read_text(encoding="utf-8")

E2B = {"kv": 15, "shared": 20, "hidden": 1536, "ffn": 6144, "layer_types": 2}


def _flags(path, after=None, before=None):
    src = (ROOT / path).read_text(encoding="utf-8")
    if after:
        src = src[src.index(after):]
    if before:
        src = src[:src.index(before)]
    return set(re.findall(r'add_argument\(\s*"(--[a-z0-9-]+)"', src))


def _used(cmd):
    """Flags after `python <cmd>` up to the end of that (continued) command."""
    m = re.search(r"python " + re.escape(cmd) + r"(.*?)(?<!\\)\n", TEXT, re.S)
    assert m, cmd
    return set(re.findall(r"\s(--[a-z0-9-]+)", m.group(1)))


def _ladder():
    block = re.search(r'LADDER="(.*?)"', TEXT, re.S).group(1)
    return [line.split() for line in block.strip().splitlines()]


def test_v1_prune_ladder_pod():
    bash = shutil.which("bash")
    if bash:
        r = subprocess.run([bash, "-n", str(SH)], capture_output=True, text=True)
        assert r.returncode == 0, r.stderr

    prune = _flags("examples/prune_kd_gemma4.py", before='k.add_argument("--student"')
    assert _used("examples/prune_kd_gemma4.py prune") <= prune
    assert _used("examples/fetch_mixture.py") <= _flags("examples/fetch_mixture.py")
    evals = _flags("examples/base_gap_eval.py")
    runs = list(re.finditer(r"python examples/base_gap_eval\.py(.*?)(?<!\\)\n", TEXT, re.S))
    assert len(runs) == 2
    for m in runs:
        used = set(re.findall(r"\s(--[a-z0-9-]+)", m.group(1)))
        assert used and used <= evals, used - evals

    rows = _ladder()
    assert len(rows) >= 3 and len({tuple(r[1:]) for r in rows}) == len(rows)
    for name, kv, sh, hidden, ffn in rows:
        kv, sh, hidden, ffn = int(kv), int(sh), int(hidden), int(ffn)
        assert E2B["layer_types"] <= kv <= E2B["kv"], name
        assert 0 <= sh <= E2B["shared"], name
        assert 0 < hidden <= E2B["hidden"] and 0 < ffn <= E2B["ffn"], name
    assert ["8", "8", "1024", "4096"] in [r[1:] for r in rows]   # the pilot

    mix = re.search(r"--mixture (\S+)", TEXT).group(1)
    m = json.loads((ROOT / mix).read_text(encoding="utf-8"))
    for phase in {s["phase"] for s in m["sources"]}:
        w = sum(s["weight"] for s in m["sources"] if s["phase"] == phase)
        assert abs(w - 1.0) < 1e-9, (phase, w)

    assert "trap finish EXIT" in TEXT
    fin = TEXT[TEXT.index("finish() {"):TEXT.index("trap finish EXIT")]
    assert fin.index("push") < fin.index("runpodctl remove pod")
    assert "runpodctl stop pod" in fin
    assert not re.search(r"hf_[A-Za-z0-9]{20,}|rpa_[A-Za-z0-9]{10,}", TEXT)
    print("PASS test_v1_prune_ladder_pod")


if __name__ == "__main__":
    test_v1_prune_ladder_pod()
