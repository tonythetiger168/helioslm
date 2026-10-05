import json
import random
import sys
import tempfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "agent"))

from finetune_data import build_dataset, gen_episode
from envs import CalcEnv
from schema import parse_tool_call
from tools import build_default_registry


def test_finetune_data_pipeline():
    reg, impls = build_default_registry()
    rng = random.Random(8)
    for _ in range(20):
        samples = gen_episode(CalcEnv(), rng)
        assert len(samples) == 2
        for s in samples:
            assert all(ord(c) < 1024 for c in s["prompt"] + s["response"])
            parse_tool_call(s["response"], reg)
        assert '"finish"' in samples[-1]["response"]
    with tempfile.TemporaryDirectory() as tmp:
        path = str(Path(tmp) / "data.jsonl")
        build_dataset(path, n_per_env=3, seed=1)
        lines = open(path, encoding="utf-8").read().splitlines()
        assert len(lines) == 21
        for ln in lines:
            parse_tool_call(json.loads(ln)["response"], reg)


if __name__ == "__main__":
    test_finetune_data_pipeline()
    print("PASS test_finetune_data_pipeline\n\n1/1 tests passed")
