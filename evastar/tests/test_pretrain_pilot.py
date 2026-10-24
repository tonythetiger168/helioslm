"""T122 — the 100M pilot pretraining loop (v5.83), tiny and on CPU.

Pins:

1. pilot_config builds every architecture; at the default width the dense
   arm is ~93M parameters and the routed-expert arms' active parameters
   stay within 25% of it (experts at half the dense width);
2. shards: write_shards / TokenShard round-trip, uint16 overflow is
   refused, batches depend only on (seed, micro-step), evaluation windows
   are fixed;
3. the learning-rate schedule (warmup, cosine to min_lr); WSD (LB2):
   flat until the decay phase, each decay shape ends at min_lr, the
   stable phase does not depend on the total, bad arguments are refused;
   a WSD run with a decay-phase data switch, stopped at step 3 of a
   planned 6 (schedule_total) and resumed, equals the straight run
   bitwise; its decay steps read the second shard; decay_train without
   wsd and schedule_total < steps are refused;
4. exact resume: 3 steps, stop, 3 more steps == 6 uninterrupted steps
   (bitwise weights and identical log lines), for the dense and the MoE
   arm; the MoE routers' biases move (update_bias is called); the loss
   falls on a learnable stream;
5. pilot_prepare_data.py end to end on a few text files (needs
   ``tokenizers``): shards, meta and a seeded tokenizer.

Run from repo root: python3 evastar/tests/test_pretrain_pilot.py
"""
import json
import math
import subprocess
import sys
import tempfile
from pathlib import Path

import numpy as np
import torch

_REPO = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(_REPO))

from evastar.src.training.pretrain import (  # noqa: E402
    ARCHS, EvastarV5, TokenShard, count_params, lr_at, pilot_config,
    pretrain, write_shards, wsd_decay_start)


def _tiny(arch):
    return pilot_config(arch, vocab_size=96, hidden_size=64, num_layers=6,
                        intermediate_size=128, max_seq=64)


def test_configs():
    for a in ARCHS:
        m = EvastarV5(_tiny(a)).eval()
        with torch.no_grad():
            out = m(torch.randint(0, 96, (2, 16)))[0]
        assert out.shape == (2, 16, 96) and torch.isfinite(out).all(), a
    counts = {}
    for a in ARCHS:
        with torch.device("meta"):
            counts[a] = count_params(EvastarV5(pilot_config(a)))
    assert 85e6 < counts["dense"]["total"] < 100e6
    for a in ("moe", "k3_block", "mimo_hybrid"):
        assert abs(counts[a]["active"] / counts["dense"]["active"] - 1) < 0.25, a
    try:
        pilot_config("rwkv")
    except ValueError:
        pass
    else:
        raise AssertionError("unknown arch accepted")
    print("PASS test_configs " + json.dumps(
        {a: round(c["active"] / 1e6, 1) for a, c in counts.items()}))


def test_shards():
    with tempfile.TemporaryDirectory() as d:
        n = write_shards([[1, 2, 3], [4, 5], [65535]], d, "t")
        assert n == 6
        sh = TokenShard(Path(d) / "t.bin")
        assert list(sh.data) == [1, 2, 3, 4, 5, 65535]
        try:
            write_shards([[70000]], d, "bad")
        except ValueError:
            pass
        else:
            raise AssertionError("uint16 overflow accepted")
        write_shards([list(range(500))], d, "big")
        big = TokenShard(Path(d) / "big.bin")
        a = big.batch(4, 16, 7, 0)
        b = big.batch(4, 16, 7, 0)
        c = big.batch(4, 16, 8, 0)
        assert torch.equal(a[0], b[0]) and not torch.equal(a[0], c[0])
        assert torch.equal(a[0][:, 1:], a[1][:, :-1])     # y is x shifted
        w = list(big.eval_windows(16, 3))
        assert len(w) == 3 and w[1][0][0, 0].item() == 16
    print("PASS test_shards")


def test_lr_schedule():
    assert abs(lr_at(0, 100, 1.0, 10) - 0.1) < 1e-12
    assert abs(lr_at(9, 100, 1.0, 10) - 1.0) < 1e-12
    assert abs(lr_at(10, 100, 1.0, 10) - 1.0) < 1e-12
    assert abs(lr_at(100, 100, 1.0, 10, 0.1) - 0.1) < 1e-12
    mid = lr_at(55, 100, 1.0, 10, 0.1)
    assert abs(mid - (0.1 + 0.9 * 0.5 * (1 + math.cos(math.pi * 0.5)))) < 1e-12
    print("PASS test_lr_schedule")


def test_wsd_schedule():
    # 100 steps, warmup 10, decay the last 20: steps 80..99
    assert wsd_decay_start(100, 10, 0.2) == 80
    assert wsd_decay_start(100, 90, 0.2) == 90       # never inside warmup
    for s in (10, 50, 79, 80):
        assert lr_at(s, 100, 1.0, 10, 0.1, "wsd") == 1.0, s
    assert abs(lr_at(0, 100, 1.0, 10, 0.1, "wsd") - 0.1) < 1e-12
    for shape in ("sqrt", "linear", "cosine"):
        lrs = [lr_at(s, 100, 1.0, 10, 0.1, "wsd", 0.2, shape)
               for s in range(80, 101)]
        assert all(a >= b for a, b in zip(lrs, lrs[1:])), shape
        assert abs(lrs[-1] - 0.1) < 1e-12, shape
    # 1 - sqrt(t) at t = 1/4: 0.1 + 0.9 * 0.5
    assert abs(lr_at(85, 100, 1.0, 10, 0.1, "wsd", 0.2, "sqrt") - 0.55) < 1e-12
    assert abs(lr_at(85, 100, 1.0, 10, 0.1, "wsd", 0.2, "linear")
               - (0.1 + 0.9 * 0.75)) < 1e-12
    # stable phase is the same whatever the total (continue-then-decay)
    for s in range(10, 160):
        assert lr_at(s, 200, 1.0, 10, 0.1, "wsd") == 1.0
    for bad in (dict(schedule="step"), dict(schedule="wsd", decay_frac=0.0),
                dict(schedule="wsd", decay_shape="exp")):
        try:
            lr_at(50, 100, 1.0, 10, 0.1, **bad)
        except ValueError:
            pass
        else:
            raise AssertionError(f"accepted {bad}")
    print("PASS test_wsd_schedule")


def test_wsd_decay_data_switch():
    torch.set_num_threads(1)
    with tempfile.TemporaryDirectory() as d:
        tr, va = _stream(d)
        write_shards([[(i * 7) % 48 for i in range(4000)]], d, "anneal")
        an = TokenShard(Path(d) / "anneal.bin")

        def run(out, steps):
            return pretrain(_tiny("dense"), tr, va, out, steps=steps,
                            batch=4, seq=32, grad_accum=2, lr=3e-3,
                            warmup=2, eval_every=3, eval_windows=4,
                            ckpt_every=3, seed=0, schedule="wsd",
                            decay_frac=0.5, decay_train=an,
                            schedule_total=6, log=lambda m: None)

        # 3 steps of a planned 6, then resume to 6 == 6 straight
        straight, resumed = Path(d) / "a", Path(d) / "b"
        run(straight, 6)
        run(resumed, 3)
        run(resumed, 6)
        sa = torch.load(straight / "ckpt.pt", weights_only=False)
        sb = torch.load(resumed / "ckpt.pt", weights_only=False)
        for k, v in sa["model"].items():
            assert torch.equal(v, sb["model"][k]), k
        log = [json.loads(x) for x in
               (straight / "log.jsonl").read_text().splitlines()]
        assert [r["phase"] for r in log] == ["main"] * 3 + ["decay"] * 3
        assert log[2]["lr"] == 3e-3 and log[-1]["lr"] < log[3]["lr"]
        # the decay steps really read the anneal shard: a run without the
        # switch ends with different weights
        plain = Path(d) / "c"
        pretrain(_tiny("dense"), tr, va, plain, steps=6, batch=4, seq=32,
                 grad_accum=2, lr=3e-3, warmup=2, eval_every=3,
                 eval_windows=4, ckpt_every=3, seed=0, schedule="wsd",
                 decay_frac=0.5, log=lambda m: None)
        sc = torch.load(plain / "ckpt.pt", weights_only=False)
        assert any(not torch.equal(v, sc["model"][k])
                   for k, v in sa["model"].items())
        for bad in (dict(decay_train=an), dict(schedule_total=1)):
            try:
                pretrain(_tiny("dense"), tr, va, Path(d) / "e", steps=2,
                         batch=4, seq=32, log=lambda m: None, **bad)
            except ValueError:
                pass
            else:
                raise AssertionError(f"accepted {bad}")
    print("PASS test_wsd_decay_data_switch")


def _stream(d):
    # a learnable stream: repeating 0..47 counting pattern
    write_shards([[i % 48 for i in range(4000)]], d, "train")
    write_shards([[i % 48 for i in range(600)]], d, "val")
    return TokenShard(Path(d) / "train.bin"), TokenShard(Path(d) / "val.bin")


def _run(arch, out, steps, d):
    tr, va = _stream(d)
    return pretrain(_tiny(arch), tr, va, out, steps=steps, batch=4, seq=32,
                    grad_accum=2, lr=3e-3, warmup=2, eval_every=3,
                    eval_windows=4, ckpt_every=3, seed=0, log=lambda m: None)


def test_exact_resume_and_learning():
    torch.set_num_threads(1)
    for arch in ("dense", "moe"):
        with tempfile.TemporaryDirectory() as d:
            straight = Path(d) / "a"
            resumed = Path(d) / "b"
            _run(arch, straight, 6, d)
            _run(arch, resumed, 3, d)
            _run(arch, resumed, 6, d)
            sa = torch.load(straight / "ckpt.pt", weights_only=False)
            sb = torch.load(resumed / "ckpt.pt", weights_only=False)
            assert sa["step"] == sb["step"] == 6
            for k, v in sa["model"].items():
                assert torch.equal(v, sb["model"][k]), (arch, k)
            la = (straight / "log.jsonl").read_text().splitlines()
            lb = (resumed / "log.jsonl").read_text().splitlines()
            timing = ("tokens_per_s", "step_tokens_per_s", "peak_mem_gb")
            strip = [{k: v for k, v in json.loads(x).items() if k not in timing}
                     for x in la]
            assert strip == [
                {k: v for k, v in json.loads(x).items() if k not in timing}
                for x in lb]
            if arch == "moe":
                biases = [v for k, v in sa["model"].items()
                          if k.endswith("route_bias")]
                assert biases and any(b.abs().sum() > 0 for b in biases)
    with tempfile.TemporaryDirectory() as d:
        s = _run("dense", Path(d) / "c", 30, d)
        log = [json.loads(x) for x in
               (Path(d) / "c" / "log.jsonl").read_text().splitlines()]
        evals = [x["val_loss"] for x in log if "val_loss" in x]
        assert evals[-1] < evals[0] * 0.7, evals
        assert s["steps"] == 30 and s["tokens"] == 30 * 4 * 32 * 2
    print("PASS test_exact_resume_and_learning")


def test_prepare_data_script():
    try:
        import tokenizers  # noqa: F401
    except ImportError:
        print("SKIP test_prepare_data_script (no tokenizers)")
        return
    with tempfile.TemporaryDirectory() as d:
        src = Path(d) / "src"
        src.mkdir()
        for i in range(12):
            (src / f"{i}.txt").write_text(
                f"文件 {i}：繁體中文與简体中文的訓練資料。The quick brown fox {i}. "
                * 20, encoding="utf-8")
        with open(src / "more.jsonl", "w", encoding="utf-8") as f:
            for i in range(30):
                f.write(json.dumps({"text": f"資料 {i} data line"}) + "\n")
        out = Path(d) / "out"
        r = subprocess.run(
            [sys.executable, str(_REPO / "examples" / "pilot_prepare_data.py"),
             "--text", str(src), "--vocab", "9000", "--zh-seed", "both",
             "--val-fraction", "0.1", "--out", str(out)],
            capture_output=True, text=True, env={"RAYON_NUM_THREADS": "1",
                                                 "PATH": "/usr/bin:/bin"})
        assert r.returncode == 0, r.stderr[-2000:]
        meta = json.loads((out / "meta.json").read_text())
        sp = meta["splits"]
        assert sp["train"]["docs"] + sp["val"]["docs"] == 42
        assert sp["val"]["docs"] >= 1
        assert len(TokenShard(out / "train.bin")) == sp["train"]["tokens"]
        assert meta["val_bytes_per_token"] > 0
        from tokenizers import Tokenizer
        tok = Tokenizer.from_file(str(out / "tokenizer.json"))
        # characters absent from the corpus are single tokens only because
        # of the seed: Traditional-only, Simplified-only, shared
        for ch in ("鬱", "龜", "龟", "乌", "山"):
            assert len(tok.encode(ch).ids) == 1, ch
    print("PASS test_prepare_data_script")


if __name__ == "__main__":
    test_configs()
    test_shards()
    test_lr_schedule()
    test_wsd_schedule()
    test_wsd_decay_data_switch()
    test_exact_resume_and_learning()
    test_prepare_data_script()
