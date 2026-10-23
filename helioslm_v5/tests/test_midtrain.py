"""T126 — LB5 mid-training data for the decay phase (v5.120).

Pins:

1. render_trace: a D1 record becomes one document in the chat format,
   with the reasoning and the /think switch; think_ratio 0 gives the
   empty think block and /no_think;
2. mix_docs: every source reaches its token budget (within one document)
   in a deterministic, interleaved order; a source too small for its
   budget repeats up to max_epochs and is then reported short; bad
   inputs are refused;
3. mixture manifests accept format "trace" (+ think_ratio) and refuse an
   unknown format or a think_ratio on a text source; the committed
   lb3_0p5b_v1.json marks distill-d1 as trace;
4. build_midtrain.py end to end (needs ``tokenizers``): the licence gate
   refuses unverified sources without --allow-unverified, the shard's
   tokens follow the phase weights, traces appear in the stream, and
   meta.json carries the tokenizer hash pilot_100m.py --decay-data
   checks.

Run from repo root: python3 helioslm_v5/tests/test_midtrain.py
"""
import hashlib
import json
import subprocess
import sys
import tempfile
from pathlib import Path

import numpy as np

_REPO = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(_REPO))

from helioslm_v5.src.training.data_mixture import (  # noqa: E402
    load_mixture, validate_mixture)
from helioslm_v5.src.training.mid_train import (  # noqa: E402
    mix_docs, render_trace)

REC = {"id": "r1", "prompt_id": "p1",
       "messages": [{"role": "user", "content": "1+1?"}],
       "reasoning": "one plus one is two", "content": "2"}


def test_render_trace():
    t = render_trace(REC)
    assert t.startswith("##user##1+1? /think\n##assistant##<think>\n")
    assert "one plus one is two" in t and t.endswith("</think>\n2")
    t = render_trace(REC, think_ratio=0.0)
    assert "/no_think" in t and "one plus" not in t
    assert render_trace(REC, mode_switch=False).startswith("##user##1+1?\n")
    print("PASS test_render_trace")


def test_mix_docs():
    a = [[1] * 10 for _ in range(20)]           # 200 tokens available
    b = [[2] * 5 for _ in range(4)]             # 20 tokens available
    order, rep = mix_docs({"a": a, "b": b}, {"a": 100, "b": 50}, seed=1,
                          max_epochs=4)
    assert rep["a"] == {"tokens": 100, "docs": 10, "epochs": 1,
                        "short": False}
    # b needs 2.5 passes over its 20 tokens: 3 epochs started
    assert rep["b"]["tokens"] == 50 and rep["b"]["epochs"] == 3
    names = [n for n, _ in order]
    # interleaved, not one source after the other
    assert names.index("b") < 3 and "a" in names[-5:] and "b" in names[-5:]
    assert mix_docs({"a": a, "b": b}, {"a": 100, "b": 50}, seed=1,
                    max_epochs=4)[0] == order
    _, rep = mix_docs({"b": b}, {"b": 500}, max_epochs=2)
    assert rep["b"] == {"tokens": 40, "docs": 8, "epochs": 2, "short": True}
    for bad in (({"a": a}, {"b": 1}), ({"a": a}, {"a": 0}),
                ({"a": [[]]}, {"a": 5})):
        try:
            mix_docs(*bad)
        except ValueError:
            pass
        else:
            raise AssertionError(f"accepted {bad}")
    print("PASS test_mix_docs")


OK = {"name": "a", "source": "path:x", "license": "MIT", "weight": 1.0,
      "license_verified": True}


def test_manifest_format():
    validate_mixture({"name": "m", "sources": [
        dict(OK, format="trace", think_ratio=0.5)]})
    for patch in ({"format": "chat"}, {"think_ratio": 0.5},
                  {"format": "trace", "think_ratio": 2}):
        try:
            validate_mixture({"name": "m", "sources": [dict(OK, **patch)]})
        except ValueError:
            pass
        else:
            raise AssertionError(f"accepted {patch}")
    mix = load_mixture(_REPO / "data" / "mixtures" / "lb3_0p5b_v1.json")
    d1 = [s for s in mix["sources"] if s["name"] == "distill-d1"][0]
    assert d1["format"] == "trace" and d1["phase"] == "decay"
    print("PASS test_manifest_format")


EN = ("The river delta supports a wide range of birds, fish and plants. "
      "Researchers measured salinity at twelve stations over three years "
      "and found that seasonal floods restore the wetland soil. ") * 6


def test_build_script():
    try:
        from tokenizers import (Tokenizer, models, pre_tokenizers,
                                trainers)
    except ImportError:
        print("SKIP test_build_script (needs tokenizers)")
        return
    with tempfile.TemporaryDirectory() as d:
        d = Path(d)
        tok = Tokenizer(models.BPE(unk_token="[UNK]"))
        tok.pre_tokenizer = pre_tokenizers.ByteLevel(add_prefix_space=False)
        tok.train_from_iterator([EN, render_trace(REC)] * 5,
                                trainers.BpeTrainer(
            vocab_size=400, show_progress=False,
            initial_alphabet=pre_tokenizers.ByteLevel.alphabet(),
            special_tokens=["[UNK]", "<|endoftext|>"]))
        tok.save(str(d / "tok.json"))
        (d / "web.jsonl").write_text("".join(
            json.dumps({"text": EN.replace("twelve", w)}) + "\n"
            for w in ("ten", "nine", "eight", "seven")))
        (d / "d1.jsonl").write_text("".join(
            json.dumps(dict(REC, id=f"r{i}")) + "\n" for i in range(6)))
        mix = {"name": "mid-test", "sources": [
            dict(OK, name="main-web", phase="main"),
            dict(OK, name="web", phase="decay", weight=0.75),
            {"name": "d1", "source": "path:d1", "license": "review",
             "license_note": "teacher outputs", "phase": "decay",
             "weight": 0.25, "format": "trace", "think_ratio": 1.0}]}
        (d / "mix.json").write_text(json.dumps(mix))
        cmd = [sys.executable, str(_REPO / "examples" / "build_midtrain.py"),
               "--mixture", str(d / "mix.json"), "--source",
               f"web={d / 'web.jsonl'}", "--source", f"d1={d / 'd1.jsonl'}",
               "--tokenizer", str(d / "tok.json"), "--tokens", "2000",
               "--out", str(d / "mid")]
        r = subprocess.run(cmd, capture_output=True, text=True)
        assert r.returncode != 0 and "licence not verified" in r.stderr
        subprocess.run(cmd + ["--allow-unverified"], check=True,
                       capture_output=True)
        meta = json.loads((d / "mid" / "meta.json").read_text())
        data = np.fromfile(d / "mid" / "train.bin", dtype=np.uint16)
        sha = hashlib.sha256((d / "tok.json").read_bytes()).hexdigest()
    assert meta["tokenizer_sha256"] == sha
    s = meta["sources"]
    assert meta["unverified_licences"] == ["d1"]
    assert s["web"]["budget"] == 1500 and s["d1"]["budget"] == 500
    assert s["web"]["tokens"] >= 1500 and s["d1"]["tokens"] >= 500
    assert len(data) == meta["tokens"] == s["web"]["tokens"] + \
        s["d1"]["tokens"]
    assert s["web"]["filter"]["kept"] == 4
    think = tok.encode(" plus one is two").ids       # the reasoning
    seq = data.tolist()
    assert any(seq[i:i + len(think)] == think for i in range(len(seq)))
    print("PASS test_build_script")


if __name__ == "__main__":
    test_render_trace()
    test_mix_docs()
    test_manifest_format()
    test_build_script()
