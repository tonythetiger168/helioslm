"""T130 — route B: continued pretraining of an open base (v5.127).

Pins:

1. every candidate base is Apache-2.0 and names an hf_id, size,
   vocabulary and llama.cpp architecture;
2. plan_cost: 6ND / (peak * MFU) for training, 2ND for forward passes,
   fixed GPU-hours as given, per-stage gpu and params overrides, and
   the contingency applies to the total only;
3. validation refuses unknown bases / GPUs / kinds, bad MFU, tokens or
   hours, duplicate stage names and a bad contingency;
4. the committed plans (data/cpt/*.json) load, name a committed mixture,
   and the B1 release run stays within the US$5-25k the owner was quoted
   (the pilot well under US$1k);
5. the B1 mixture is mostly Taiwanese and keeps general-data replay;
6. the B1.0 pre-registration gates every MediaTek rival on every task
   and the base itself on the Taiwanese tasks;
7. uint32 shards: a 248K-vocabulary stream round-trips through
   write_shards / TokenShard (torch only).

Run from repo root: python3 -m pytest helioslm_v5/tests/test_cpt_plan.py
"""
import json
import sys
import tempfile
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(ROOT))

from helioslm_v5.eval.prereg import pending, validate  # noqa: E402
from helioslm_v5.src.training.cpt_plan import (  # noqa: E402
    BASES, GPUS, load_plan, plan_cost, validate_plan)
from helioslm_v5.src.training.data_mixture import (  # noqa: E402
    load_mixture, token_budget)


def _plan(**kw):
    p = {"name": "t", "base": "gemma4-e4b", "gpu": "h100", "mfu": 0.5,
         "stages": [{"name": "a", "kind": "train", "tokens": 1e9}]}
    p.update(kw)
    return p


def test_bases():
    for k, b in BASES.items():
        assert b["license"] == "Apache-2.0", k
        assert b["hf_id"] and b["llama_cpp_arch"]
        assert b["params"] > 1e9 and b["vocab"] > 65536


def test_cost_formula():
    g = GPUS["h100"]
    c = plan_cost(_plan())
    want = 6 * 4.5e9 * 1e9 / (g["peak_flops"] * 0.5 * 3600)
    assert c["rows"][0]["gpu_hours"] == pytest.approx(want)
    assert c["usd"] == pytest.approx(want * g["usd_per_hour"])
    p = _plan(contingency=0.5, stages=[
        {"name": "f", "kind": "forward", "tokens": 1e9, "params": 8e9,
         "gpu": "h200"},
        {"name": "x", "kind": "fixed", "gpu_hours": 10}])
    c = plan_cost(p)
    fh = 2 * 8e9 * 1e9 / (GPUS["h200"]["peak_flops"] * 0.5 * 3600)
    assert c["rows"][0]["gpu_hours"] == pytest.approx(fh)
    assert c["rows"][1]["gpu_hours"] == 10
    usd = fh * GPUS["h200"]["usd_per_hour"] + 10 * g["usd_per_hour"]
    assert c["usd"] == pytest.approx(usd)
    assert c["usd_with_contingency"] == pytest.approx(1.5 * usd)


@pytest.mark.parametrize("kw", [
    dict(base="llama-3.2-3b"), dict(gpu="tpu"),
    dict(mfu=0.0),
    dict(mfu=1.5), dict(stages=[]), dict(contingency=-0.1),
    dict(stages=[{"name": "a", "kind": "sing", "tokens": 1}]),
    dict(stages=[{"name": "a", "kind": "train", "tokens": 0}]),
    dict(stages=[{"name": "a", "kind": "train", "tokens": True}]),
    dict(stages=[{"name": "a", "kind": "fixed"}]),
    dict(stages=[{"name": "a", "kind": "train", "tokens": 1, "gpu": "x"}]),
    dict(stages=[{"name": "a", "kind": "train", "tokens": 1, "params": 0}]),
    dict(stages=[{"name": "a", "kind": "train", "tokens": 1},
                 {"name": "a", "kind": "train", "tokens": 1}]),
])
def test_validation(kw):
    with pytest.raises(ValueError):
        validate_plan(_plan(**kw))


@pytest.mark.parametrize("name,lo,hi", [
    ("b1_gemma4_e4b", 5_000, 25_000),
    ("b0_gemma4_e2b_pilot", 0, 1_000),
])
def test_committed_plans(name, lo, hi):
    plan = load_plan(ROOT / "data" / "cpt" / f"{name}.json")
    load_mixture(ROOT / plan["mixture"])
    c = plan_cost(plan)
    assert lo <= c["usd_with_contingency"] <= hi, c
    if plan.get("prereg"):
        validate(json.loads((ROOT / plan["prereg"]).read_text("utf-8")))


def test_b1_mixture():
    mix = load_mixture(ROOT / "data" / "mixtures" / "b1_gemma4_e4b_cpt_v1.json")
    main = [s for s in mix["sources"] if s.get("phase", "main") == "main"]
    tw = sum(s["weight"] for s in main if "tw" in (s.get("zh_variants") or []))
    assert 0.5 <= tw <= 0.7, tw                         # Taiwanese first
    assert 1.0 - tw >= 0.3                              # replay kept
    b = token_budget(mix, 40e9, decay_frac=0.2)
    assert sum(b.values()) == pytest.approx(40e9, rel=1e-6)
    assert not any("tmmlu" in s["source"].lower() for s in mix["sources"])


def test_b1_prereg():
    plan = json.loads((ROOT / "docs" / "prereg" / "b1.0.json")
                      .read_text("utf-8"))
    validate(plan)
    tasks = {t["name"] for t in plan["tasks"]}
    rivals = [b["name"] for b in plan["baselines"]
              if b["name"].startswith("MediaTek-Research/")]
    assert len(rivals) == 4
    gates = {(g["task"], g["vs"]): g["margin"] for g in plan["gates"]}
    for r in rivals:
        for t in tasks:
            assert gates[(t, r)] == 0.0
    base = BASES["gemma4-e4b"]["hf_id"]
    assert gates[("tmmluplus", base)] == 0.0
    assert gates[("tw_justice", base)] == 0.0
    assert gates[("ceval", base)] < 0 and gates[("cmmlu", base)] < 0
    assert base in pending(plan)


def test_uint32_shards():
    pytest.importorskip("torch")
    from helioslm_v5.src.training.pretrain import (
        TokenShard, token_dtype, write_shards)
    assert token_dtype(65536) == "uint16"
    assert token_dtype(248320) == "uint32"
    with tempfile.TemporaryDirectory() as d:
        ids = [248000, 1, 70000, 248319] * 20
        with pytest.raises(ValueError):
            write_shards([ids], d, "narrow")
        assert write_shards([ids], d, "wide", dtype="uint32") == len(ids)
        sh = TokenShard(Path(d) / "wide.bin", dtype="uint32")
        assert len(sh) == len(ids)
        x, y = sh.batch(2, 8, 0, 0)
        assert int(x.max()) > 65535 and x.shape == (2, 8)
