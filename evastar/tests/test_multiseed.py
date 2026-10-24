"""--seeds for the A/B scripts (#55).

Pins:

1. aggregate: equal values kept, differing numbers -> mean / sample std /
   min / max / n, dicts and equal-length lists merged recursively,
   anything else listed per seed;
2. the CLI: --seed and --seeds are mutually exclusive, seeds must be
   distinct non-negative integers, every A/B script has both flags;
3. legacy behaviour: without --seeds the payload is the old one (no
   multi-seed keys, no "seeds" in the recorded args);
4. reproducibility and pairing on two real scripts: a --seeds run's
   per-seed results equal separate --seed runs bit for bit, and the
   aggregate follows from them.
"""
import hashlib
import json
import math
import subprocess
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from evastar.eval.multiseed import add_seed_args, aggregate  # noqa: E402

AB_SCRIPTS = sorted(p.name for p in (ROOT / "examples").glob("*_ab*.py"))

# tiny settings per script for the end-to-end checks
FAST = {
    "smebu_balance_ab.py": ["--lambdas", "0.01", "--steps", "10", "--batch", "32"],
    "csa_ab_passkey.py": ["--seq-lens", "64", "--ratios", "4", "--topks", "8",
                          "--trials", "6"],
}


def test_aggregate():
    assert aggregate([3, 3, 3]) == 3
    assert aggregate(["a", "a"]) == "a"
    s = aggregate([1.0, 2.0, 4.0])
    assert s["mean"] == pytest.approx(7 / 3) and s["n"] == 3
    assert s["std"] == pytest.approx(math.sqrt(((1 - 7 / 3) ** 2 + (2 - 7 / 3) ** 2
                                                + (4 - 7 / 3) ** 2) / 2))
    assert (s["min"], s["max"]) == (1.0, 4.0)
    nested = aggregate([{"arm": "x", "acc": {"L64": 0.5}, "rows": [1, 2]},
                        {"arm": "x", "acc": {"L64": 0.7}, "rows": [1, 3]}])
    assert nested["arm"] == "x"
    assert nested["acc"]["L64"]["mean"] == pytest.approx(0.6)
    assert nested["rows"][0] == 1 and nested["rows"][1]["max"] == 3.0
    assert aggregate([True, False]) == {"per_seed": [True, False]}
    assert aggregate([[1], [1, 2]]) == {"per_seed": [[1], [1, 2]]}


def test_cli_validation():
    import argparse
    ap = add_seed_args(argparse.ArgumentParser())
    assert ap.parse_args([]).seed == 0 and ap.parse_args([]).seeds is None
    assert ap.parse_args(["--seeds", "3", "4"]).seeds == [3, 4]
    for bad in (["--seed", "1", "--seeds", "2"], ["--seeds", "-1"],
                ["--seeds", "x"], ["--seeds"]):
        with pytest.raises(SystemExit):
            ap.parse_args(bad)
    from evastar.eval.multiseed import resolve_seeds
    with pytest.raises(SystemExit):
        resolve_seeds(ap.parse_args(["--seeds", "2", "2"]))


def test_every_ab_script_has_both_flags():
    # static check for all of them (importing each costs seconds), one --help
    assert len(AB_SCRIPTS) >= 8
    for name in AB_SCRIPTS:
        src = (ROOT / "examples" / name).read_text(encoding="utf-8")
        assert "multiseed.add_seed_args(ap" in src, name
        assert "multiseed.run(args, " in src, name
        assert 'add_argument("--seed"' not in src, name
    r = subprocess.run([sys.executable, str(ROOT / "examples" / "smebu_balance_ab.py"),
                        "--help"], capture_output=True, text=True, timeout=120, cwd=ROOT)
    assert r.returncode == 0 and "--seeds" in r.stdout and "--seed " in r.stdout


def _run(name, extra, out):
    r = subprocess.run([sys.executable, str(ROOT / "examples" / name), *FAST[name],
                        *extra, "--output", str(out)],
                       capture_output=True, text=True, timeout=300, cwd=ROOT,
                       env={"OMP_NUM_THREADS": "1", "PATH": "/usr/bin:/bin"})
    assert r.returncode == 0, (name, extra, r.stderr[-1000:])
    return json.loads(out.read_text())


@pytest.mark.parametrize("name", sorted(FAST))
def test_seeds_match_single_runs(name, tmp_path):
    multi = _run(name, ["--seeds", "0", "1"], tmp_path / "multi.json")
    # legacy payload, seed 0 by default and seed 1 explicitly
    single = [_run(name, [], tmp_path / "s0.json"),
              _run(name, ["--seed", "1"], tmp_path / "s1.json")]
    for p in single:
        assert "per_seed" not in p and "aggregate" not in p
        assert "seeds" not in json.dumps({k: v for k, v in p.items() if k != "results"})
    # multi-seed payload
    assert multi["seeds"] == [0, 1]
    assert [p["seed"] for p in multi["per_seed"]] == [0, 1]
    for got, ref in zip(multi["per_seed"], single):
        assert got["results"] == ref["results"]
        if "results_sha256" in ref:
            assert got["results_sha256"] == ref["results_sha256"]
    assert multi["aggregate"] == aggregate([p["results"] for p in single])
    assert multi["per_seed_results_sha256"] == hashlib.sha256(
        json.dumps([p["results"] for p in single], sort_keys=True).encode()).hexdigest()
    assert {"python", "platform", "argv"} <= set(multi["environment"])


def test_csa_check_refuses_seeds():
    r = subprocess.run([sys.executable, str(ROOT / "examples" / "csa_ab_passkey.py"),
                        "--check", "--seeds", "0", "1"],
                       capture_output=True, text=True, timeout=120, cwd=ROOT)
    assert r.returncode != 0 and "--check" in r.stderr
