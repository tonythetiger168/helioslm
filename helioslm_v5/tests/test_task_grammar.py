"""T34 - v5.35: the task grammar oracles.

One table serves render (envs) and parse (grounding/policies): if they
ever disagree, this suite fails. Covers roundtrip identity per kind,
cross-kind contamination (a write_read text must NOT parse as calc),
and agreement with the live env generators (the real consumers).
Run from repo root: python3 helioslm_v5/tests/test_task_grammar.py
"""
import random
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "agent"))

import task_grammar as G
from envs import make_envs, make_long_envs

SPECS = [
    ("calc", {"expr": "-87 % -37 % 4 * -70 + -24 // 3"}),
    ("str", {"op": "upper", "n": 0, "s": "hello world"}),
    ("str", {"op": "repeat", "n": 3, "s": "ab"}),
    ("compose", {"expr": "3 * 4 + 5", "op": "reverse"}),
    ("write_read", {"expr": "21 * 17 + 10", "path": "f.txt"}),
    ("write_transform", {"s": "joker harbor", "a": "a.txt", "op": "repeat",
                         "n": 2, "b": "b.txt"}),
    ("write_transform", {"s": "kiwi", "a": "a.txt", "op": "reverse",
                         "n": 0, "b": "b.txt"}),
    ("accumulate", {"e1": "-5 * 5", "p1": "a.txt", "e2": "-10 + 36",
                    "p2": "b.txt"}),
    ("echo", {"s": "x9Qz! mnp"}),
]


def test_roundtrip_identity():
    for kind, spec in SPECS:
        text = G.render(kind, **spec)
        back = G.parse(text)
        assert back is not None, f"{kind}: parse(render) failed: {text!r}"
        assert back["kind"] == kind, (back["kind"], kind)
        for k, v in spec.items():
            assert back.get(k) == v, f"{kind}.{k}: {back.get(k)!r} != {v!r}"
    print(f"PASS test_roundtrip_identity ({len(SPECS)} specs)")


def test_no_cross_contamination():
    wr = G.render("write_read", expr="21 * 17 + 10", path="f.txt")
    assert G.parse(wr)["kind"] == "write_read", "write_read leaked to calc"
    acc = G.render("accumulate", e1="1+1", p1="a.txt", e2="2+2", p2="b.txt")
    assert G.parse(acc)["kind"] == "accumulate"
    wt = G.render("write_transform", s="x", a="a.txt", op="upper", n=0,
                  b="b.txt")
    assert G.parse(wt)["kind"] == "write_transform"
    # bare prefix and empty expr must not parse as calc
    assert G.parse("Compute the value of: ") is None
    assert G.parse("Compute the value of: and write the rest") is None
    print("PASS test_no_cross_contamination")


def test_agrees_with_env_generators():
    """The envs' actual sampled task texts must all parse with the
    correct kind -- this is the live-consumer contract."""
    rng = random.Random(4242)
    counts = {}
    for env in make_envs() + make_long_envs():
        for _ in range(40):
            task = env.sample(rng)
            spec = G.parse(task.text)
            assert spec is not None, f"env text unparsed: {task.text!r}"
            counts[spec["kind"]] = counts.get(spec["kind"], 0) + 1
            if spec["kind"] == "calc":
                assert spec["expr"] in task.text
    assert counts.get("calc") and counts.get("str")
    print(f"PASS test_agrees_with_env_generators {counts}")


def test_echo_kind():
    text = G.render("echo", s="42 * 13")
    # echo content must not be treated as an expression anywhere
    spec = G.parse(text)
    assert spec == {"kind": "echo", "s": "42 * 13"}
    print("PASS test_echo_kind")


if __name__ == "__main__":
    test_roundtrip_identity()
    test_no_cross_contamination()
    test_agrees_with_env_generators()
    test_echo_kind()
    print("\n4/4 task-grammar tests passed")
