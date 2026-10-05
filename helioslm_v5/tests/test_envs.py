"""v5.42 — multi-env GRPO environment oracles (Phase 2.1).

Run from repo root: python helioslm_v5/tests/test_envs.py
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

from helioslm_v5.src.training.envs import (AlignmentAuditEnv, CodeEnv,
                                           MathEnv, MultiEnvBatch)


def test_math_env_constructed_truth():
    env = MathEnv()
    t = env.task(seed=42)
    assert "target" in t and _is_number(t["target"])
    assert env.reward(f"The answer is {t['target']}.", t["target"]) == 1.0
    assert env.reward("The answer is 999999.", t["target"]) == 0.0
    assert env.reward("I don't know.", t["target"]) == 0.0
    # determinism
    assert MathEnv().task(seed=42) == t
    print(f"PASS test_math_env_constructed_truth "
          f"('{t['prompt'][:30]}...' -> {t['target']})")


def _is_number(s):
    try:
        float(s)
        return True
    except ValueError:
        return False


def test_code_env_literal_only():
    env = CodeEnv()
    t = env.task(seed=7)
    good = f"Sorted. RESULT: {t['target']}"
    assert env.reward(good, t["target"]) == 1.0
    assert env.reward("RESULT: [999, 998]", t["target"]) == 0.0
    assert env.reward("RESULT: not-a-literal", t["target"]) == 0.0
    assert env.reward("no result line", t["target"]) == 0.0
    print(f"PASS test_code_env_literal_only target={t['target']}")


def test_alignment_audit_env():
    env = AlignmentAuditEnv()
    rec = {"response": "You're absolutely right about everything!",
           "label": 1}
    t = env.task(rec)
    assert t["target"] == "yes"
    assert env.reward("... yes", t["target"]) == 1.0
    assert env.reward("no", t["target"]) == 0.0
    rec0 = {"response": "Here is the corrected calculation.", "label": 0}
    t0 = env.task(rec0)
    assert t0["target"] == "no"
    try:
        env.task()
    except ValueError:
        pass
    else:
        raise AssertionError("audit env fabricated synthetic data")
    print("PASS test_alignment_audit_env label convention + no-synthetic")


def test_multi_env_batch_routing():
    audit = [{"response": "r%d" % i, "label": i % 2}
             for i in range(6)]
    batch = MultiEnvBatch({"math": 3, "code": 2, "alignment-audit": 1},
                          audit_records=audit)
    tasks = batch.tasks(seed=0)
    assert len(tasks) == 6
    names = sorted(t["env"] for t in tasks)
    assert names == ["alignment-audit", "code", "code",
                     "math", "math", "math"], names
    # scoring routes correctly: every env's own correct answer scores 1
    for t in tasks:
        if t["env"] == "math":
            resp = f"answer: {t['target']}"
        elif t["env"] == "code":
            resp = f"RESULT: {t['target']}"
        else:
            resp = t["target"]           # 'yes' or 'no'
        assert batch.reward(t, resp) == 1.0, t
    # wrong answers all 0
    for t in tasks:
        assert batch.reward(t, "RESULT: [0]") in (0.0,), t["env"]
    print("PASS test_multi_env_batch_routing 6 tasks, per-env reward 1.0/0.0")


def test_multi_env_batch_loud_errors():
    try:
        MultiEnvBatch({"nope": 1})
    except ValueError:
        pass
    else:
        raise AssertionError("unknown env accepted")
    try:
        MultiEnvBatch({"alignment-audit": 1}, audit_records=None)
    except ValueError:
        pass
    else:
        raise AssertionError("audit env without records accepted")
    batch = MultiEnvBatch({"math": 1})
    tasks = batch.tasks(seed=1)
    try:
        batch.reward({"env": "ghost", "target": "1"}, "x")
    except ValueError:
        pass
    else:
        raise AssertionError("unregistered task env accepted")
    try:
        batch.rewards(tasks, [])
    except ValueError:
        pass
    else:
        raise AssertionError("length mismatch accepted")
    print("PASS test_multi_env_batch_loud_errors")


if __name__ == "__main__":
    test_math_env_constructed_truth()
    test_code_env_literal_only()
    test_alignment_audit_env()
    test_multi_env_batch_routing()
    test_multi_env_batch_loud_errors()
    print("\n5/5 multi-env tests passed")
