"""T29 - v5.32 P2: RLCD outcome loop oracles.

The closing of the loop T28 opened: Brier calibration bites when its
target is a replay-verified OUTCOME. Here an overconfident confidence_fn
(0.95 on everything) drives a ThresholdGate; broken tasks end wrong
(outcome 0.0); the DecisionHead trained on these records must lower its
confidence on the failure-bound state class while a lambda=0 twin pins
at 1.0. T28's A/B could not separate these (labels == outcomes); this
test can, because targets come from env.verify.
Run from repo root: python3 helioslm_v5/tests/test_decision_data.py
"""
import random
import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "agent"))

from decision_data import (RecordingGate, records_from_runs,
                           rlcd_reward_adjustment)
from decision_head import DecisionHead
from envs import make_envs
from gate import ThresholdGate
from loop import AgentLoop
from schema import ToolCall, render_tool_call
from tools import build_default_registry
from trajectory import text_to_ids


def _scripted():
    """Broken-ness is TEXTUALLY MARKED: the loop receives 'RISKY '+text
    for the broken subset, and the policy breaks exactly those. The state
    text must let the head learn which states end wrong — an unmarked
    broken set would only teach base-rate regression (found by T29 v1)."""
    def policy(prompt, seed, step):
        import re
        obs = re.findall(r"step \d+: (.*)", prompt)
        m = re.search(r"Compute the value of: (.*)", prompt)
        expr = m.group(1)
        risky = "RISKY" in prompt
        if not obs:
            return render_tool_call([ToolCall("calc", {"expr": expr})])
        ans = obs[-1]
        if risky:
            ans = str(int(ans) + 1) if ans.lstrip("-").isdigit() else ans
        return render_tool_call([ToolCall("finish", {"answer": ans})])
    return policy


def _run_tasks(n, seed, broken_frac=0.4):
    """Returns (records, outcomes, task_texts). Broken tasks are RISKY-
    prefixed; outcomes keyed by the SAME text the gate saw (the prefixed
    one), so records and outcomes align."""
    rng = random.Random(seed)
    env = make_envs()[0]
    reg, impls = build_default_registry("/tmp/helioslm_t29")
    tasks = [env.sample(rng) for _ in range(n)]
    broken = {t.text for t in tasks if rng.random() < broken_frac}
    gate = RecordingGate(ThresholdGate(0.5, oracle=lambda c, x: "ORACLE"))
    outcomes = {}
    policy = _scripted()
    for t in tasks:
        run_text = ("RISKY " + t.text) if t.text in broken else t.text
        conf_fn = lambda p, s: 0.95   # overconfident on EVERYTHING
        loop = AgentLoop(policy, reg, impls, gate,
                         max_steps=t.step_budget, confidence_fn=conf_fn)
        traj = loop.run(run_text, seed=1)
        ok = env.verify(t, traj.final_answer) if traj.final_answer else False
        outcomes[run_text] = ok
    return records_from_runs(gate.log, outcomes), outcomes, list(outcomes)


def test_recording_gate_and_records():
    recs, outcomes, _ = _run_tasks(6, seed=3, broken_frac=0.5)
    assert recs, "no records collected"
    assert all("route" in r["answers"] for r in recs)
    assert all("route__target_conf" in r["answers"] for r in recs)
    vals = {r["answers"]["route__target_conf"] for r in recs}
    assert vals == {0.0, 1.0}, vals   # both outcomes observed
    print(f"PASS test_recording_gate_and_records ({len(recs)} records, "
          f"outcomes {sum(outcomes.values())}/{len(outcomes)})")


def test_rlcd_loop_lowers_overconfidence():
    torch.manual_seed(5)
    recs, outcomes, _ = _run_tasks(40, seed=11, broken_frac=0.4)
    heads = {}
    for name, lam in (("brier", 10.0), ("plain", 0.0)):
        h = DecisionHead({"route": ("choice", ("DIRECT", "ESCALATE"))},
                         brier_lambda=lam)
        h.fit(recs, epochs=700, lr=5e-2)
        heads[name] = h

    # held-out states from the same distribution; RISKY marker decides
    # the class, so the oracle is a clean separation, not base-rate
    _, out2, _ = _run_tasks(20, seed=99, broken_frac=0.4)
    report = {}
    for name, h in heads.items():
        h.eval()
        confs = {"right": [], "wrong": []}
        with torch.no_grad():
            for t, ok in out2.items():
                ids = torch.tensor(text_to_ids(t + " [calc]"))
                _, conf = h.decide("route", ids)
                confs["right" if ok else "wrong"].append(conf)
        report[name] = {k: sum(v) / len(v) for k, v in confs.items()}
    b, p = report["brier"], report["plain"]
    # FINDING: binary Choice confidence = max softmax prob has a HARD
    # FLOOR at 1/K = 0.5 — it can never express "20% sure". The RISKY
    # class is pulled from 1.0 toward that floor; sub-0.5 uncertainty
    # requires the Noul/Score primitives (why Jev's set has three types).
    assert b["wrong"] < 0.62, \
        f"RLCD head did not pull RISKY confidence down: {report}"
    assert b["right"] - b["wrong"] > 0.3, \
        f"no class separation: {report}"
    assert b["right"] > 0.6, \
        f"RLCD head confidence collapsed on clean states: {report}"
    assert p["wrong"] > 0.9, f"plain head should pin high: {report}"
    print(f"PASS test_rlcd_loop_lowers_overconfidence {report} "
          f"(binary conf floor=0.5)")


def test_rlcd_reward_adjustment():
    adj = rlcd_reward_adjustment([1.0, 1.0, 0.5], [0.9, 0.2, None],
                                 [1.0, 1.0, 0.0], lam=2.0)
    assert abs(adj[0] - (1.0 - 2.0 * 0.01)) < 1e-9
    assert abs(adj[1] - (1.0 - 2.0 * 0.64)) < 1e-9
    assert adj[2] == 0.5, "None confidence must be skipped"
    print("PASS test_rlcd_reward_adjustment")


if __name__ == "__main__":
    test_recording_gate_and_records()
    test_rlcd_loop_lowers_overconfidence()
    test_rlcd_reward_adjustment()
    print("\n3/3 decision-data (RLCD) tests passed")
