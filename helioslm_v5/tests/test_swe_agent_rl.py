"""T100 — roadmap Phase AC, AC2: multi-turn agent RL on the SWE sandbox.

Pins:

1. GRPO ``action_mask``: None is bit-identical to the old sum, an
   all-True mask equals None, and a mask sums exactly the policy tokens;
2. environment semantics on every task: broken candidates fail the
   visible tests, a fix applied on top of a broken patch still fails,
   REVERT + fix passes visible AND hidden tests; outcomes are memoized
   from real sandbox runs;
3. rollout structure: actions and observations alternate, the mask marks
   exactly the action positions, old_logprob equals the policy's masked
   log-prob of the trajectory;
4. a scripted apply-test-revert policy scores 1.0 on all 20 tasks, while
   "apply slot 0 then DONE" scores about 1/3 — the gap RL has to close;
5. the committed training artifact (when present): held-out greedy pass
   rate after training above before, and the hash.

Run from repo root: python3 helioslm_v5/tests/test_swe_agent_rl.py
"""
import hashlib
import json
import random
import sys
from pathlib import Path

import torch

_REPO = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(_REPO))
sys.path.insert(0, str(_REPO / "examples"))

from helioslm_v5.eval import swe_bench as sb  # noqa: E402
from helioslm_v5.src.model_v5 import HeliosLMv5  # noqa: E402
from helioslm_v5.src.training import swe_agent_rl as ar  # noqa: E402
from helioslm_v5.src.training.grpo import GRPOTrainer  # noqa: E402

_ART = _REPO / "benchmarks" / "swe_agent_rl_2026-10-16.json"


class _Args:
    group, kl, lr = 4, 0.0, 1e-3


def _policy(seed=0):
    import swe_agent_rl_train as script
    torch.manual_seed(seed)
    return HeliosLMv5(script.policy_config(_Args())).eval()


def test_action_mask():
    model = _policy()
    ids = torch.tensor([[1, 2, 8, 6, 10, 5, 8, 7]])
    lp = GRPOTrainer._sequence_logprob
    with torch.no_grad():
        base = lp(model, ids, 1)
        same = lp(model, ids, 1, torch.ones(7, dtype=torch.bool))
        mask = torch.tensor([True, False, True, False, True, False, True])
        part = lp(model, ids, 1, mask)
        logp = torch.log_softmax(model(ids)[0][0, :-1].float(), -1)
        tok = logp.gather(-1, ids[0, 1:, None]).squeeze(-1)
    assert torch.equal(base, tok.sum()) and torch.equal(base, same)
    assert torch.allclose(part, tok[mask].sum())
    print("PASS test_action_mask")


def test_env_semantics():
    for t in sb.TASKS:
        env = ar.SweMacroEnv(t, random.Random(3))
        bad = [k for k in range(3) if k != env.fix_slot]
        for b in bad:
            env.step(ar.REVERT)
            env.step(ar.APPLY[b])
            assert env.step(ar.TEST) == ar.OBS_FAIL and not env.verify(), t["id"]
        env.step(ar.APPLY[env.fix_slot])
        assert env.step(ar.TEST) == ar.OBS_FAIL, t["id"]   # broken still there
        env.step(ar.REVERT)
        env.step(ar.APPLY[env.fix_slot])
        assert env.step(ar.TEST) == ar.OBS_PASS and env.verify(), t["id"]
    assert len(ar._OUTCOMES) >= 20 * 4
    print(f"PASS test_env_semantics ({len(sb.TASKS)} tasks)")


def test_rollout_structure():
    model = _policy()
    r = ar.rollout(model, sb.TASKS[0], random.Random(0))
    ids, mask = r["full_ids"][0, 1:], r["action_mask"]
    assert len(ids) == len(mask) and r["steps"] == int(mask.sum())
    assert all(int(t) in ar.ACTIONS for t in ids[mask])
    assert all(int(t) in (ar.OBS_NONE, ar.OBS_PASS, ar.OBS_FAIL)
               for t in ids[~mask])
    assert list(mask[::2]) == [True] * len(mask[::2])
    with torch.no_grad():
        logits = model(r["full_ids"])[0][0, :-1].float()
    allowed = torch.full((logits.shape[-1],), float("-inf"))
    allowed[list(ar.ACTIONS)] = 0.0
    logp = torch.log_softmax(logits + allowed, -1)
    want = logp.gather(-1, ids[:, None]).squeeze(-1)[mask].sum()
    assert torch.allclose(r["old_logprob"], want, atol=1e-4)
    print("PASS test_rollout_structure")


def _scripted(env, careful):
    if not careful:
        env.step(ar.APPLY[0])
        return env.verify()
    for k in range(3):
        env.step(ar.REVERT)
        env.step(ar.APPLY[k])
        if env.step(ar.TEST) == ar.OBS_PASS:
            return env.verify()
    return False


def test_scripted_policies():
    careful = naive = n = 0
    for s in range(3):
        rng = random.Random(10_000 + s)
        for t in sb.TASKS:
            careful += _scripted(ar.SweMacroEnv(t, rng), True)
            naive += _scripted(ar.SweMacroEnv(t, rng), False)
            n += 1
    assert careful == n
    assert 0.15 < naive / n < 0.55, naive / n
    print(f"PASS test_scripted_policies (careful 1.0, naive {naive / n:.2f})")


def test_training_artifact():
    if not _ART.exists():
        print("SKIP test_training_artifact (no artifact yet)")
        return
    art = json.loads(_ART.read_text())
    res = art["results"]
    assert art["results_sha256"] == hashlib.sha256(
        json.dumps(res, sort_keys=True).encode()).hexdigest()
    assert res["after"]["held_out"] > res["before"]["held_out"]
    assert res["after"]["train"] > res["before"]["train"]
    print("PASS test_training_artifact "
          f"(held-out {res['before']['held_out']:.3f} -> "
          f"{res['after']['held_out']:.3f})")


if __name__ == "__main__":
    test_action_mask()
    test_env_semantics()
    test_rollout_structure()
    test_scripted_policies()
    test_training_artifact()
