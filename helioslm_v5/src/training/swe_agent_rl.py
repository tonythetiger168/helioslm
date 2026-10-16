"""Multi-turn agent RL on the mini SWE sandbox (roadmap Phase AC, AC2).

A toy-scale but real agent-RL loop: the policy acts over several turns
in a real sandbox, the environment answers with the outcome of each
action, and the reward is the SWE verifier (visible + hidden tests).
The update is ``GRPOTrainer._learn_from_samples`` unchanged — group
advantages, clipped surrogate, k3 KL — with an ``action_mask`` so only
policy-emitted tokens enter the log-probs (observation tokens the
environment interleaves are excluded).

SETTING (macro actions, candidate-patch selection)
--------------------------------------------------
A CPU toy model cannot learn to write shell from scratch, so the action
space is the selection stage of a generate-then-select agent: each
episode offers three candidate patches in a SHUFFLED order — the
reference fix and two broken patches (each makes the module raise on
import, so the visible tests fail). Actions:

    APPLY_0 / APPLY_1 / APPLY_2   apply candidate k (shell patch command)
    REVERT                        restore the repository
    TEST                          run the visible tests -> OBS_PASS / OBS_FAIL
    DONE                          stop; reward = 1 if the verifier passes

Reward (知止, the AC5 view): +1 for DONE with a passing verifier,
``-false_done_penalty`` for DONE with a failing one, 0 for running out
of steps. With the penalty at 0, "apply any candidate, DONE" earns 1/3
for free and GRPO collapses onto it (observed: v5.63 run 1); at 1 a
blind guess is worth -1/3 and only stopping after a passing TEST pays.

Applying after a broken patch without REVERT leaves the repo broken, and
the candidate order is reshuffled every episode, so no fixed action
sequence works: a good policy must apply, test, revert on failure and
stop only after a pass — the AC5 "know when you are done" habit, learned
from the reward. Random policies rarely manage it.

Observation tokens follow each action (OBS_NONE after apply/revert).
The policy sees no task id: train and held-out tasks share the
procedure, which is what the held-out evaluation checks.
"""
import random
import tempfile
from pathlib import Path

import torch

from helioslm_v5.eval import swe_bench as sb
from helioslm_v5.eval import term_bench as tb

PAD, BOS = 0, 1
APPLY = (2, 3, 4)
REVERT, TEST, DONE = 5, 6, 7
OBS_NONE, OBS_PASS, OBS_FAIL = 8, 9, 10
VOCAB = 16
ACTIONS = APPLY + (REVERT, TEST, DONE)
ACTION_NAMES = {2: "APPLY_0", 3: "APPLY_1", 4: "APPLY_2", 5: "REVERT",
                6: "TEST", 7: "DONE"}
MAX_STEPS = 10


def broken_patches(task):
    """Two patches that make the fixed module raise on import."""
    target = task["fix"][0][0]
    src = task["files"][target]
    tail = src.rstrip("\n").splitlines()[-1]
    return [
        [(target, tail, tail + "\nraise RuntimeError('broken patch')")],
        [(target, tail, tail + "\ndef broken(:\n    pass")],
    ]


_OUTCOMES = {}   # (task id, applied patch ids, "test"|"verify") -> bool


class SweMacroEnv:
    """One episode on one task (use as a context manager).

    The repository state is fully determined by the candidate patches
    applied since the last REVERT, in order, so the env tracks that
    sequence and the outcome of TEST / verify is memoized per (task,
    sequence). A cache miss materializes the state in a fresh sandbox —
    seed the files, run each patch command, then the real visible tests
    or the real verifier — so every recorded outcome came from an actual
    sandbox run; repeats only skip re-running it.
    """

    def __init__(self, raw_task, rng):
        self.raw = raw_task
        self.term = {"id": raw_task["id"], "files": raw_task["files"]}
        self.patches = [raw_task["fix"]] + broken_patches(raw_task)
        order = list(range(3))
        rng.shuffle(order)
        self.order = order                  # slot -> patch id (0 = fix)
        self.fix_slot = order.index(0)
        self.applied = ()

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        return False

    def _outcome(self, kind):
        key = (self.raw["id"], self.applied, kind)
        if key not in _OUTCOMES:
            with tempfile.TemporaryDirectory(prefix="swe_rl_") as root:
                tb.seed_sandbox(self.term, root)
                for pid in self.applied:
                    tb.run_command(sb.patch_command(self.patches[pid]), root)
                if kind == "test":
                    ok = all(tb.run_command(f"python3 {v}", root)[0] == 0
                             for v in self.raw["visible"])
                else:
                    ok = sb._verify(self.raw, root)[0]
            _OUTCOMES[key] = ok
        return _OUTCOMES[key]

    def step(self, action):
        if action in APPLY:
            self.applied += (self.order[action - APPLY[0]],)
            return OBS_NONE
        if action == REVERT:
            self.applied = ()
            return OBS_NONE
        if action == TEST:
            return OBS_PASS if self._outcome("test") else OBS_FAIL
        raise ValueError(f"not a stepping action: {action}")

    def verify(self):
        return self._outcome("verify")


def rollout(model, raw_task, rng, greedy=False, temperature=1.0,
            false_done_penalty=0.0):
    """-> sample dict for GRPOTrainer plus episode stats."""
    ids = [BOS]
    mask = []
    old_lp = 0.0
    done = False
    with SweMacroEnv(raw_task, rng) as env:
        for _ in range(MAX_STEPS):
            x = torch.tensor([ids])
            with torch.no_grad():
                logits = model(x)[0][0, -1].float()
            allowed = torch.full_like(logits, float("-inf"))
            allowed[list(ACTIONS)] = 0.0
            logp = torch.log_softmax(logits / temperature + allowed, -1)
            if greedy:
                a = int(logp.argmax())
            else:
                g = torch.Generator().manual_seed(rng.randrange(2 ** 31))
                a = int(torch.multinomial(logp.exp(), 1, generator=g))
            old_lp += float(torch.log_softmax(logits + allowed, -1)[a])
            ids.append(a)
            mask.append(True)
            if a == DONE:
                done = True
                break
            ids.append(env.step(a))
            mask.append(False)
        passed = done and env.verify()
    return {
        "full_ids": torch.tensor([ids]),
        "prompt_len": 1,
        "action_mask": torch.tensor(mask, dtype=torch.bool),
        "old_logprob": torch.tensor(old_lp),
        "reward": 1.0 if passed else (-false_done_penalty if done else 0.0),
        "passed": passed,
        "done": done,
        "steps": sum(mask),
        "trace": [ACTION_NAMES.get(t, "?") for t in ids[1::2]],
    }


def evaluate(model, tasks, seeds=(0, 1, 2), greedy=True):
    """Pass rate over tasks x shuffled candidate orders."""
    model.eval()
    wins = n = 0
    for s in seeds:
        rng = random.Random(10_000 + s)
        for t in tasks:
            r = rollout(model, t, rng, greedy=greedy)
            wins += r["passed"]
            n += 1
    return wins / n
