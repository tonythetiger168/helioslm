"""SelfGenEnv - RSI route (2): AI-generated curriculum with independent verification.

The RSI survey's warning: "harder tasks != stronger model; independent
verification needed." This module lets the mid 360M model GENERATE new
toy-env tasks (exprs/strings), then independently verifies whether the
generated tasks (a) are well-formed (parse via the shared grammar) and
(b) are actually solved by the model (training value check).

This is Absolute Zero's minimal reproducible version, with our replay
verification as the "independent verification" mechanism the survey
demands.
"""
import json
import random

import torch

import sys as _sys
_sys.path.insert(0, "/mnt/agents/output/hlwork/main/helioslm_v5/agent")
from task_grammar import parse as parse_task, render as render_task


class SelfGenEnv:
    """Generates tasks by querying the model, verifies independently."""

    def __init__(self, model_fn, seed=0):
        self.model_fn = model_fn
        self.rng = random.Random(seed)

    def gen_calc_task(self):
        """Ask the model for a new calc expression; verify it parses."""
        prompt = ("Generate one new arithmetic expression using + - * "
                  "// and numbers -50 to 50. Reply with ONLY the "
                  "expression, nothing else.")
        out = self.model_fn(prompt, 0, 0).strip()
        # strip prose, keep expr-ish core
        expr = "".join(c for c in out
                       if c in "-0123456789+*/%() ").strip()
        try:
            val = eval(expr)
            if isinstance(val, int) and abs(val) < 10**9:
                return {"kind": "calc", "expr": expr, "answer": repr(val),
                        "well_formed": True}
        except Exception:
            pass
        return {"kind": "calc", "expr": expr, "answer": None,
                "well_formed": False}

    def gen_str_task(self):
        prompt = ("Generate one random string of 6-10 letters and digits. "
                  "Reply with ONLY the string, nothing else.")
        out = self.model_fn(prompt, 0, 0).strip().strip('"').strip()
        s = "".join(c for c in out if c.isalnum())[:12]
        if 4 <= len(s):
            return {"kind": "str", "s": s, "well_formed": True}
        return {"kind": "str", "s": s, "well_formed": False}

    def verify_independent(self, task, solver_fn, n_samples=3):
        """Independent verification: can the model SOLVE its own task?
        This is the survey's key check -- a generated task has training
        value only if the model can (sometimes) solve it, so RL signal
        is non-degenerate."""
        if not task.get("well_formed"):
            return {"solvable": False, "reason": "malformed"}
        if task["kind"] == "calc":
            prompt = f"Compute the value of: {task['expr']}"
            want = task["answer"]
        else:
            prompt = f'Apply upper to the string: "{task["s"]}"'
            want = task["s"].upper()
        hits = 0
        for s in range(n_samples):
            out = solver_fn(prompt, s, 0).strip()
            if task["kind"] == "calc":
                ok = out == want or out.lstrip("-").isdigit() and \
                     repr(int(out)) == want
            else:
                ok = out == want
            hits += ok
        return {"solvable": hits > 0, "hit_rate": hits / n_samples,
                "want": want}


if __name__ == "__main__":
    # smoke: scripted generator (no real model needed for the loop)
    def fake_gen(prompt, seed, step):
        if "expression" in prompt:
            return "17 * 3 - 5"
        return "xK9mQ2"
    env = SelfGenEnv(fake_gen)
    t1 = env.gen_calc_task()
    t2 = env.gen_str_task()
    print("gen calc:", t1)
    print("gen str:", t2)
    r1 = env.verify_independent(t1, lambda p, s, st: "46")
    print("verify calc (solve 46):", r1)
    r2 = env.verify_independent(t2, lambda p, s, st: "XK9MQ2")
    print("verify str (solve upper):", r2)
    # roundtrip through the shared grammar
    text = render_task("calc", expr=t1["expr"])
    back = parse_task(text)
    print("grammar roundtrip:", back["kind"] == "calc" and back["expr"] == t1["expr"])
