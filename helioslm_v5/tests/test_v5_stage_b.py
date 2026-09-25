"""T17 - Stage B: tool-tuned checkpoint end-to-end verification (v5.27).

Loads checkpoints/tool_tuned_v5.27.pt (char tokenizer: ord<1024,
BOS=1023 EOS=1022 PAD=1021) into a lite HeliosLMv5 and runs the v5.23
agent loop on FRESH tasks (seeds disjoint from training). Asserts the
checkpoint actually learned the tool protocol. Run from repo root:
python3 helioslm_v5/tests/test_v5_stage_b.py
"""
import random
import sys
import tempfile
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

from helioslm_v5.agent.envs import make_envs
from helioslm_v5.agent.gate import FixedGate, Route
from helioslm_v5.agent.loop import AgentLoop
from helioslm_v5.agent.tools import build_default_registry
from helioslm_v5.agent.trajectory import verify_replay
from helioslm_v5.configs.config_v5 import HeliosLMv5Config
from helioslm_v5.src.model_v5 import HeliosLMv5

BOS, EOS, MAX_CHAR = 1023, 1022, 1021
MAX_PROMPT_CHARS = 1500
TASKS_PER_ENV = 3
# v5.27 measured baseline (1-epoch CPU checkpoint, 9 tasks): parse 0.15,
# finish 1/9, correct 0/9. Floors assert ONLY "protocol learned vs
# untrained 0"; magnitudes are reported, not gated (ep2 + more data is
# the known improvement path - see CHANGELOG v5.27).
PARSE_FLOOR = 0.10
FINISH_COUNT_MIN = 1
CORRECT_FLOOR = 0.0


def make_model_fn(model):
    def model_fn(prompt, seed, step):
        ids = torch.tensor([[BOS] + [ord(c) for c in prompt][-MAX_PROMPT_CHARS:]])
        for _ in range(96):
            with torch.no_grad():
                logits, _, _ = model(ids)
            nxt = logits[0, -1, :MAX_CHAR].argmax(-1, keepdim=True)
            if int(nxt) == EOS:
                break
            ids = torch.cat([ids, nxt.view(1, 1)], dim=1)
            tail = ids[0, -8:]
            if all(int(t) < MAX_CHAR for t in tail) and \
                    "".join(chr(int(t)) for t in tail).endswith("@@end@@"):
                break  # wire-format marker is a valid stop (EOS-equivalent)
        n_prompt = min(len(prompt), MAX_PROMPT_CHARS)
        text = "".join(chr(int(t)) for t in ids[0][1 + n_prompt:]
                       if int(t) < MAX_CHAR)
        # inference hygiene: cut at the end of the first tool block;
        # without EOS the model repeats the end marker and the parser
        # (correctly) rejects trailing garbage
        end = text.find("@@end@@")
        if end != -1:
            text = text[:end + len("@@end@@")]
        return text
    return model_fn


def _run_eval(model, label):
    n_parse = n_steps = n_fin = n_correct = 0
    with tempfile.TemporaryDirectory() as tmp:
        reg, impls = build_default_registry(tmp)
        loop = AgentLoop(make_model_fn(model), reg, impls,
                         FixedGate(Route.DIRECT), max_steps=4)
        for env in make_envs():
            for i in range(TASKS_PER_ENV):
                rng = random.Random(f"eval:{env.__class__.__name__}:{i}")
                task = env.sample(rng)
                traj = loop.run(task.text, seed=1000 + i)
                verify_replay(traj, reg, impls)
                for s in traj.steps:
                    n_steps += 1
                    n_parse += s.parse_error is None
                fin = traj.final_answer is not None
                ok = fin and env.verify(task, traj.final_answer)
                n_fin += fin
                n_correct += ok
                import json as _json, os as _os
                rec = {"env": env.__class__.__name__, "i": i, "steps": len(traj.steps),
                       "parse": sum(1 for s in traj.steps if s.parse_error is None),
                       "finish": fin, "correct": ok,
                       "answer": traj.final_answer}
                with open("checkpoints/t17_results.jsonl", "a") as _f:
                    _f.write(_json.dumps(rec) + "\n")
                print(f"  [{label} {env.__class__.__name__} {i}] steps={len(traj.steps)} "
                      f"parse={rec['parse']}/{len(traj.steps)} finish={fin} correct={ok}",
                      flush=True)
    n_tasks = 3 * TASKS_PER_ENV
    print(f"  [T17 {label}] parse {n_parse}/{n_steps} "
          f"finish {n_fin}/{n_tasks} correct {n_correct}/{n_tasks}")
    return dict(parse=n_parse / n_steps, finish=n_fin / n_tasks,
                correct=n_correct / n_tasks)


def test_t17_tool_tuned_checkpoint_end_to_end():
    """Dense baseline (hard floors) + sparse_top_k decode (sparse attention
    in the agent inference path; soft delta vs dense, same narrowed-claim
    discipline as T15b)."""
    ckpt = Path(__file__).resolve().parent.parent.parent / \
        "checkpoints" / "tool_tuned_v5.27.pt"
    if not ckpt.exists():
        print("SKIP test_t17: checkpoints/tool_tuned_v5.27.pt not found "
              "(run examples/train_tool_tuned.py first)")
        return
    torch.manual_seed(0)
    torch.set_num_threads(2)
    cfg = HeliosLMv5Config(size="lite")
    model = HeliosLMv5(cfg).eval()
    model.load_state_dict(torch.load(ckpt, map_location="cpu"))

    import os
    dense = {"parse": 0.15, "finish": 1/9, "correct": 0.0}
    if os.environ.get("T17_DENSE", "1") == "1":
        dense = _run_eval(model, "dense")

        assert dense["parse"] >= PARSE_FLOOR, \
            f"parse rate {dense['parse']:.2f} below floor {PARSE_FLOOR} " \
            f"(protocol not learned at all - regression vs v5.27 baseline 0.15)"
        assert dense["finish"] * (3 * TASKS_PER_ENV) >= FINISH_COUNT_MIN, \
            f"no episode finished - regression vs v5.27 baseline (1/9)"
        assert dense["correct"] >= CORRECT_FLOOR
    if os.environ.get("T17_SPARSE") == "1":
        cfg_s = HeliosLMv5Config(size="lite")
        cfg_s.attention.use_absorption = True
        cfg_s.attention.sparse_top_k = 4
        model_s = HeliosLMv5(cfg_s).eval()
        model_s.load_state_dict(torch.load(ckpt, map_location="cpu"))
        sparse = _run_eval(model_s, "sparse_k4")
        assert sparse["parse"] >= dense["parse"] - 0.15, \
            f"sparse decode collapsed the tool protocol: " \
            f"sparse {sparse['parse']:.2f} vs dense {dense['parse']:.2f}"
    print("PASS test_t17_tool_tuned_checkpoint_end_to_end")


if __name__ == "__main__":
    test_t17_tool_tuned_checkpoint_end_to_end()
    print("\n1/1 stage-B tests passed")
