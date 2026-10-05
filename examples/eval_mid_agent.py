"""eval_mid_agent.py - v5.33: run the trained mid model through AgentLoop.

The REAL agency test (the wall Qwen3-0.6B hit at 0/15): multi-step tool
loops with observations fed back, replay-verified. Single-prompt SFT
eval already passed 40/40 -- this measures what that did NOT.

Usage (local, after train_mid_sft.py):
    python examples/eval_mid_agent.py
Writes checkpoints/mid_agent_eval.json and prints it.
"""
import json
import os
import random
import sys
import time
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "helioslm_v5" / "agent"))

from envs import make_envs, make_long_envs
from gate import FixedGate, Route
from helioslm_v5.configs.config_v5 import HeliosLMv5Config
from helioslm_v5.src.model_v5 import HeliosLMv5
from helioslm_v5.src.tokenizer.bpe import BOS, HeliosBPE, build_corpus
from loop import AgentLoop
from tools import build_default_registry
from trajectory import verify_replay

FEWSHOT = """Solved example:
Task: Compute the value of: 2 + 3
Assistant: @@tool@@{"calls":[{"name":"calc","args":{"expr":"2 + 3"}}]}@@end@@
step 0: 5
Assistant: @@tool@@{"calls":[{"name":"finish","args":{"answer":"5"}}]}@@end@@
Final answer: 5

"""


def main():
    torch.manual_seed(0)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    corpus = build_corpus()
    tok = HeliosBPE.train([corpus[-30000:] + corpus[:170000]], vocab_size=16000)
    bos = tok.vocab[BOS]
    model = HeliosLMv5(HeliosLMv5Config(size="mid")).to(device)
    ck = Path(os.environ.get("HELIOS_CKPT_DIR", "checkpoints"))
    model.load_state_dict(torch.load(ck / "mid_sft_v5.33.pt", map_location=device))
    model.eval()
    print("mid checkpoint loaded", flush=True)

    state = {"conf": None}

    def model_fn(prompt, seed, step):
        ids = [bos] + tok.encode(FEWSHOT + prompt)
        input_ids = torch.tensor([ids]).to(device)
        with torch.no_grad():
            for _ in range(120):
                logits, _, _ = model(input_ids,
                                     attention_mask=torch.ones_like(input_ids))
                probs = torch.softmax(logits[0, -1], dim=-1)
                conf, nxt = float(probs.max()), int(probs.argmax())
                if nxt == tok.vocab["<eos>"]:
                    break
                input_ids = torch.cat(
                    [input_ids, torch.tensor([[nxt]]).to(device)], dim=1)
        state["conf"] = conf
        return tok.decode(input_ids[0, len(ids):].tolist())

    import tempfile
    n_tasks = int(os.environ.get("EVAL_TASKS", "12"))
    rng = random.Random(20260928)
    results = []
    with tempfile.TemporaryDirectory() as root:
        for env in make_envs() + make_long_envs():
            for _ in range(n_tasks // 4):
                task = env.sample(rng)
                reg, impls = build_default_registry(root)
                loop = AgentLoop(model_fn, reg, impls, FixedGate(Route.DIRECT),
                                 max_steps=task.step_budget)
                t0 = time.time()
                traj = loop.run(task.text, seed=1)
                parsed = sum(1 for s in traj.steps if s.parsed is not None)
                replay_ok = True
                try:
                    verify_replay(traj, reg, impls)
                except Exception:
                    replay_ok = False
                ok = (traj.final_answer is not None
                      and env.verify(task, traj.final_answer))
                results.append({
                    "family": getattr(task, "family", type(task).__name__),
                    "correct": ok, "steps": len(traj.steps),
                    "budget": task.step_budget,
                    "parse_rate": parsed / max(1, len(traj.steps)),
                    "finished": traj.final_answer is not None,
                    "replay_ok": replay_ok, "conf": state["conf"],
                    "wall_s": round(time.time() - t0, 1)})
                print(f"{results[-1]['family']} correct={ok} "
                      f"steps={len(traj.steps)}/{task.step_budget} "
                      f"conf={state['conf']}", flush=True)
    corr = [r for r in results if r["correct"]]
    fin = [r for r in results if r["finished"]]
    conf_wrong = [r["conf"] for r in results if not r["correct"] and r["conf"]]
    summary = {
        "n": len(results),
        "correct": f"{len(corr)}/{len(results)}",
        "finished": f"{len(fin)}/{len(results)}",
        "mean_parse_rate": round(sum(r["parse_rate"] for r in results)
                                 / len(results), 3),
        "replay_ok": all(r["replay_ok"] for r in results),
        "conf_on_wrong_mean": (round(sum(conf_wrong) / len(conf_wrong), 3)
                               if conf_wrong else None),
        "results": results}
    out = ck / "mid_agent_eval.json"
    out.write_text(json.dumps(summary, indent=2))
    print(json.dumps({k: v for k, v in summary.items() if k != "results"},
                     indent=2), flush=True)


if __name__ == "__main__":
    main()
