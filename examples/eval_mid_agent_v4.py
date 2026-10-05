"""eval_mid_agent_v4.py - v5.33: agency eval with FULL evidence capture.

v2 recorded metrics but NOT the model's actual answers -- the autopsy
question (copy-observation failure vs arithmetic confabulation) needs
the raw finish answers. v4 captures everything: task text, expected
answer, per-step generated text, final answer, transcript tail.

Usage: python examples/eval_mid_agent_v4.py
Writes checkpoints/mid_agent_eval_v4.json -- upload THAT for autopsy.
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

from chat import ChatSession
from envs import make_envs, make_long_envs
from gate import FixedGate, Route
from helioslm_v5.configs.config_v5 import HeliosLMv5Config
from helioslm_v5.src.model_v5 import HeliosLMv5
from helioslm_v5.src.tokenizer.bpe import BOS, HeliosBPE
from trajectory import ids_to_text
from tools import build_default_registry

MAX_NEW = 128


def main():
    torch.manual_seed(0)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    ck = Path(os.environ.get("HELIOS_CKPT_DIR", "checkpoints"))
    tok = HeliosBPE.load(str(ck / "mid_sft_v5.33.tok.json"))
    print("loaded paired tokenizer from checkpoint", flush=True)
    bos = tok.vocab[BOS]
    model = HeliosLMv5(HeliosLMv5Config(size="mid")).to(device)
    model.load_state_dict(torch.load(ck / "mid_sft_v5.33.pt",
                                     map_location=device))
    model.eval()
    print("mid checkpoint loaded (v4, full capture)", flush=True)

    state = {"conf": None}

    def model_fn(prompt, seed, step):
        ids = [bos] + tok.encode(prompt)
        input_ids = torch.tensor([ids]).to(device)
        with torch.no_grad():
            for _ in range(MAX_NEW):
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

    n_tasks = int(os.environ.get("EVAL_TASKS", "12"))
    rng = random.Random(20260928)   # SAME seed as v2: directly comparable
    results = []
    for env in make_envs() + make_long_envs():
        reg, impls = build_default_registry()
        for _ in range(max(1, n_tasks // 4)):
            task = env.sample(rng)
            session = ChatSession(model_fn, reg, impls,
                                  FixedGate(Route.DIRECT),
                                  max_steps=task.step_budget)
            final = session.send(task.text, seed=1)
            ok = final is not None and env.verify(task, final)
            results.append({
                "family": getattr(task, "family", type(task).__name__),
                "task": task.text,
                "expected": task.answer,
                "final": final,
                "correct": ok,
                "steps": len(session.steps),
                "budget": task.step_budget,
                "conf": state["conf"],
                "gens": [ids_to_text(s.generated_ids) for s in session.steps],
                "transcript_tail": "\n".join(
                    f"{t.role} {t.content}" for t in session.turns[-6:]),
            })
            print(f"{results[-1]['family']} correct={ok} "
                  f"expected={task.answer!r} final={str(final)[:40]!r}",
                  flush=True)
    corr = [r for r in results if r["correct"]]
    summary = {"n": len(results),
               "correct": f"{len(corr)}/{len(results)}",
               "results": results}
    out = ck / "mid_agent_eval_v4.json"
    out.write_text(json.dumps(summary, indent=2))
    print("wrote", out, flush=True)
    os._exit(0)


if __name__ == "__main__":
    main()
