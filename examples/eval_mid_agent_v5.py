"""eval_mid_agent_v5.py - v5.34: mid + GroundingGate, the therapy applied.

v4 named the disease (content confabulation at 0.9999 confidence); this
eval applies the therapy: the SAME mid checkpoint driven through
ChatSession with GroundingGate wrapped around the DIRECT route -- the
model keeps the tool-sequence decision, policy in code fills both ends.

Usage (local, after a train_mid_sft.py run):
    python examples/eval_mid_agent_v5.py
Writes checkpoints/mid_agent_eval_v5.json. Compare against
mid_agent_eval_v4.json (0/12 ungrounded).
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
from grounding import GroundingGate
from helioslm_v5.configs.config_v5 import HeliosLMv5Config
from helioslm_v5.src.model_v5 import HeliosLMv5
from helioslm_v5.src.tokenizer.bpe import BOS, HeliosBPE
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
    print("mid checkpoint loaded (v5, grounded)", flush=True)

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
    rng = random.Random(20260928)   # same seed as v2/v4
    results = []
    for env in make_envs() + make_long_envs():
        reg, impls = build_default_registry()
        for _ in range(max(1, n_tasks // 4)):
            task = env.sample(rng)
            session = ChatSession(model_fn, reg, impls,
                                  GroundingGate(FixedGate(Route.DIRECT)),
                                  max_steps=task.step_budget)
            final = session.send(task.text, seed=1)
            ok = final is not None and env.verify(task, final)
            results.append({"family": getattr(task, "family",
                                              type(task).__name__),
                            "task": task.text[:80], "expected": task.answer,
                            "final": final, "correct": ok,
                            "steps": len(session.steps), "conf": state["conf"]})
            print(f"{results[-1]['family']} correct={ok} "
                  f"final={str(final)[:30]!r}", flush=True)
    corr = [r for r in results if r["correct"]]
    summary = {"n": len(results),
               "correct": f"{len(corr)}/{len(results)}",
               "grounded": True,
               "compare": "mid_agent_eval_v4.json was 0/12 ungrounded",
               "results": results}
    (ck / "mid_agent_eval_v5.json").write_text(json.dumps(summary, indent=2))
    print(json.dumps({k: v for k, v in summary.items() if k != "results"},
                     indent=2), flush=True)
    os._exit(0)


if __name__ == "__main__":
    main()
