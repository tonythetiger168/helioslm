"""eval_mid_agent_v2.py - v5.33: mid agency test IN THE MODEL'S FORMAT.

v1 (eval_mid_agent.py) drove the mid model through AgentLoop's tool
pipeline format ("Task:/step i:") and got 0/12 with parse_rate 0.0 --
NOT a capacity verdict but train/serve format skew: the mid SFT data
(gen_chat_episode) renders chat transcripts (##user##/##assistant##/
##tool##), which the v1 harness never used. Lite's T17 parse 0.67 came
from train_tool_tuned.py, whose data IS AgentLoop format -- data format
decides capability ownership, not parameter count (recorded finding).

This v2 drives ChatSession -- the harness that matches the training
distribution -- so the agency question is tested in-format.
Usage: python examples/eval_mid_agent_v2.py
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
from helioslm_v5.src.tokenizer.bpe import BOS, HeliosBPE, build_corpus
from tools import build_default_registry

MAX_NEW = 128


def main():
    torch.manual_seed(0)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    ck = Path(os.environ.get("HELIOS_CKPT_DIR", "checkpoints"))
    # PAIRED tokenizer first: never retrain at eval time (train/serve
    # tokenizer skew, 2026-09-28). Retrain only as a warned fallback.
    tok_path = ck / "mid_sft_v5.33.tok.json"
    if tok_path.exists():
        tok = HeliosBPE.load(str(tok_path))
        print("loaded paired tokenizer from checkpoint", flush=True)
    else:
        print("WARNING: no saved tokenizer -- retraining (may mismatch!",
              flush=True)
        corpus = build_corpus()
        tok = HeliosBPE.train([corpus[-30000:] + corpus[:170000]],
                              vocab_size=16000)
    bos = tok.vocab[BOS]
    model = HeliosLMv5(HeliosLMv5Config(size="mid")).to(device)
    model.load_state_dict(torch.load(ck / "mid_sft_v5.33.pt",
                                     map_location=device))
    model.eval()
    print("mid checkpoint loaded (v2, ChatSession format)", flush=True)

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
    rng = random.Random(20260928)
    results = []
    for env in make_envs() + make_long_envs():
        reg, impls = build_default_registry()
        for _ in range(max(1, n_tasks // 4)):
            task = env.sample(rng)
            session = ChatSession(model_fn, reg, impls,
                                  FixedGate(Route.DIRECT),
                                  max_steps=task.step_budget)
            t0 = time.time()
            final = session.send(task.text, seed=1)
            parsed = sum(1 for s in session.steps
                         if s.kind == "tool")
            ok = final is not None and env.verify(task, final)
            results.append({
                "family": getattr(task, "family", type(task).__name__),
                "correct": ok, "steps": len(session.steps),
                "budget": task.step_budget,
                "tool_steps": parsed, "finished": final is not None,
                "conf": state["conf"],
                "wall_s": round(time.time() - t0, 1)})
            print(f"{results[-1]['family']} correct={ok} "
                  f"steps={len(session.steps)}/{task.step_budget} "
                  f"tool_steps={parsed} conf={state['conf']}", flush=True)
    corr = [r for r in results if r["correct"]]
    fin = [r for r in results if r["finished"]]
    conf_wrong = [r["conf"] for r in results if not r["correct"] and r["conf"]]
    summary = {
        "n": len(results),
        "correct": f"{len(corr)}/{len(results)}",
        "finished": f"{len(fin)}/{len(results)}",
        "mean_tool_steps": round(sum(r["tool_steps"] for r in results)
                                 / len(results), 2),
        "conf_on_wrong_mean": (round(sum(conf_wrong) / len(conf_wrong), 4)
                               if conf_wrong else None),
        "harness": "ChatSession (training distribution)",
        "results": results}
    out = ck / "mid_agent_eval_v2.json"
    out.write_text(json.dumps(summary, indent=2))
    print(json.dumps({k: v for k, v in summary.items() if k != "results"},
                     indent=2), flush=True)
    # Windows + CUDA teardown hang: results are on disk; skip the
    # driver's slow exit cleanup (user-reported 2026-09-29). os comes
    # from the module-top import (a function-local import would shadow
    # every os.environ use above -- UnboundLocalError, found on first
    # real run 2026-09-29).
    os._exit(0)


if __name__ == "__main__":
    main()
