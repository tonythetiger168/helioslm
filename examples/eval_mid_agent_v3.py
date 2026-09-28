"""eval_mid_agent_v3.py - v5.33: PROMPT-DIFF ORACLE.

v1 (AgentLoop format): 0/12, parse 0.0 -- diagnosed as format skew.
v2 (ChatSession format): 0/12, finished 12/12, tool_steps 0.0 -- the
model answers every task with plain TEXT in one step, confidence 1.0.
BUT the same checkpoint scored mode-tool 20/20 in train_mid_sft's
evaluate() on the SAME distribution. Two harnesses, one model,
contradictory behavior => the prompts must differ somewhere the eye
missed. This oracle finds it mechanically:

1. Reconstructs the TRAINING-format prompt for a task text and asserts
   byte-equality with ChatSession.build_prompt's render (prints the
   first divergence + unified diff on mismatch).
2. Runs single-turn generation on BOTH prompt strings, prints raw
   outputs -- same-prompt-different-output would indict generation;
   prompt-diff indicts the harness.
3. Runs the ChatSession drive with raw-output capture.

Usage: python examples/eval_mid_agent_v3.py
"""
import difflib
import json
import os
import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "helioslm_v5" / "agent"))

import random

from chat import CHAT_SYSTEM, ChatSession, ChatTurn
from envs import make_envs
from gate import FixedGate, Route
from helioslm_v5.configs.config_v5 import HeliosLMv5Config
from helioslm_v5.src.model_v5 import HeliosLMv5
from helioslm_v5.src.tokenizer.bpe import BOS, HeliosBPE, build_corpus
from loop import render_tool_docs
from tools import build_default_registry


def main():
    torch.manual_seed(0)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    corpus = build_corpus()
    tok = HeliosBPE.train([corpus[-30000:] + corpus[:170000]], vocab_size=16000)
    bos = tok.vocab[BOS]
    model = HeliosLMv5(HeliosLMv5Config(size="mid")).to(device)
    ck = Path(os.environ.get("HELIOS_CKPT_DIR", "checkpoints"))
    model.load_state_dict(torch.load(ck / "mid_sft_v5.33.pt",
                                     map_location=device))
    model.eval()

    def gen(prompt, max_new=160):
        ids = [bos] + tok.encode(prompt)
        input_ids = torch.tensor([ids]).to(device)
        with torch.no_grad():
            for _ in range(max_new):
                logits, _, _ = model(input_ids,
                                     attention_mask=torch.ones_like(input_ids))
                probs = torch.softmax(logits[0, -1], dim=-1)
                conf, nxt = float(probs.max()), int(probs.argmax())
                if nxt == tok.vocab["<eos>"]:
                    break
                input_ids = torch.cat(
                    [input_ids, torch.tensor([[nxt]]).to(device)], dim=1)
        return tok.decode(input_ids[0, len(ids):].tolist()), conf

    reg, impls = build_default_registry()
    system = CHAT_SYSTEM.replace("%%TOOLS%%", render_tool_docs(reg))
    rng = random.Random(20260928)
    env = make_envs()[0]
    report = {"prompt_equal": [], "gens": []}
    for i in range(3):
        task = env.sample(rng)
        # (a) training-format prompt: system + first user turn
        train_prompt = "\n".join([system, f"##user## {task.text}"])
        # (b) ChatSession render of the same state
        session = ChatSession(lambda p, s, st: "", reg, impls,
                              FixedGate(Route.DIRECT))
        session.turns.append(session.turns.__class__.__mro__ and
                             __import__("chat").ChatTurn("##user##", task.text))
        cs_prompt = session.build_prompt()
        equal = train_prompt == cs_prompt
        report["prompt_equal"].append(equal)
        print(f"[task {i}] prompt_equal={equal}", flush=True)
        if not equal:
            for k, (a, b) in enumerate(zip(train_prompt, cs_prompt)):
                if a != b:
                    print(f"  first divergence at char {k}: "
                          f"{train_prompt[max(0,k-30):k+30]!r} vs "
                          f"{cs_prompt[max(0,k-30):k+30]!r}", flush=True)
                    break
            print("\n".join(difflib.unified_diff(
                train_prompt.splitlines(), cs_prompt.splitlines(),
                "train", "chatsession", lineterm="")), flush=True)
        out_a, conf_a = gen(train_prompt)
        out_b, conf_b = gen(cs_prompt)
        print(f"  gen(train_prompt): {out_a[:120]!r} conf={conf_a:.3f}",
              flush=True)
        print(f"  gen(cs_prompt)   : {out_b[:120]!f}" if False else
              f"  gen(cs_prompt)   : {out_b[:120]!r} conf={conf_b:.3f}",
              flush=True)
        report["gens"].append({"task": task.text[:80],
                               "train_out": out_a[:200], "cs_out": out_b[:200]})
    out = ck / "mid_agent_eval_v3.json"
    out.write_text(json.dumps(report, indent=2))
    print("wrote", out, flush=True)


if __name__ == "__main__":
    main()
