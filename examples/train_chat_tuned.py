"""train_chat_tuned.py - v5.30: chat SFT on top of the tool-tuned checkpoint.

Data: finetune_data.build_chat_dataset — episodes mixing env tool tasks,
follow-up questions (transcript memory), and direct text answers. Prompts
are the EXACT ChatSession.build_prompt render at inference, so training
format == inference format by construction (same discipline as v5.27).

Init: hot-start from checkpoints/tool_tuned_v5.27.pt so tool ability is
preserved; chat capability is additive. Lower LR than the from-scratch
tool stage.

Output: checkpoints/chat_tuned_v5.30.pt (state_dict) + .json summary.
Resume: checkpoint + .step file saved every 50 steps (sandbox kill-safe).

Run from repo root:  python3 examples/train_chat_tuned.py
"""
import json
import math
import random
import sys
import time
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from helioslm_v5.agent.finetune_data import build_chat_dataset
from helioslm_v5.configs.config_v5 import HeliosLMv5Config
from helioslm_v5.src.model_v5 import HeliosLMv5

sys.path.insert(0, str(Path(__file__).resolve().parent))
from train_tool_tuned import BOS, EOS, PAD, MAX_CHAR, collate, encode, generate

EPISODES_PER_ENV = 400
EVAL_PER_TYPE = 20   # stratified: 20 tool-target + 20 text-target evals
EPOCHS = 2
BATCH = 8
LR = 5e-4            # hot-start: half of the from-scratch tool stage
SEED = 6
DATA_PATH = "/tmp/chat_sft_v5302.jsonl"
PROMPT_CAP = 1500    # matches generate()'s [-1500:] tail window; v5.30.2 fix:
                     # the old <700 total filter silently dropped the
                     # magic-word text samples (longest transcripts),
                     # leaving the mode-choice eval with zero text targets


def gen_dataset():
    build_chat_dataset(DATA_PATH, n_per_env=EPISODES_PER_ENV, seed=SEED)
    tool, text = [], []
    with open(DATA_PATH, encoding="utf-8") as f:
        for line in f:
            d = json.loads(line)
            if len(d["prompt"]) <= PROMPT_CAP:
                (tool if d["response"].strip().startswith("@@tool@@")
                 else text).append((d["prompt"], d["response"]))
    rng = random.Random(SEED + 1)
    rng.shuffle(tool)
    rng.shuffle(text)
    evals = tool[:EVAL_PER_TYPE] + text[:EVAL_PER_TYPE]
    train = tool[EVAL_PER_TYPE:] + text[EVAL_PER_TYPE:]
    rng.shuffle(train)
    print(f"dataset: {len(tool)} tool / {len(text)} text "
          f"(kept {len(tool)+len(text)} of built)", flush=True)
    return train, evals


def eval_exact_and_modes(model, evals):
    """Exact-match plus the chat-specific mode-choice metric, per type.

    mode_choice: for eval samples whose target is a tool block, the
    generation must parse as a tool block; for text targets, the
    generation must be marker-free. This is the v5.30 capability T20
    proves at the protocol level; here it is measured on real weights.
    generate() is greedy (argmax), so failures are the model's, not
    sampling noise.
    """
    from helioslm_v5.agent.schema import (ToolCallError, parse_chat_turn)
    res = {"exact": 0, "tool_n": 0, "tool_ok": 0, "text_n": 0, "text_ok": 0}
    for prompt, want in evals:
        got = generate(model, prompt, max_new=len(want) + 20)
        res["exact"] += got.strip() == want.strip()
        if want.strip().startswith("@@tool@@"):
            res["tool_n"] += 1
            try:
                parse_chat_turn(got, None)
                res["tool_ok"] += 1
            except ToolCallError:
                pass
        else:
            res["text_n"] += 1
            res["text_ok"] += ("@@tool@@" not in got
                               and "@@end@@" not in got)
    return res


def main():
    torch.manual_seed(SEED)
    torch.set_num_threads(2)
    train, evals = gen_dataset()
    print(f"split: {len(train)} train / {len(evals)} eval "
          f"(20 tool + 20 text)", flush=True)

    cfg = HeliosLMv5Config(size="lite")
    model = HeliosLMv5(cfg)
    out_dir = Path(__file__).resolve().parent.parent / "checkpoints"
    out_dir.mkdir(exist_ok=True)
    ckpt = out_dir / "chat_tuned_v5.30.2.pt"
    init = out_dir / "tool_tuned_v5.27.pt"
    if not init.exists():
        sys.exit(f"hot-start checkpoint missing: {init} — "
                 "run examples/train_tool_tuned.py first")
    model.load_state_dict(torch.load(init, map_location="cpu"))
    print(f"hot-start from {init}", flush=True)
    opt = torch.optim.AdamW(model.parameters(), lr=LR)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(
        opt, T_max=EPOCHS * math.ceil(len(train) / BATCH))
    model.train()
    t0 = time.time()
    step = 0
    for ep in range(EPOCHS):
        rng = random.Random(SEED + ep)
        order = list(range(len(train)))
        rng.shuffle(order)
        tot_loss, nb = 0.0, 0
        for b0 in range(0, len(order), BATCH):
            chunk = [encode(*train[i]) for i in order[b0:b0 + BATCH]]
            input_ids, labels, attn = collate(chunk)
            logits, _, _ = model(input_ids, attention_mask=attn)
            loss = torch.nn.functional.cross_entropy(
                logits[:, :-1].reshape(-1, logits.size(-1)),
                labels[:, 1:].reshape(-1), ignore_index=-100)
            opt.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
            sched.step()
            tot_loss += loss.item()
            nb += 1
            step += 1
            if step % 50 == 0:
                torch.save(model.state_dict(), ckpt)
                (out_dir / "chat_tuned_v5.30.2.step").write_text(str(step))
                print(f"  step {step} ep{ep} loss {tot_loss/nb:.4f} "
                      f"({time.time()-t0:.0f}s) [saved]", flush=True)
    print(f"train done in {time.time()-t0:.0f}s, final loss {tot_loss/nb:.4f}",
          flush=True)

    torch.save(model.state_dict(), ckpt)
    model.eval()
    r = eval_exact_and_modes(model, evals)
    mode_all = r["tool_ok"] + r["text_ok"]
    print(f"eval exact-match: {r['exact']}/{len(evals)}", flush=True)
    print(f"eval mode-choice: {mode_all}/{len(evals)} "
          f"(tool {r['tool_ok']}/{r['tool_n']}, text {r['text_ok']}/{r['text_n']})",
          flush=True)

    meta = {"arch": "helioslm_v5_lite", "params": sum(p.numel()
            for p in model.parameters()), "vocab_size": cfg.vocab_size,
            "tokenizer": "char(ord<1024); BOS=1023 EOS=1022 PAD=1021",
            "init_from": "tool_tuned_v5.27.pt (hot-start)",
            "train_samples": len(train),
            "eval_exact": f"{r['exact']}/{len(evals)}",
            "eval_mode_choice": f"{mode_all}/{len(evals)}",
            "eval_mode_tool": f"{r['tool_ok']}/{r['tool_n']}",
            "eval_mode_text": f"{r['text_ok']}/{r['text_n']}",
            "seed": SEED, "version": "v5.30.2"}
    (out_dir / "chat_tuned_v5.30.2.json").write_text(json.dumps(meta, indent=2))
    print("saved:", ckpt, flush=True)


if __name__ == "__main__":
    main()
