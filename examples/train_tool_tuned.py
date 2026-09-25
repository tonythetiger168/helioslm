"""train_tool_tuned.py - Stage B: train a tool-tuned HeliosLMv5 checkpoint.

Data: replay the v5.23 agent loop with the scripted correct policy as the
model (AgentLoop records EXACT prompt/response text in trajectory steps, so
training format == inference format by construction). Envs: calc/str/compose.

Tokenizer: char-level. finetune_data guarantees ord(c) < 1024 for all
prompt/response chars. Control ids live at the top of vocab:
    BOS = 1023, EOS = 1022, PAD = 1021  (never appear in printable data)

Output: checkpoints/tool_tuned_v5.27.pt (state_dict) + .json (config/summary)

Run from repo root:  python3 examples/train_tool_tuned.py
"""
import json
import math
import random
import sys
import time
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from helioslm_v5.agent.benchmark import scripted_correct_policy
from helioslm_v5.agent.envs import make_envs
from helioslm_v5.agent.gate import FixedGate, Route
from helioslm_v5.agent.loop import AgentLoop
from helioslm_v5.agent.tools import build_default_registry
from helioslm_v5.agent.trajectory import ids_to_text
from helioslm_v5.configs.config_v5 import HeliosLMv5Config
from helioslm_v5.src.model_v5 import HeliosLMv5

BOS, EOS, PAD = 1023, 1022, 1021
MAX_CHAR = 1021          # logits masked to ids < MAX_CHAR at inference
EPISODES_PER_ENV = 400
EVAL_EPISODES = 40
EPOCHS = 2
BATCH = 8
LR = 1e-3
SEED = 5


def gen_dataset():
    """Replay the agent loop with the scripted policy; emit (prompt, response)."""
    import tempfile
    samples = []
    with tempfile.TemporaryDirectory() as tmp:
        reg, impls = build_default_registry(tmp)
        loop = AgentLoop(scripted_correct_policy, reg, impls,
                         FixedGate(Route.DIRECT), max_steps=8)
        for env in make_envs():
            for i in range(EPISODES_PER_ENV):
                rng = random.Random(f"{SEED}:{env.__class__.__name__}:{i}")
                task = env.sample(rng)
                traj = loop.run(task.text, seed=i)
                for s in traj.steps:
                    samples.append((ids_to_text(s.prompt_ids),
                                    ids_to_text(s.generated_ids)))
    samples = [(p_, r_) for p_, r_ in samples
               if len(p_) + len(r_) < 700]
    rng = random.Random(SEED + 1)
    rng.shuffle(samples)
    return samples


def encode(prompt, response):
    for ch in prompt + response:
        assert ord(ch) < 1021, \
            f"char {ch!r} (ord={ord(ch)}) out of tokenizer range"
    ids = [BOS] + [ord(c) for c in prompt] + [ord(c) for c in response] + [EOS]
    labels = [-100] * (1 + len(prompt)) + [ord(c) for c in response] + [EOS]
    return ids, labels


def collate(batch):
    n = max(len(x[0]) for x in batch)
    input_ids = torch.full((len(batch), n), PAD, dtype=torch.long)
    labels = torch.full((len(batch), n), -100, dtype=torch.long)
    attn = torch.zeros((len(batch), n), dtype=torch.long)
    for i, (ids, lab) in enumerate(batch):
        input_ids[i, :len(ids)] = torch.tensor(ids)
        labels[i, :len(lab)] = torch.tensor(lab)
        attn[i, :len(ids)] = 1
    return input_ids, labels, attn


@torch.no_grad()
def generate(model, prompt, max_new=200):
    ids = torch.tensor([[BOS] + [ord(c) for c in prompt][-1500:]])
    for _ in range(max_new):
        logits, _, _ = model(ids)
        nxt = logits[0, -1, :MAX_CHAR].argmax(-1, keepdim=True)
        if int(nxt) == EOS:
            break
        ids = torch.cat([ids, nxt.view(1, 1)], dim=1)
    out = [int(t) for t in ids[0]]
    return "".join(chr(t) for t in out[1 + len(prompt[-1500:]):] if t < MAX_CHAR)


def main():
    torch.manual_seed(SEED)
    torch.set_num_threads(2)
    samples = gen_dataset()
    split = len(samples) - EVAL_EPISODES
    train, evals = samples[:split], samples[split:]
    print(f"dataset: {len(train)} train / {len(evals)} eval", flush=True)

    cfg = HeliosLMv5Config(size="lite")
    model = HeliosLMv5(cfg)
    out_dir = Path(__file__).resolve().parent.parent / "checkpoints"
    out_dir.mkdir(exist_ok=True)
    ckpt = out_dir / "tool_tuned_v5.27.pt"
    start_step = 0
    if ckpt.exists():
        model.load_state_dict(torch.load(ckpt, map_location="cpu"))
        start_step = int((out_dir / "tool_tuned_v5.27.step").read_text().strip()) \
            if (out_dir / "tool_tuned_v5.27.step").exists() else 0
        print(f"resumed from {ckpt} at step {start_step}", flush=True)
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
                (out_dir / "tool_tuned_v5.27.step").write_text(str(step))
                print(f"  step {step} ep{ep} loss {tot_loss/nb:.4f} "
                      f"({time.time()-t0:.0f}s) [saved]", flush=True)
    print(f"train done in {time.time()-t0:.0f}s, final loss {tot_loss/nb:.4f}",
          flush=True)

    torch.save(model.state_dict(), ckpt)
    print("training done, checkpoint at:", ckpt, flush=True)

    model.eval()
    exact = 0
    for prompt, want in evals:
        got = generate(model, prompt, max_new=len(want) + 20)
        exact += got.strip() == want.strip()
    print(f"eval exact-match: {exact}/{len(evals)}", flush=True)

    meta = {"arch": "helioslm_v5_lite", "params": sum(p.numel()
            for p in model.parameters()), "vocab_size": cfg.vocab_size,
            "tokenizer": "char(ord<1024); BOS=1023 EOS=1022 PAD=1021",
            "train_samples": len(train), "eval_exact": f"{exact}/{len(evals)}",
            "seed": SEED, "version": "v5.27"}
    (out_dir / "tool_tuned_v5.27.json").write_text(json.dumps(meta, indent=2))
    print("saved:", ckpt, flush=True)


if __name__ == "__main__":
    main()
