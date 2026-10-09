"""train_mid_sft_mix.py - v1.29: mixed SFT for open-domain chat.

Mid 360M is trained on toy agent/chat -- it cannot converse freely.
This script adds Alpaca/Dolly-style open-domain data to the mix so mid
becomes a general chat assistant while keeping its tool ability.

Data sources (auto-download, no manual prep):
  - databricks/databricks-dolly-15k (Alpaca-style, ~15k samples)
  - existing chat SFT data from train_mid_sft.py

Usage (local, ~1h on 4060):
  python examples/train_mid_sft_mix.py
"""
import json
import os
import random
import sys
import time
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

# reuse the training backbone from train_mid_sft
from examples.train_mid_sft import (BATCH, EPISODES_PER_ENV, LR, MAXLEN,
                                    SEED, collate, gen_dataset)
from helioslm_v5.configs.config_v5 import HeliosLMv5Config
from helioslm_v5.src.model_v5 import HeliosLMv5
from helioslm_v5.src.tokenizer.bpe import BOS, HeliosBPE, build_corpus

OPEN_PATH = "/tmp/mid_open_dolly.jsonl"
OUT_NAME = "mid_sft_v5.29_mix"


def load_open_domain():
    """Download databricks-dolly-15k and convert to chat SFT format."""
    cache = Path(OPEN_PATH)
    if not cache.exists():
        print("downloading databricks-dolly-15k...", flush=True)
        import urllib.request
        url = ("https://huggingface.co/datasets/databricks/databricks-"
               "dolly-15k/resolve/main/databricks-dolly-15k.jsonl")
        req = urllib.request.Request(url, headers={
            "User-Agent": "Mozilla/5.0"})
        data = urllib.request.urlopen(req, timeout=120).read()
        cache.write_bytes(data)
    samples = []
    with open(cache, encoding="utf-8") as f:
        for line in f:
            d = json.loads(line)
            instr = d.get("instruction", "").strip()
            resp = d.get("response", "").strip()
            ctx = d.get("context", "").strip()
            if not instr or not resp:
                continue
            if ctx:
                instr = f"{ctx}\n\n{instr}"
            # format: simple user/assistant (chat SFT format)
            prompt = f"##user## {instr}"
            response = f"##assistant## {resp}"
            samples.append({"prompt": prompt, "response": response})
    return samples


def main():
    torch.manual_seed(SEED)
    corpus = build_corpus()
    tok = HeliosBPE.train([corpus[-30000:] + corpus[:170000]],
                          vocab_size=16000)
    tok_bos = tok.vocab[BOS]

    # existing chat SFT data
    chat_train, chat_eval = gen_dataset(tok)

    # open-domain data
    open_samples = load_open_domain()
    rng = random.Random(SEED + 99)
    rng.shuffle(open_samples)
    # encode open samples
    open_encoded = []
    for s in open_samples[:3000]:   # cap at 3k open samples
        p = tok.encode(s["prompt"])
        if len(p) > MAXLEN - 8:
            continue
        r = tok.encode(s["response"])
        open_encoded.append((p, r))
    print(f"open-domain: {len(open_encoded)} samples", flush=True)

    # merge: chat (toy) + open (dolly)
    train = chat_train + open_encoded
    rng.shuffle(train)
    print(f"total train: {len(train)}", flush=True)

    out_dir = Path(os.environ.get("HELIOS_CKPT_DIR", "checkpoints"))
    out_dir.mkdir(exist_ok=True, parents=True)
    ckpt = out_dir / f"{OUT_NAME}.pt"
    model = HeliosLMv5(HeliosLMv5Config(size="mid"))
    # hot start from existing mid
    init = out_dir / "mid_sft_v5.33.pt"
    if init.exists():
        model.load_state_dict(torch.load(init, map_location="cpu"))
        print("hot-start from mid_sft_v5.33.pt", flush=True)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = model.to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=LR, foreach=False)
    import math
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(
        opt, T_max=2 * math.ceil(len(train) / BATCH))

    model.train()
    step = 0
    t0 = time.time()
    for ep in range(2):
        rng2 = random.Random(SEED + ep)
        order = list(range(len(train)))
        rng2.shuffle(order)
        for b0 in range(0, len(order), BATCH):
            chunk = [train[i] for i in order[b0:b0 + BATCH]]
            ids, labels, attn = collate(chunk, pad_id=tok_bos)
            ids, labels, attn = (t.to(device) for t in (ids, labels, attn))
            logits, _, _ = model(ids, attention_mask=attn)
            loss = torch.nn.functional.cross_entropy(
                logits[:, :-1].reshape(-1, logits.size(-1)),
                labels[:, 1:].reshape(-1), ignore_index=-100)
            opt.zero_grad(); loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step(); sched.step()
            step += 1
            if step % 100 == 0:
                torch.save(model.state_dict(), ckpt)
                print(f"  step {step} loss {loss.item():.4f} "
                      f"({time.time()-t0:.0f}s)", flush=True)
    print(f"done in {time.time()-t0:.0f}s", flush=True)
    torch.save(model.state_dict(), ckpt)
    print("saved:", ckpt, flush=True)


if __name__ == "__main__":
    main()
