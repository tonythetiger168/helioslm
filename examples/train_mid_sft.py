"""train_mid_sft.py - v5.33: mid (360M, BPE) SFT on local GPU.

One-command local run:
    python3 examples/train_mid_sft.py            # GPU (bf16 autocast)
    CUDA_VISIBLE_DEVICES= python3 examples/train_mid_sft.py   # CPU smoke

Pipeline: gen_chat_episode data -> HeliosBPE tokenize -> mid config ->
AdamW + cosine + grad clip (v5.30.2 discipline: resume via .step,
HELIOS_CKPT_DIR escape hatch, _hf_sync every save). Prints the v5.33
acceptance metrics on a stratified eval split:
    tool parse rate (>0.8 target) | mode-choice text AND tool (>0.6)
    | exact-match reference | params/vocab/RAM summary
The script is self-contained: tokenizer trains from build_corpus() at
startup (~1 min CPU), dataset generates from seeded env episodes.
"""
import json
import math
import os
import random
import sys
import time
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from helioslm_v5.agent.finetune_data import build_chat_dataset
from helioslm_v5.agent.schema import ToolCallError, parse_chat_turn
from helioslm_v5.configs.config_v5 import HeliosLMv5Config
from helioslm_v5.src.model_v5 import HeliosLMv5
from helioslm_v5.src.tokenizer.bpe import BOS, EOS, HeliosBPE, build_corpus

EPISODES_PER_ENV = int(os.environ.get("MID_EPISODES", "400"))
EVAL_PER_TYPE = int(os.environ.get("MID_EVAL", "20"))
EPOCHS = int(os.environ.get("MID_EPOCHS", "4"))
BATCH = int(os.environ.get("MID_BATCH", "8"))
LR = 2e-4
SEED = 7
MAXLEN = 640
DATA_PATH = "/tmp/mid_sft_v533.jsonl"
OUT_NAME = "mid_sft_v5.33"


def _hf_sync(path, repo_path):
    if os.environ.get("HELIOS_HF_SYNC") != "1":
        return
    import subprocess as sp
    try:
        sp.Popen(["python3", "/tmp/hf_upload_lfs.py", str(path), repo_path],
                 stdout=open("hf_sync.log", "a"), stderr=sp.STDOUT,
                 start_new_session=True)
    except Exception as e:
        print(f"  [hf-sync] start failed: {e!r}", flush=True)


def gen_dataset(tok):
    build_chat_dataset(DATA_PATH, n_per_env=EPISODES_PER_ENV, seed=SEED)
    tool, text = [], []
    with open(DATA_PATH, encoding="utf-8") as f:
        for line in f:
            d = json.loads(line)
            p = tok.encode(d["prompt"])
            if len(p) > MAXLEN - 8:
                continue
            r = tok.encode(d["response"])
            (tool if d["response"].strip().startswith("@@tool@@")
             else text).append((p, r))
    rng = random.Random(SEED + 1)
    rng.shuffle(tool)
    rng.shuffle(text)
    evals = tool[:EVAL_PER_TYPE] + text[:EVAL_PER_TYPE]
    train = tool[EVAL_PER_TYPE:] + text[EVAL_PER_TYPE:]
    rng.shuffle(train)
    print(f"dataset: {len(tool)} tool / {len(text)} text "
          f"-> {len(train)} train / {len(evals)} eval", flush=True)
    return train, evals


def collate(batch, pad_id=0):
    maxlen = max(len(p) + len(r) for p, r in batch) + 2
    ids = torch.full((len(batch), maxlen), pad_id, dtype=torch.long)
    labels = torch.full((len(batch), maxlen), -100, dtype=torch.long)
    attn = torch.zeros((len(batch), maxlen), dtype=torch.long)
    for i, (p, r) in enumerate(batch):
        seq = [tok_bos] + p + r + [tok_eos]
        ids[i, :len(seq)] = torch.tensor(seq)
        attn[i, :len(seq)] = 1
        labels[i, len(p) + 1:len(seq)] = torch.tensor(seq[len(p) + 1:])
    return ids, labels, attn


def evaluate(model, tok, evals, device):
    model.eval()
    res = {"exact": 0, "tool_n": 0, "tool_ok": 0, "text_n": 0, "text_ok": 0}
    with torch.no_grad():
        for p, r in evals:
            want = tok.decode(r)
            seq = [tok_bos] + p
            input_ids = torch.tensor([seq]).to(device)
            for _ in range(len(r) + 16):
                logits, _, _ = model(input_ids,
                                     attention_mask=torch.ones_like(input_ids))
                nxt = int(logits[0, -1].argmax())
                if nxt == tok_eos:
                    break
                input_ids = torch.cat(
                    [input_ids, torch.tensor([[nxt]]).to(device)], dim=1)
            got = tok.decode(input_ids[0, len(seq):].tolist())
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
    global tok_bos, tok_eos
    torch.manual_seed(SEED)
    t0 = time.time()
    print("training HeliosBPE tokenizer...", flush=True)
    corpus = build_corpus()
    # pure-python trainer is O(merges x corpus): train on the protocol
    # tail + a 170K source head (domain coverage kept, wall-clock sane;
    # recorded deviation -- a GPU-side run may train on the full corpus)
    tok = HeliosBPE.train([corpus[-30000:] + corpus[:170000]],
                          vocab_size=16000)
    tok_bos, tok_eos = tok.vocab[BOS], tok.vocab[EOS]
    train, evals = gen_dataset(tok)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"device: {device}", flush=True)
    cfg = HeliosLMv5Config(size="mid")
    model = HeliosLMv5(cfg).to(device)
    n_params = sum(p.numel() for p in model.parameters())
    print(f"mid params: {n_params/1e6:.1f}M | vocab {cfg.vocab_size}",
          flush=True)

    out_dir = Path(os.environ.get("HELIOS_CKPT_DIR", "checkpoints"))
    out_dir.mkdir(exist_ok=True, parents=True)
    ckpt = out_dir / f"{OUT_NAME}.pt"
    # tokenizer travels WITH the checkpoint: eval processes must load
    # this file, never retrain (tokenizer train/serve skew, 2026-09-28)
    tok_path = out_dir / f"{OUT_NAME}.tok.json"
    tok.save(tok_path)
    # RESUME GUARD (found 2026-09-29): resuming old weights against a
    # newly trained tokenizer silently corrupts the model (88 mismatch
    # steps drove loss 0.47 -> 15.0). Fingerprint the tokenizer; refuse
    # to resume on any difference -- loud, never silent.
    import hashlib
    tok_fp = hashlib.sha256(tok_path.read_bytes()).hexdigest()
    fp_file = out_dir / f"{OUT_NAME}.tokfp"
    step_file = out_dir / f"{OUT_NAME}.step"
    # Legacy-hole fix (found 2026-09-29, second corruption): checkpoints
    # saved BEFORE the .tokfp mechanism have no fingerprint file, so the
    # check below silently passed and re-corrupted the model again. Also
    # compare against the checkpoint's OWN saved .tok.json whenever it
    # exists -- no trust in sidecar freshness.
    if ckpt.exists():
        legacy_tok = out_dir / f"{OUT_NAME}.tok.json"
        if legacy_tok.exists() \
                and hashlib.sha256(legacy_tok.read_bytes()).hexdigest() != tok_fp:
            sys.exit("REFUSING to resume: this checkpoint's saved tokenizer "
                     "differs from the one just trained. Run with "
                     "MID_FRESH=1 to start clean.")
        if step_file.exists() and fp_file.exists() \
                and fp_file.read_text() != tok_fp:
            sys.exit("REFUSING to resume: tokenizer fingerprint differs "
                     "from the checkpoint's training tokenizer. Run with "
                     "MID_FRESH=1 to start clean.")
    fp_file.write_text(tok_fp)
    # MID_FRESH=1: hard-ignore any resume state. Belt-and-braces for
    # exactly the confusion seen 2026-09-29 (a corrupted checkpoint was
    # re-evaluated repeatedly because 'delete the old files first' is a
    # silent prerequisite). Fresh runs should ALWAYS use MID_FRESH=1.
    if os.environ.get("MID_FRESH") == "1":
        start_step = 0
        for stale in (ckpt, step_file):
            if stale.exists():
                stale.unlink()
        print("MID_FRESH=1: cleared resume state, starting from step 0",
              flush=True)
    else:
        start_step = int(step_file.read_text()) if (ckpt.exists()
                                                    and step_file.exists()) else 0
    if start_step:
        model.load_state_dict(torch.load(ckpt, map_location=device))
        print(f"RESUME from step {start_step}", flush=True)
    opt = torch.optim.AdamW(model.parameters(), lr=LR)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(
        opt, T_max=EPOCHS * math.ceil(len(train) / BATCH))
    if start_step:
        for _ in range(start_step):
            sched.step()

    step = 0
    for ep in range(EPOCHS):
        rng = random.Random(SEED + ep)
        order = list(range(len(train)))
        rng.shuffle(order)
        tot, nb = 0.0, 0
        for b0 in range(0, len(order), BATCH):
            if start_step and step < start_step:
                step += 1
                continue
            chunk = [train[i] for i in order[b0:b0 + BATCH]]
            ids, labels, attn = collate(chunk)
            ids, labels, attn = ids.to(device), labels.to(device), attn.to(device)
            with torch.autocast(device_type=device,
                                dtype=torch.bfloat16,
                                enabled=(device == "cuda")):
                logits, _, _ = model(ids, attention_mask=attn)
                loss = torch.nn.functional.cross_entropy(
                    logits[:, :-1].reshape(-1, logits.size(-1)),
                    labels[:, 1:].reshape(-1), ignore_index=-100)
            opt.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
            sched.step()
            tot += loss.item()
            nb += 1
            step += 1
            if step % 100 == 0:
                torch.save(model.state_dict(), ckpt)
                step_file.write_text(str(step))
                _hf_sync(ckpt, f"checkpoints/{OUT_NAME}.pt")
                print(f"  step {step} ep{ep} loss {tot/nb:.4f} "
                      f"({time.time()-t0:.0f}s) [saved]", flush=True)
    print(f"train done in {time.time()-t0:.0f}s, final loss {tot/nb:.4f}",
          flush=True)
    torch.save(model.state_dict(), ckpt)
    _hf_sync(ckpt, f"checkpoints/{OUT_NAME}.pt")

    res = evaluate(model, tok, evals, device)
    mode_all = res["tool_ok"] + res["text_ok"]
    acc = {"params": n_params, "train_samples": len(train),
           "device": device,
           "eval_exact": f"{res['exact']}/{len(evals)}",
           "eval_mode_tool": f"{res['tool_ok']}/{res['tool_n']}",
           "eval_mode_text": f"{res['text_ok']}/{res['text_n']}",
           "eval_mode_choice": f"{mode_all}/{len(evals)}",
           "seed": SEED, "version": "v5.33",
           "acceptance": {"tool_parse_target": ">0.8",
                          "mode_choice_target": "text AND tool >0.6"}}
    (out_dir / f"{OUT_NAME}.json").write_text(json.dumps(acc, indent=2))
    print(json.dumps(acc, indent=2), flush=True)


if __name__ == "__main__":
    main()
