"""Train a reproducible BPE toy checkpoint for Issue #5.

Requires: pip install tokenizers torch
Example: python examples/train_toy_bpe.py --steps 400
Artifacts: checkpoints/toy_bpe.pt, checkpoints/toy_bpe.tokenizer.json,
           checkpoints/toy_bpe.metrics.json

Tokenizer is trained on the training split only. Held-out text is not
included in vocabulary learning. Char and BPE token losses are NOT directly
comparable; report per-byte NLL for an apples-to-apples text likelihood.

--tokenizer char trains the character baseline through the same split,
loop and evaluation (token id == ord(char)); the char vocabulary only
holds characters below the model vocabulary, so --max-ord drops the rest
from BOTH splits. Pass the same --max-ord to the BPE run to compare the two
on identical text (examples/toy_tokenizer_compare.py does this).
"""
import argparse
import hashlib
import json
from pathlib import Path
import random
import re
import sys
import time

import torch
import torch.nn.functional as F

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from evastar.configs.config_v5 import EvastarV5Config
from evastar.src.model_v5 import EvastarV5


def corpus_files():
    excluded = {".git", "__pycache__", "checkpoints", ".venv", "venv", "node_modules"}
    return sorted(p for p in ROOT.rglob("*") if p.suffix in (".py", ".md")
                  and not any(part in excluded for part in p.relative_to(ROOT).parts)
                  and p.is_file())


def split_corpus(seed, max_ord=None):
    """File-level train / validation split. max_ord drops every character
    with ord >= max_ord from both splits (the char baseline's vocabulary)."""
    paths = corpus_files()
    if len(paths) < 3:
        raise ValueError("Need at least three source files for train/validation split")
    rng = random.Random(seed)
    rng.shuffle(paths)
    n_val = max(1, len(paths) // 10)
    train_paths, val_paths = paths[n_val:], paths[:n_val]
    def read(items):
        text = "\n\n".join(p.read_text(encoding="utf-8", errors="replace") for p in items)
        if max_ord is not None:
            text = "".join(c for c in text if ord(c) < max_ord)
        return text
    return read(train_paths), read(val_paths), [str(p.relative_to(ROOT)) for p in val_paths]


# Smallest vocabulary that can hold the seed at all (measured: seed alone
# plus a little English needs 6030 / 4359 / 7357 tokens). The seed's merges
# have the lowest frequency, so next to a real corpus they are learned last
# and need room beyond the corpus's own merges; the reported
# zh_seed_single_token_coverage says whether they fit (~32k does for the
# repo corpus).
ZH_SEED_MIN_VOCAB = {"tw": 6100, "cn": 4400, "both": 7400}


def train_tokenizer(train_text, vocab_size, zh_seed=None):
    """zh_seed (v5.80): also learn the common Chinese characters as single
    tokens (evastar/eval/zh_tokenizer_coverage.py; 3 -> 1 token per
    character for text the corpus barely covers): "tw" = 5401 Big5
    Traditional, "cn" = 3755 GB2312 Simplified, "both" = their 6722-
    character union. A vocabulary too small to hold the seed is refused.
    """
    from tokenizers import Tokenizer, models, pre_tokenizers, decoders, trainers
    if zh_seed:
        need = ZH_SEED_MIN_VOCAB.get(zh_seed)
        if need is None:
            raise ValueError(f"--zh-seed must be tw, cn or both, got {zh_seed!r}")
        if vocab_size < need:
            raise ValueError(f"--zh-seed {zh_seed} needs --vocab-size >= "
                             f"{need} (the characters plus their byte merges)")
        from evastar.eval.zh_tokenizer_coverage import zh_seed_text
        train_text = train_text + "\n" + zh_seed_text(zh_seed)
    tokenizer = Tokenizer(models.BPE(unk_token="[UNK]"))
    tokenizer.pre_tokenizer = pre_tokenizers.ByteLevel(add_prefix_space=False)
    tokenizer.decoder = decoders.ByteLevel()
    trainer = trainers.BpeTrainer(vocab_size=vocab_size, min_frequency=2,
                                 initial_alphabet=pre_tokenizers.ByteLevel.alphabet(),
                                 special_tokens=["[UNK]"], show_progress=False)
    tokenizer.train_from_iterator([train_text], trainer=trainer)
    if tokenizer.get_vocab_size() > vocab_size:
        raise ValueError("Tokenizer exceeds model embedding budget")
    return tokenizer


class CharTokenizer:
    """The toy checkpoint's char vocabulary (id == ord(char)) behind the
    tokenizers.Tokenizer calls this script uses."""

    class _Encoding:
        def __init__(self, ids):
            self.ids = ids

    def __init__(self, vocab_size):
        self.vocab_size = vocab_size

    def encode(self, text):
        bad = next((c for c in text if ord(c) >= self.vocab_size), None)
        if bad is not None:
            raise ValueError(f"character {bad!r} is outside the char vocabulary "
                             f"(ord >= {self.vocab_size}); filter with max_ord")
        return self._Encoding([ord(c) for c in text])

    def decode(self, ids, skip_special_tokens=False):
        return "".join(chr(int(i)) for i in ids)

    def get_vocab_size(self):
        return self.vocab_size

    def save(self, path):
        Path(path).write_text(json.dumps({"type": "char", "vocab_size": self.vocab_size}),
                              encoding="utf-8")


WORD = re.compile(r"[A-Za-z]{3,}")


def real_word_rate(sample, train_text):
    """Fraction of the sample's alphabetic words (3+ letters) that occur as
    words in the training text: 0 for gibberish, 1 if every word is real."""
    words = WORD.findall(sample)
    if not words:
        return 0.0
    vocab = set(WORD.findall(train_text))
    return sum(w in vocab for w in words) / len(words)


def train_model(ids, val_ids, cfg, steps, batch, seq, lr, seed):
    torch.manual_seed(seed)
    model = EvastarV5(cfg)
    model.train()
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    gen = torch.Generator().manual_seed(seed)
    if len(ids) < seq + 2 or len(val_ids) < 3:
        raise ValueError("Insufficient train or validation tokens")
    start = time.perf_counter()
    for step in range(steps):
        ix = torch.randint(0, len(ids) - seq - 1, (batch,), generator=gen)
        data = torch.stack([ids[i:i+seq+1] for i in ix])
        logits, _, _ = model(data[:, :-1])
        loss = F.cross_entropy(logits.reshape(-1, logits.size(-1)), data[:, 1:].reshape(-1))
        opt.zero_grad()
        loss.backward()
        opt.step()
        if step % 50 == 0 or step == steps-1:
            print(f"step={step} loss={loss.item():.4f}", flush=True)
    model.eval()
    # Chunked evaluation with context resets; all tokens except first of each
    # chunk are scored. Byte normalization makes tokenizer comparisons valid.
    total_nll, predicted = 0.0, 0
    with torch.inference_mode():
        for offset in range(0, len(val_ids)-1, seq):
            chunk = val_ids[offset:offset+seq+1]
            if len(chunk) < 2:
                continue
            logits, _, _ = model(chunk[:-1].unsqueeze(0))
            total_nll += F.cross_entropy(logits.reshape(-1, logits.size(-1)),
                                        chunk[1:], reduction="sum").item()
            predicted += len(chunk)-1
    return model, {"val_nll": total_nll, "val_tokens": predicted,
                   "val_loss_per_token": total_nll / predicted,
                   "train_seconds": round(time.perf_counter()-start, 2)}


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--steps", type=int, default=400)
    ap.add_argument("--batch", type=int, default=8)
    ap.add_argument("--seq", type=int, default=64)
    ap.add_argument("--lr", type=float, default=3e-3)
    ap.add_argument("--seed", type=int, default=13)
    ap.add_argument("--out", default="checkpoints/toy_bpe.pt")
    ap.add_argument("--vocab-size", type=int, default=None,
                    help="tokenizer/model vocabulary (default: lite config)")
    ap.add_argument("--tokenizer", choices=["bpe", "char"], default="bpe")
    ap.add_argument("--max-ord", type=int, default=None,
                    help="drop characters with ord >= N from both splits "
                         "(char: defaults to the model vocabulary)")
    ap.add_argument("--zh-seed", choices=["tw", "cn", "both"],
                    help="v5.80: common Chinese characters as single tokens "
                         "(tw = Big5 Traditional, cn = GB2312 Simplified)")
    args = ap.parse_args()
    if min(args.steps, args.batch, args.seq, args.lr) <= 0:
        ap.error("steps, batch, seq and lr must be positive")
    if args.tokenizer == "char" and args.zh_seed:
        ap.error("--zh-seed applies to the BPE tokenizer only")
    print(json.dumps(run(args), indent=2))


def run(args):
    """One training run; returns the metrics (also written next to --out)."""
    cfg = EvastarV5Config(size="lite")
    if args.vocab_size is not None:
        cfg.vocab_size = args.vocab_size
    max_ord = args.max_ord
    if args.tokenizer == "char":
        if max_ord is None:
            max_ord = cfg.vocab_size
        if max_ord > cfg.vocab_size:
            raise ValueError("--max-ord cannot exceed the char vocabulary")
    train_text, val_text, val_paths = split_corpus(args.seed, max_ord)
    if args.tokenizer == "char":
        tok = CharTokenizer(cfg.vocab_size)
    else:
        tok = train_tokenizer(train_text, cfg.vocab_size, args.zh_seed)
    train_ids = torch.tensor(tok.encode(train_text).ids, dtype=torch.long)
    val_ids = torch.tensor(tok.encode(val_text).ids, dtype=torch.long)
    model, metrics = train_model(train_ids, val_ids, cfg, args.steps,
                                 args.batch, args.seq, args.lr, args.seed)
    if args.zh_seed:
        from evastar.eval.zh_tokenizer_coverage import Bpe, zh_seed_chars
        bpe = Bpe(tok)
        chars = zh_seed_chars(args.zh_seed)
        metrics["zh_seed"] = args.zh_seed
        metrics["zh_seed_single_token_coverage"] = round(
            sum(bpe.covers(c) for c in chars) / len(chars), 4)
    kind = "char" if args.tokenizer == "char" else "bytelevel-bpe"
    metrics.update({"tokenizer": kind, "seed": args.seed, "steps": args.steps,
                    "batch": args.batch, "seq": args.seq, "lr": args.lr, "max_ord": max_ord,
                    "train_utf8_bytes": len(train_text.encode("utf-8")),
                    "train_tokens": len(train_ids),
                    "vocab_size": tok.get_vocab_size(),
                    "val_utf8_bytes": len(val_text.encode("utf-8")),
                    "val_nll_per_utf8_byte": metrics["val_nll"] / len(val_text.encode("utf-8")),
                    "val_files": val_paths,
                    "val_text_sha256": hashlib.sha256(val_text.encode("utf-8")).hexdigest()})
    with torch.inference_mode():
        prompt = "HeliosLM"
        pids = torch.tensor([tok.encode(prompt).ids])
        generated = model.generate(pids, max_new_tokens=80, temperature=0)[0].tolist()
        metrics["sample"] = tok.decode(generated, skip_special_tokens=False)
        metrics["sample_real_word_rate"] = round(
            real_word_rate(metrics["sample"][len(prompt):], train_text), 4)
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    tokenizer_path = out.with_suffix(".tokenizer.json")
    tok.save(str(tokenizer_path))
    torch.save({"state_dict": model.state_dict(), "config_size": cfg.size,
                "tokenizer": kind, "tokenizer_file": tokenizer_path.name,
                "vocab_size": cfg.vocab_size, "meta": metrics}, out)
    out.with_suffix(".metrics.json").write_text(json.dumps(metrics, indent=2)+"\n", encoding="utf-8")
    return metrics


if __name__ == "__main__":
    main()
