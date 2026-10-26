"""Traditional-Chinese training speed by tokenizer (v5.80).

The same tiny HeliosLMv5 is trained for the same number of steps on the
same zh-TW text (every line with a CJK character in the repo's training
corpus, held-out documents excluded) with each tokenizer. Per step the
model sees a fixed number of TOKENS, so a tokenizer that spends fewer
tokens per character feeds it more text per step. Measured per arm:

  chars_per_token    on the training text;
  chars_per_second   training text consumed per wall-clock second;
  bits_per_char      held-out cross-entropy, summed over tokens and divided
                     by characters (comparable across tokenizers), after
                     the same number of steps.
"""
import math
import time

import torch
import torch.nn.functional as F

from helioslm_v5.configs.config_v5 import (
    HeliosLMv5Config, HybridAttentionConfig, MoEConfig, MTPConfig)
from helioslm_v5.eval.zh_tokenizer_coverage import _CJK, eval_split
from helioslm_v5.src.model_v5 import HeliosLMv5


def zh_training_text(corpus):
    """Lines of the training corpus that contain a CJK character."""
    return "\n".join(line for line in corpus.splitlines() if _CJK.search(line))


def model_config(vocab_size, hidden=128, layers=2):
    return HeliosLMv5Config(
        size="lite", num_hidden_layers=layers, vocab_size=vocab_size,
        hidden_size=hidden, intermediate_size=2 * hidden,
        hybrid_attention=HybridAttentionConfig(enabled=False),
        mtp=MTPConfig(enabled=False),
        moe=MoEConfig(num_experts=2, num_activated_experts=1,
                      expert_hidden_size=hidden),
        use_attention_residuals=False, use_hyper_connections=False)


@torch.no_grad()
def bits_per_char(model, ids, n_chars, seq):
    """Held-out NLL in bits, summed over every token, per character."""
    model.eval()
    total = 0.0
    for s in range(0, len(ids) - 1, seq):
        chunk = torch.tensor([ids[s:s + seq + 1]])
        if chunk.shape[1] < 2:
            break
        logits = model(chunk[:, :-1])[0].float()
        total += F.cross_entropy(logits.reshape(-1, logits.shape[-1]),
                                 chunk[0, 1:], reduction="sum").item()
    model.train()
    return total / math.log(2) / n_chars


def run(tok, train_text, root, steps, seq, batch, lr, seed, eval_every):
    """Train one model with ``tok`` (a tokenizers.Tokenizer)."""
    ids = tok.encode(train_text).ids
    zh, _ = eval_split(root)
    held = "\n".join(zh)
    held_ids = tok.encode(held).ids
    torch.manual_seed(seed)
    model = HeliosLMv5(model_config(tok.get_vocab_size()))
    opt = torch.optim.AdamW(model.parameters(), lr=lr)
    gen = torch.Generator().manual_seed(seed)
    data = torch.tensor(ids)
    curve, train_seconds = [], 0.0
    model.train()
    for step in range(steps):
        t0 = time.perf_counter()
        starts = torch.randint(0, len(ids) - seq - 1, (batch,), generator=gen)
        x = torch.stack([data[s:s + seq] for s in starts.tolist()])
        y = torch.stack([data[s + 1:s + seq + 1] for s in starts.tolist()])
        logits = model(x)[0]
        loss = F.cross_entropy(logits.reshape(-1, logits.shape[-1]),
                               y.reshape(-1))
        opt.zero_grad()
        loss.backward()
        opt.step()
        train_seconds += time.perf_counter() - t0
        if (step + 1) % eval_every == 0:
            curve.append({"step": step + 1, "bits_per_char": round(
                bits_per_char(model, held_ids, len(held), seq), 4)})
    chars_per_token = len(train_text) / len(ids)
    return {"vocab": tok.get_vocab_size(),
            "train_tokens": len(ids),
            "chars_per_token": round(chars_per_token, 4),
            "chars_per_step": round(chars_per_token * batch * seq, 1),
            "bits_per_char_curve": curve,
            "timing": {"train_seconds": round(train_seconds, 1),
                       "tokens_per_second": round(
                           steps * batch * seq / train_seconds, 1),
                       "chars_per_second": round(
                           steps * batch * seq * chars_per_token
                           / train_seconds, 1)}}
