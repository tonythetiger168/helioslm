"""Student SFT on filtered teacher answers (v5.113, D3).

Sequence-level distillation: the student is fine-tuned with cross-entropy
on the teacher's answers (D1 records that survived the D2 filter), the
loss on the answer tokens only.

  format        the repo's chat transcript (``hsh.llm.render_transcript``
                tags), so the student can be dropped into hsh with
                ``TextModelAdapter``::

                    ##user##<question>
                    ##assistant##<think>
                    <teacher reasoning>
                    </think>
                    <teacher answer><eot>

  keep_thinking "all": every example carries the reasoning; "none": the
                answer only (an empty think block, so one model learns
                both modes); "mix": a fixed share ``think_ratio`` keeps
                it, chosen by a hash of the record id (reproducible);
  mode_switch   (v5.118, LB6 hybrid thinking) appends a soft switch to
                the last user turn, " /think" when the target carries
                reasoning and " /no_think" when its think block is empty,
                so with keep_thinking="mix" the user, not chance, picks
                the mode at inference (MiniCPM5-1B's hybrid-thinking SFT;
                the tag convention is Qwen3's). Off by default;
  mask          1 on the target tokens (everything after
                ``##assistant##``, the eot included), 0 on the prompt;
  packing       examples are packed whole into windows of seq + 1 tokens
                (x = w[:-1], y = w[1:], mask aligned with y), the rest of
                a window padded with eot under mask 0; an example longer
                than a window is dropped and counted, never cut, so the
                student never learns a truncated answer;
  shards        ``<name>.bin`` (uint16 tokens) + ``<name>.mask.bin``
                (uint8), window-aligned, read by ``SFTShard``, which the
                pretraining loop takes in place of ``TokenShard``
                (``pretrain(..., init_from=<base ckpt>)``).

The formatting and packing need no torch; ``SFTShard`` imports it.
"""
import hashlib
import json
from array import array
from pathlib import Path

USER, ASSISTANT = "##user##", "##assistant##"
THINK_OPEN, THINK_CLOSE = "<think>\n", "\n</think>\n"
EMPTY_THINK = "<think>\n</think>\n"
THINK_TAG, NO_THINK_TAG = " /think", " /no_think"
KEEP = ("all", "none", "mix")

# Starting points; check them against the first eval before a long run.
SFT_PRESETS = {
    # the v5.91 pilot (93M dense, seq 1024 pretraining); 2048 fits one
    # answer with a ~1.5k-token reasoning chain in the student's tokens
    "pilot_100m": {"seq": 2048, "batch": 4, "grad_accum": 8, "lr": 1e-4,
                   "warmup": 100, "epochs": 2, "keep_thinking": "mix",
                   "think_ratio": 0.7},
    # v1.0 (#40, 0.5B): longer chains, lower lr
    "v1_0p5b": {"seq": 4096, "batch": 4, "grad_accum": 16, "lr": 5e-5,
                "warmup": 200, "epochs": 2, "keep_thinking": "mix",
                "think_ratio": 0.7},
}


def keeps_thinking(rec_id, mode, think_ratio=0.7):
    if mode not in KEEP:
        raise ValueError(f"keep_thinking must be one of {KEEP}")
    if mode != "mix":
        return mode == "all"
    h = int.from_bytes(hashlib.sha256(rec_id.encode("utf-8")).digest()[:8], "big")
    return h / 2 ** 64 < think_ratio


def format_example(rec, keep_thinking="all", think_ratio=0.7,
                   mode_switch=False):
    """(prompt text, target text) for one D1 record. A record that is
    already text (``prompt_text`` / ``target_text``, the D4 agent steps)
    is passed through, untagged by ``mode_switch``."""
    if "prompt_text" in rec:
        return rec["prompt_text"], rec["target_text"]
    reasoning = (rec.get("reasoning") or "").strip()
    thinks = keeps_thinking(rec["id"], keep_thinking, think_ratio) \
        and bool(reasoning)
    last_user = max((i for i, m in enumerate(rec["messages"])
                     if m["role"] == "user"), default=None)
    parts = []
    for i, m in enumerate(rec["messages"]):
        if m["role"] == "user":
            tag = (THINK_TAG if thinks else NO_THINK_TAG) \
                if mode_switch and i == last_user else ""
            parts.append(USER + m["content"] + tag)
        elif m["role"] == "assistant":
            parts.append(ASSISTANT + m.get("content", ""))
        elif m["role"] == "system":
            parts.append("##system##" + m["content"])
    prompt = "\n".join(parts) + "\n" + ASSISTANT
    if thinks:
        think = THINK_OPEN + reasoning + THINK_CLOSE
    else:
        think = EMPTY_THINK
    return prompt, think + rec["content"].strip()


def encode_example(prompt, target, encode, eot):
    """Token ids and loss mask. Prompt and target are encoded apart so the
    mask boundary is exact (a merge across it cannot blur it)."""
    p, t = list(encode(prompt)), list(encode(target)) + [eot]
    return p + t, [0] * len(p) + [1] * len(t)


def pack(examples, seq, eot):
    """Greedy packing of whole examples into windows of seq + 1 tokens.
    Returns (tokens, mask, stats); both flat, a multiple of seq + 1."""
    w = seq + 1
    toks, mask = array("H"), array("B")
    cur_t, cur_m = [], []
    stats = {"examples": 0, "packed": 0, "too_long": 0, "windows": 0,
             "target_tokens": 0, "pad_tokens": 0}

    def flush():
        pad = w - len(cur_t)
        toks.extend(cur_t + [eot] * pad)
        mask.extend(cur_m + [0] * pad)
        stats["windows"] += 1
        stats["pad_tokens"] += pad

    for ids, m in examples:
        stats["examples"] += 1
        if len(ids) > w:
            stats["too_long"] += 1
            continue
        if any(i < 0 or i > 65535 for i in ids):
            raise ValueError("token ids must fit in uint16")
        if len(cur_t) + len(ids) > w:
            flush()
            cur_t, cur_m = [], []
        cur_t += ids
        cur_m += m
        stats["packed"] += 1
        stats["target_tokens"] += sum(m)
    if cur_t:
        flush()
    return toks, mask, stats


def build(records, encode, eot, seq, keep_thinking="all", think_ratio=0.7,
          utf8_target_bytes=True, mode_switch=False):
    """Format, encode and pack records. Returns (tokens, mask, stats);
    stats include the UTF-8 bytes per target token, the denominator that
    makes the SFT val bits/byte comparable across tokenizers."""
    examples, n_bytes, thinking = [], 0, 0
    for r in records:
        prompt, target = format_example(r, keep_thinking, think_ratio,
                                        mode_switch)
        thinking += not target.startswith(EMPTY_THINK)
        ids, m = encode_example(prompt, target, encode, eot)
        if len(ids) <= seq + 1 and utf8_target_bytes:
            n_bytes += len(target.encode("utf-8"))
        examples.append((ids, m))
    toks, mask, st = pack(examples, seq, eot)
    st["with_thinking"] = thinking
    st["target_bytes_per_token"] = round(n_bytes / st["target_tokens"], 6) \
        if st["target_tokens"] else 0.0
    return toks, mask, st


def write_sft_shard(toks, mask, out_dir, name):
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    with open(out / f"{name}.bin", "wb") as fh:
        toks.tofile(fh)
    with open(out / f"{name}.mask.bin", "wb") as fh:
        mask.tofile(fh)


def split_by_prompt(records, val_fraction=0.02):
    """Train / val split by prompt id hash, so samples of one prompt never
    straddle the split."""
    tr, va = [], []
    for r in records:
        h = int(hashlib.sha256(r["prompt_id"].encode("utf-8")).hexdigest()[:8], 16)
        (va if h / 16 ** 8 < val_fraction else tr).append(r)
    return tr, va


def write_meta(out_dir, meta):
    (Path(out_dir) / "meta.json").write_text(json.dumps(meta, indent=2, ensure_ascii=False) + "\n")


class SFTShard:
    """Window-aligned SFT shard for ``pretrain``: ``batch`` and
    ``eval_windows`` yield (x, y, mask) with mask aligned with y."""

    def __init__(self, path, seq):
        import numpy as np
        path = Path(path)
        if path.stat().st_size == 0:
            raise ValueError(f"{path} is empty: no example fit a window of seq + 1 "
                             "(see the prepare stats' too_long)")
        self.seq = seq
        self.data = np.memmap(path, dtype=np.uint16, mode="r")
        self.mask = np.memmap(path.with_suffix(".mask.bin"), dtype=np.uint8, mode="r")
        if len(self.data) != len(self.mask) or len(self.data) % (seq + 1):
            raise ValueError("shard and mask must hold whole windows of seq + 1")
        self.n_windows = len(self.data) // (seq + 1)

    def __len__(self):
        return len(self.data)

    def _window(self, i):
        import numpy as np
        import torch
        w = self.seq + 1
        t = torch.from_numpy(self.data[i * w:(i + 1) * w].astype(np.int64))
        m = torch.from_numpy(self.mask[i * w:(i + 1) * w].astype(np.int64))
        return t[:-1], t[1:], m[1:]

    def batch(self, batch, seq, micro_step, seed):
        """Random whole windows; indices depend only on (seed, micro_step)."""
        import torch
        if seq != self.seq:
            raise ValueError(f"shard was packed for seq {self.seq}, asked {seq}")
        gen = torch.Generator().manual_seed(seed * 1_000_003 + micro_step)
        idx = torch.randint(0, self.n_windows, (batch,), generator=gen).tolist()
        xs, ys, ms = zip(*(self._window(i) for i in idx))
        return torch.stack(xs), torch.stack(ys), torch.stack(ms)

    def eval_windows(self, seq, max_windows):
        if seq != self.seq:
            raise ValueError(f"shard was packed for seq {self.seq}, asked {seq}")
        for i in range(min(max_windows, self.n_windows)):
            x, y, m = self._window(i)
            yield x.unsqueeze(0), y.unsqueeze(0), m.unsqueeze(0)
