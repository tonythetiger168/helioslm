"""D3 (v5.111): student SFT on teacher answers (src/training/distill_sft.py
and the masked / init_from path of src/training/pretrain.py).

Pins:
1. format: the ##user## / ##assistant## transcript, a think block with the
   reasoning, an empty think block without it; "mix" keeps the requested
   share, reproducibly by record id;
2. encode: mask 0 on the prompt and 1 on target + eot, boundary exact;
3. pack: whole examples only (too long ones dropped and counted), windows
   of seq + 1 padded with eot under mask 0, uint16 range enforced; the
   train / val split never separates samples of one prompt;
4. (torch) SFTShard: shapes, mask aligned with y, batches a function of
   (seed, micro-step);
5. (torch) pretrain on an SFT shard: evaluation equals a hand-computed
   masked cross-entropy; init_from with lr 0 reproduces the base weights;
   kill + resume is bitwise equal to an uninterrupted run;
6. (torch + tokenizers) the CLI end to end: prepare then train from a
   tiny base run.
"""
import json
import math
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from helioslm_v5.src.training import distill_sft as ds  # noqa: E402


def rec(i, reasoning="先想。", content="答案。", pid=None, prompt="問題？"):
    return {"id": f"r{i}", "prompt_id": pid or f"p{i}",
            "messages": [{"role": "user", "content": prompt}],
            "reasoning": reasoning, "content": content}


def enc(text):                      # one token per character, ids < 300
    return [ord(c) % 250 + 1 for c in text]


EOT = 0


def test_format_and_keep_thinking():
    p, t = ds.format_example(rec(1), "all")
    assert p == "##user##問題？\n##assistant##"
    assert t == "<think>\n先想。\n</think>\n答案。"
    assert ds.format_example(rec(1), "none")[1] == "<think>\n</think>\n答案。"
    assert ds.format_example(rec(1, reasoning=""), "all")[1] == "<think>\n</think>\n答案。"
    multi = {"id": "m", "messages": [{"role": "system", "content": "s"},
                                     {"role": "user", "content": "a"},
                                     {"role": "assistant", "content": "b"},
                                     {"role": "user", "content": "c"}],
             "reasoning": "", "content": "d"}
    assert ds.format_example(multi)[0] == "##system##s\n##user##a\n##assistant##b\n##user##c\n##assistant##"
    kept = sum(ds.keeps_thinking(f"id{i}", "mix", 0.7) for i in range(2000))
    assert 1300 < kept < 1500
    assert ds.keeps_thinking("x", "mix", 0.7) == ds.keeps_thinking("x", "mix", 0.7)
    with pytest.raises(ValueError):
        ds.keeps_thinking("x", "some")


def test_encode_mask_boundary():
    ids, m = ds.encode_example("ab", "cde", enc, EOT)
    assert ids == enc("ab") + enc("cde") + [EOT]
    assert m == [0, 0, 1, 1, 1, 1]


def test_pack_and_split():
    exs = [ds.encode_example("p" * 3, "t" * 2, enc, EOT),        # 6 tokens
           ds.encode_example("p" * 2, "t" * 1, enc, EOT),        # 4
           ds.encode_example("p" * 9, "t" * 5, enc, EOT),        # 15: too long
           ds.encode_example("p", "t", enc, EOT)]                # 3
    toks, mask, st = ds.pack(exs, seq=8, eot=EOT)
    assert st == {"examples": 4, "packed": 3, "too_long": 1, "windows": 2,
                  "target_tokens": 3 + 2 + 2, "pad_tokens": 18 - 13}
    assert len(toks) == len(mask) == 18
    assert list(mask[:9]) == [0, 0, 0, 1, 1, 1, 0, 0, 0]          # ex1 + pad
    assert list(mask[9:]) == [0, 0, 1, 1, 0, 1, 1, 0, 0]          # ex2 + ex4 + pad
    assert list(toks[6:9]) == [EOT] * 3
    with pytest.raises(ValueError):
        ds.pack([([70000], [1])], 8, EOT)
    toks, mask, st = ds.build([rec(i) for i in range(4)], enc, EOT, seq=64,
                              keep_thinking="all")
    assert st["with_thinking"] == 4 and st["target_bytes_per_token"] > 1
    recs = [rec(i, pid=f"q{i // 3}") for i in range(300)]
    tr, va = ds.split_by_prompt(recs, 0.2)
    assert len(tr) + len(va) == 300 and va
    assert not {r["prompt_id"] for r in tr} & {r["prompt_id"] for r in va}


# ---------------------------------------------------------------- torch

def _shard(tmp_path, n=40, seq=32, name="train"):
    recs = [rec(i, reasoning="想" * (i % 5), content=f"答{i % 7}" * 3,
                prompt=f"題目{i}？") for i in range(n)]
    toks, mask, st = ds.build(recs, enc, EOT, seq=seq, keep_thinking="all")
    ds.write_sft_shard(toks, mask, tmp_path, name)
    return st


def test_sft_shard(tmp_path):
    torch = pytest.importorskip("torch")
    _shard(tmp_path)
    sh = ds.SFTShard(tmp_path / "train.bin", 32)
    x, y, m = sh.batch(4, 32, 3, 0)
    assert x.shape == y.shape == m.shape == (4, 32)
    assert torch.equal(x[:, 1:], y[:, :-1])
    assert torch.equal(sh.batch(4, 32, 3, 0)[0], x)
    assert not torch.equal(sh.batch(4, 32, 4, 0)[0], x)
    i = 0
    full = torch.tensor(sh.data[:33].astype("int64"))
    fm = torch.tensor(sh.mask[:33].astype("int64"))
    x0, y0, m0 = sh._window(i)
    assert torch.equal(y0, full[1:]) and torch.equal(m0, fm[1:])
    assert int(m0.sum()) > 0
    with pytest.raises(ValueError):
        sh.batch(2, 16, 0, 0)
    with pytest.raises(ValueError):
        ds.SFTShard(tmp_path / "train.bin", 31)


def _tiny():
    from helioslm_v5.src.training.pretrain import pilot_config
    return pilot_config("dense", vocab_size=256, hidden_size=64, num_layers=2,
                        intermediate_size=128, max_seq=32)


def test_pretrain_masked_init_and_resume(tmp_path):
    torch = pytest.importorskip("torch")
    import torch.nn.functional as F
    from helioslm_v5.src.training.pretrain import HeliosLMv5, evaluate, pretrain
    torch.set_num_threads(1)
    _shard(tmp_path, name="train")
    _shard(tmp_path, n=12, name="val")
    tr = ds.SFTShard(tmp_path / "train.bin", 32)
    va = ds.SFTShard(tmp_path / "val.bin", 32)

    torch.manual_seed(0)
    model = HeliosLMv5(_tiny()).eval()
    got = evaluate(model, va, 32, 10, 1.0, torch.device("cpu"), torch.float32)
    tot, n = 0.0, 0
    with torch.no_grad():
        for x, y, m in va.eval_windows(32, 10):
            ce = F.cross_entropy(model(x)[0].float().reshape(-1, 256), y.reshape(-1),
                                 reduction="none")
            tot += (ce * m.reshape(-1)).sum().item()
            n += int(m.sum())
    assert got["val_loss"] == pytest.approx(round(tot / n, 5), abs=1e-5)

    kw = dict(batch=2, seq=32, grad_accum=2, warmup=1, eval_every=2, eval_windows=3,
              ckpt_every=2, seed=0, log=lambda m: None)
    base = tmp_path / "base"
    pretrain(_tiny(), tr, va, base, steps=2, lr=3e-3, **kw)
    same = tmp_path / "same"
    pretrain(_tiny(), tr, va, same, steps=1, lr=0.0, init_from=str(base / "ckpt.pt"), **kw)
    sb = torch.load(base / "ckpt.pt", weights_only=False)["model"]
    ss = torch.load(same / "ckpt.pt", weights_only=False)["model"]
    assert all(torch.equal(v, ss[k]) for k, v in sb.items())

    a, b = tmp_path / "a", tmp_path / "b"
    init = str(base / "ckpt.pt")
    pretrain(_tiny(), tr, va, a, steps=4, lr=1e-3, init_from=init, **kw)
    pretrain(_tiny(), tr, va, b, steps=2, lr=1e-3, init_from=init, **kw)
    pretrain(_tiny(), tr, va, b, steps=4, lr=1e-3, init_from=init, **kw)
    sa = torch.load(a / "ckpt.pt", weights_only=False)["model"]
    sbb = torch.load(b / "ckpt.pt", weights_only=False)["model"]
    assert all(torch.equal(v, sbb[k]) for k, v in sa.items())
    logs = [json.loads(x) for x in (a / "log.jsonl").read_text().splitlines()]
    assert all(math.isfinite(r["train_loss"]) for r in logs)
    assert not any(torch.equal(v, sa[k]) for k, v in sb.items() if v.dim() >= 2)


def test_cli_prepare_and_train(tmp_path):
    torch = pytest.importorskip("torch")
    pytest.importorskip("tokenizers")
    from tokenizers import Tokenizer, models, pre_tokenizers, trainers
    from helioslm_v5.src.training.pretrain import pretrain
    sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "examples"))
    import distill_sft as cli

    texts = [f"##user##題目{i}？\n##assistant##<think>\n想\n</think>\n答案{i}" for i in range(50)]
    tok = Tokenizer(models.BPE(unk_token="[UNK]"))
    tok.pre_tokenizer = pre_tokenizers.ByteLevel(add_prefix_space=False)
    tok.train_from_iterator(texts, trainers.BpeTrainer(
        vocab_size=300, special_tokens=["<|endoftext|>", "[UNK]"],
        initial_alphabet=pre_tokenizers.ByteLevel.alphabet(), show_progress=False))
    tok.save(str(tmp_path / "tokenizer.json"))
    recs = tmp_path / "kept.jsonl"
    recs.write_text("\n".join(json.dumps(rec(i, prompt=f"題目{i}？", content=f"答案{i}"),
                                         ensure_ascii=False) for i in range(60)),
                    encoding="utf-8")
    data = tmp_path / "sft"
    cli.main(["prepare", "--records", str(recs), "--tokenizer", str(tmp_path / "tokenizer.json"),
              "--out", str(data), "--seq", "64", "--val-fraction", "0.2"])
    meta = json.loads((data / "meta.json").read_text())
    assert meta["splits"]["train"]["packed"] > 0 and meta["splits"]["val"]["windows"] > 0
    assert meta["keep_thinking"] == "mix"

    from helioslm_v5.src.training.pretrain import pilot_config
    vocab = meta["vocab_size"]
    base = tmp_path / "base"
    cfg = pilot_config("dense", vocab_size=vocab, hidden_size=64, num_layers=2,
                       intermediate_size=128, max_seq=64)
    sh = ds.SFTShard(data / "train.bin", 64)
    pretrain(cfg, sh, sh, base, steps=1, batch=2, seq=64, lr=1e-3, warmup=1,
             eval_every=1, eval_windows=1, ckpt_every=1, log=lambda m: None)
    (base / "run.json").write_text(json.dumps({"arch": "dense", "hidden": 64, "layers": 2,
                                               "intermediate": 128, "balance": "smebu"}))
    out = tmp_path / "run"
    cli.main(["train", "--data", str(data), "--base-run", str(base), "--out", str(out),
              "--steps", "2", "--batch", "2", "--grad-accum", "1", "--eval-every", "2",
              "--eval-windows", "2", "--ckpt-every", "2", "--device", "cpu"])
    last = [json.loads(x) for x in (out / "log.jsonl").read_text().splitlines()][-1]
    assert last["step"] == 2 and math.isfinite(last["val_bits_per_byte"])
    assert torch.load(out / "ckpt.pt", weights_only=False)["step"] == 2
