"""Plan E1: the lm-evaluation-harness adapter (evastar/eval/lm_harness.py).

Offline: a tiny BPE and model built in the test; the end-to-end check runs
lm-eval's own simple_evaluate on a local multiple-choice task.

Pins:
1. loglikelihood equals the log-softmax computed directly from the logits,
   the greedy flag is right, and long contexts are cut from the left;
2. loglikelihood_rolling equals one full pass when the text fits, and
   scores every token exactly once when it is split into windows;
3. generate_until stops at a stop string and returns the text before it;
4. simple_evaluate on a local task gives the accuracy implied by the
   adapter's own loglikelihoods;
5. a pilot_100m.py-style run directory loads (config rebuilt from
   run.json, tokenizer hash checked).
"""
import json
import os
import sys
from pathlib import Path

import pytest
import torch
import torch.nn.functional as F

pytest.importorskip("lm_eval")
pytest.importorskip("tokenizers")

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from evastar.eval.lm_harness import HeliosHarnessLM  # noqa: E402

TEXT = ("the cat sat on the mat. the dog ran in the park. a bird sang in the "
        "tree. the sun is hot and the sky is blue. ") * 40


class Req:
    def __init__(self, *args):
        self.args = args


@pytest.fixture(scope="module")
def lm():
    from tokenizers import Tokenizer, decoders, models, pre_tokenizers, trainers
    from evastar.configs.config_v5 import EvastarV5Config
    from evastar.src.model_v5 import EvastarV5
    tok = Tokenizer(models.BPE(unk_token="[UNK]"))
    tok.pre_tokenizer = pre_tokenizers.ByteLevel(add_prefix_space=False)
    tok.decoder = decoders.ByteLevel()
    tok.train_from_iterator([TEXT], trainers.BpeTrainer(
        vocab_size=300, show_progress=False,
        initial_alphabet=pre_tokenizers.ByteLevel.alphabet(),
        special_tokens=["[UNK]", "<|endoftext|>"]))
    torch.manual_seed(0)
    model = EvastarV5(EvastarV5Config(size="lite")).eval()
    return HeliosHarnessLM(model=model, tokenizer=tok, max_length=64)


def _direct(lm, ids, n):
    with torch.no_grad():
        logits = lm.model(torch.tensor([ids[:-1]]))[0][0].float()
    lp = F.log_softmax(logits, -1)[-n:]
    tgt = torch.tensor(ids[-n:])
    return lp.gather(-1, tgt[:, None]).sum().item(), bool((lp.argmax(-1) == tgt).all())


def test_loglikelihood_matches_logits(lm):
    ctx, cont = "the cat sat on", " the mat."
    (got, greedy), = lm.loglikelihood([Req(ctx, cont)])
    whole = lm.tok_encode(ctx + cont)
    n = len(whole) - len(lm.tok_encode(ctx))
    ref, ref_greedy = _direct(lm, whole, n)
    assert abs(got - ref) < 1e-5 and greedy == ref_greedy
    # the model's own argmax continuation is greedy by construction
    ids = lm.tok_encode(ctx)
    nxt = int(lm._logprobs(ids)[-1].argmax())
    assert lm._score(ids, [nxt])[1] is True
    # a trailing space in the context moves to the continuation
    (a, _), (b, _) = lm.loglikelihood([Req("the cat sat on ", "the mat."),
                                       Req("the cat sat on", " the mat.")])
    assert abs(a - b) < 1e-6


def test_left_truncation(lm):
    long_ctx = TEXT[:2000]
    (got, _), = lm.loglikelihood([Req(long_ctx, " the end")])
    ctx_ids = lm.tok_encode(long_ctx)
    whole = lm.tok_encode(long_ctx + " the end")
    n = len(whole) - len(ctx_ids)
    ref, _ = _direct(lm, whole[-(lm.max_length + 1):], n)
    assert len(whole) > lm.max_length + 1 and abs(got - ref) < 1e-5


def test_rolling(lm):
    short = "the dog ran in the park."
    ids = lm.tok_encode(short)
    (got,) = lm.loglikelihood_rolling([Req(short)])
    ref, _ = _direct(lm, [lm.eot_token_id] + ids, len(ids))
    assert abs(got - ref) < 1e-5
    # long text: every token scored exactly once across the windows
    long = TEXT[:1500]
    n_ids = len(lm.tok_encode(long))
    scored = []
    orig = lm._score
    lm._score = lambda c, t: (scored.append(len(t)), orig(c, t))[1]
    try:
        (total,) = lm.loglikelihood_rolling([Req(long)])
    finally:
        lm._score = orig
    assert n_ids > lm.max_length and sum(scored) == n_ids and len(scored) > 1
    assert total < 0 and total == total


def test_generate_until(lm):
    ctx = "the sun is"
    (free,) = lm.generate_until([Req(ctx, {"until": [], "max_gen_toks": 12})])
    assert free
    stop = free[len(free) // 2:len(free) // 2 + 2]
    (cut,) = lm.generate_until([Req(ctx, {"until": [stop], "max_gen_toks": 12})])
    assert stop not in cut and free.startswith(cut)


def test_simple_evaluate_local_task(lm, tmp_path):
    from lm_eval import simple_evaluate
    from lm_eval.tasks import TaskManager
    docs = [{"question": "the cat sat on", "choices": [" the mat.", " the sky."], "answer": 0},
            {"question": "the dog ran in", "choices": [" the park.", " the sun."], "answer": 0},
            {"question": "a bird sang in", "choices": [" the hot.", " the tree."], "answer": 1},
            {"question": "the sky is", "choices": [" blue.", " cat."], "answer": 0}]
    data = tmp_path / "toy.jsonl"
    data.write_text("".join(json.dumps(d) + "\n" for d in docs))
    (tmp_path / "toy.yaml").write_text(f"""task: helios_toy_mc
dataset_path: json
dataset_kwargs:
  data_files:
    test: {data}
test_split: test
output_type: multiple_choice
doc_to_text: "{{{{question}}}}"
doc_to_choice: "{{{{choices}}}}"
doc_to_target: answer
metric_list:
  - metric: acc
""")
    os.environ.setdefault("HF_DATASETS_OFFLINE", "1")
    res = simple_evaluate(model=lm, tasks=["helios_toy_mc"],
                          task_manager=TaskManager(include_path=str(tmp_path)),
                          num_fewshot=0, log_samples=False)
    acc = res["results"]["helios_toy_mc"]["acc,none"]
    picks = []
    for d in docs:
        scores = [s for s, _ in lm.loglikelihood(
            [Req(d["question"], c) for c in d["choices"]])]
        picks.append(int(max(range(len(scores)), key=scores.__getitem__)))
    expected = sum(p == d["answer"] for p, d in zip(picks, docs)) / len(docs)
    assert abs(acc - expected) < 1e-9


def test_load_pilot_run(tmp_path):
    from tokenizers import Tokenizer, decoders, models, pre_tokenizers, trainers
    from evastar.src.training.pretrain import (
        TokenShard, pilot_config, pretrain, write_shards)
    import hashlib
    data = tmp_path / "data"
    data.mkdir()
    tok = Tokenizer(models.BPE(unk_token="[UNK]"))
    tok.pre_tokenizer = pre_tokenizers.ByteLevel(add_prefix_space=False)
    tok.decoder = decoders.ByteLevel()
    tok.train_from_iterator([TEXT], trainers.BpeTrainer(
        vocab_size=300, show_progress=False,
        initial_alphabet=pre_tokenizers.ByteLevel.alphabet(),
        special_tokens=["[UNK]", "<|endoftext|>"]))
    tok.save(str(data / "tokenizer.json"))
    sha = hashlib.sha256((data / "tokenizer.json").read_bytes()).hexdigest()
    ids = tok.encode(TEXT).ids
    write_shards([ids], data, "train")
    write_shards([ids[:400]], data, "val")
    (data / "meta.json").write_text(json.dumps(
        {"tokenizer_sha256": sha, "vocab_size": tok.get_vocab_size()}))
    run_meta = {"arch": "dense", "hidden": 64, "layers": 2, "intermediate": 128,
                "seq": 32, "balance": "smebu", "data": str(data),
                "tokenizer_sha256": sha}
    run = tmp_path / "run"
    cfg = pilot_config("dense", vocab_size=tok.get_vocab_size(), hidden_size=64,
                       num_layers=2, intermediate_size=128, max_seq=32)
    pretrain(cfg, TokenShard(data / "train.bin"), TokenShard(data / "val.bin"), run,
             steps=2, batch=2, seq=32, warmup=1, eval_every=2, eval_windows=2,
             ckpt_every=2, log=lambda m: None)
    (run / "run.json").write_text(json.dumps(run_meta))
    lm = HeliosHarnessLM(run=str(run), data=str(data))
    assert lm.max_length == 32
    (score, _), = lm.loglikelihood([Req("the cat sat on", " the mat.")])
    assert score < 0
    run_meta["tokenizer_sha256"] = "0" * 64
    (run / "run.json").write_text(json.dumps(run_meta))
    with pytest.raises(ValueError, match="tokenizer"):
        HeliosHarnessLM(run=str(run), data=str(data))
