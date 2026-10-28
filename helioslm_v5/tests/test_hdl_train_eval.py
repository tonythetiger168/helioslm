"""T157 — HDL track: SFT on prompt / response records and the
three-benchmark eval runner (v5.178, v5.179).

Pins:

1. sft_hf.render / pack: chat-template prompt tokens get mask 0, response
   + EOS mask 1; a record whose prompt fills --seq is dropped, a long
   response is cut; windows() cuts [n, seq + 1] id / mask rows;
2. sft_hf.masked_loss scores mask-1 targets only (prompt tokens do not
   change it); train() on a tiny torch LM lowers the masked loss and
   reports a validation loss;
3. cpt_hf.cast_frozen (v5.179): frozen parameters in bf16 stay fixed
   while the rest trains under autocast;
4. hdl_eval_all.collect / score_all: every VerilogEval and RTLLM fixture
   gets one job with its harness's own prompt, RTLLM exclusions are
   skipped, and generated references score pass@1 = 1 (needs iverilog).

Run from repo root: python3 -m pytest helioslm_v5/tests/test_hdl_train_eval.py
"""
import shutil
import sys
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "examples"))

import sft_hf  # noqa: E402

FIX = ROOT / "helioslm_v5" / "tests" / "fixtures"


class Tok:
    """Characters as ids; the chat template is '<u>' + text + '<a>'."""
    eos_token_id = 1
    chat_template = "x"

    def apply_chat_template(self, msgs, tokenize, add_generation_prompt):
        if msgs[0]["role"] == "system":
            raise ValueError("no system role")
        return "<u>" + msgs[0]["content"] + "<a>"

    def __call__(self, text, add_special_tokens=False):
        return {"input_ids": [2 + (ord(c) % 50) for c in text]}


def test_render_and_pack():
    tok = Tok()
    ids, mask = sft_hf.render(tok, "sys", "hi", "ok")
    assert len(ids) == len(mask) == len("<u>sys\n\nhi<a>") + 3
    assert mask[-3:] == [1, 1, 1] and ids[-1] == 1 and sum(mask) == 3
    recs = [{"prompt": "p", "response": "r"},
            {"prompt": "x" * 50, "response": "r"},           # prompt too long
            {"prompt": "p", "response": "y" * 40}]           # cut response
    ids, mask, st = sft_hf.pack(recs, tok, 20, ("system", "prompt", "response"))
    assert st["kept"] == 2 and st["dropped_long_prompt"] == 1
    assert st["cut_response"] == 1 and len(ids) == len("<u>p<a>") + 2 + 20
    assert ids.dtype == np.uint32 and mask.dtype == np.uint8
    wi, wm = sft_hf.windows(ids, mask, 8)
    assert wi.shape == wm.shape == ((len(ids) - 1) // 8, 9)
    assert (wi[1, 0] == ids[8]) and (wi[0, -1] == ids[8])



def test_no_think_and_drop_long():
    tok = Tok()
    assert sft_hf.strip_think("<think>a\nb</think>\n\nmodule m;") == "module m;"
    assert sft_hf.strip_think("module m;") == "module m;"
    recs = [{"prompt": "p", "response": "<think>" + "z" * 30 + "</think>ok"},
            {"prompt": "p", "response": "y" * 40}]
    ids, mask, st = sft_hf.pack(recs, tok, 20, ("system", "prompt", "response"),
                                think=False, drop_long=True)
    assert st["kept"] == 1 and st["dropped_long"] == 1 and st["cut_response"] == 0
    assert int(mask.sum()) == 3                            # "ok" + EOS

torch = pytest.importorskip("torch")


class TinyLM(torch.nn.Module):
    def __init__(self, vocab=64, dim=32):
        super().__init__()
        self.emb = torch.nn.Embedding(vocab, dim)
        self.rnn = torch.nn.GRU(dim, dim, batch_first=True)
        self.head = torch.nn.Linear(dim, vocab)

    def forward(self, input_ids):
        h, _ = self.rnn(self.emb(input_ids))
        return (self.head(h),)


def test_masked_loss_ignores_prompt():
    torch.manual_seed(0)
    m = TinyLM()
    x = torch.randint(0, 64, (1, 10))
    y = torch.randint(0, 64, (1, 10))
    mk = torch.zeros(1, 10, dtype=torch.long)
    mk[0, 6:] = 1
    a = sft_hf.masked_loss(m, x, y, mk)
    y2 = y.clone()
    y2[0, :6] = (y2[0, :6] + 1) % 64                   # change prompt targets
    assert torch.allclose(a, sft_hf.masked_loss(m, x, y2, mk))


def test_train_lowers_masked_loss():
    torch.manual_seed(0)
    rng = np.random.default_rng(0)
    seq, n = 16, 64
    ids = np.zeros((n, seq + 1), dtype=np.uint32)
    mask = np.zeros((n, seq + 1), dtype=np.uint8)
    for i in range(n):
        p = rng.integers(2, 60, size=8)
        ids[i, :8], ids[i, 8:] = p, 7                  # response: always 7
        mask[i, 8:] = 1
    m = TinyLM()
    rows = []
    res = sft_hf.train(m, ids[:56], mask[:56], epochs=8, micro_batch=4,
                       grad_accum=1, lr=1e-2, warmup=2, val_ids=ids[56:],
                       val_mask=mask[56:], autocast=False, log=rows.append,
                       eval_every=0)
    assert res["steps"] == 8 * 56 // 4 and res["stopped"] is None
    assert rows[-1]["loss"] < rows[0]["loss"] / 4
    assert res["val_loss"] < 0.5


def test_frozen_bf16_trains():
    """v5.179: frozen parameters stored in bf16 stay fixed; training of the
    rest still runs under autocast."""
    from cpt_hf import cast_frozen, freeze
    torch.manual_seed(0)
    m = TinyLM()
    m.norm = torch.nn.LayerNorm(32)
    tr, fr = freeze(m, ["emb"])
    saved = cast_frozen(m, torch.bfloat16)
    assert fr == 64 * 32 and saved == fr * 2
    assert m.emb.weight.dtype == torch.bfloat16
    assert m.head.weight.dtype == torch.float32
    before = m.emb.weight.detach().clone()
    ids = np.full((8, 9), 7, dtype=np.uint32)
    mask = np.ones((8, 9), dtype=np.uint8)
    res = sft_hf.train(m, ids, mask, epochs=2, micro_batch=4, grad_accum=1,
                       lr=1e-2, warmup=1, autocast=True, eval_every=0)
    assert res["steps"] == 4 and res["stopped"] is None
    assert torch.equal(m.emb.weight, before)


@pytest.mark.skipif(shutil.which("iverilog") is None, reason="needs iverilog")
def test_eval_all_references(tmp_path):
    import json
    import hdl_eval_all as ea
    from helioslm_v5.eval import hdl_bench as hb
    from helioslm_v5.eval import rtllm_bench as rb
    names = ("adder_8bit", "multi_booth_8bit", "multi_pipe_4bit")
    rb.build_cache(FIX / "rtllm", tmp_path / "rt", designs=names)
    jobs, data = ea.collect(("rtllm",), None, tmp_path / "rt", None,
                            exclude=("multi_pipe_4bit",))
    designs = data["rtllm"]
    assert [designs[k]["name"] for (_, k, _, _) in jobs] == list(names[:2])
    assert jobs[0][2] == hb.SYSTEM_MSG
    assert jobs[0][3] == rb.build_prompt(designs[0]["prompt"])
    outs = [["```verilog\n" + rb.reference_sample(designs[k]) + "\n```"]
            for (_, k, _, _) in jobs]
    res = ea.score_all(jobs, outs, data, 1, (1,), exclude=("multi_pipe_4bit",))
    assert res["rtllm"]["pass@1"] == 1.0 and res["rtllm"]["n_designs"] == 2
    assert ea.summary(res)["rtllm"]["pass@1"] == 1.0
    json.dumps(res)
    # --samples rescores saved generations without a GPU (v5.194)
    samples = tmp_path / "s.samples.json"
    samples.write_text(json.dumps(
        [{"bench": b, "key": k, "outputs": o} for (b, k, _, _), o in zip(jobs, outs)]
        + [{"bench": "cvdp", "key": "unused", "outputs": ["x"]}]))
    out = tmp_path / "r.json"
    assert ea.main(["--model", "m", "--samples", str(samples), "--out", str(out),
                    "--benches", "rtllm", "--rtllm-cache", str(tmp_path / "rt"),
                    "--rtllm-exclude", "multi_pipe_4bit", "--workers", "1"]) == 0
    assert json.loads(out.read_text())["summary"]["rtllm"]["pass@1"] == 1.0
