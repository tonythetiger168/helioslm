"""T145 — ``cpt_hf.py --ce-chunk``: route-B continued pretraining without
the full [tokens, vocab] logits, v5.152.

A tiny model with the Hugging Face causal-LM surface (``logits_to_keep``,
``output_hidden_states``, ``get_output_embeddings``, ``config``) pins:

1. the chunked loss and every parameter gradient equal the full-logits
   ones, with and without Gemma's final logit soft cap (read from
   ``config.get_text_config()``);
2. the forward runs with ``logits_to_keep=1`` (no full logits) and the
   output head still gets its gradient;
3. ``check_chunked_head`` accepts that model and refuses one whose last
   hidden state is not the head input (pre-norm), and a head with a bias;
4. ``train(ce_chunk=...)`` follows the same loss curve as full logits.
"""
import json
import sys
import tempfile
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
import torch

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from examples import cpt_hf  # noqa: E402


class TinyHF(torch.nn.Module):
    def __init__(self, vocab=50, dim=16, softcap=None, prenorm_hidden=False, bias=False):
        super().__init__()
        torch.manual_seed(0)
        self.embed = torch.nn.Embedding(vocab, dim)
        self.mix = torch.nn.Linear(dim, dim)
        self.norm = torch.nn.LayerNorm(dim)
        self.lm_head = torch.nn.Linear(dim, vocab, bias=bias)
        text = SimpleNamespace(final_logit_softcapping=softcap)
        self.config = SimpleNamespace(get_text_config=lambda: text)
        self.prenorm_hidden = prenorm_hidden
        self.kept = []

    def get_output_embeddings(self):
        return self.lm_head

    def forward(self, input_ids, output_hidden_states=False, logits_to_keep=0):
        h = torch.tanh(self.mix(self.embed(input_ids)))
        n = self.norm(h)
        keep = n[:, -logits_to_keep:] if logits_to_keep else n
        self.kept.append(keep.shape[1])
        logits = self.lm_head(keep)
        cap = self.config.get_text_config().final_logit_softcapping
        if cap is not None:
            logits = cap * torch.tanh(logits / cap)
        hidden = (h if self.prenorm_hidden else n,)
        return SimpleNamespace(logits=logits,
                               hidden_states=hidden if output_hidden_states else None)


def _batch(vocab=50):
    g = torch.Generator().manual_seed(1)
    x = torch.randint(0, vocab, (3, 21), generator=g)
    return x, torch.randint(0, vocab, (3, 21), generator=g)


def _grads(model, loss):
    model.zero_grad()
    loss.backward()
    return {n: p.grad.clone() for n, p in model.named_parameters()}


@pytest.mark.parametrize("softcap", [None, 3.0])
def test_chunked_equals_full(softcap):
    model = TinyHF(softcap=softcap)
    assert cpt_hf.final_softcap(model) == softcap
    x, y = _batch()
    full = cpt_hf.lm_loss(model, x, y)
    gf = _grads(model, full)
    model.kept.clear()
    chunked = cpt_hf.lm_loss(model, x, y, ce_chunk=8)
    assert model.kept == [1]                     # no full logits
    gc = _grads(model, chunked)
    assert torch.allclose(full, chunked, atol=1e-6)
    assert gf.keys() == gc.keys()
    for n in gf:
        assert torch.allclose(gf[n], gc[n], atol=1e-6), n
    assert gc["lm_head.weight"].abs().sum() > 0


def test_check_chunked_head():
    x, y = _batch()
    full, chunked = cpt_hf.check_chunked_head(TinyHF(softcap=3.0), x, y, 8)
    assert abs(full - chunked) < 1e-5
    with pytest.raises(ValueError, match="not the output head input"):
        cpt_hf.check_chunked_head(TinyHF(prenorm_hidden=True), x, y, 8)
    with pytest.raises(ValueError, match="without bias"):
        cpt_hf.lm_loss(TinyHF(bias=True), x, y, ce_chunk=8)


def test_ddp_style_wrapper_is_unwrapped():
    class Wrap(torch.nn.Module):
        def __init__(self, m):
            super().__init__()
            self.module = m

        def forward(self, **kw):
            return self.module(**kw)

    inner = TinyHF(softcap=2.0)
    x, y = _batch()
    assert cpt_hf.final_softcap(Wrap(inner)) == 2.0
    assert torch.allclose(cpt_hf.lm_loss(Wrap(inner), x, y, ce_chunk=5),
                          cpt_hf.lm_loss(inner, x, y), atol=1e-6)


def test_train_with_ce_chunk():
    with tempfile.TemporaryDirectory() as d:
        (Path(d) / "main").mkdir()
        np.tile(np.arange(4, 12), 200).astype(np.uint32).tofile(Path(d) / "main" / "m.bin")
        (Path(d) / "manifest.json").write_text(json.dumps({"dtype": "uint32", "sources": {
            "m": {"phase": "main", "path": "main/m.bin", "tokens": 1600, "weight": 1.0}}}))
        phases = cpt_hf.load_phases(d, seq=16)
        runs = []
        for chunk in (0, 7):
            rows = []
            cpt_hf.train(TinyHF(), phases, steps=12, micro_batch=2, seq=16,
                         grad_accum=2, lr=3e-2, warmup=1, log=rows.append,
                         autocast=False, ce_chunk=chunk)
            runs.append([r["loss"] for r in rows])
        assert runs[1][-1] < runs[1][0] * 0.8
        assert np.allclose(runs[0], runs[1], atol=1e-3)
