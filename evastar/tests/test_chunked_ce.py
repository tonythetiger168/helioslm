"""T140 — fused chunked output head + cross-entropy, v5.142.

``chunked_lm_loss`` computes the LM loss (and, in the same pass, its
gradients) ``chunk`` tokens at a time, so the [tokens, vocab] logits never
exist at once. Pins:

1. loss and gradients (hidden and head weight) equal full-logits
   cross-entropy, with a target mask, the final logit soft cap, ragged
   chunks and bf16 hidden states;
2. ``EvastarV5.forward(return_logits=False)`` returns no logits and the
   final-normed hidden states the logits come from;
3. model gradients equal the full-logits loss's, and ``pretrain(
   ce_chunk=...)`` logs the same losses as the full-logits path (4 steps;
   weights are not compared: AdamW turns float noise in near-zero
   gradients into lr-sized steps), for dense and MoE;
4. ``ce_chunk`` is validated.

Run from repo root: python3 -m pytest evastar/tests/test_chunked_ce.py
"""
import json
import sys
import tempfile
from pathlib import Path

import pytest
import torch
import torch.nn.functional as F

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from evastar.src.training.pretrain import chunked_lm_loss, pretrain  # noqa: E402
from evastar.src.model_v5 import EvastarV5  # noqa: E402
from evastar.tests.test_pretrain_pilot import _stream, _tiny  # noqa: E402


def _full(h, w, t, mask, cap):
    z = (h @ w.T).float()
    if cap is not None:
        z = cap * torch.tanh(z / cap)
    ce = F.cross_entropy(z, t, reduction="none")
    return (ce * mask).sum() / mask.sum() if mask is not None else ce.mean()


@pytest.mark.parametrize("masked", [False, True])
@pytest.mark.parametrize("cap", [None, 5.0])
@pytest.mark.parametrize("chunk", [1, 64, 77, 10_000])
def test_matches_full_logits(masked, cap, chunk):
    g = torch.Generator().manual_seed(0)
    N, d, V = 300, 32, 1000
    h = torch.randn(N, d, generator=g, requires_grad=True)
    w = torch.randn(V, d, generator=g, requires_grad=True)
    t = torch.randint(0, V, (N,), generator=g)
    mask = (torch.rand(N, generator=g) > 0.3).float() if masked else None
    ref = _full(h, w, t, mask, cap)
    gh, gw = torch.autograd.grad(1.7 * ref, (h, w))
    loss = chunked_lm_loss(h, w, t, mask, cap, chunk)
    gh2, gw2 = torch.autograd.grad(1.7 * loss, (h, w))
    assert torch.allclose(loss, ref, atol=1e-5)
    assert torch.allclose(gh, gh2, atol=1e-6) and torch.allclose(gw, gw2, atol=1e-6)


def test_bf16_hidden():
    g = torch.Generator().manual_seed(1)
    h = torch.randn(2, 50, 64, generator=g).bfloat16().requires_grad_(True)
    w = torch.randn(500, 64, generator=g, requires_grad=True)
    t = torch.randint(0, 500, (2, 50), generator=g)
    loss = chunked_lm_loss(h, w, t, chunk=32)
    loss.backward()
    ref = F.cross_entropy((h.detach().reshape(-1, 64) @ w.detach().bfloat16().T).float(),
                          t.reshape(-1))
    assert abs(loss.item() - ref.item()) < 1e-3
    assert h.grad.dtype == torch.bfloat16 and w.grad.dtype == torch.float32


def test_forward_without_logits():
    torch.manual_seed(0)
    model = EvastarV5(_tiny("dense")).eval()
    ids = torch.randint(3, 96, (2, 10))
    with torch.no_grad():
        logits, hidden, _ = model(ids)
        none, hidden2, _ = model(ids, return_logits=False)
    assert none is None and torch.equal(hidden, hidden2)
    assert torch.allclose(model.lm_head(hidden2), logits)


@pytest.mark.parametrize("arch", ["dense", "moe"])
def test_pretrain_same_as_full_logits(arch):
    torch.set_num_threads(1)
    runs = []
    with tempfile.TemporaryDirectory() as d:
        tr, va = _stream(d)
        for chunk in (0, 40):
            out = Path(d) / f"c{chunk}"
            pretrain(_tiny(arch), tr, va, out, steps=4, batch=4, seq=32,
                     grad_accum=2, lr=3e-3, warmup=2, eval_every=2,
                     eval_windows=4, ckpt_every=4, seed=0, ce_chunk=chunk,
                     log=lambda m: None)
            rows = [json.loads(x) for x in (out / "log.jsonl").read_text().splitlines()]
            runs.append([r["train_loss"] for r in rows if "train_loss" in r])
    assert len(runs[0]) == 4 and runs[0] == pytest.approx(runs[1], abs=1e-4)


@pytest.mark.parametrize("arch", ["dense", "moe"])
def test_model_gradients_match(arch):
    torch.manual_seed(0)
    model = EvastarV5(_tiny(arch)).train()
    ids = torch.randint(3, 96, (2, 17))
    x, y = ids[:, :-1], ids[:, 1:]
    grads = []
    for chunked in (False, True):
        model.zero_grad(set_to_none=True)
        if chunked:
            h = model(x, return_logits=False)[1]
            loss = chunked_lm_loss(h, model.lm_head.weight, y, chunk=7)
        else:
            logits = model(x)[0]
            loss = F.cross_entropy(logits.float().reshape(-1, 96), y.reshape(-1))
        loss.backward()
        grads.append({n: p.grad.clone() for n, p in model.named_parameters()
                      if p.grad is not None})
    assert grads[0].keys() == grads[1].keys()
    for n in grads[0]:
        assert torch.allclose(grads[0][n], grads[1][n], atol=1e-6), n


def test_ce_chunk_validated():
    with pytest.raises(ValueError, match="ce_chunk"):
        pretrain(_tiny("dense"), None, None, "/nonexistent", steps=1, batch=1,
                 seq=8, ce_chunk=-1)
