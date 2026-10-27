"""v5.41 — DPO trainer oracles (Phase 1.4).

Tests helioslm_v5/src/training/dpo.py. Uses a tiny trainable stub LM that
follows the HeliosLMv5 call convention (returns (logits, None, None)) so
the oracles run on CPU in milliseconds — the point is the LOSS MATH, not
model scale.

Run from repo root: python helioslm_v5/tests/test_dpo.py
"""
import math
import sys
from pathlib import Path

import torch
import torch.nn as nn

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

from helioslm_v5.src.training.dpo import DPOTrainer


class _StubLM(nn.Module):
    """(1, T) token ids -> (1, T, vocab) logits from an embedding table.

    Every input row is looked up independently, so log-probs are exact and
    the model is trivially trainable — a pure test fixture."""

    def __init__(self, vocab: int = 64, d: int = 16, seed: int = 0):
        super().__init__()
        g = torch.Generator().manual_seed(seed)
        self.emb = nn.Embedding(vocab, vocab)
        with torch.no_grad():
            self.emb.weight.copy_(
                0.1 * torch.randn(vocab, vocab, generator=g))

    def forward(self, full_ids):
        return self.emb(full_ids), None, None


def _pairs(n=8, prompt=(1, 2), chosen=(3,), rejected=(4,)):
    return [{"prompt": list(prompt), "chosen": list(chosen),
             "rejected": list(rejected)} for _ in range(n)]


def test_policy_equal_ref_gives_log2():
    """When pi_theta == pi_ref every margin is 0, sigma(0) = 0.5, and the
    DPO loss is exactly log 2 per pair — the analytic anchor value."""
    torch.manual_seed(0)
    lm = _StubLM()
    tr = DPOTrainer(lm, ref_model=lm, beta=0.1, lr=1e-3)
    loss, m = tr.loss(_pairs(4))
    assert abs(m["loss"] - math.log(2)) < 1e-6, m
    assert m["acc"] == 0.0 and abs(m["margin"]) < 1e-6, m
    print(f"PASS test_policy_equal_ref_gives_log2 loss={m['loss']:.6f} "
          f"(log2={math.log(2):.6f})")


def test_loss_matches_manual_computation():
    """Rebuild the margin by hand from the same stub weights; the trainer
    must agree to float precision."""
    torch.manual_seed(1)
    lm = _StubLM()
    ref = _StubLM(seed=7)          # different weights -> nonzero margins
    tr = DPOTrainer(lm, ref_model=ref, beta=0.3, lr=1e-3)
    pairs = [{"prompt": [1, 2], "chosen": [3], "rejected": [4]}]
    _, m = tr.loss(pairs)

    ids = [1, 2]
    c_ids, r_ids = [3], [4]
    c_full = torch.tensor([ids + c_ids])
    r_full = torch.tensor([ids + r_ids])
    pl = len(ids)
    with torch.no_grad():
        pi_c = tr._sequence_logprob(lm, c_full, pl)
        pi_r = tr._sequence_logprob(lm, r_full, pl)
        ref_c = tr._sequence_logprob(ref, c_full, pl)
        ref_r = tr._sequence_logprob(ref, r_full, pl)
    margin = 0.3 * ((pi_c - ref_c) - (pi_r - ref_r))
    manual = float(-torch.nn.functional.logsigmoid(margin))
    assert abs(m["loss"] - manual) < 1e-6, (m["loss"], manual)
    print(f"PASS test_loss_matches_manual_computation loss={m['loss']:.6f} "
          f"margin={m['margin']:.4f}")


def test_training_prefers_chosen():
    """A few DPO steps must increase the chosen-vs-rejected margin and
    push implied accuracy toward 1.0."""
    torch.manual_seed(2)
    lm = _StubLM()
    ref = _StubLM()                # identical init: start at margin 0
    tr = DPOTrainer(lm, ref_model=ref, beta=0.5, lr=5e-2)
    pairs = _pairs(8)
    _, m0 = tr.loss(pairs)
    history = tr.fit(pairs, epochs=40)
    _, m1 = tr.loss(pairs)
    assert m1["margin"] > m0["margin"] + 0.5, (m0, m1)
    assert m1["acc"] >= m0["acc"], (m0, m1)
    # loss went down, not just the metric
    assert history[-1]["loss"] < history[0]["loss"]
    print(f"PASS test_training_prefers_chosen margin {m0['margin']:.3f} "
          f"-> {m1['margin']:.3f}, acc {m0['acc']:.2f} -> {m1['acc']:.2f}, "
          f"loss {history[0]['loss']:.4f} -> {history[-1]['loss']:.4f}")


def test_reference_is_frozen():
    torch.manual_seed(3)
    lm, ref = _StubLM(), _StubLM()
    before = [p.detach().clone() for p in ref.parameters()]
    tr = DPOTrainer(lm, ref_model=ref, lr=1e-2)
    tr.fit(_pairs(4), epochs=3)
    for b, p in zip(before, ref.parameters()):
        assert torch.equal(b, p), "reference model moved during training"
    print("PASS test_reference_is_frozen")


def test_loud_errors():
    lm = _StubLM()
    try:
        DPOTrainer(lm, ref_model=lm, beta=0.0)
    except ValueError:
        pass
    else:
        raise AssertionError("beta=0 accepted")
    tr = DPOTrainer(lm, ref_model=lm)
    for bad in ([], None):
        try:
            tr.loss(bad)
        except ValueError:
            pass
        else:
            raise AssertionError(f"empty pairs accepted: {bad!r}")
    print("PASS test_loud_errors beta=0 and empty pairs raise")


def test_string_inputs_use_tokenizer():
    """Text prompts route through the byte-level fallback tokenizer; with
    a vocab covering all byte values the loss must be finite and equal to
    the id-list form of the same bytes."""
    torch.manual_seed(4)
    lm = _StubLM(vocab=300)
    ref = _StubLM(vocab=300, seed=9)
    tr = DPOTrainer(lm, ref_model=ref, beta=0.1)
    byte_ids = list("Q".encode("utf-8"))
    text_pairs = [{"prompt": "Q", "chosen": "y", "rejected": "n"}]
    id_pairs = [{"prompt": byte_ids,
                 "chosen": list("y".encode("utf-8")),
                 "rejected": list("n".encode("utf-8"))}]
    _, m_text = tr.loss(text_pairs)
    _, m_ids = tr.loss(id_pairs)
    assert math.isfinite(m_text["loss"])
    assert abs(m_text["loss"] - m_ids["loss"]) < 1e-6, (m_text, m_ids)
    print(f"PASS test_string_inputs_use_tokenizer loss={m_text['loss']:.6f}")


if __name__ == "__main__":
    test_policy_equal_ref_gives_log2()
    test_loss_matches_manual_computation()
    test_training_prefers_chosen()
    test_reference_is_frozen()
    test_loud_errors()
    test_string_inputs_use_tokenizer()
    print("\n6/6 DPO tests passed")
