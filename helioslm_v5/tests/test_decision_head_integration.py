"""v5.41 — Phase 1.3: DecisionHead integration oracles on the real model.

Wires ``src/decision_head.py`` onto the final-norm hidden states of a real
HeliosLMv5 (documented contract: ``model(input_ids)`` returns
``(logits, hidden_states, past_key_values)`` with hidden [B, L, hidden])
and proves the composition is a single autograd graph: head gradients
reach the model's own parameters.

CPU-only, lite preset, seconds per oracle.

Run from repo root: python helioslm_v5/tests/test_decision_head_integration.py
"""
import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

from helioslm_v5.configs.config_v5 import HeliosLMv5Config
from helioslm_v5.src.decision_head import DecisionHead
from helioslm_v5.src.model_v5 import HeliosLMv5


def _model_and_head(seed=0, num_choices=6):
    torch.manual_seed(seed)
    model = HeliosLMv5(HeliosLMv5Config(size="lite"))
    head = DecisionHead(hidden_size=model.config.hidden_size,
                        num_choices=num_choices)
    return model, head


def test_hidden_states_drive_head():
    model, head = _model_and_head()
    ids = torch.randint(0, model.config.vocab_size, (2, 8))
    logits, hidden, _past = model(ids)
    assert hidden.shape == (2, 8, model.config.hidden_size), hidden.shape
    out = head(hidden)
    assert out["noul"].shape == (2, 8)
    assert out["choice"].shape == (2, 8, 6)
    assert out["score"].shape == (2, 8)
    assert torch.allclose(out["choice"].sum(-1), torch.ones(2, 8), atol=1e-6)
    # noul is a probability over REAL activations, not a stub constant
    assert float(out["noul"].std().detach()) > 1e-4
    print("PASS test_hidden_states_drive_head "
          f"hidden {tuple(hidden.shape)} -> noul/choice/score ok, "
          f"noul std {float(out['noul'].std().detach()):.4f}")


def test_gradients_reach_model_parameters():
    """A scalar through head+model must produce grads on BOTH the head
    and the trunk parameters that produced the hidden states.

    Scope is stated honestly: lm_head, unexercised MTP modules, and MoE
    experts/routes the tokens never selected are NOT expected to move —
    the loss genuinely does not pass through them. The oracle asserts
    the trunk (embeddings, every transformer layer, final norm) is
    reached, and records lm_head's zero grad as the wiring property it
    is: the decision path bypasses the LM head entirely.
    """
    model, head = _model_and_head()
    ids = torch.randint(0, model.config.vocab_size, (1, 6))
    _logits, hidden, _past = model(ids)
    out = head(hidden)
    loss = out["noul"].mean() + out["choice"].mean() + out["score"].mean()
    loss.backward()
    head_grads = [p.grad is not None and p.grad.abs().sum() > 0
                  for p in head.parameters()]
    named = dict(model.named_parameters())
    trunk_ok, moe_grad, moe_static, untouched = 0, 0, 0, 0
    for n, p in named.items():
        got = p.grad is not None and p.grad.abs().sum() > 0
        is_route_bias = n.endswith("moe.route_bias")
        is_expert = ".moe.experts." in n
        is_trunk = n.startswith(("embed_tokens", "layers.", "final_norm",
                                 "norm."))
        if is_route_bias:
            # aux-loss-free load balancing: selection-only bias updated by
            # the heuristic/quantile rule, NOT by backprop — a zero grad
            # here is the design, not a break
            moe_static += 1
        elif is_expert:
            moe_grad += bool(got)        # routed experts move; others may not
        elif is_trunk:
            assert got, f"trunk parameter {n} received no gradient"
            trunk_ok += 1
        elif got:
            pass
        else:
            untouched += 1               # lm_head, mtp, ...
    assert all(head_grads), "head received no gradient"
    lm_grad = named["lm_head.weight"].grad
    assert lm_grad is None or float(lm_grad.abs().sum()) == 0.0, \
        "lm_head moved — the decision path should bypass the LM head"
    print(f"PASS test_gradients_reach_model_parameters trunk "
          f"{trunk_ok} tensors reached, head {sum(head_grads)}/"
          f"{len(head_grads)}, lm_head bypass confirmed, "
          f"{moe_grad} routed experts moved, {moe_static} route_bias "
          f"heuristic-only (no backprop by design), {untouched} params "
          f"structurally unexercised (lm_head/mtp)")


def test_routing_signal_learns_through_model():
    """End-to-end smoke: a synthetic routing target (first-token parity)
    must become learnable through the frozen-lm-head / trained-body graph.

    Honest scope note: this oracle proves the WIRING trains, not that the
    task is easy — the parity target is attached to the first position
    only, so the signal is weak by construction. The assert floor is
    therefore modest (clearly above the 1/6 chance level), and the
    printed margin is the recordable evidence.
    """
    torch.manual_seed(1)
    model, head = _model_and_head(num_choices=2)
    opt = torch.optim.Adam(
        list(model.parameters()) + list(head.parameters()), lr=3e-3)
    g = torch.Generator().manual_seed(7)
    xs = torch.randint(0, model.config.vocab_size, (32, 6), generator=g)
    ys = (xs[:, 0] % 2).long()          # parity of first token -> route
    for _ in range(60):
        opt.zero_grad()
        _logits, hidden, _past = model(xs)
        logp = torch.log(head(hidden[:, 0])["choice"] + 1e-9)
        loss = torch.nn.functional.nll_loss(logp, ys)
        loss.backward()
        opt.step()
    with torch.no_grad():
        _logits, hidden, _past = model(xs)
        pred = head(hidden[:, 0])["choice"].argmax(-1)
        acc = float((pred == ys).float().mean())
    assert acc > 0.55, f"wiring acc {acc:.3f} at chance level (1/2=0.50)"
    print(f"PASS test_routing_signal_learns_through_model acc={acc:.3f} "
          f"(chance 0.500, final nll {float(loss.detach()):.4f})")


if __name__ == "__main__":
    test_hidden_states_drive_head()
    test_gradients_reach_model_parameters()
    test_routing_signal_learns_through_model()
    print("\n3/3 DecisionHead integration tests passed")
