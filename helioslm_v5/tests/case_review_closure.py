"""Review-plan closure oracles (2026-10-10 review items B1-2/B1-3/B1-4/B1-6).

Each test takes one open item from HeliosLM_改進計畫_2026-10.md and pins
the shipped mechanism (or adds the missing oracle) with real numbers:

1. B1-2 MLA vs HCA bitwise contrast: one shared input through the lite
   model's own MLA layer and a same-width HighCompressionAttention
   (128x-class chunk compression + top-k exact rescue). Both arms are
   pinned deterministic (eval forward twice, max|diff| == 0), the HCA
   per-token state accounting reports the 128x compression and the
   rescue budget separately, and the two mechanisms' outputs are
   distinct — a reproducible contrast data point, not a quality claim.
2. B1-3 NVFP4 QAT end-to-end training: apply_qat(nvfp4) on the real
   lite model, a fixed memorization batch, 25 AdamW steps — CE loss
   drops below 20% of its start, gradients land on the fp32 masters,
   the MTP lm_head tie survives the wrap, and baking fake-quant weights
   into a plain nn.Linear runs a deploy forward.
3. B1-4 batched MTP + acceptance telemetry: one B=3 generate (the
   batched path landed with the engine), per-row greedy output
   bitwise-equal to plain greedy, batched counters equal the sum of
   solo counters, and the telemetry payload shape matches the web_ui_v2
   badge contract (mtp_acceptance float in [0, 1]).
4. B1-6 swarm delegate_all true parallelism: three 120 ms sleep
   workers finish in well under the 360 ms serial floor, the parent
   effect log carries worker-prefixed kinds merged in worker-name order
   (byte-identical replay regardless of thread scheduling), and the
   all/any/quorum join semantics over completion_order hold.

Run from repo root: python3 helioslm_v5/tests/case_review_closure.py
"""
import sys
import time
from pathlib import Path

import torch
import torch.nn as nn

_REPO = Path(__file__).resolve().parent.parent.parent
if str(_REPO) not in sys.path:
    sys.path.insert(0, str(_REPO))

from helioslm_v5.tests.case_common import _lite_model, _pass


def test_b12_mla_vs_hca_contrast():
    """MLA vs HCA on one shared input: deterministic, accounted, distinct."""
    from helioslm_v5.src.attention.kv_compress import HighCompressionAttention

    model, config = _lite_model(41)
    layer0 = model.layers[0]
    assert type(layer0.attention).__name__ == "MLA", \
        "lite hybrid_attention.enabled is False: layer 0 must be MLA"

    torch.manual_seed(42)
    x = torch.randn(1, 48, config.hidden_size)

    with torch.no_grad():
        mla1 = layer0(x)[0]
        mla2 = layer0(x)[0]
    mla_diff = (mla1 - mla2).abs().max().item()
    assert mla_diff == 0.0, f"MLA eval forward not deterministic: {mla_diff}"

    torch.manual_seed(43)
    hca = HighCompressionAttention(config.hidden_size, n_heads=2, window=8,
                                   chunk=128, rescue_topk=2).eval()
    with torch.no_grad():
        h1 = hca(x)
        h2 = hca(x)
    h_diff = (h1 - h2).abs().max().item()
    assert h_diff == 0.0, f"HCA eval forward not deterministic: {h_diff}"

    # Honest per-token state accounting (values, not bytes — the module's
    # own convention): 128x slot compression + a separate rescue budget.
    stats = hca.get_compression_stats(x.shape[1])
    assert stats["compression_vs_dense"] == 128.0
    # rescue budget = rescue_topk * 2(K,V) * n_heads * head_dim
    assert stats["rescue_values_per_query"] == 2 * 2 * 2 * \
        (config.hidden_size // 2)
    assert stats["slot_values_per_token"] * 128 == \
        stats["dense_values_per_token"]

    # The contrast data point itself: same input, two mechanisms.
    assert torch.isfinite(mla1).all() and torch.isfinite(h1).all()
    cross = (mla1 - h1).abs().max().item()
    assert cross > 1e-4, f"untrained MLA vs HCA must differ: {cross}"
    _pass("test_b12_mla_vs_hca_contrast",
          f"MLA vs HCA(128x+rescue2) both deterministic; "
          f"cross-mechanism max|diff|={cross:.3f}")


def test_b13_nvfp4_qat_end_to_end_training():
    """NVFP4 QAT training loop on the real lite model: loss collapses."""
    from helioslm_v5.src.quantization.qat import apply_qat

    model, config = _lite_model(52)
    applied = apply_qat(model, method="nvfp4")
    assert len(applied) > 0 and all(m.method == "nvfp4"
                                    for _, m in applied)
    if getattr(model, "mtp_modules", None):
        assert model.mtp_modules[0].lm_head is model.lm_head, \
            "apply_qat must re-bind the MTP lm_head tie"

    # Memorization curriculum: one fixed batch, so a falling loss directly
    # evidences trainable fp32 masters under the fake-quant STE.
    g = torch.Generator().manual_seed(99)
    ids = torch.randint(3, config.vocab_size, (2, 16), generator=g)
    opt = torch.optim.AdamW(model.parameters(), lr=1e-2)
    losses = []
    for _ in range(25):
        out = model(ids)
        logits = out[0] if isinstance(out, tuple) else out
        loss = torch.nn.functional.cross_entropy(
            logits[:, :-1].reshape(-1, logits.shape[-1]).float(),
            ids[:, 1:].reshape(-1))
        opt.zero_grad()
        loss.backward()
        opt.step()
        losses.append(float(loss.detach()))
    assert all(torch.isfinite(torch.tensor(losses))), losses
    assert losses[-1] < 0.2 * losses[0], \
        f"NVFP4 QAT must memorize the fixed batch: {losses[0]:.3f} -> {losses[-1]:.3f}"
    for name, m in applied[:4]:
        assert m.weight.grad is not None \
            and torch.isfinite(m.weight.grad).all(), name

    # Deploy path: bake fake-quant weights into a plain Linear and run it.
    with torch.no_grad():
        first = applied[0][1]
        baked = nn.Linear(first.weight.shape[1], first.weight.shape[0],
                          bias=first.bias is not None)
        baked.weight.copy_(first.fake_quant_weight())
        if first.bias is not None:
            baked.bias.copy_(first.bias)
        probe = torch.randn(3, config.hidden_size)
        assert torch.isfinite(baked(probe)).all()
    _pass("test_b13_nvfp4_qat_end_to_end_training",
          f"nvfp4 QAT loss {losses[0]:.3f} -> {losses[-1]:.3f} in 25 steps, "
          f"{len(applied)} linears wrapped, deploy bake forward OK")


def test_b14_batched_mtp_acceptance_telemetry():
    """Batched MTP: per-row equality with plain greedy; clean telemetry."""
    from helioslm_v5.src.inference.mtp import MTPDecoder

    model, config = _lite_model(64)
    decoder = MTPDecoder(model, model.mtp_modules, config)
    torch.manual_seed(65)
    ids = torch.randint(3, config.vocab_size, (3, 8))

    res = decoder.generate(ids, max_new_tokens=6, temperature=0)
    assert res.sequences.shape == (3, 14)
    assert 0.0 <= res.acceptance_rate <= 1.0
    assert res.num_drafted >= res.num_accepted >= 0 and res.num_rounds >= 1

    for row in range(3):
        plain = model.generate(ids[row:row + 1], max_new_tokens=6,
                               temperature=0)
        n = res.sequences.shape[1]
        assert torch.equal(plain[:, :n], res.sequences[row:row + 1]), \
            f"row {row}: batched MTP greedy diverged from plain greedy"

    # Telemetry isolation: batched counters equal the sum of solo runs.
    solo = [decoder.generate(ids[r:r + 1], max_new_tokens=6, temperature=0)
            for r in range(3)]
    for r, s in enumerate(solo):
        assert torch.equal(s.sequences, res.sequences[r:r + 1]), \
            f"row {r}: solo and in-batch sequences differ"
    assert sum(s.num_drafted for s in solo) == res.num_drafted
    assert sum(s.num_accepted for s in solo) == res.num_accepted

    # web_ui_v2 badge contract: a plain float acceptance rate.
    assert isinstance(res.acceptance_rate, float)
    _pass("test_b14_batched_mtp_acceptance_telemetry",
          f"B=3 batched MTP, acceptance={res.acceptance_rate:.3f} "
          f"({res.num_accepted}/{res.num_drafted} drafted), per-row == plain")


def test_b16_swarm_delegate_all_parallel():
    """delegate_all: sub-serial wall time, name-ordered replay, joins."""
    from helioslm_v5.harness.harness_core import Context
    from helioslm_v5.harness.harness_plugins import (
        AgentSwarmPlugin, AgentWorkflowPlugin, ModelPlugin, PresetPlugin)

    def sleeper(answer, delay):
        def fn(prompt, seed, step):
            if step == 0:
                time.sleep(delay)
            return answer
        return fn

    ctx = Context()
    ctx.use(PresetPlugin("full", model_fn=None))
    ctx.use(AgentSwarmPlugin(n_workers=3))
    ctx.use(AgentWorkflowPlugin())
    fns = {n: sleeper(n.upper(), 0.12) for n in ("a", "b", "c")}
    for n, f in fns.items():
        ctx.use(ModelPlugin(n, f))
    ctx.swarm.spawn("parallel probe", fns)

    t0 = time.monotonic()
    results = ctx.swarm.delegate_all(
        [(n, "sub " + n, getattr(ctx.model, n)) for n in ("a", "b", "c")])
    wall = time.monotonic() - t0
    assert results == {"a": "A", "b": "B", "c": "C"}
    # 0.30s is 2.5x the single-worker 0.12s and 83% of the 0.36s serial
    # floor — generous to slow CI, impossible without real overlap.
    assert wall < 0.30, f"3 x 120 ms workers must overlap: {wall:.3f}s"

    swarm = ctx.state["swarm"]
    assert sorted(swarm["completion_order"]) == ["a", "b", "c"]
    kinds = [e.kind for e in ctx.effects]
    prefixed = [k for k in kinds if k.split(":", 1)[0] in ("a", "b", "c")]
    order = [k.split(":", 1)[0] for k in prefixed]
    assert order == sorted(order), \
        f"child logs must merge in worker-name order: {order}"
    assert swarm["blackboard"] == {"a": "A", "b": "B", "c": "C"}

    agg_all = ctx.swarm.aggregate("all")
    assert isinstance(agg_all, list) and set(agg_all) == {"A", "B", "C"}
    assert ctx.swarm.aggregate(mode="any") in ("A", "B", "C")
    quorum = ctx.swarm.aggregate(mode="quorum")
    assert isinstance(quorum, list) and len(quorum) == 2, quorum
    _pass("test_b16_swarm_delegate_all_parallel",
          f"3x120ms in {wall:.3f}s (serial floor 0.36s), name-ordered "
          "replay, all/any/quorum joins OK")


def test_review_closure_suite():
    """Run all four review-plan closure oracles in the daily gate."""
    test_b12_mla_vs_hca_contrast()
    test_b13_nvfp4_qat_end_to_end_training()
    test_b14_batched_mtp_acceptance_telemetry()
    test_b16_swarm_delegate_all_parallel()
    _pass("test_review_closure_suite",
          "2026-10-10 review plan B1-2/B1-3/B1-4/B1-6 closure oracles")


if __name__ == "__main__":
    test_review_closure_suite()
