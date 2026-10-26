"""HeliosLM v5.1 — cross-module integration tests with a REAL lite model (CPU).

Every test instantiates the real HeliosLMv5(size="lite") model (or the real
module under test) — no dummy stand-ins. Each test prints PASS/FAIL plus key
numbers; the script exits non-zero if any test fails.

Run:  python /mnt/agents/output/integration_test_v51.py
"""
import copy
import sys
import traceback

sys.path.insert(0, "/mnt/agents/output")

import torch
import torch.nn as nn

from helioslm_v5.configs.config_v5 import HeliosLMv5Config

torch.manual_seed(0)

RESULTS = []


def report(name, ok, detail=""):
    RESULTS.append((name, ok))
    print(f"[{'PASS' if ok else 'FAIL'}] {name}" + (f" — {detail}" if detail else ""))


def make_lite_model(seed=0):
    torch.manual_seed(seed)
    config = HeliosLMv5Config(size="lite")
    model = HeliosLMv5_shim(config)
    model.eval()
    return model, config


def HeliosLMv5_shim(config):
    # Local import so a broken model import fails inside the test, not at load.
    from helioslm_v5.src.model_v5 import HeliosLMv5
    return HeliosLMv5(config)


# ----------------------------------------------------------------------
# T1: cached decode vs full-forward logits consistency
# ----------------------------------------------------------------------
def test_cache_consistency():
    model, config = make_lite_model(seed=1)
    ids = torch.randint(3, config.vocab_size, (1, 12))
    with torch.no_grad():
        full_logits, _, _ = model(ids, use_cache=False)
        past = None
        step_logits = []
        for i in range(ids.shape[1]):
            logits, _, past = model(ids[:, i:i + 1], past_key_values=past,
                                    use_cache=True)
            step_logits.append(logits[:, -1])
        step_logits = torch.stack(step_logits, dim=1)
    diff = (full_logits - step_logits).abs().max().item()
    ok = diff < 1e-4
    report("T1 cached-decode vs full-forward", ok, f"max|diff|={diff:.3e} (atol 1e-4)")


# ----------------------------------------------------------------------
# T2: MTP end-to-end (use_mtp=True) == plain greedy, acceptance in [0,1]
# ----------------------------------------------------------------------
def test_mtp_end_to_end():
    from helioslm_v5.src.inference.mtp import MTPDecoder
    model, config = make_lite_model(seed=2)
    ids = torch.randint(3, config.vocab_size, (1, 6))

    # shared weights between main model and MTP modules
    shared = (model.mtp_modules[0].embed_tokens is model.embed_tokens
              and model.mtp_modules[0].lm_head is model.lm_head)

    plain = model.generate(ids, max_new_tokens=10, temperature=0)
    mtp_seq = model.generate(ids, max_new_tokens=10, temperature=0, use_mtp=True)
    assert isinstance(mtp_seq, torch.Tensor), \
        f"generate(use_mtp=True) must return a token tensor, got {type(mtp_seq)}"
    n = mtp_seq.shape[1]
    match = torch.equal(plain[:, :n], mtp_seq)

    # acceptance-rate plumbing via a direct decoder run
    decoder = MTPDecoder(model, model.mtp_modules, config)
    res = decoder.generate(ids, max_new_tokens=10, temperature=0)
    rate_ok = 0.0 <= res.acceptance_rate <= 1.0
    rounds_ok = res.num_rounds >= 1

    # v5.2: batched MTP is supported. Positive contract: use_mtp=True with
    # batch=2 must produce exactly the per-row greedy generate results
    # (early-stopped rows are right-padded with pad_token_id).
    ids2 = torch.randint(3, config.vocab_size, (2, 6))
    mtp_batch = model.generate(ids2, max_new_tokens=10, temperature=0,
                               use_mtp=True)
    batch_match = True
    for i in range(2):
        ref_i = model.generate(ids2[i:i + 1], max_new_tokens=10, temperature=0)
        L_i = ref_i.shape[1]
        row_ok = torch.equal(mtp_batch[i, :L_i], ref_i[0]) and bool(
            (mtp_batch[i, L_i:] == config.pad_token_id).all())
        batch_match = batch_match and row_ok

    ok = shared and match and rate_ok and rounds_ok and batch_match
    report("T2 MTP end-to-end", ok,
           f"shared_weights={shared}, greedy_match={match} (len {n} vs "
           f"{plain.shape[1]}), acceptance={res.acceptance_rate:.3f} "
           f"({res.num_accepted}/{res.num_drafted} drafted, {res.num_rounds} rounds), "
           f"batch2==per-row-greedy={batch_match}")


# ----------------------------------------------------------------------
# T3: VLLMEngine end-to-end == per-request greedy generate, no block leak
# ----------------------------------------------------------------------
def test_vllm_engine():
    from helioslm_v5.src.inference.vllm_engine import VLLMEngine
    model, config = make_lite_model(seed=3)
    prompts = [[5, 100, 200, 7], [42, 900]]  # unequal lengths
    max_new = 6
    engine = VLLMEngine(model, config, block_size=4, max_num_blocks=64)
    ids = [engine.add_request(p, max_new_tokens=max_new, temperature=0.0)
           for p in prompts]
    results = engine.run()

    match = True
    details = []
    for p, rid in zip(prompts, ids):
        ref = model.generate(torch.tensor([p]), max_new_tokens=max_new,
                             temperature=0)
        ref_new = ref[0, len(p):].tolist()
        got = results[rid]
        # engine stops at EOS; generate freezes with pad afterwards — the
        # engine output must be an exact prefix of the greedy reference.
        pref = ref_new[:len(got)] == got
        match = match and pref
        details.append(f"req{rid}: engine={got} ref[:len]={ref_new[:len(got)]}")

    bm = engine.block_manager
    leak_free = (len(bm.block_tables) == 0 and len(bm.refcounts) == 0
                 and bm.num_free_blocks() == 64)

    # PagedAttention module itself: cache dtype must follow the input (fp32
    # here), and the write/read path must run end-to-end.
    from helioslm_v5.src.inference.paged_attention import BlockManager, PagedAttention
    torch.manual_seed(31)
    pa = PagedAttention(config)
    bm2 = BlockManager(block_size=4, num_blocks=16, device="cpu")  # dtype=None
    bm2.allocate(0, 0, pa.num_heads, pa.head_dim)
    h = torch.randn(1, 5, config.hidden_size)
    with torch.no_grad():
        pa_out = pa(h, bm2, [0])
    dtype_ok = bm2.k_cache is not None and bm2.k_cache.dtype == torch.float32
    pa_ok = dtype_ok and pa_out.shape == h.shape and torch.isfinite(pa_out).all()

    ok = match and leak_free and pa_ok
    report("T3 VLLMEngine end-to-end", ok,
           f"{' ; '.join(details)} ; blocks free={bm.num_free_blocks()}/64, "
           f"tables={len(bm.block_tables)} ; PagedAttention dtype_follows={dtype_ok}")


# ----------------------------------------------------------------------
# T4: GRPO end-to-end with the real lite model
# ----------------------------------------------------------------------
def test_grpo():
    from helioslm_v5.src.training.grpo import GRPOTrainer
    model, config = make_lite_model(seed=4)
    ref_model, _ = make_lite_model(seed=14)  # different init -> non-trivial k3 KL
    config.grpo.group_size = 4
    config.grpo.max_new_tokens = 8
    trainer = GRPOTrainer(model, ref_model, config)  # byte-level tokenizer fallback
    out = trainer.train_step(["What is 2+2?"], ["4"])
    loss = out["loss"]
    grads = [p.grad for p in model.parameters() if p.grad is not None]
    grad_ok = len(grads) > 0 and all(torch.isfinite(g).all() for g in grads)
    # embed_tokens must carry grad (used by the grad-carrying logprob forward)
    emb_grad = model.embed_tokens.weight.grad is not None
    ok = (isinstance(loss, float) and loss == loss and abs(loss) != float("inf")
          and grad_ok and emb_grad)
    report("T4 GRPO train_step", ok,
           f"loss={loss:.4f} policy={out['policy_loss']:.4f} "
           f"kl={out['kl_penalty']:.6f} reward={out['mean_reward']:.3f}, "
           f"params_with_grad={len(grads)}, embed_grad={emb_grad}")


# ----------------------------------------------------------------------
# T5: DualPipe end-to-end over a real lite-model layer
# ----------------------------------------------------------------------
def test_dualpipe():
    from helioslm_v5.src.training.dualpipe import (
        DualPipeScheduler, DualPipeStage, LayerWrap)
    model, config = make_lite_model(seed=5)

    # LayerWrap (dualpipe) adapts HeliosLMv5Layer ((hidden, kv, attn_res)
    # output) to the tensor->tensor stage contract; with attention
    # residuals off (lite default) it threads nothing (v5.4 behaviour).
    stage = DualPipeStage(nn.ModuleList([LayerWrap(model.layers[0]),
                                         LayerWrap(model.layers[1])]))
    sched = DualPipeScheduler([stage], num_micro_batches=4)
    torch.manual_seed(55)
    inputs = [torch.randn(2, 5, config.hidden_size) for _ in range(4)]
    loss = sched.run_dual(inputs, lambda out: out.pow(2).mean())
    p = model.layers[0].attention.q_a_proj.weight
    p2 = model.layers[1].moe.router.weight
    grad_ok = (p.grad is not None and torch.isfinite(p.grad).all()
               and p.grad.abs().sum() > 0
               and p2.grad is not None and p2.grad.abs().sum() > 0)
    n_f = sum(1 for e, _ in sched.trace if e == "F")
    n_b = sum(1 for e, _ in sched.trace if e == "B")
    ok = grad_ok and abs(loss) != float("inf") and loss == loss
    report("T5 DualPipe run_dual", ok,
           f"loss={loss:.4f}, q_a_grad_norm={p.grad.norm():.4e} "
           f"router_grad_norm={p2.grad.norm():.4e}, trace F/B={n_f}/{n_b}")


# ----------------------------------------------------------------------
# T6: FP8Trainer end-to-end, finite loss, no cross-step grad accumulation
# ----------------------------------------------------------------------
def test_fp8():
    from helioslm_v5.src.training.fp8_trainer import FP8Trainer
    model, config = make_lite_model(seed=6)
    trainer = FP8Trainer(model, config)
    ids = torch.randint(3, config.vocab_size, (2, 10))
    batch = {"input_ids": ids}
    probe = model.embed_tokens.weight  # survives FP8 conversion (not Linear)
    losses, grad_norms = [], []
    for _ in range(3):
        losses.append(trainer.train_step(batch))
        grad_norms.append(probe.grad.norm().item())
    finite = all(l == l and abs(l) != float("inf") for l in losses)
    # With zero_grad each step, grad norms stay ~constant on a repeated batch;
    # accumulation would make step-3 norm ~3x step-1.
    ratio = grad_norms[-1] / max(grad_norms[0], 1e-12)
    no_accum = ratio < 1.8
    ok = finite and no_accum and grad_norms[0] > 0
    report("T6 FP8Trainer 3 steps", ok,
           f"losses={[f'{l:.4f}' for l in losses]}, "
           f"grad_norms={[f'{g:.4e}' for g in grad_norms]}, ratio={ratio:.2f}")


# ----------------------------------------------------------------------
# T7: Quantization (AWQ + GPTQ) on the real lite model, then generate
# ----------------------------------------------------------------------
def test_quantization():
    from helioslm_v5.src.quantization.standard_quant import QuantizationManager
    ok_all, details = True, []
    for method in ("awq", "gptq"):
        model, config = make_lite_model(seed=7)
        qm = QuantizationManager(method=method)
        qm.quantize_model(model, group_size=64)
        model.eval()
        ids = torch.randint(3, config.vocab_size, (1, 5))
        with torch.no_grad():
            logits, _, _ = model(ids)
            gen = model.generate(ids, max_new_tokens=4, temperature=0)
        finite = torch.isfinite(logits).all().item()
        ran = gen.shape[1] == 5 + 4
        # MTP weight tying must survive quantization: mtp lm_head/embed must
        # be the main model's CURRENT (quantized) modules, not stale copies.
        mtp_ok = (model.mtp_modules is None) or all(
            m.lm_head is model.lm_head and m.embed_tokens is model.embed_tokens
            for m in model.mtp_modules)
        ok_all = ok_all and finite and ran and mtp_ok
        details.append(f"{method}: logits_finite={finite}, gen_len={gen.shape[1]}, "
                       f"mtp_rebound={mtp_ok}")
    report("T7 quantization AWQ+GPTQ", ok_all, " ; ".join(details))


# ----------------------------------------------------------------------
# T8: streaming audio — 3 chunks vs one-shot
# ----------------------------------------------------------------------
def test_audio_streaming():
    from helioslm_v5.src.audio.streaming_encoder import StreamingAudioEncoder
    config = HeliosLMv5Config(size="lite")
    config.multimodal.audio_hidden_size = 128
    config.multimodal.audio_num_layers = 2
    torch.manual_seed(8)
    enc = StreamingAudioEncoder(config).eval()
    mel = torch.randn(1, config.multimodal.audio_n_mels, 60)
    with torch.no_grad():
        enc.reset_state()
        full = enc(mel)
        enc.reset_state()
        chunks = [enc(mel[:, :, i:i + 20]) for i in range(0, 60, 20)]
        streamed = torch.cat(chunks, dim=1)
    diff = (full - streamed).abs().max().item()
    ok = diff < 1e-3 and full.shape == streamed.shape
    report("T8 audio 3-chunk streaming", ok,
           f"shape={tuple(full.shape)}, max|diff|={diff:.3e} (tol 1e-3)")


# ----------------------------------------------------------------------
# T9: NaViT — 512x512 needs max_grid>=40; default 32 must raise ValueError
# ----------------------------------------------------------------------
def test_navit():
    from helioslm_v5.src.vision.navit import NaViTEncoder
    img = torch.randn(1, 3, 512, 512)  # 36x36 patch grid at patch 14

    config = HeliosLMv5Config(size="lite")
    config.multimodal.vision_hidden_size = 128
    config.multimodal.vision_num_layers = 2
    config.multimodal.vision_num_heads = 4

    # (a) max_grid=40 -> runs
    config.multimodal.vision_max_grid = 40
    torch.manual_seed(9)
    enc = NaViTEncoder(config).eval()
    with torch.no_grad():
        out = enc(img)
    ran_ok = out.shape == (1, 36 * 36, 128) and torch.isfinite(out).all()

    # (b) default 32 -> ValueError
    config.multimodal.vision_max_grid = 32
    enc2 = NaViTEncoder(config).eval()
    raised = False
    try:
        with torch.no_grad():
            enc2(img)
    except ValueError:
        raised = True

    # (c) model-level wiring: enabling multimodal builds matching encoders
    cfg_mm = HeliosLMv5Config(size="lite")
    cfg_mm.multimodal.enabled = True
    cfg_mm.multimodal.vision_hidden_size = 128
    cfg_mm.multimodal.vision_num_layers = 1
    cfg_mm.multimodal.vision_num_heads = 4
    cfg_mm.multimodal.audio_hidden_size = 64
    cfg_mm.multimodal.audio_num_layers = 1
    torch.manual_seed(9)
    model = HeliosLMv5_shim(cfg_mm).eval()
    wired = (model.vision_encoder.max_grid == cfg_mm.multimodal.vision_max_grid
             and model.vision_proj.in_features == 128)
    with torch.no_grad():
        logits, _, _ = model(torch.randint(3, 1024, (1, 4)),
                             images=torch.randn(1, 3, 224, 224))
    mm_ok = wired and torch.isfinite(logits).all().item()

    ok = ran_ok and raised and mm_ok
    report("T9 NaViT grid limits + model wiring", ok,
           f"512px@grid40 out={tuple(out.shape)}, grid32 raises={raised}, "
           f"model vision path wired={wired}, mm logits finite={mm_ok}")


TESTS = [
    test_cache_consistency,
    test_mtp_end_to_end,
    test_vllm_engine,
    test_grpo,
    test_dualpipe,
    test_fp8,
    test_quantization,
    test_audio_streaming,
    test_navit,
]


def main():
    for t in TESTS:
        try:
            t()
        except Exception:
            report(t.__name__, False, "EXCEPTION")
            traceback.print_exc()
    n_ok = sum(1 for _, ok in RESULTS if ok)
    print("=" * 70)
    print(f"INTEGRATION SUMMARY: {n_ok}/{len(RESULTS)} PASS")
    for name, ok in RESULTS:
        if not ok:
            print(f"  FAILED: {name}")
    sys.exit(0 if n_ok == len(RESULTS) else 1)


if __name__ == "__main__":
    main()
