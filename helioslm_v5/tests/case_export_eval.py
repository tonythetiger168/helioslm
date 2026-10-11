"""HeliosLM v5 test cases — GGUF export + eval harness.

Split out of the monolithic test_v5.py on 2026-10-11; the test_v5.py
runner imports every case module and executes the tests in the
historical order with unchanged exit-code semantics. The case_*.py
filename deliberately avoids pytest's default test_*.py collection
pattern so the CI pytest step does not double-run these.
"""

import math
from pathlib import Path

import torch
import torch.nn as nn

from helioslm_v5.configs.config_v5 import HeliosLMv5Config
from helioslm_v5.tests.case_common import (_expect_raises, _lite_model, _pass)


# ----------------------------------------------------------------------
# GGUF v3 export (limitations task): spec-faithful container writer +
# reader, verified by bit-exact round-trip and raw-header inspection.
# ----------------------------------------------------------------------
def test_gguf_export():
    import struct as _struct
    import tempfile
    from helioslm_v5.src.export.gguf import (
        export_gguf, read_gguf, GGML_TYPE_F32, GGML_TYPE_F16,
    )

    model, config = _lite_model(777)
    sd = model.state_dict()

    with tempfile.TemporaryDirectory() as tmp:
        p32 = Path(tmp) / "model-f32.gguf"
        export_gguf(model, p32, config=config,
                    extra_metadata={"custom.tags": ["reference", "cpu"],
                                    "custom.lr": 0.5, "custom.count": 7,
                                    "custom.flag": True})
        raw = p32.read_bytes()

        # 1) Raw header, parsed with plain struct (independent of the
        #    module's own reader): magic, version 3, counts.
        magic, version, n_tensors, n_kv = _struct.unpack_from("<4sIQQ", raw, 0)
        assert magic == b"GGUF", f"bad magic {magic!r}"
        assert version == 3, f"bad version {version}"
        assert n_tensors == len(sd), f"tensor count {n_tensors} != {len(sd)}"
        assert n_kv > 0
        # Raw dim-order check: GGML stores dims innermost-first, so the
        # (vocab, hidden) embedding must appear as [hidden, vocab] on disk.
        needle = (_struct.pack("<Q", len(b"embed_tokens.weight"))
                  + b"embed_tokens.weight"
                  + _struct.pack("<IQQI", 2, config.hidden_size,
                                 config.vocab_size, GGML_TYPE_F32))
        assert needle in raw, "embed_tokens tensor info not found in GGML dim order"

        # 2) Full round-trip through the module reader.
        gf = read_gguf(p32)
        assert gf.metadata["general.architecture"] == "helioslm"
        assert gf.metadata["general.name"] == config.model_name
        assert gf.metadata["general.license"] == "Apache-2.0"
        assert gf.alignment == 32 and gf.metadata["general.alignment"] == 32
        assert gf.metadata["general.file_type"] == 0  # ALL_F32
        assert gf.metadata["helioslm.context_length"] == config.max_position_embeddings
        assert gf.metadata["helioslm.embedding_length"] == config.hidden_size
        assert gf.metadata["helioslm.block_count"] == config.num_hidden_layers
        assert gf.metadata["helioslm.feed_forward_length"] == config.intermediate_size
        assert gf.metadata["helioslm.vocab_size"] == config.vocab_size
        assert gf.metadata["helioslm.attention.head_count"] == config.attention.num_attention_heads
        assert gf.metadata["helioslm.attention.head_count_kv"] == config.attention.num_key_value_heads
        assert gf.metadata["helioslm.attention.key_length"] == config.attention.head_dim
        assert gf.metadata["helioslm.attention.value_length"] == config.attention.v_head_dim
        assert gf.metadata["helioslm.attention.kv_lora_rank"] == config.attention.kv_latent_dim
        assert gf.metadata["helioslm.rope.dimension_count"] == config.attention.rope_head_dim
        assert gf.metadata["helioslm.expert_count"] == config.moe.num_experts
        assert gf.metadata["helioslm.expert_used_count"] == config.moe.num_activated_experts
        # FLOAT32 metadata: compare against the f32 rounding of the config eps.
        eps_f32 = _struct.unpack("<f", _struct.pack("<f", config.rms_norm_eps))[0]
        assert gf.metadata["helioslm.attention.layer_norm_rms_epsilon"] == eps_f32
        # extra_metadata round-trip (array/string/float64/uint64/bool).
        assert gf.metadata["custom.tags"] == ["reference", "cpu"]
        assert gf.metadata["custom.lr"] == 0.5
        assert gf.metadata["custom.count"] == 7
        assert gf.metadata["custom.flag"] is True

        assert set(gf.tensor_names()) == set(sd.keys())
        assert all(t.offset % 32 == 0 for t in gf.tensors), "misaligned tensor offset"
        assert gf._data_start % 32 == 0, "tensor-data block misaligned"
        n_bad = 0
        for k, v in sd.items():
            t = gf.get_tensor(k)
            if t.shape != v.shape or not torch.equal(t, v):
                n_bad += 1
        assert n_bad == 0, f"{n_bad} tensors did not round-trip bit-exactly"

        # 3) Determinism: exporting the same model twice is byte-identical.
        p32b = Path(tmp) / "model-f32-b.gguf"
        export_gguf(model, p32b, config=config,
                    extra_metadata={"custom.tags": ["reference", "cpu"],
                                    "custom.lr": 0.5, "custom.count": 7,
                                    "custom.flag": True})
        assert p32b.read_bytes() == raw, "two exports of the same model differ"

        # 4) F16 export: MOSTLY_F16, exact half-precision values, ~half size.
        p16 = Path(tmp) / "model-f16.gguf"
        export_gguf(model, p16, config=config, dtype="f16")
        gf16 = read_gguf(p16)
        assert gf16.metadata["general.file_type"] == 1  # MOSTLY_F16
        assert all(t.ggml_type == GGML_TYPE_F16 for t in gf16.tensors)
        n_bad16 = 0
        for k, v in sd.items():
            t = gf16.get_tensor(k)
            if t.dtype != torch.float16 or not torch.equal(t, v.half()):
                n_bad16 += 1
        assert n_bad16 == 0, f"{n_bad16} f16 tensors mismatch the half() cast"
        assert p16.stat().st_size < p32.stat().st_size * 0.6, (
            f"f16 file not smaller: {p16.stat().st_size} vs {p32.stat().st_size}")

        # 5) Loud export errors.
        _expect_raises(ValueError, lambda: export_gguf(
            model, Path(tmp) / "x.gguf", dtype="q4_0"), "unknown dtype")
        _expect_raises(ValueError, lambda: export_gguf(
            {"x": torch.zeros(4, dtype=torch.int64)}, Path(tmp) / "x.gguf"),
            "int64 tensor")
        _expect_raises(ValueError, lambda: export_gguf(
            {"a" * 65: torch.zeros(4)}, Path(tmp) / "x.gguf"), "name > 64 bytes")
        _expect_raises(ValueError, lambda: export_gguf(
            [("dup", torch.zeros(4)), ("dup", torch.zeros(4))],
            Path(tmp) / "x.gguf"), "duplicate names")
        _expect_raises(ValueError, lambda: export_gguf(
            {"scalar": torch.tensor(1.0)}, Path(tmp) / "x.gguf"), "0-d tensor")
        _expect_raises(ValueError, lambda: export_gguf(
            {"deep": torch.zeros(1, 1, 1, 1, 1)}, Path(tmp) / "x.gguf"),
            "5-d tensor")
        _expect_raises(ValueError, lambda: export_gguf(
            {"big": torch.full((4,), 1e30)}, Path(tmp) / "x.gguf", dtype="f16"),
            "f16 overflow")
        _expect_raises(ValueError, lambda: export_gguf(
            {"x": torch.zeros(4)}, Path(tmp) / "x.gguf",
            extra_metadata={"Bad Key": 1}), "invalid metadata key")
        _expect_raises(ValueError, lambda: export_gguf(
            {"x": torch.zeros(4)}, Path(tmp) / "x.gguf",
            extra_metadata={"general.name": "clash"}), "duplicate metadata key")
        _expect_raises(ValueError, lambda: export_gguf(
            {"x": torch.zeros(4)}, Path(tmp) / "x.gguf",
            extra_metadata={"custom.bad": object()}), "unsupported metadata value")

        # 6) Loud reader errors: bad magic, wrong version, truncation, and a
        #    hand-crafted misaligned tensor offset (offset 4 % 32 != 0).
        bad = Path(tmp) / "bad-magic.gguf"
        bad.write_bytes(b"NOPE" + raw[4:])
        _expect_raises(ValueError, lambda: read_gguf(bad), "bad magic")
        badv = Path(tmp) / "bad-version.gguf"
        badv.write_bytes(raw[:4] + _struct.pack("<I", 2) + raw[8:])
        _expect_raises(ValueError, lambda: read_gguf(badv), "version 2")
        trunc = Path(tmp) / "truncated.gguf"
        trunc.write_bytes(raw[:-100])
        _expect_raises(ValueError, lambda: read_gguf(trunc), "truncated file")

        mis = bytearray()
        mis += b"GGUF" + _struct.pack("<IQQ", 3, 1, 1)
        key = b"general.alignment"
        mis += _struct.pack("<Q", len(key)) + key + _struct.pack("<II", 4, 32)
        mis += _struct.pack("<Q", 1) + b"x"  # name
        mis += _struct.pack("<IQQ", 1, 4, GGML_TYPE_F32)  # 1 dim, 4 elems, F32
        mis += _struct.pack("<Q", 4)  # misaligned offset (not a multiple of 32)
        mis += bytes((32 - len(mis) % 32) % 32) + bytes(64)
        p_mis = Path(tmp) / "misaligned.gguf"
        p_mis.write_bytes(bytes(mis))
        _expect_raises(ValueError, lambda: read_gguf(p_mis), "misaligned offset")

    _pass("test_gguf_export",
          f"{len(sd)} tensors bit-exact (f32) + exact f16, header/metadata/"
          f"alignment/determinism OK, 15 loud-error paths OK")


def test_eval_harness():
    """v5.11: loglikelihood harness correctness + learning-signal detection."""
    from helioslm_v5.eval.harness import (
        TASKS, loglikelihood, multiple_choice, run_harness)
    from helioslm_v5.src.model_v5 import HeliosLMv5
    torch.manual_seed(11)
    cfg = HeliosLMv5Config(size="lite")
    model = HeliosLMv5(cfg)
    model.eval()

    # 1) loglikelihood == manual teacher-forced computation from forward()
    ctx, cont = [5, 6, 7], [9, 9]
    manual = 0.0
    with torch.no_grad():
        logits, _, _ = model(torch.tensor([ctx + cont]))
        lp = torch.log_softmax(logits[0, :len(ctx) + len(cont) - 1].float(), -1)
        manual = sum(lp[len(ctx) - 1 + i, t].item() for i, t in enumerate(cont))
    assert abs(loglikelihood(model, ctx, cont) - manual) < 1e-5

    # 2) multiple_choice / run_harness plumbing on the synthetic task
    docs = TASKS["copy_vs_reverse"](n_samples=8, seed=3)
    report = run_harness(model, {"copy_vs_reverse": docs})
    r = report["copy_vs_reverse"]
    assert r["n"] == 8 and 0.0 <= r["acc"] <= 1.0
    pred, scores = multiple_choice(model, docs[0]["context"], docs[0]["choices"])
    assert 0 <= pred < len(scores) and all(math.isfinite(s) for s in scores)

    # 3) harness detects a learned signal: memorize context -> [9, 9]
    model.train()
    opt = torch.optim.Adam(model.parameters(), lr=0.05)
    x = torch.tensor([ctx + cont])
    for _ in range(120):
        logits, _, _ = model(x)
        loss = torch.nn.functional.cross_entropy(
            logits[0, :2].reshape(-1, logits.shape[-1]),
            torch.tensor([9, 9]))
        opt.zero_grad()
        loss.backward()
        opt.step()
    model.eval()
    ll_good = loglikelihood(model, ctx, [9, 9])
    ll_bad = max(loglikelihood(model, ctx, [t, t])
                 for t in (8, 10, 42) )
    assert ll_good > ll_bad, "harness should prefer the memorized continuation"
    _pass("test_eval_harness",
          "loglikelihood==manual; harness plumbing; memorized-signal detected")
