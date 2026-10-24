"""v5.122: ggml Q8_0 / Q4_K block quantization in the GGUF writer.

The Q4_K reference decoder below is a line-by-line port of ggml's
``dequantize_row_q4_K`` and ``get_scale_min_k4`` (scalar C loops), kept
separate from the vectorized module so a layout mistake in one does not
hide in the other.
"""
import struct

import numpy as np
import torch

from evastar.src.export import ggml_quants as q
from evastar.src.export.gguf import (
    GGML_TYPE_F32, GGML_TYPE_Q4_K, GGML_TYPE_Q8_0, export_gguf, read_gguf)
from evastar.tests.case_common import _lite_model


def _get_scale_min_k4(j, s):
    if j < 4:
        return s[j] & 63, s[j + 4] & 63
    return ((s[j + 4] & 0xF) | ((s[j - 4] >> 6) << 4),
            (s[j + 4] >> 4) | ((s[j] >> 6) << 4))


def _ref_dequant_q4_k(data):
    out = []
    for off in range(0, len(data), 144):
        blk = data[off:off + 144]
        d, dmin = struct.unpack("<ee", blk[0:4])
        scales, qs = blk[4:16], blk[16:144]
        is_, qi = 0, 0
        for _ in range(0, 256, 64):
            sc, m = _get_scale_min_k4(is_, scales)
            d1, m1 = d * sc, dmin * m
            sc, m = _get_scale_min_k4(is_ + 1, scales)
            d2, m2 = d * sc, dmin * m
            out += [d1 * (qs[qi + l] & 0xF) - m1 for l in range(32)]
            out += [d2 * (qs[qi + l] >> 4) - m2 for l in range(32)]
            qi += 32
            is_ += 2
    return np.array(out, np.float32)


def test_q8_0_exact_layout():
    # Values exactly representable as d * q with d = 2^-4 and one |q| = 127:
    # the encoder must recover d and q, and the bytes are fixed by the spec.
    qv = np.arange(-16, 16, dtype=np.int8) * 4
    qv[0] = -127
    x = qv.astype(np.float32) / 16.0
    raw = q.quantize_q8_0(x)
    assert len(raw) == 34
    assert raw[:2] == np.float16(127 / 16 / 127).tobytes()
    assert raw[2:] == qv.tobytes()
    assert np.array_equal(q.dequantize_q8_0(raw, 32), x)


def test_q8_0_error_bound():
    rng = np.random.default_rng(0)
    x = rng.standard_normal((8, 128)).astype(np.float32) * 3
    y = q.dequantize_q8_0(q.quantize_q8_0(x), x.size).reshape(x.shape)
    blocks = x.reshape(-1, 32)
    d = np.abs(blocks).max(axis=1) / 127
    # Half a step from rounding plus the f16 rounding of d across |q| <= 127.
    bound = d * (0.5 + 127 * 2.0 ** -11) * (1 + 1e-6)
    err = np.abs(blocks - y.reshape(-1, 32)).max(axis=1)
    assert (err <= bound).all()


def test_q4_k_scale_packing_roundtrip():
    sc = np.array(np.meshgrid(np.arange(64), np.arange(64))).reshape(2, -1).T
    sc = sc.astype(np.uint8)
    # Spread every (scale, min) pair over all eight sub-block slots.
    s8 = np.repeat(sc[:, :1], 8, axis=1)
    m8 = np.repeat(sc[:, 1:], 8, axis=1)
    s8 = np.roll(s8, 1, axis=0)
    packed = q._pack_scales(s8, m8)
    us, um = q._unpack_scales(packed)
    assert np.array_equal(us, s8) and np.array_equal(um, m8)


def test_q4_k_decoder_matches_ggml_reference_on_random_bytes():
    rng = np.random.default_rng(1)
    blocks = rng.integers(0, 256, size=(6, 144), dtype=np.uint8)
    blocks[:, 0:2] = np.array([0.0371], np.float16).view(np.uint8)
    blocks[:, 2:4] = np.array([0.0123], np.float16).view(np.uint8)
    raw = blocks.tobytes()
    assert np.allclose(q.dequantize_q4_k(raw, 6 * 256), _ref_dequant_q4_k(raw),
                       rtol=0, atol=1e-6)


def test_q4_k_roundtrip_error():
    rng = np.random.default_rng(2)
    x = rng.standard_normal((4, 512)).astype(np.float32)
    x[0, :256] += 5.0          # all-positive super-block: min clamps to 0
    raw = q.quantize_q4_k(x)
    assert len(raw) == q.nbytes(GGML_TYPE_Q4_K, x.size) == 8 * 144
    y = q.dequantize_q4_k(raw, x.size)
    assert np.allclose(y, _ref_dequant_q4_k(raw), rtol=0, atol=1e-6)
    sub = x.reshape(-1, 32)
    rng_sub = sub.max(axis=1) - np.minimum(sub.min(axis=1), 0)
    err = np.abs(sub - y.reshape(-1, 32)).max(axis=1)
    # One 4-bit step is range/15; 6-bit scales and f16 add under one more.
    assert (err <= rng_sub / 15 * 1.5 + 1e-6).all()
    rel = np.sqrt(((x.reshape(-1) - y) ** 2).mean()) / x.std()
    assert rel < 0.12, rel


def test_quantizers_reject_bad_input():
    for fn, n in ((q.quantize_q8_0, 32), (q.quantize_q4_k, 256)):
        for bad in (np.zeros(n + 1, np.float32), np.full(n, np.inf, np.float32)):
            try:
                fn(bad)
            except ValueError:
                continue
            raise AssertionError(f"{fn.__name__} accepted {bad[:2]}")


def test_export_quantized_files(tmp_path):
    model, config = _lite_model(777)
    sd = model.state_dict()
    f32 = export_gguf(model, tmp_path / "m-f32.gguf", config=config)
    sizes = {}
    for dtype, ftype, main in (("q8_0", 7, GGML_TYPE_Q8_0),
                               ("q4_k", 14, GGML_TYPE_Q4_K)):
        path = export_gguf(model, tmp_path / f"m-{dtype}.gguf", config=config,
                           dtype=dtype)
        assert path.read_bytes() == export_gguf(
            model, tmp_path / "again.gguf", config=config, dtype=dtype).read_bytes()
        gf = read_gguf(path)
        assert gf.metadata["general.file_type"] == ftype
        types = {t.name: t.ggml_type for t in gf.tensors}
        assert set(types) == set(sd)
        for name, tensor in sd.items():
            if tensor.dim() < 2 or name.endswith("router.weight"):
                assert types[name] == GGML_TYPE_F32, name
            elif tensor.shape[-1] % 32 == 0:
                assert types[name] in (main, GGML_TYPE_Q8_0), name
        assert main in types.values()
        tol = 0.02 if dtype == "q8_0" else 0.15
        for name, tensor in sd.items():
            back = gf.get_tensor(name).to(torch.float32)
            assert back.shape == tensor.shape
            scale = tensor.float().abs().max().item() or 1.0
            assert (back - tensor.float()).abs().max().item() <= tol * scale, name
        sizes[dtype] = path.stat().st_size
    assert sizes["q4_k"] < sizes["q8_0"] < f32.stat().st_size * 0.4


def test_q8_0_model_logits_close(tmp_path):
    model, config = _lite_model(778)
    path = export_gguf(model, tmp_path / "m.gguf", config=config, dtype="q8_0")
    gf = read_gguf(path)
    clone, _ = _lite_model(778)
    clone.load_state_dict({n: gf.get_tensor(n).to(t.dtype)
                           for n, t in model.state_dict().items()})
    ids = torch.randint(0, config.vocab_size, (2, 24),
                        generator=torch.Generator().manual_seed(0))
    with torch.no_grad():
        a = model(ids)
        b = clone(ids)
    a = a["logits"] if isinstance(a, dict) else getattr(a, "logits", a)
    b = b["logits"] if isinstance(b, dict) else getattr(b, "logits", b)
    if isinstance(a, tuple):
        a, b = a[0], b[0]
    cos = torch.nn.functional.cosine_similarity(a.flatten(), b.flatten(), dim=0)
    # A random-init model has near-flat logits, so top-1 agreement says
    # little; cosine over all logits is the stable signal (0.994-0.9999
    # over seeds 1, 2, 3, 778).
    assert cos.item() > 0.99, cos.item()
