"""GGML block quantization (Q8_0, Q4_K) for the GGUF writer (numpy only).

llama.cpp-family runtimes load quantized weights as GGML *block* formats.
This module encodes and decodes two of them, byte for byte as ggml lays
them out (``ggml-common.h``):

  - **Q8_0** (type 8): blocks of 32 values, ``{f16 d; int8 qs[32]}`` =
    34 bytes. ``x ~= d * q`` with ``d = max|x| / 127``.
  - **Q4_K** (type 12): super-blocks of 256 values, ``{f16 d; f16 dmin;
    uint8 scales[12]; uint8 qs[128]}`` = 144 bytes. Eight sub-blocks of 32
    each carry a 6-bit scale ``sc`` and 6-bit min ``m``; ``x ~= d*sc*q -
    dmin*m`` with 4-bit ``q``. The twelve scale bytes pack the sixteen
    6-bit values as ggml's ``get_scale_min_k4`` reads them.

Rows are quantized along the innermost (last) dimension, which is the
dimension ggml blocks run along, so a tensor's last dim must be a multiple
of the block size.

Deliberate simplifications (loud where it matters):

  - The Q4_K encoder picks each sub-block's scale and min from its own
    min/max (``min`` clamped to <= 0, as ggml requires). llama.cpp's
    ``quantize`` searches for a lower-error scale per sub-block, so its
    files decode slightly closer to the source; the format is identical
    and any ggml decoder reads ours.
  - Q8_0 rounds half away from zero (C ``roundf``), as ggml's reference
    quantizer does; Q4_K uses the same rounding.
  - Non-finite inputs and scales that overflow f16 raise ValueError
    instead of writing inf/NaN blocks.
"""
import numpy as np

GGML_TYPE_Q8_0 = 8
GGML_TYPE_Q4_K = 12

QK8_0 = 32
QK_K = 256
K_SCALE_SIZE = 12

# ggml_type -> (values per block, bytes per block)
BLOCK_LAYOUT = {
    GGML_TYPE_Q8_0: (QK8_0, 2 + QK8_0),
    GGML_TYPE_Q4_K: (QK_K, 2 + 2 + K_SCALE_SIZE + QK_K // 2),
}


def _round_away(x):
    """C ``roundf``: halves round away from zero (numpy rounds to even)."""
    return np.sign(x) * np.floor(np.abs(x) + 0.5)


def _rows(x, block):
    x = np.ascontiguousarray(x, dtype=np.float32)
    if x.ndim == 0 or x.shape[-1] % block != 0:
        raise ValueError(
            f"innermost dim {x.shape[-1] if x.ndim else None} is not a "
            f"multiple of the block size {block}"
        )
    if not np.isfinite(x).all():
        raise ValueError("cannot block-quantize non-finite values")
    return x.reshape(-1, block)


def _f16(values, what):
    out = values.astype(np.float16)
    if not np.isfinite(out).all():
        raise ValueError(f"{what} overflows f16; tensor magnitudes too large")
    return out


def nbytes(ggml_type, n_elements):
    """Payload size of ``n_elements`` values in a block type."""
    block, size = BLOCK_LAYOUT[ggml_type]
    if n_elements % block:
        raise ValueError(f"{n_elements} values is not a whole number of "
                         f"{block}-value blocks")
    return n_elements // block * size


# ---------------------------------------------------------------------------
# Q8_0
# ---------------------------------------------------------------------------
def quantize_q8_0(x):
    """Encode a float array (last dim % 32 == 0) as Q8_0 bytes."""
    b = _rows(x, QK8_0)
    amax = np.abs(b).max(axis=1)
    d = amax / 127.0
    inv = np.divide(1.0, d, out=np.zeros_like(d), where=d > 0)
    q = _round_away(b * inv[:, None]).astype(np.int8)
    out = np.empty((b.shape[0], BLOCK_LAYOUT[GGML_TYPE_Q8_0][1]), np.uint8)
    out[:, :2] = _f16(d, "Q8_0 scale").view(np.uint8).reshape(-1, 2)
    out[:, 2:] = q.view(np.uint8)
    return out.tobytes()


def dequantize_q8_0(data, n_elements):
    blocks = np.frombuffer(data, np.uint8).reshape(-1, 2 + QK8_0)
    assert blocks.shape[0] * QK8_0 == n_elements
    d = blocks[:, :2].copy().view(np.float16).astype(np.float32)
    q = blocks[:, 2:].copy().view(np.int8).astype(np.float32)
    return (d * q).reshape(-1)


# ---------------------------------------------------------------------------
# Q4_K
# ---------------------------------------------------------------------------
def _pack_scales(sc, m):
    """Pack (n, 8) 6-bit scales and mins into ggml's (n, 12) layout.

    Inverse of ggml's ``get_scale_min_k4``: sub-blocks 0-3 keep their low
    6 bits in bytes 0-3 (scale) and 4-7 (min); sub-blocks 4-7 put their low
    4 bits in bytes 8-11 and their top 2 bits in the top of bytes 0-7.
    """
    out = np.zeros((sc.shape[0], K_SCALE_SIZE), np.uint8)
    out[:, 0:4] = sc[:, 0:4]
    out[:, 4:8] = m[:, 0:4]
    out[:, 8:12] = (sc[:, 4:8] & 0xF) | ((m[:, 4:8] & 0xF) << 4)
    out[:, 0:4] |= (sc[:, 4:8] >> 4) << 6
    out[:, 4:8] |= (m[:, 4:8] >> 4) << 6
    return out


def _unpack_scales(s):
    sc = np.empty((s.shape[0], 8), np.uint8)
    m = np.empty((s.shape[0], 8), np.uint8)
    sc[:, 0:4] = s[:, 0:4] & 63
    m[:, 0:4] = s[:, 4:8] & 63
    sc[:, 4:8] = (s[:, 8:12] & 0xF) | ((s[:, 0:4] >> 6) << 4)
    m[:, 4:8] = (s[:, 8:12] >> 4) | ((s[:, 4:8] >> 6) << 4)
    return sc, m


def quantize_q4_k(x):
    """Encode a float array (last dim % 256 == 0) as Q4_K bytes."""
    b = _rows(x, QK_K).reshape(-1, 8, 32)
    lo = np.minimum(b.min(axis=2), 0.0)          # ggml: min <= 0
    hi = b.max(axis=2)
    scale = (hi - lo) / 15.0
    neg_min = -lo
    d16 = _f16(scale.max(axis=1) / 63.0, "Q4_K super-block scale")
    dmin16 = _f16(neg_min.max(axis=1) / 63.0, "Q4_K super-block min")
    d = d16.astype(np.float32)[:, None]
    dmin = dmin16.astype(np.float32)[:, None]
    sc = np.clip(_round_away(np.divide(scale, d, out=np.zeros_like(scale),
                                       where=d > 0)), 0, 63).astype(np.uint8)
    m = np.clip(_round_away(np.divide(neg_min, dmin, out=np.zeros_like(neg_min),
                                      where=dmin > 0)), 0, 63).astype(np.uint8)
    step = (d * sc)[:, :, None]
    offset = (dmin * m)[:, :, None]
    q = np.clip(_round_away(np.divide(b + offset, step, out=np.zeros_like(b),
                                      where=step > 0)), 0, 15).astype(np.uint8)
    # qs: each 32-byte group covers two sub-blocks (2g low nibble, 2g+1 high).
    qs = (q[:, 0::2, :] | (q[:, 1::2, :] << 4)).reshape(-1, QK_K // 2)
    out = np.empty((b.shape[0], BLOCK_LAYOUT[GGML_TYPE_Q4_K][1]), np.uint8)
    out[:, 0:2] = d16.view(np.uint8).reshape(-1, 2)
    out[:, 2:4] = dmin16.view(np.uint8).reshape(-1, 2)
    out[:, 4:16] = _pack_scales(sc, m)
    out[:, 16:] = qs
    return out.tobytes()


def dequantize_q4_k(data, n_elements):
    blocks = np.frombuffer(data, np.uint8).reshape(-1, BLOCK_LAYOUT[GGML_TYPE_Q4_K][1])
    assert blocks.shape[0] * QK_K == n_elements
    d = blocks[:, 0:2].copy().view(np.float16).astype(np.float32).reshape(-1)
    dmin = blocks[:, 2:4].copy().view(np.float16).astype(np.float32).reshape(-1)
    sc, m = _unpack_scales(blocks[:, 4:16])
    qs = blocks[:, 16:].reshape(-1, 4, 32)
    q = np.empty((blocks.shape[0], 8, 32), np.float32)
    q[:, 0::2, :] = qs & 0xF
    q[:, 1::2, :] = qs >> 4
    y = d[:, None, None] * sc[:, :, None] * q - dmin[:, None, None] * m[:, :, None]
    return y.reshape(-1)


QUANTIZE = {GGML_TYPE_Q8_0: quantize_q8_0, GGML_TYPE_Q4_K: quantize_q4_k}
DEQUANTIZE = {GGML_TYPE_Q8_0: dequantize_q8_0, GGML_TYPE_Q4_K: dequantize_q4_k}
