"""GGUF v3 export/import for HeliosLM checkpoints (PyTorch + numpy + struct).

GGUF (GGML Universal Format) is the single-file model container used by
llama.cpp and other GGML-based executors. This module implements a minimal,
spec-faithful subset of GGUF v3 (little-endian, as mandated by the spec for
files that do not declare otherwise):

  - header: magic ``b"GGUF"``, version 3, tensor count, metadata-kv count
    (all counts uint64, per the v2->v3 struct definitions);
  - metadata key/value pairs with the 13 standard value types (uint8 ..
    float64, bool, string, nested arrays);
  - tensor infos: name (<= 64 bytes), up to 4 dimensions, GGML type code,
    data offset relative to the tensor-data block;
  - tensor data padded to ``general.alignment`` (default 32, the spec's
    fallback when the key is absent).

Deliberate simplifications (documented, loud):

  - Only GGML_TYPE_F32 and GGML_TYPE_F16 tensor payloads are written and
    read. The quantized GGML types (Q4_0, Q8_0, K-quants, ...) are block
    formats that would need their own encoders; passing HeliosLM
    GPTQ/AWQ-packed buffers here raises ValueError instead of silently
    writing a mislabeled file.
  - Only little-endian files are produced and parsed (GGUF's default).
  - ``general.architecture`` is ``"helioslm"``. No GGML executor (llama.cpp
    included) implements a "helioslm" architecture, so the produced file
    cannot be *run* by those executors — it is a spec-compliant,
    self-describing checkpoint container: any GGUF reader (including the
    official ``gguf`` Python package) can enumerate its metadata and
    tensors, and ``read_gguf`` below round-trips every byte. Do not expect
    ``llama-cli helioslm.gguf`` to work.
  - Dimensions follow the GGML convention: the info block lists them
    innermost-first, i.e. a PyTorch tensor of shape ``(out, in)`` is stored
    with ``dimensions = [in, out]`` while the raw bytes stay row-major
    contiguous. ``read_gguf`` reverses them back, so round-tripped tensors
    have their original shape.

Verification: ``tests/test_v5.py::test_gguf_export`` round-trips a lite
model bit-exactly (F32), checks F16 casting, alignment of every tensor
offset, determinism (two exports are byte-identical) and every loud error
path. During development the written files were additionally parsed with
the official ``gguf`` package reader (``gguf.GGUFReader``) as an
independent format-conformance check.

Spec: https://github.com/ggml-org/ggml/blob/master/docs/gguf.md
"""
import re
import struct
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch

GGUF_MAGIC = b"GGUF"
GGUF_VERSION = 3
DEFAULT_ALIGNMENT = 32          # spec fallback when general.alignment is absent
MAX_TENSOR_NAME_BYTES = 64      # spec: tensor names are at most 64 bytes
MAX_METADATA_KEY_BYTES = 65535  # spec: metadata keys are at most 2^16-1 bytes
MAX_TENSOR_DIMS = 4             # spec: "currently at most 4"

ARCHITECTURE = "helioslm"

# ggml_type codes (subset we can encode/decode).
GGML_TYPE_F32 = 0
GGML_TYPE_F16 = 1
_GGML_ITEMSIZE = {GGML_TYPE_F32: 4, GGML_TYPE_F16: 2}
_GGML_TORCH_DTYPE = {GGML_TYPE_F32: torch.float32, GGML_TYPE_F16: torch.float16}
_GGML_NAME = {GGML_TYPE_F32: "F32", GGML_TYPE_F16: "F16"}

# gguf_metadata_value_type codes.
_VAL_UINT8, _VAL_INT8, _VAL_UINT16, _VAL_INT16 = 0, 1, 2, 3
_VAL_UINT32, _VAL_INT32, _VAL_FLOAT32, _VAL_BOOL = 4, 5, 6, 7
_VAL_STRING, _VAL_ARRAY, _VAL_UINT64, _VAL_INT64, _VAL_FLOAT64 = 8, 9, 10, 11, 12

_VAL_STRUCT = {
    _VAL_UINT8: "<B", _VAL_INT8: "<b", _VAL_UINT16: "<H", _VAL_INT16: "<h",
    _VAL_UINT32: "<I", _VAL_INT32: "<i", _VAL_FLOAT32: "<f", _VAL_BOOL: "<?",
    _VAL_UINT64: "<Q", _VAL_INT64: "<q", _VAL_FLOAT64: "<d",
}

_FILE_DTYPES = {"f32": (GGML_TYPE_F32, torch.float32, "<f4"),
                "f16": (GGML_TYPE_F16, torch.float16, "<f2")}
_FILE_TYPE_ENUM = {"f32": 0, "f16": 1}  # general.file_type: ALL_F32 / MOSTLY_F16

# Source tensor dtypes we accept. Anything else (int buffers, float64, ...)
# raises loudly rather than being silently re-typed.
_SRC_DTYPES = {torch.float32, torch.float16, torch.bfloat16, torch.float64}

_KEY_RE = re.compile(r"^[a-z0-9_]+(\.[a-z0-9_]+)*$")


# ---------------------------------------------------------------------------
# low-level packing helpers
# ---------------------------------------------------------------------------
def _pack_str(s):
    data = s.encode("utf-8") if isinstance(s, str) else s
    return struct.pack("<Q", len(data)) + data


def _validate_key(key):
    encoded = key.encode("utf-8") if isinstance(key, str) else b""
    if not isinstance(key, str) or not key:
        raise ValueError(f"metadata key must be a non-empty str, got {key!r}")
    if len(encoded) > MAX_METADATA_KEY_BYTES:
        raise ValueError(
            f"metadata key {key[:40]!r}... is {len(encoded)} bytes; the GGUF "
            f"limit is {MAX_METADATA_KEY_BYTES}"
        )
    if not _KEY_RE.match(key):
        raise ValueError(
            f"metadata key {key!r} is not valid GGUF lower_snake_case dotted "
            "ASCII (segments of [a-z0-9_] joined by '.')"
        )


def _pack_value(vtype, value):
    """Pack one metadata value. Arrays recurse (nested arrays are legal)."""
    if vtype in _VAL_STRUCT:
        return struct.pack(_VAL_STRUCT[vtype], value)
    if vtype == _VAL_STRING:
        return _pack_str(value)
    if vtype == _VAL_ARRAY:
        sub_type, items = value
        out = struct.pack("<IQ", sub_type, len(items))
        for item in items:
            out += _pack_value(sub_type, item)
        return out
    raise ValueError(f"cannot pack metadata value type {vtype}")


def _python_to_typed(value):
    """Map a plain Python value to (vtype, packed-ready value), losslessly.

    int -> UINT64 when >= 0 else INT64; float -> FLOAT64 (FLOAT32 would
    silently round); str -> STRING; bool -> BOOL; lists/tuples become ARRAY
    with a homogeneous element type.
    """
    if isinstance(value, bool):
        return _VAL_BOOL, value
    if isinstance(value, int):
        if value >= 0:
            if value >= 2**64:
                raise ValueError(f"metadata int {value} does not fit uint64")
            return _VAL_UINT64, value
        if value < -(2**63):
            raise ValueError(f"metadata int {value} does not fit int64")
        return _VAL_INT64, value
    if isinstance(value, float):
        return _VAL_FLOAT64, value
    if isinstance(value, str):
        return _VAL_STRING, value
    if isinstance(value, (list, tuple)):
        if not value:
            raise ValueError("empty metadata arrays are not written (loud)")
        typed = [_python_to_typed(v) for v in value]
        sub_types = {t for t, _ in typed}
        if len(sub_types) != 1:
            raise ValueError(
                f"metadata array elements must share one type, got {sub_types}"
            )
        return _VAL_ARRAY, (sub_types.pop(), [v for _, v in typed])
    raise ValueError(
        f"unsupported metadata value type {type(value).__name__} for value "
        f"{value!r}; use str/int/float/bool or a homogeneous list of them"
    )


# ---------------------------------------------------------------------------
# metadata
# ---------------------------------------------------------------------------
def _config_metadata(config):
    """Standard GGUF LLM keys, namespaced under the helioslm architecture."""
    def need(obj, field):
        value = getattr(obj, field, None)
        if value is None:
            raise ValueError(
                f"config is missing required field {field!r}; pass a "
                "HeliosLMv5Config (or an object with the same fields)"
            )
        return value

    a = need(config, "attention")
    m = need(config, "moe")
    kv = []  # ordered list of (key, vtype, value)

    def u32(key, value):
        kv.append((key, _VAL_UINT32, int(value)))

    kv.append(("general.architecture", _VAL_STRING, ARCHITECTURE))
    u32("general.alignment", DEFAULT_ALIGNMENT)
    kv.append(("general.license", _VAL_STRING, "Apache-2.0"))
    kv.append((
        "general.description", _VAL_STRING,
        "HeliosLM reference-stack checkpoint (MLA + sigmoid-MoE). Container "
        "only: no GGML executor implements the 'helioslm' architecture.",
    ))

    arch = ARCHITECTURE
    u32(f"{arch}.context_length", need(config, "max_position_embeddings"))
    u32(f"{arch}.embedding_length", need(config, "hidden_size"))
    u32(f"{arch}.block_count", need(config, "num_hidden_layers"))
    u32(f"{arch}.feed_forward_length", need(config, "intermediate_size"))
    u32(f"{arch}.vocab_size", need(config, "vocab_size"))
    u32(f"{arch}.attention.head_count", need(a, "num_attention_heads"))
    u32(f"{arch}.attention.head_count_kv", need(a, "num_key_value_heads"))
    u32(f"{arch}.attention.key_length", a.no_rope_head_dim + a.rope_head_dim)
    u32(f"{arch}.attention.value_length", need(a, "v_head_dim"))
    u32(f"{arch}.attention.kv_lora_rank", need(a, "kv_latent_dim"))
    u32(f"{arch}.attention.q_lora_rank", need(a, "q_lora_rank"))
    kv.append((f"{arch}.attention.layer_norm_rms_epsilon", _VAL_FLOAT32,
               float(need(config, "rms_norm_eps"))))
    u32(f"{arch}.rope.dimension_count", need(a, "rope_head_dim"))
    u32(f"{arch}.expert_count", need(m, "num_experts"))
    u32(f"{arch}.expert_used_count", need(m, "num_activated_experts"))
    u32(f"{arch}.expert_shared_count", need(m, "num_shared_experts"))
    return kv


# ---------------------------------------------------------------------------
# export
# ---------------------------------------------------------------------------
def _resolve_tensors(model_or_tensors):
    """Normalize the input into an ordered list of (name, tensor) pairs."""
    if hasattr(model_or_tensors, "state_dict"):
        items = list(model_or_tensors.state_dict().items())
    elif isinstance(model_or_tensors, Mapping):
        items = list(model_or_tensors.items())
    else:
        items = list(model_or_tensors)  # iterable of (name, tensor)
    seen = set()
    out = []
    for name, tensor in items:
        if not isinstance(name, str) or not name:
            raise ValueError(f"tensor name must be a non-empty str, got {name!r}")
        if name in seen:
            raise ValueError(f"duplicate tensor name {name!r}")
        seen.add(name)
        if not isinstance(tensor, torch.Tensor):
            raise ValueError(
                f"tensor {name!r} is a {type(tensor).__name__}, expected "
                "torch.Tensor"
            )
        out.append((name, tensor))
    if not out:
        raise ValueError("no tensors to export (empty state dict / mapping)")
    return out


def export_gguf(model_or_tensors, path, *, config=None, dtype="f32", name=None,
                extra_metadata=None):
    """Export a HeliosLM model (or an explicit tensor mapping) to GGUF v3.

    Args:
        model_or_tensors: a ``torch.nn.Module`` (its ``state_dict()`` is
            exported), a mapping name -> tensor, or an iterable of
            ``(name, tensor)`` pairs.
        path: destination file path (overwritten; parent must exist).
        config: optional HeliosLMv5Config; when given, the standard GGUF LLM
            hyperparameter keys are written under the ``helioslm.`` prefix.
        dtype: ``"f32"`` (default, bit-exact) or ``"f16"`` (casts every
            tensor; raises loudly if a finite value overflows the f16 range).
        name: ``general.name``; defaults to ``config.model_name`` when a
            config is given, else "HeliosLM".
        extra_metadata: optional mapping of additional GGUF keys (must be
            lower_snake_case dotted ASCII) to str/int/float/bool/list values.

    Returns:
        The path written (``pathlib.Path``).

    Loud errors (ValueError) on: unknown ``dtype``, unsupported tensor dtype
    (int/float64 buffers), non-ASCII or >64-byte tensor names, duplicate
    names, 0-d or >4-d tensors, f16 overflow, invalid metadata keys/values.
    """
    if dtype not in _FILE_DTYPES:
        raise ValueError(f"dtype must be one of {sorted(_FILE_DTYPES)}, got {dtype!r}")
    ggml_type, torch_dtype, np_dtype = _FILE_DTYPES[dtype]
    path = Path(path)

    pairs = _resolve_tensors(model_or_tensors)

    # -- metadata -----------------------------------------------------------
    kv = _config_metadata(config) if config is not None else [
        ("general.architecture", _VAL_STRING, ARCHITECTURE),
        ("general.alignment", _VAL_UINT32, DEFAULT_ALIGNMENT),
    ]
    if name is None and config is not None:
        name = getattr(config, "model_name", None)
    kv.append(("general.name", _VAL_STRING, name or "HeliosLM"))
    kv.append(("general.file_type", _VAL_UINT32, _FILE_TYPE_ENUM[dtype]))
    if extra_metadata:
        if not isinstance(extra_metadata, Mapping):
            raise ValueError("extra_metadata must be a mapping key -> value")
        for key, value in extra_metadata.items():
            _validate_key(key)
            if any(key == k for k, _, _ in kv):
                raise ValueError(f"extra_metadata key {key!r} duplicates a built-in key")
            vtype, typed_value = _python_to_typed(value)
            kv.append((key, vtype, typed_value))

    # -- tensor conversion ---------------------------------------------------
    converted = []  # (name, shape, data_bytes)
    for tname, tensor in pairs:
        encoded = tname.encode("utf-8")
        if len(encoded) > MAX_TENSOR_NAME_BYTES or not encoded.isascii():
            raise ValueError(
                f"tensor name {tname!r} must be ASCII and at most "
                f"{MAX_TENSOR_NAME_BYTES} bytes (got {len(encoded)})"
            )
        if tensor.dtype not in _SRC_DTYPES:
            raise ValueError(
                f"tensor {tname!r} has dtype {tensor.dtype}; only float "
                "tensors (f32/f16/bf16/f64) can be exported. Quantized "
                "(GPTQ/AWQ-packed) or integer buffers are not representable "
                "by this writer — dequantize or skip them explicitly."
            )
        if tensor.dim() == 0 or tensor.dim() > MAX_TENSOR_DIMS:
            raise ValueError(
                f"tensor {tname!r} has {tensor.dim()} dims; GGUF supports 1.."
                f"{MAX_TENSOR_DIMS}. Reshape scalars to [1] explicitly if you "
                "want them exported."
            )
        t = tensor.detach().to(torch_dtype).contiguous().cpu()
        if torch_dtype == torch.float16 and tensor.dtype != torch.float16:
            overflow = torch.isinf(t) & torch.isfinite(tensor.detach())
            if overflow.any():
                raise ValueError(
                    f"tensor {tname!r} has {int(overflow.sum())} finite values "
                    "outside the f16 range; refusing lossy f16 cast"
                )
        data = t.numpy().astype(np.dtype(np_dtype), copy=False).tobytes()
        converted.append((tname, tuple(tensor.shape), data))

    # -- write ---------------------------------------------------------------
    def align_up(offset):
        return offset + (DEFAULT_ALIGNMENT - offset % DEFAULT_ALIGNMENT) % DEFAULT_ALIGNMENT

    blob = bytearray()
    blob += GGUF_MAGIC
    blob += struct.pack("<IQQ", GGUF_VERSION, len(converted), len(kv))
    for key, vtype, value in kv:
        blob += _pack_str(key)
        blob += struct.pack("<I", vtype)
        blob += _pack_value(vtype, value)

    offsets = []
    cursor = 0
    for _, _, data in converted:
        offsets.append(cursor)
        cursor = align_up(cursor + len(data))

    for (tname, shape, _), offset in zip(converted, offsets):
        blob += _pack_str(tname)
        blob += struct.pack("<I", len(shape))
        for dim in reversed(shape):  # GGML lists dims innermost-first
            blob += struct.pack("<Q", dim)
        blob += struct.pack("<IQ", ggml_type, offset)

    blob += bytes(align_up(len(blob)) - len(blob))  # tensor-data block alignment
    for _, _, data in converted:
        assert len(blob) % DEFAULT_ALIGNMENT == 0, "tensor-data misalignment"
        blob += data
        pad = align_up(len(data)) - len(data)
        if pad:
            blob += bytes(pad)

    path.write_bytes(bytes(blob))
    return path


# ---------------------------------------------------------------------------
# reader (verification + inspection)
# ---------------------------------------------------------------------------
@dataclass
class GGUFTensorInfo:
    """One tensor entry. ``shape`` is in PyTorch (outermost-first) order."""
    name: str
    shape: tuple
    ggml_type: int
    offset: int          # relative to the tensor-data block, bytes
    nbytes: int          # payload size in bytes


class GGUFFile:
    """Parsed GGUF file: ``metadata`` dict + ``tensors`` list + accessors."""

    def __init__(self, metadata, tensors, raw, data_start, alignment):
        self.metadata = metadata
        self.tensors = tensors
        self._raw = raw
        self._data_start = data_start
        self.alignment = alignment

    def tensor_names(self):
        return [t.name for t in self.tensors]

    def tensor_bytes(self, name):
        for t in self.tensors:
            if t.name == name:
                start = self._data_start + t.offset
                return bytes(self._raw[start:start + t.nbytes])
        raise ValueError(f"no tensor named {name!r} in file")

    def get_tensor(self, name):
        """Return the tensor as a torch.Tensor (F32/F16 payloads only)."""
        for t in self.tensors:
            if t.name == name:
                if t.ggml_type not in _GGML_TORCH_DTYPE:
                    raise ValueError(
                        f"tensor {name!r} has GGML type {t.ggml_type}; only "
                        "F32/F16 payloads can be materialized as torch tensors"
                    )
                raw = bytearray(self.tensor_bytes(name))
                out = torch.frombuffer(raw, dtype=_GGML_TORCH_DTYPE[t.ggml_type])
                return out.reshape(t.shape).clone()
        raise ValueError(f"no tensor named {name!r} in file")


class _Cursor:
    def __init__(self, data):
        self.data = data
        self.pos = 0

    def read(self, fmt):
        size = struct.calcsize(fmt)
        if self.pos + size > len(self.data):
            raise ValueError(
                f"truncated GGUF file: need {size} bytes at offset {self.pos}, "
                f"file has {len(self.data)}"
            )
        value = struct.unpack_from(fmt, self.data, self.pos)
        self.pos += size
        return value[0] if len(value) == 1 else value

    def read_str(self):
        n = self.read("<Q")
        if self.pos + n > len(self.data):
            raise ValueError(
                f"truncated GGUF file: string of {n} bytes at offset {self.pos}"
            )
        out = bytes(self.data[self.pos:self.pos + n]).decode("utf-8")
        self.pos += n
        return out


def _read_value(cur, vtype):
    if vtype in _VAL_STRUCT:
        return cur.read(_VAL_STRUCT[vtype])
    if vtype == _VAL_STRING:
        return cur.read_str()
    if vtype == _VAL_ARRAY:
        sub_type = cur.read("<I")
        n = cur.read("<Q")
        return [_read_value(cur, sub_type) for _ in range(n)]
    raise ValueError(f"unknown metadata value type {vtype} in file")


def read_gguf(path):
    """Parse a GGUF v3 file, loudly.

    Returns a :class:`GGUFFile`. Raises ValueError on a bad magic, an
    unsupported version, truncation, an invalid ``general.alignment``, a
    misaligned tensor offset, an out-of-bounds tensor payload, duplicate
    tensor names, or unknown value types.
    """
    raw = Path(path).read_bytes()
    cur = _Cursor(raw)
    magic = cur.read("<4s")
    if magic != GGUF_MAGIC:
        raise ValueError(f"not a GGUF file: magic {magic!r} != b'GGUF'")
    version = cur.read("<I")
    if version != GGUF_VERSION:
        raise ValueError(f"unsupported GGUF version {version} (this reader: v3)")
    tensor_count = cur.read("<Q")
    kv_count = cur.read("<Q")

    metadata = {}
    for _ in range(kv_count):
        key = cur.read_str()
        vtype = cur.read("<I")
        metadata[key] = _read_value(cur, vtype)

    alignment = metadata.get("general.alignment", DEFAULT_ALIGNMENT)
    alignment = int(alignment)
    if alignment <= 0 or alignment % 8 != 0:
        raise ValueError(
            f"general.alignment must be a positive multiple of 8, got {alignment}"
        )

    infos = []
    seen = set()
    for _ in range(tensor_count):
        name = cur.read_str()
        n_dims = cur.read("<I")
        if not (1 <= n_dims <= MAX_TENSOR_DIMS):
            raise ValueError(
                f"tensor {name!r} has {n_dims} dims; GGUF allows 1..{MAX_TENSOR_DIMS}"
            )
        dims_ggml = [cur.read("<Q") for _ in range(n_dims)]
        ggml_type = cur.read("<I")
        offset = cur.read("<Q")
        if name in seen:
            raise ValueError(f"duplicate tensor name {name!r} in file")
        seen.add(name)
        if offset % alignment != 0:
            raise ValueError(
                f"tensor {name!r} offset {offset} is not a multiple of the "
                f"alignment {alignment}"
            )
        itemsize = _GGML_ITEMSIZE.get(ggml_type)
        nbytes = None
        if itemsize is not None:
            n_elems = 1
            for d in dims_ggml:
                n_elems *= d
            nbytes = n_elems * itemsize
        infos.append(GGUFTensorInfo(
            name=name, shape=tuple(reversed(dims_ggml)), ggml_type=ggml_type,
            offset=offset, nbytes=nbytes if nbytes is not None else -1,
        ))

    data_start = cur.pos + (alignment - cur.pos % alignment) % alignment
    for t in infos:
        if t.nbytes >= 0 and data_start + t.offset + t.nbytes > len(raw):
            raise ValueError(
                f"tensor {t.name!r} payload [{t.offset}, {t.offset + t.nbytes}) "
                f"extends past the end of file ({len(raw)} bytes)"
            )
    return GGUFFile(metadata, infos, raw, data_start, alignment)
