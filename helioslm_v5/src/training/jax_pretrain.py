"""jax_pretrain.py — own small base model, pretrained from scratch in JAX
on TPU (Kaggle TPU v5e-8 / TRC) or CPU (v5.266).

Why JAX: the cheapest compute for an own from-scratch base is the free
Kaggle TPU v5e-8 quota (about 20 h a week, 9 h per session), then TRC /
NCHC if those come through. The torch trainer (``pretrain.py``) targets
CUDA; this module is the TPU path. It is deliberately plain: a dense
Llama-shaped decoder (RMSNorm, RoPE, GQA, SwiGLU, tied embeddings), so
the result exports to a standard ``LlamaForCausalLM`` checkpoint that
transformers, vLLM, lm-eval and llama.cpp load without custom code
(``export_hf``), and the repo's evaluation scripts run on it unchanged.

Pieces:
  - ``ModelConfig`` / ``PRESETS`` ("tiny" for tests, "0.3b" = 304M);
  - ``init_params`` / ``forward`` / ``loss_fn`` (layers under
    ``lax.scan`` + ``jax.checkpoint``; cross-entropy in sequence chunks
    so the B x T x V logits are never held in memory at once);
  - ``adamw_init`` / ``adamw_update`` (no optax dependency; decoupled
    weight decay on matrices only, global-norm clipping);
  - ``lr_at`` (linear warmup, cosine to ``min_lr_ratio``);
  - ``MixtureLoader``: token shards per source (uint16 / uint32
    ``.bin`` as written by ``examples/pilot_prepare_data.py``), mixture
    weights, a resumable cursor, optional download from a Hugging Face
    dataset repo one shard at a time (the TPU VM disk does not have to
    hold the whole corpus);
  - ``save_checkpoint`` / ``load_checkpoint`` (``.npz``, no pickle);
  - ``export_hf`` (numpy state dict with Llama names + config.json).

The CLI is ``examples/jax_pretrain_tpu.py``; the Kaggle notebook is
``examples/notebooks/kaggle_tpu_own_base.ipynb``; the plan and budget
are in ``docs/OWN_BASE_0P3B.md``.
"""
import dataclasses
import json
import math
import os
from pathlib import Path

import numpy as np

try:  # the module stays importable (and its numpy parts testable) without jax
    import jax
    import jax.numpy as jnp
except ImportError:  # pragma: no cover - exercised only where jax is absent
    jax = None
    jnp = None


# ----------------------------------------------------------------- config

@dataclasses.dataclass(frozen=True)
class ModelConfig:
    vocab_size: int = 32768
    hidden: int = 1024
    layers: int = 24
    heads: int = 16
    kv_heads: int = 4
    ffn: int = 2816
    max_seq: int = 2048
    rope_theta: float = 10000.0
    norm_eps: float = 1e-5
    init_std: float = 0.02

    @property
    def head_dim(self):
        return self.hidden // self.heads

    def n_params(self):
        """Parameter count (tied embedding counted once)."""
        d, f, hd = self.hidden, self.ffn, self.head_dim
        attn = d * self.heads * hd * 2 + d * self.kv_heads * hd * 2
        per_layer = attn + 3 * d * f + 2 * d
        return self.vocab_size * d + self.layers * per_layer + d

    def flops_per_token(self, seq):
        """Training FLOPs per token: 6N plus the attention score terms."""
        return 6 * self.n_params() + 12 * self.layers * self.hidden * seq


PRESETS = {
    # unit tests and CPU smoke runs
    "tiny": ModelConfig(vocab_size=512, hidden=64, layers=2, heads=4,
                        kv_heads=2, ffn=176, max_seq=128),
    # ~25M: a 1-2 h Kaggle shake-down run
    "25m": ModelConfig(hidden=384, layers=8, heads=6, kv_heads=2, ffn=1024),
    # the own-base demo: 304M parameters with the 32,768-token BPE
    "0.3b": ModelConfig(),
}


# ------------------------------------------------------------------ model

def _require_jax():
    if jax is None:
        raise ImportError("jax is required: pip install jax (TPU: jax[tpu])")


def init_params(cfg, seed=0):
    """fp32 parameters; layer weights stacked on a leading [layers] axis
    so the forward pass can ``lax.scan`` over them."""
    _require_jax()
    d, f, hd, L = cfg.hidden, cfg.ffn, cfg.head_dim, cfg.layers
    keys = jax.random.split(jax.random.PRNGKey(seed), 8)
    std = cfg.init_std
    out_std = std / math.sqrt(2 * L)  # residual-branch output projections

    def n(k, shape, s):
        return s * jax.random.normal(k, shape, jnp.float32)

    return {
        "embed": n(keys[0], (cfg.vocab_size, d), std),
        "final_norm": jnp.ones((d,), jnp.float32),
        "layers": {
            "attn_norm": jnp.ones((L, d), jnp.float32),
            "wq": n(keys[1], (L, d, cfg.heads * hd), std),
            "wk": n(keys[2], (L, d, cfg.kv_heads * hd), std),
            "wv": n(keys[3], (L, d, cfg.kv_heads * hd), std),
            "wo": n(keys[4], (L, cfg.heads * hd, d), out_std),
            "mlp_norm": jnp.ones((L, d), jnp.float32),
            "w_gate": n(keys[5], (L, d, f), std),
            "w_up": n(keys[6], (L, d, f), std),
            "w_down": n(keys[7], (L, f, d), out_std),
        },
    }


def _rms_norm(x, w, eps):
    xf = x.astype(jnp.float32)
    xf = xf * jax.lax.rsqrt(jnp.mean(xf * xf, axis=-1, keepdims=True) + eps)
    return (xf * w).astype(x.dtype)


def _rope_tables(seq, head_dim, theta):
    inv = 1.0 / (theta ** (np.arange(0, head_dim, 2, dtype=np.float64)
                           / head_dim))
    ang = np.outer(np.arange(seq, dtype=np.float64), inv)  # [T, hd/2]
    ang = np.concatenate([ang, ang], axis=-1)              # HF rotate_half
    return np.cos(ang).astype(np.float32), np.sin(ang).astype(np.float32)


def _apply_rope(x, cos, sin):
    # x [B, T, H, hd]; the Hugging Face Llama "rotate_half" convention, so
    # export_hf needs no weight permutation
    half = x.shape[-1] // 2
    rot = jnp.concatenate([-x[..., half:], x[..., :half]], axis=-1)
    c = cos[None, :, None, :].astype(x.dtype)
    s = sin[None, :, None, :].astype(x.dtype)
    return x * c + rot * s


# "xla": plain einsum attention (any backend; materialises T x T scores
# per head). "flash": the Pallas TPU flash-attention kernel (O(T) memory);
# set by examples/jax_pretrain_tpu.py --attn after a probe on the device.
ATTN_IMPL = "xla"


def _attention_xla(q, k, v):
    scale = 1.0 / math.sqrt(q.shape[-1])
    s = jnp.einsum("bqhd,bkhd->bhqk", q, k).astype(jnp.float32) * scale
    T = q.shape[1]
    mask = jnp.tril(jnp.ones((T, T), bool))
    s = jnp.where(mask[None, None], s, jnp.finfo(jnp.float32).min)
    p = jax.nn.softmax(s, axis=-1).astype(v.dtype)
    return jnp.einsum("bhqk,bkhd->bqhd", p, v)


def _attention_flash(q, k, v):
    from jax.experimental.pallas.ops.tpu.flash_attention import (
        flash_attention)
    o = flash_attention(q.swapaxes(1, 2), k.swapaxes(1, 2),
                        v.swapaxes(1, 2), causal=True,
                        sm_scale=1.0 / math.sqrt(q.shape[-1]))
    return o.swapaxes(1, 2)


def _attention(q, k, v):
    # q [B,T,H,hd], k/v [B,T,KV,hd] -> [B,T,H,hd]; causal, GQA by repeat
    rep = q.shape[2] // k.shape[2]
    if rep > 1:
        k = jnp.repeat(k, rep, axis=2)
        v = jnp.repeat(v, rep, axis=2)
    if ATTN_IMPL == "flash":
        return _attention_flash(q, k, v)
    return _attention_xla(q, k, v)


def hidden_states(params, tokens, cfg, dtype=None, remat=True):
    """Final-norm hidden states [B, T, d] (before the tied LM head)."""
    _require_jax()
    dtype = dtype or jnp.float32
    B, T = tokens.shape
    hd = cfg.head_dim
    cos, sin = _rope_tables(T, hd, cfg.rope_theta)
    x = params["embed"].astype(dtype)[tokens]

    def layer(x, p):
        p = jax.tree_util.tree_map(lambda a: a.astype(dtype), p)
        h = _rms_norm(x, p["attn_norm"], cfg.norm_eps)
        q = (h @ p["wq"]).reshape(B, T, cfg.heads, hd)
        k = (h @ p["wk"]).reshape(B, T, cfg.kv_heads, hd)
        v = (h @ p["wv"]).reshape(B, T, cfg.kv_heads, hd)
        q, k = _apply_rope(q, cos, sin), _apply_rope(k, cos, sin)
        a = _attention(q, k, v).reshape(B, T, cfg.heads * hd)
        x = x + a @ p["wo"]
        h = _rms_norm(x, p["mlp_norm"], cfg.norm_eps)
        x = x + (jax.nn.silu(h @ p["w_gate"]) * (h @ p["w_up"])) @ p["w_down"]
        return x, None

    step = jax.checkpoint(layer) if remat else layer
    x, _ = jax.lax.scan(step, x, params["layers"])
    return _rms_norm(x, params["final_norm"].astype(dtype), cfg.norm_eps)


def forward(params, tokens, cfg, dtype=None):
    """Logits [B, T, V] in fp32 (tests / small batches; training uses the
    chunked loss instead)."""
    h = hidden_states(params, tokens, cfg, dtype, remat=False)
    return (h @ params["embed"].astype(h.dtype).T).astype(jnp.float32)


def loss_fn(params, tokens, targets, cfg, dtype=None, ce_chunk=512):
    """Mean next-token cross-entropy. The LM head + log-softmax run over
    ``ce_chunk`` positions at a time under ``jax.checkpoint``, so the
    full logits tensor is never materialised."""
    h = hidden_states(params, tokens, cfg, dtype)
    B, T, d = h.shape
    c = min(ce_chunk, T)
    if T % c:
        raise ValueError(f"seq {T} must be a multiple of ce_chunk {c}")
    emb = params["embed"].astype(h.dtype)
    hs = h.reshape(B, T // c, c, d).swapaxes(0, 1)       # [n, B, c, d]
    ts = targets.reshape(B, T // c, c).swapaxes(0, 1)    # [n, B, c]

    @jax.checkpoint
    def chunk(carry, xs):
        hc, tc = xs
        logits = (hc @ emb.T).astype(jnp.float32)
        lse = jax.nn.logsumexp(logits, axis=-1)
        gold = jnp.take_along_axis(logits, tc[..., None], axis=-1)[..., 0]
        return carry + jnp.sum(lse - gold), None

    total, _ = jax.lax.scan(chunk, jnp.zeros((), jnp.float32), (hs, ts))
    return total / (B * T)


# -------------------------------------------------------------- optimizer

def adamw_init(params):
    _require_jax()
    zeros = jax.tree_util.tree_map(jnp.zeros_like, params)
    return {"m": zeros, "v": jax.tree_util.tree_map(jnp.zeros_like, params),
            "step": jnp.zeros((), jnp.int32)}


def global_norm(tree):
    leaves = jax.tree_util.tree_leaves(tree)
    return jnp.sqrt(sum(jnp.sum(jnp.square(x.astype(jnp.float32)))
                        for x in leaves))


def adamw_update(params, grads, opt, lr, b1=0.9, b2=0.95, eps=1e-8,
                 weight_decay=0.1, clip=1.0):
    """One AdamW step with global-norm clipping. Norm weights (the
    per-layer [layers, d] vectors and final_norm) are not decayed."""
    gnorm = global_norm(grads)
    scale = jnp.minimum(1.0, clip / (gnorm + 1e-6)) if clip else 1.0
    step = opt["step"] + 1
    bc1 = 1 - b1 ** step.astype(jnp.float32)
    bc2 = 1 - b2 ** step.astype(jnp.float32)

    def upd(path, p, g, m, v):
        g = g * scale
        m = b1 * m + (1 - b1) * g
        v = b2 * v + (1 - b2) * g * g
        u = (m / bc1) / (jnp.sqrt(v / bc2) + eps)
        name = jax.tree_util.keystr(path)
        if "norm" not in name:
            u = u + weight_decay * p
        return p - lr * u, m, v

    out = jax.tree_util.tree_map_with_path(upd, params, grads, opt["m"],
                                           opt["v"])
    is_t = lambda x: isinstance(x, tuple)  # noqa: E731
    new_p = jax.tree_util.tree_map(lambda t: t[0], out, is_leaf=is_t)
    new_m = jax.tree_util.tree_map(lambda t: t[1], out, is_leaf=is_t)
    new_v = jax.tree_util.tree_map(lambda t: t[2], out, is_leaf=is_t)
    return new_p, {"m": new_m, "v": new_v, "step": step}, gnorm


def lr_at(step, peak, warmup, total, min_lr_ratio=0.1):
    """Linear warmup to ``peak``, cosine decay to ``peak*min_lr_ratio``
    at ``total``; flat afterwards (python float, host side)."""
    if step < warmup:
        return peak * (step + 1) / warmup
    t = min(1.0, (step - warmup) / max(1, total - warmup))
    return peak * (min_lr_ratio + (1 - min_lr_ratio)
                   * 0.5 * (1 + math.cos(math.pi * t)))


# ------------------------------------------------------------------- data

TOKEN_DTYPES = {"uint16": np.uint16, "uint32": np.uint32}


class MixtureLoader:
    """Random windows from per-source token shards, mixed by weight.

    ``mix`` is a dict (or a JSON file path) like::

        {"dtype": "uint16", "hf_repo": "user/own-base-tokens",
         "sources": [{"name": "zhtw", "weight": 0.5,
                      "files": ["zhtw/000.bin", "zhtw/001.bin"]}, ...],
         "val": "val.bin"}

    ``files`` are local paths, or paths inside ``hf_repo`` (a Hugging
    Face *dataset* repo) fetched with ``huggingface_hub`` into
    ``cache_dir`` one at a time; the previous shard of a source is
    deleted when it advances (``keep_files=False``), so disk use is about
    one shard per source. A source moves to its next shard after
    drawing as many tokens as the shard holds (each window is random,
    so a shard is sampled, not read sequentially); after the last shard
    it wraps around (a second epoch of that source).

    The state (``state()`` / ``load_state()``) is the per-source shard
    index and tokens drawn plus the batch counter; with the same seed a
    resumed run draws the same windows as an uninterrupted one.
    """

    def __init__(self, mix, seq, seed=0, cache_dir="token_cache",
                 keep_files=True, token=None):
        if isinstance(mix, (str, Path)):
            mix = json.loads(Path(mix).read_text(encoding="utf-8"))
        self.mix = mix
        self.seq = seq
        self.seed = seed
        self.dtype = TOKEN_DTYPES[mix.get("dtype", "uint16")]
        self.repo = mix.get("hf_repo")
        self.cache = Path(cache_dir)
        self.keep = keep_files
        self.token = token
        srcs = [s for s in mix["sources"] if s.get("weight", 0) > 0]
        if not srcs:
            raise ValueError("mixture has no source with weight > 0")
        w = np.array([float(s["weight"]) for s in srcs])
        self.sources = srcs
        self.weights = w / w.sum()
        self.shard = [0] * len(srcs)
        self.drawn = [0] * len(srcs)
        self.batches = 0
        self._mm = [None] * len(srcs)
        self._mm_path = [None] * len(srcs)

    # -- files
    def _local(self, rel):
        p = Path(rel)
        if self.repo is None or p.is_absolute() and p.exists():
            return p
        target = self.cache / rel
        if not target.exists():
            from huggingface_hub import hf_hub_download
            hf_hub_download(self.repo, rel, repo_type="dataset",
                            local_dir=str(self.cache), token=self.token)
        return target

    def _open(self, i):
        files = self.sources[i]["files"]
        rel = files[self.shard[i] % len(files)]
        path = self._local(rel)
        if self._mm_path[i] != path:
            old = self._mm_path[i]
            self._mm[i] = np.memmap(path, dtype=self.dtype, mode="r")
            self._mm_path[i] = path
            if (old is not None and not self.keep and self.repo
                    and old != path and old.exists()):
                old.unlink()
            if len(self._mm[i]) < self.seq + 2:
                raise ValueError(f"shard {path} has {len(self._mm[i])} "
                                 f"tokens, needs more than {self.seq + 1}")
        return self._mm[i]

    # -- batches
    def next_batch(self, batch):
        """(inputs, targets) int32 [batch, seq]."""
        rng = np.random.default_rng([self.seed, self.batches])
        src = rng.choice(len(self.sources), size=batch, p=self.weights)
        rows = np.empty((batch, self.seq + 1), np.int64)
        for i in np.unique(src):
            idx = np.nonzero(src == i)[0]
            data = self._open(i)
            starts = rng.integers(0, len(data) - self.seq - 1, size=len(idx))
            for r, s in zip(idx, starts):
                rows[r] = data[s:s + self.seq + 1]
            self.drawn[i] += len(idx) * self.seq
            if self.drawn[i] >= len(data):
                self.shard[i] += 1
                self.drawn[i] = 0
        self.batches += 1
        rows = rows.astype(np.int32)
        return rows[:, :-1], rows[:, 1:]

    def val_windows(self, n):
        """Fixed consecutive windows from ``mix['val']`` (or None)."""
        rel = self.mix.get("val")
        if not rel:
            return None
        data = np.memmap(self._local(rel), dtype=self.dtype, mode="r")
        n = min(n, (len(data) - 1) // self.seq)
        rows = np.stack([np.asarray(data[i * self.seq:(i + 1) * self.seq + 1])
                         for i in range(n)]).astype(np.int32)
        return rows[:, :-1], rows[:, 1:]

    def state(self):
        return {"shard": list(self.shard), "drawn": list(self.drawn),
                "batches": self.batches}

    def load_state(self, st):
        self.shard = list(st["shard"])
        self.drawn = list(st["drawn"])
        self.batches = int(st["batches"])


# ------------------------------------------------------------ checkpoints

def _flatten(tree, prefix=""):
    out = {}
    if isinstance(tree, dict):
        for k in sorted(tree):
            out.update(_flatten(tree[k], f"{prefix}{k}/"))
    else:
        out[prefix[:-1]] = np.asarray(tree)
    return out


def _unflatten(flat):
    tree = {}
    for key, val in flat.items():
        node = tree
        parts = key.split("/")
        for p in parts[:-1]:
            node = node.setdefault(p, {})
        node[parts[-1]] = val
    return tree


def save_checkpoint(path, params, opt, step, cfg, loader_state=None,
                    extra=None):
    """``path``/ckpt.npz (params + optimizer) and ``path``/state.json."""
    path = Path(path)
    path.mkdir(parents=True, exist_ok=True)
    flat = _flatten({"params": params, "opt": opt})
    tmp = path / "ckpt.tmp.npz"
    np.savez(tmp, **flat)
    os.replace(tmp, path / "ckpt.npz")
    meta = {"step": int(step), "config": dataclasses.asdict(cfg),
            "loader": loader_state, "extra": extra or {}}
    (path / "state.json").write_text(json.dumps(meta, indent=1),
                                     encoding="utf-8")


def load_checkpoint(path):
    """(params, opt, meta) as numpy trees, or None when absent."""
    path = Path(path)
    if not (path / "ckpt.npz").exists():
        return None
    with np.load(path / "ckpt.npz", allow_pickle=False) as z:
        tree = _unflatten({k: z[k] for k in z.files})
    meta = json.loads((path / "state.json").read_text(encoding="utf-8"))
    return tree["params"], tree["opt"], meta


# ----------------------------------------------------------------- export

def export_hf(params, cfg, out_dir, tokenizer_json=None, dtype="bfloat16",
              bos_id=None, eos_id=None):
    """Write a ``LlamaForCausalLM`` checkpoint (config.json +
    model.safetensors, or pytorch-free ``model.npz`` when safetensors is
    missing) into ``out_dir``. Our weights are [in, out]; HF Linear is
    [out, in], hence the transposes."""
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    p = jax.tree_util.tree_map(np.asarray, params) if jax else params
    L = cfg.layers
    sd = {"model.embed_tokens.weight": p["embed"],
          "model.norm.weight": p["final_norm"]}
    names = {"wq": "self_attn.q_proj", "wk": "self_attn.k_proj",
             "wv": "self_attn.v_proj", "wo": "self_attn.o_proj",
             "w_gate": "mlp.gate_proj", "w_up": "mlp.up_proj",
             "w_down": "mlp.down_proj"}
    lay = p["layers"]
    for i in range(L):
        pre = f"model.layers.{i}."
        sd[pre + "input_layernorm.weight"] = lay["attn_norm"][i]
        sd[pre + "post_attention_layernorm.weight"] = lay["mlp_norm"][i]
        for k, hf in names.items():
            sd[pre + hf + ".weight"] = np.ascontiguousarray(lay[k][i].T)
    hf_cfg = {
        "architectures": ["LlamaForCausalLM"], "model_type": "llama",
        "vocab_size": cfg.vocab_size, "hidden_size": cfg.hidden,
        "intermediate_size": cfg.ffn, "num_hidden_layers": cfg.layers,
        "num_attention_heads": cfg.heads,
        "num_key_value_heads": cfg.kv_heads, "head_dim": cfg.head_dim,
        "max_position_embeddings": cfg.max_seq,
        "rms_norm_eps": cfg.norm_eps, "rope_theta": cfg.rope_theta,
        "hidden_act": "silu", "tie_word_embeddings": True,
        "attention_bias": False, "mlp_bias": False,
        "torch_dtype": dtype,
    }
    if bos_id is not None:
        hf_cfg["bos_token_id"] = bos_id
    if eos_id is not None:
        hf_cfg["eos_token_id"] = eos_id
    (out / "config.json").write_text(json.dumps(hf_cfg, indent=1),
                                     encoding="utf-8")
    try:
        from safetensors.numpy import save_file
    except ImportError:
        np.savez(out / "model.npz", **{k: np.asarray(v, np.float32)
                                       for k, v in sd.items()})
    else:
        if dtype == "bfloat16":
            import ml_dtypes
            conv = {k: np.asarray(v, np.float32).astype(ml_dtypes.bfloat16)
                    for k, v in sd.items()}
        else:
            conv = {k: np.asarray(v, np.float32) for k, v in sd.items()}
        save_file(conv, str(out / "model.safetensors"),
                  metadata={"format": "pt"})
    if tokenizer_json:
        import shutil
        shutil.copy(tokenizer_json, out / "tokenizer.json")
    return sd, hf_cfg
