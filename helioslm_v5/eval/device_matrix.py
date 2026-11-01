"""Cross-accelerator capability matrix, numeric oracle and cost record (#175).

Three parts, all CPU-runnable:

1. ``probe_backends()`` reports which torch backends this machine has
   (cpu, cuda, mps, xpu, xla/TPU via torch_xla, npu via torch_npu). An
   absent backend carries a ``reason``; nothing here imports torch_xla or
   torch_npu unless they are installed.
2. ``FEATURES`` is a checked-in matrix of what each model path does on each
   backend: ``native`` (runs there, with the evidence that showed it),
   ``fallback`` (runs, but on a slower generic path), ``unsupported`` or
   ``untested``. ``native`` without evidence is rejected by the tests, so
   the matrix cannot claim a backend works because a package imports.
   ``resolve(feature, backend, strict=True)`` refuses silent fallbacks.
3. ``forward_oracle`` compares a model on a target device/dtype against the
   CPU float32 reference (full-sequence logits and prefill+decode with the
   KV cache), with per-dtype tolerances fixed here before any run.
   ``measure_generation`` times TTFT and decode tokens/s, and
   ``make_record`` writes them with the hardware, precision, batch and
   context they were measured at. ``cost_per_mtok`` needs an explicit
   price; there is no default, so no cost figure appears without one.

TPU and NPU rows stay ``untested`` until someone runs the oracle on real
hardware and records the evidence; a skipped test is not a pass.
"""
import copy
import importlib
import importlib.util
import platform
import statistics
import time

import torch

BACKENDS = ("cpu", "cuda", "mps", "xpu", "xla", "npu")
STATUSES = ("native", "fallback", "unsupported", "untested")

# Relative error bound max|got - ref| / max|ref| per dtype, fixed before
# measuring. float32 allows TF32/kernel-order noise; the half types allow
# one rounding of a ~unit-scale activation per layer on the lite config.
TOLERANCES = {
    torch.float32: 1e-4,
    torch.float16: 2e-2,
    torch.bfloat16: 6e-2,
}

_CUDA_EVIDENCE = "docs/CUDA_VERIFICATION_2026-10-21.md (RTX 6000 Ada)"
_CI = "CI CPU suite (helioslm_v5/tests)"


def _row(module, cpu, cuda, others="untested", note="", **over):
    row = {"module": module, "note": note, "backends": {}}
    defaults = {"cpu": cpu, "cuda": cuda}
    for b in BACKENDS:
        row["backends"][b] = over.get(b, defaults.get(b, others))
    return row


def _native(evidence):
    return {"status": "native", "evidence": evidence}


def _status(status, reason):
    return {"status": status, "reason": reason}


# feature -> module that implements it and per-backend status.
FEATURES = {
    "mla_attention": _row(
        "helioslm_v5.src.attention.mla",
        _native(_CI), _native(_CUDA_EVIDENCE + ": forward_fp32, greedy_generate"),
        others=_status("untested", "no run on this backend yet")),
    "kv_cache_decode": _row(
        "helioslm_v5.src.model_v5",
        _native(_CI), _native(_CUDA_EVIDENCE + ": greedy_generate == CPU"),
        others=_status("untested", "no run on this backend yet")),
    "fp8_kv_cache": _row(
        "helioslm_v5.src.attention.mla",
        _native(_CI), _native(_CUDA_EVIDENCE + ": fp8_kv_cache"),
        others=_status("untested", "float8 storage dtype support varies "
                       "by backend")),
    "moe_dispatch": _row(
        "helioslm_v5.src.moe.sigmoid_moe",
        _native(_CI), _native(_CUDA_EVIDENCE + ": pilot_moe"),
        others=_status("untested", "no run on this backend yet")),
    "kda_linear_attention_torch": _row(
        "helioslm_v5.src.attention.linear_attention",
        _native(_CI), _native(_CUDA_EVIDENCE + ": pilot_k3_block"),
        others=_status("untested", "no run on this backend yet")),
    "kda_fla_kernel": _row(
        "helioslm_v5.src.attention.linear_attention",
        _status("fallback", "FLA Triton kernels are CUDA-only; "
                "chunked_delta_rule runs instead"),
        _native("CHANGELOG v5.160: FLA 0.5.2 chunk_kda vs chunked_delta_rule "
                "on an RTX A5000, bf16 rel err 4.6e-3"),
        others=_status("fallback", "FLA Triton kernels are CUDA-only; "
                       "chunked_delta_rule runs instead"),
        note="optional: hybrid_attention.kernel='fla'"),
    "fused_quant_awq_gptq_mxfp4": _row(
        "helioslm_v5.src.quantization.standard_quant",
        _native(_CI), _native(_CUDA_EVIDENCE + ": quant_awq/gptq/mxfp4"),
        others=_status("untested", "no run on this backend yet")),
    "fp8_training": _row(
        "helioslm_v5.src.training.fp8_trainer",
        _status("untested", "CPU tests check layer placement only, not "
                "FP8 matmul numerics"),
        _native(_CUDA_EVIDENCE + ": fp8_trainer (fixed in v5.90)"),
        others=_status("untested", "no run on this backend yet")),
    "gpu_monitor": _row(
        "helioslm_v5.src.inference.vllm_engine",
        _status("unsupported", "reports CUDA devices only"),
        _native(_CUDA_EVIDENCE + ": gpu_monitor (fixed in v5.90)"),
        others=_status("unsupported", "reports CUDA devices only")),
}


class FallbackError(RuntimeError):
    """A strict run hit a fallback or unsupported path."""


def resolve(feature, backend, strict=False):
    """Status entry for ``feature`` on ``backend``.

    ``unsupported`` always raises. ``fallback`` raises when ``strict`` so
    a benchmark cannot report a generic path as the accelerated one.
    """
    if feature not in FEATURES:
        raise KeyError(f"unknown feature {feature!r}")
    if backend not in BACKENDS:
        raise KeyError(f"unknown backend {backend!r}")
    entry = FEATURES[feature]["backends"][backend]
    if entry["status"] == "unsupported":
        raise FallbackError(f"{feature} is unsupported on {backend}: "
                            f"{entry['reason']}")
    if strict and entry["status"] == "fallback":
        raise FallbackError(f"{feature} falls back on {backend}: "
                            f"{entry['reason']}")
    return entry


def validate_matrix(features=None):
    """Raise ValueError on a malformed matrix (used by the tests)."""
    features = FEATURES if features is None else features
    for name, row in features.items():
        if set(row["backends"]) != set(BACKENDS):
            raise ValueError(f"{name}: backends must be exactly {BACKENDS}")
        for b, e in row["backends"].items():
            if e.get("status") not in STATUSES:
                raise ValueError(f"{name}/{b}: bad status {e.get('status')!r}")
            if e["status"] == "native" and not e.get("evidence"):
                raise ValueError(f"{name}/{b}: native needs evidence")
            if e["status"] != "native" and not e.get("reason"):
                raise ValueError(f"{name}/{b}: {e['status']} needs a reason")


# --------------------------------------------------------------------------
# Backend probe
# --------------------------------------------------------------------------

def _probe_cpu():
    return {"available": True, "device_name": platform.processor()
            or platform.machine(), "version": torch.__version__}


def _probe_cuda():
    if not torch.cuda.is_available():
        return {"available": False,
                "reason": "torch.cuda.is_available() is False"
                + ("" if torch.version.cuda else " (CPU-only torch build)")}
    return {"available": True, "device_name": torch.cuda.get_device_name(0),
            "version": torch.version.cuda}


def _probe_mps():
    mps = getattr(torch.backends, "mps", None)
    if mps is None or not mps.is_available():
        return {"available": False, "reason": "torch.backends.mps unavailable"}
    return {"available": True, "device_name": "Apple MPS", "version": None}


def _probe_xpu():
    xpu = getattr(torch, "xpu", None)
    if xpu is None or not xpu.is_available():
        return {"available": False, "reason": "torch.xpu unavailable"}
    return {"available": True, "device_name": xpu.get_device_name(0),
            "version": None}


def _probe_plugin(pkg, check):
    if importlib.util.find_spec(pkg) is None:
        return {"available": False, "reason": f"{pkg} is not installed"}
    try:
        mod = importlib.import_module(pkg)
        name = check(mod)
    except Exception as e:  # installed but no device / driver mismatch
        return {"available": False, "reason": f"{pkg} present but {e!r}"}
    return {"available": True, "device_name": name,
            "version": getattr(mod, "__version__", None)}


def _xla_check(mod):
    import torch_xla.core.xla_model as xm
    return str(xm.xla_device())


def _npu_check(mod):
    if not torch.npu.is_available():
        raise RuntimeError("torch.npu.is_available() is False")
    return torch.npu.get_device_name(0)


def probe_backends():
    """{backend: {available, device_name|reason, version}} for this host."""
    out = {"cpu": _probe_cpu(), "cuda": _probe_cuda(), "mps": _probe_mps(),
           "xpu": _probe_xpu(),
           "xla": _probe_plugin("torch_xla", _xla_check),
           "npu": _probe_plugin("torch_npu", _npu_check)}
    for b, r in out.items():
        r["backend"] = b
    return out


# --------------------------------------------------------------------------
# Numeric oracle
# --------------------------------------------------------------------------

def _rel_err(got, ref):
    got = got.detach().float().cpu()
    ref = ref.detach().float().cpu()
    return float((got - ref).abs().max() / ref.abs().max().clamp_min(1e-12))


def oracle_config(dense=True):
    """The lite config the oracle runs on.

    ``dense=True`` makes every layer a dense FFN (``moe.first_k_dense``).
    Half-precision checks need it: top-k MoE routing on near-tied router
    scores flips experts in bf16 and moves the logits by a discrete step.
    On lite (seed 0) bf16 gives 0.33 relative error with MoE and 0.011
    dense; the jump is a routing difference, not a kernel error. float32
    passes either way.
    """
    from helioslm_v5.configs.config_v5 import HeliosLMv5Config
    cfg = HeliosLMv5Config(size="lite")
    if dense:
        cfg.moe.first_k_dense = cfg.num_hidden_layers
    return cfg


def forward_oracle(model, device="cpu", dtype=torch.float32, seed=0,
                   batch=2, seq=12, decode_steps=4, tolerance=None,
                   target=None):
    """Compare ``model`` on ``device``/``dtype`` with CPU float32.

    ``model`` is the reference (left untouched on its device); a deep copy
    runs on the target. Two checks, each against the same tolerance:
    ``full`` (logits of one forward over the sequence) and ``decode``
    (prefill seq - decode_steps tokens, then one token at a time with the
    KV cache; the target's step logits are compared with the reference's
    full-sequence logits at the same positions). Returns a dict with the
    errors, top-1 agreement and ``passed``. ``target`` replaces the
    deep copy (e.g. a model loaded through a backend's own path).
    """
    tol = TOLERANCES[dtype] if tolerance is None else tolerance
    vocab = model.config.vocab_size
    g = torch.Generator().manual_seed(seed)
    ids = torch.randint(0, vocab, (batch, seq), generator=g)
    ref_model = copy.deepcopy(model).to("cpu", torch.float32).eval()
    tgt = copy.deepcopy(model) if target is None else target
    tgt = tgt.to(device=device, dtype=dtype).eval()
    with torch.no_grad():
        ref = ref_model(ids)[0]
        full = tgt(ids.to(device))[0]
        n0 = seq - decode_steps
        logits, _, past = tgt(ids[:, :n0].to(device), use_cache=True)
        steps = [logits[:, -1]]
        for t in range(n0, seq - 1):
            logits, _, past = tgt(ids[:, t:t + 1].to(device),
                                  past_key_values=past, use_cache=True)
            steps.append(logits[:, -1])
        dec = torch.stack(steps, dim=1)
    dec_ref = ref[:, n0 - 1:seq - 1]
    full_err = _rel_err(full, ref)
    dec_err = _rel_err(dec, dec_ref)
    top1 = float((full.float().cpu().argmax(-1) == ref.argmax(-1))
                 .float().mean())
    return {"device": str(device), "dtype": str(dtype).replace("torch.", ""),
            "tolerance": tol, "full_rel_err": full_err,
            "decode_rel_err": dec_err, "top1_agreement": top1,
            "passed": full_err <= tol and dec_err <= tol}


# --------------------------------------------------------------------------
# Measurement + record
# --------------------------------------------------------------------------

def _sync(device):
    dev = torch.device(device)
    if dev.type == "cuda":
        torch.cuda.synchronize(dev)
    elif dev.type == "xpu":
        torch.xpu.synchronize(dev)
    elif dev.type == "mps":
        torch.mps.synchronize()


def _pct(xs, q):
    xs = sorted(xs)
    i = min(len(xs) - 1, max(0, round(q * (len(xs) - 1))))
    return xs[i]


def measure_generation(model, device="cpu", batch=1, prompt_len=32,
                       new_tokens=16, repeats=3, warmup=1, seed=0):
    """Greedy-generate timings on ``device`` (model must already be there).

    TTFT is a 1-token generate; decode time is (full - TTFT) over
    ``new_tokens - 1`` tokens. Peak memory is measured on CUDA only and
    reported as None elsewhere (with ``peak_mem_source`` saying so).
    """
    if new_tokens < 2:
        raise ValueError("new_tokens must be >= 2 to time decode")
    g = torch.Generator().manual_seed(seed)
    ids = torch.randint(0, model.config.vocab_size, (batch, prompt_len),
                        generator=g).to(device)
    is_cuda = torch.device(device).type == "cuda"
    model.eval()
    ttft, decode = [], []
    with torch.no_grad():
        for _ in range(warmup):
            model.generate(ids[:, :4], max_new_tokens=2, temperature=0)
        _sync(device)
        if is_cuda:
            torch.cuda.reset_peak_memory_stats()
        for _ in range(repeats):
            t0 = time.perf_counter()
            model.generate(ids, max_new_tokens=1, temperature=0)
            _sync(device)
            t1 = time.perf_counter()
            model.generate(ids, max_new_tokens=new_tokens, temperature=0)
            _sync(device)
            t2 = time.perf_counter()
            ttft.append(t1 - t0)
            decode.append(max((t2 - t1) - (t1 - t0), 1e-9))
    dec_tok_s = [batch * (new_tokens - 1) / d for d in decode]
    return {
        "ttft_s_p50": statistics.median(ttft), "ttft_s_p95": _pct(ttft, .95),
        "decode_tok_s_p50": statistics.median(dec_tok_s),
        "decode_tok_s_p05": _pct(dec_tok_s, .05),
        "prefill_tok_s_p50": batch * prompt_len / statistics.median(ttft),
        "peak_mem_gb": (round(torch.cuda.max_memory_allocated() / 2**30, 3)
                        if is_cuda else None),
        "peak_mem_source": ("torch.cuda.max_memory_allocated" if is_cuda
                            else "not measured on this backend"),
        "repeats": repeats, "warmup": warmup,
    }


RECORD_SCHEMA = "device-bench/1"
REQUIRED = ("backend", "device_name", "dtype", "batch", "prompt_len",
            "new_tokens", "model", "torch_version", "commit", "oracle",
            "timing")


def cost_per_mtok(tokens_per_s, usd_per_hour):
    """USD per million generated tokens at an explicit hourly price."""
    if usd_per_hour is None or usd_per_hour <= 0:
        raise ValueError("cost needs an explicit usd_per_hour > 0")
    if tokens_per_s <= 0:
        raise ValueError("tokens_per_s must be > 0")
    return usd_per_hour / (tokens_per_s * 3600) * 1e6


def make_record(*, usd_per_hour=None, **fields):
    """A device-bench/1 record; raises on missing context fields.

    A timing without the oracle result is rejected: speed of a path whose
    numbers were not checked is not reported.
    """
    missing = [k for k in REQUIRED if fields.get(k) in (None, "")]
    if missing:
        raise ValueError(f"record missing {missing}")
    if not fields["oracle"].get("passed"):
        raise ValueError("oracle did not pass; timing not recordable")
    rec = {"schema": RECORD_SCHEMA, **fields}
    rec["usd_per_hour"] = usd_per_hour
    rec["usd_per_mtok_decode"] = (
        cost_per_mtok(fields["timing"]["decode_tok_s_p50"], usd_per_hour)
        if usd_per_hour is not None else None)
    return rec
