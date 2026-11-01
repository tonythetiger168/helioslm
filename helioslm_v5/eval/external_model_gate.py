"""Source / licence gate and memory-fit check for external reference
models (#185), starting with Mistral Large 4 (1T total / 52B active).

An external model here is a comparison baseline or a teacher *candidate*.
It is not HeliosLM's own work (the v6.0 1T target in #45 stays open), and
it never enters the distillation allowlist from this module:
``teacher_license_audit.REVIEWS`` is the only route there, and an entry
lands in it only after a human review.

Three checks, all offline-testable:

1. ``check_source(rid, fetch)``: the live model id, commit, card licence
   and LICENSE text sha256 must equal the pinned ones. Unknown model,
   moved revision or changed licence text -> ``GateError`` (fail closed).
2. ``use_allowed(rid, use)``: each candidate records a reading per use
   (internal research, redistributing weights, commercial MaaS, official
   API, training on its outputs). Anything not ``allowed`` is refused,
   and the candidate's ``status`` stays ``pending_review`` until a person
   signs it off.
3. ``fit(rid, quant, tier, context)``: GGUF size from the pinned listing,
   plus the vision projector, the KV cache at ``context`` and a runtime
   buffer, against an L128 / L256 / L512 tier. ``fits_static`` is weights
   alone <= tier memory; ``fits_with_reserve`` keeps the OS / desktop
   share ``1 - package.FIT_SHARE`` free. ``tested_runtime`` is always
   "not run" here: a size that fits says nothing about speed or output
   quality.
"""
from helioslm_v5.src.export.package import FIT_SHARE, TIERS

GIB = 2 ** 30
USES = ("internal_research", "redistribute_weights", "commercial_maas",
        "official_api", "train_on_outputs")
USE_VERDICTS = ("allowed", "allowed_with_conditions", "needs_license",
                "not_allowed", "pending_review")
RUNTIME_BUFFER_GIB = 4.0   # llama.cpp compute buffers, assumed

CANDIDATES = {
    "mistralai/Mistral-Large-4-1T-A52B": {
        "status": "pending_review",
        "role": "reference baseline; teacher candidate only after review",
        "revision": "f12dabdc0c1e99e7f7924461a6f308859b05af4d",
        "card_license": "other",
        "license": "Mistral Ecosystem Open-Weight License (modified MIT)",
        # LICENSE at the revision above, fetched 2026-11-01 (2632 bytes)
        "license_sha256": "d242ebedb53ffd7e26c220e3305665ca47e4e5267fa62ff4cdeb3953b8353c78",
        "uses": {
            "internal_research": (
                "allowed",
                "s.3(a): internal use, nothing made available to third "
                "parties, is exempt from the s.2 MaaS clause"),
            "redistribute_weights": (
                "allowed_with_conditions",
                "keep the copyright and permission notice (s.1); a third "
                "party offering it as MaaS needs its own licence"),
            "commercial_maas": (
                "needs_license",
                "s.2: an operator (or affiliate) of a Model-as-a-Service "
                "business needs a separate commercial licence from Mistral "
                "before any commercial use of the model or derivatives; "
                "no revenue threshold"),
            "official_api": (
                "pending_review",
                "outputs from Mistral's own API fall under its terms of "
                "service, not this licence; not read"),
            "train_on_outputs": (
                "pending_review",
                "the licence does not mention outputs; a distilled student "
                "treated as a derivative work would inherit s.2 for any "
                "MaaS operator -- needs a person's (legal) review"),
        },
        "arch": {   # text_config of config.json at the revision above
            "layers": 72, "kv_lora_rank": 512, "qk_rope_head_dim": 64,
            "max_position_embeddings": 1048576, "experts": 128,
            "experts_per_token": 4, "vision": True,
        },
        "gguf": {
            "repo": "ggml-org/Mistral-Large-4-1T-A52B-GGUF",
            "revision": "7d4f13b81e55e1223b3c125a58af4c19bb1ddf6a",
            # summed shard sizes from the Hub tree listing, 2026-11-01
            "quants": {
                "IQ2_XXS": (6, 297676048832),
                "Q2_K": (8, 380706433472),
                "Q4_K_M": (13, 634802244608),
                "Q8_0": (24, 1115150340608),
                "BF16": (43, 2098861021920),
            },
            "mmproj_bytes": 3444696416,   # mmproj-BF16.gguf
        },
    },
}


class GateError(ValueError):
    """Unknown model, moved revision, changed licence or refused use."""


def _candidate(rid):
    if rid not in CANDIDATES:
        raise GateError(f"{rid} has no pinned review; add it to "
                        "CANDIDATES after reading its licence")
    return CANDIDATES[rid]


def check_source(rid, fetch):
    """Compare a live fetch with the pin. ``fetch(rid)`` returns
    ``{"sha", "card_license", "license_sha256"}``
    (``teacher_license_audit.hub_fetch`` plus a hash of its text)."""
    c = _candidate(rid)
    live = fetch(rid)
    for key, pinned in (("sha", c["revision"]),
                        ("card_license", c["card_license"]),
                        ("license_sha256", c["license_sha256"])):
        if live.get(key) != pinned:
            raise GateError(f"{rid}: {key} is {live.get(key)!r}, pinned "
                            f"{pinned!r}; re-read before use")
    return {"id": rid, "status": c["status"], "pinned": True}


def use_allowed(rid, use, accept_conditions=False):
    """True only for an ``allowed`` reading (or ``allowed_with_conditions``
    when the caller accepts them). Raises on an unknown use."""
    c = _candidate(rid)
    if use not in USES:
        raise GateError(f"unknown use {use!r}; one of {USES}")
    verdict = c["uses"][use][0]
    ok = ("allowed", "allowed_with_conditions") if accept_conditions \
        else ("allowed",)
    return verdict in ok


def require_use(rid, use, accept_conditions=False):
    if not use_allowed(rid, use, accept_conditions):
        verdict, why = _candidate(rid)["uses"][use]
        raise GateError(f"{rid}: {use} is {verdict}: {why}")


def kv_bytes_per_token(arch, elt_bytes=2):
    """MLA latent cache: (kv_lora_rank + rope dim) per layer. Assumes the
    runtime caches the latent (llama.cpp's MLA path); an expanded K/V
    cache is far larger."""
    return arch["layers"] * (arch["kv_lora_rank"]
                             + arch["qk_rope_head_dim"]) * elt_bytes


def fit(rid, quant, tier, context=131072, kv_elt_bytes=2):
    c = _candidate(rid)
    if tier not in TIERS:
        raise KeyError(f"unknown tier {tier!r}")
    if context > c["arch"]["max_position_embeddings"]:
        raise ValueError("context beyond the model's maximum")
    files, weights = c["gguf"]["quants"][quant]
    mem = TIERS[tier].memory_gb * GIB
    kv = kv_bytes_per_token(c["arch"], kv_elt_bytes) * context
    extra = c["gguf"]["mmproj_bytes"] if c["arch"]["vision"] else 0
    total = weights + extra + kv + RUNTIME_BUFFER_GIB * GIB
    budget = mem * FIT_SHARE
    return {
        "id": rid, "quant": quant, "tier": tier, "context": context,
        "weights_gib": round(weights / GIB, 1), "files": files,
        "mmproj_gib": round(extra / GIB, 1), "kv_gib": round(kv / GIB, 1),
        "runtime_buffer_gib": RUNTIME_BUFFER_GIB,
        "total_gib": round(total / GIB, 1),
        "tier_gib": TIERS[tier].memory_gb,
        "budget_gib": round(budget / GIB, 1),
        "fits_static": weights <= mem,
        "fits_with_reserve": total <= budget,
        "tested_runtime": "not run",
    }


def fit_table(rid, tiers=("L128", "L256", "L512"), context=131072):
    return [fit(rid, q, t, context) for t in tiers
            for q in CANDIDATES[rid]["gguf"]["quants"]]


def conclusion(rid, tiers=("L128", "L256", "L512"), context=131072):
    """``feasible`` needs a recorded runtime test, which this module never
    has; a configuration that fits on paper is ``unverified``, none at
    all is ``blocked``."""
    rows = fit_table(rid, tiers, context)
    fitting = [(r["tier"], r["quant"]) for r in rows if r["fits_with_reserve"]]
    return {"id": rid, "status": "unverified" if fitting else "blocked",
            "fits_on_paper": fitting,
            "missing": ["a 512 GB unified-memory machine (not available)",
                        "llama.cpp / vLLM / SGLang runtime test",
                        "a person's licence review for train_on_outputs "
                        "and commercial use"]}
