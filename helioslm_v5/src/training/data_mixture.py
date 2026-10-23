"""Pretraining data mixture manifests (v5.116, LB3).

A mixture is a JSON file naming every source of a pretraining corpus, its
licence, its share of the tokens and its filters, so the #40 corpus is a
reviewed, hashed artifact instead of a command line. Two phases match the
LB2 WSD schedule: ``main`` sources fill the stable phase and ``decay``
sources the decay phase (``pretrain(decay_train=...)``).

    {"name": "lb3-0.5b-v1",
     "sources": [
       {"name": "fineweb2-cmn", "source": "hf:HuggingFaceFW/fineweb-2:cmn_Hani",
        "license": "ODC-BY-1.0", "license_verified": false,
        "phase": "main", "weight": 0.3,
        "zh_variants": ["tw", "mixed"], "quality_threshold": 0.5}, ...]}

Rules ``load_mixture`` enforces: unique names; phase main or decay;
weights > 0 summing to 1 within each phase; a licence from
``ALLOWED_LICENSES``, or ``"license": "review"`` with a ``license_note``
(then the source cannot be used until ``license_verified`` is true and
the licence is filled in). ``license_verified`` false is allowed for
planning; ``usable_sources`` returns only verified ones.
"""
import hashlib
import json
from pathlib import Path

ALLOWED_LICENSES = {
    "Apache-2.0", "MIT", "ODC-BY-1.0", "CC-BY-4.0", "CC-BY-SA-4.0",
    "CC0-1.0", "public-domain",
}
PHASES = ("main", "decay")
ZH_VARIANTS = {"tw", "cn", "mixed", "none"}


def validate_mixture(mix):
    if not isinstance(mix.get("name"), str) or not mix["name"]:
        raise ValueError("mixture needs a name")
    srcs = mix.get("sources") or []
    if not srcs:
        raise ValueError("mixture has no sources")
    names = [s.get("name") for s in srcs]
    if len(set(names)) != len(names) or not all(names):
        raise ValueError("source names must be present and unique")
    totals = {}
    for s in srcs:
        n = s["name"]
        if not s.get("source"):
            raise ValueError(f"{n}: no source")
        ph = s.get("phase", "main")
        if ph not in PHASES:
            raise ValueError(f"{n}: phase must be one of {PHASES}")
        w = s.get("weight")
        if not isinstance(w, (int, float)) or w <= 0:
            raise ValueError(f"{n}: weight must be > 0")
        totals[ph] = totals.get(ph, 0.0) + w
        lic = s.get("license")
        if lic == "review":
            if not s.get("license_note"):
                raise ValueError(f"{n}: license 'review' needs license_note")
            if s.get("license_verified"):
                raise ValueError(f"{n}: a 'review' licence cannot be verified")
        elif lic not in ALLOWED_LICENSES:
            raise ValueError(f"{n}: licence {lic!r} not in the allow list")
        bad = set(s.get("zh_variants") or []) - ZH_VARIANTS
        if bad:
            raise ValueError(f"{n}: unknown zh_variants {sorted(bad)}")
        q = s.get("quality_threshold")
        if q is not None and not 0.0 <= q <= 1.0:
            raise ValueError(f"{n}: quality_threshold must be in [0, 1]")
    if "main" not in totals:
        raise ValueError("mixture needs at least one main-phase source")
    for ph, t in totals.items():
        if abs(t - 1.0) > 1e-6:
            raise ValueError(f"{ph} weights sum to {t}, not 1")
    return mix


def load_mixture(path):
    return validate_mixture(json.loads(Path(path).read_text(encoding="utf-8")))


def mixture_sha256(mix):
    """Hash of the canonical JSON: two equal mixtures hash equal."""
    return hashlib.sha256(json.dumps(mix, sort_keys=True,
                                     ensure_ascii=False).encode()).hexdigest()


def token_budget(mix, total_tokens, decay_frac=0.2):
    """Tokens per source: main sources share (1 - decay_frac) of the
    total, decay sources share decay_frac (or the whole total goes to the
    main phase when the mixture has no decay sources)."""
    has_decay = any(s.get("phase", "main") == "decay" for s in mix["sources"])
    if has_decay and not 0.0 < decay_frac < 1.0:
        raise ValueError("decay_frac must be in (0, 1) with decay sources")
    share = {"main": 1.0 - decay_frac if has_decay else 1.0,
             "decay": decay_frac}
    return {s["name"]: int(round(total_tokens * share[s.get("phase", "main")]
                                 * s["weight"]))
            for s in mix["sources"]}


def usable_sources(mix):
    """Sources cleared to download: verified licence, not under review."""
    return [s for s in mix["sources"]
            if s.get("license_verified") and s.get("license") != "review"]
