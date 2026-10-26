"""Offline token-cost estimates and provenance for saved API responses.

No network, credentials, tokenization or live model selection. Rates are
explicit caller-supplied USD/1M token assumptions, never current-price claims.
Missing cache counters/rates yield an unknown cost, not a fabricated zero.

Field semantics checked 2026-10-25 against:
https://developers.openai.com/api/docs/guides/prompt-caching
https://platform.claude.com/docs/en/build-with-claude/prompt-caching
"""
import hashlib
from datetime import datetime, timezone
from decimal import Decimal, InvalidOperation


def _count(obj, key):
    value = obj.get(key)
    if value is not None and (type(value) is not int or value < 0):
        raise ValueError(f"{key} must be a non-negative integer")
    return value


def normalize_usage(response, provider):
    """Disjoint token buckets; None means the source did not report enough.

    OpenAI input totals INCLUDE cache reads/writes; Anthropic input_tokens
    EXCLUDES them. Keep cache-write lifetimes separate when reported.
    """
    if not isinstance(response, dict):
        raise ValueError("response must be an object")
    u = response.get("usage") or {}
    if not isinstance(u, dict):
        raise ValueError("usage must be an object")
    if provider == "openai":
        prefix = "prompt" if "prompt_tokens" in u else "input"
        total = _count(u, prefix + "_tokens")
        output = _count(u, "completion_tokens" if prefix == "prompt"
                        else "output_tokens")
        details = u.get(prefix + "_tokens_details") or {}
        if not isinstance(details, dict):
            raise ValueError("input token details must be an object")
        read = _count(details, "cached_tokens")
        write = _count(details, "cache_write_tokens")
        split = details.get("cache_write_tokens_details")
        fields = {"cache_write_30m": "ttl_30m_tokens",
                  "cache_write_12h": "ttl_12h_tokens"}
        ordinary = (total - read - write
                    if all(v is not None for v in (total, read, write)) else None)
    elif provider == "anthropic":
        ordinary = _count(u, "input_tokens")
        output = _count(u, "output_tokens")
        read = _count(u, "cache_read_input_tokens")
        write = _count(u, "cache_creation_input_tokens")
        split = u.get("cache_creation")
        fields = {"cache_write_5m": "ephemeral_5m_input_tokens",
                  "cache_write_1h": "ephemeral_1h_input_tokens"}
        total = (ordinary + read + write if all(
            v is not None for v in (ordinary, read, write)) else None)
    else:
        raise ValueError("provider must be openai or anthropic")
    if ordinary is not None and ordinary < 0:
        raise ValueError("cache reads and writes exceed total input")
    buckets = {"input": ordinary, "cache_read": read, "output": output}
    if split is not None:
        if not isinstance(split, dict):
            raise ValueError("cache write breakdown must be an object")
        counts = {name: _count(split, field) for name, field in fields.items()}
        if all(v is not None for v in counts.values()):
            if write is not None and sum(counts.values()) != write:
                raise ValueError("cache write breakdown does not match total")
        else:
            # A partial breakdown must not conceal writes in the other TTL.
            counts = {name: None for name in fields}
        buckets.update(counts)
        if write is None:
            buckets["cache_write"] = None
    else:
        buckets["cache_write"] = write
    return {"input_tokens_total": total, "buckets": buckets}


def estimate_cost(response, provider, rate_card=None):
    """Return an estimate and its assumptions. Never a provider invoice."""
    usage = normalize_usage(response, provider)
    reasons = [f"missing usage: {k}" for k, v in usage["buckets"].items()
               if v is None]
    components = {}
    if rate_card is None:
        reasons.append("no rate card supplied")
    else:
        if not isinstance(rate_card, dict):
            raise ValueError("rate card must be an object")
        for key in ("model", "provider", "as_of", "source", "rates"):
            if key not in rate_card or rate_card[key] is None:
                raise ValueError(f"rate card needs {key}")
        for key in ("model", "provider", "as_of", "source"):
            if not isinstance(rate_card[key], str) or not rate_card[key].strip():
                raise ValueError(f"rate card needs non-empty {key}")
        if not isinstance(rate_card["rates"], dict):
            raise ValueError("rates must be an object")
        if rate_card["provider"] != provider or not response.get("model") or \
                rate_card["model"] != response["model"]:
            reasons.append("rate card does not match the reported model/provider")
        if "service_tier" in rate_card and rate_card["service_tier"] != \
                response.get("service_tier"):
            reasons.append("rate card service tier mismatch or missing")
        limit = rate_card.get("max_input_tokens")
        if limit is not None:
            _count(rate_card, "max_input_tokens")
            if usage["input_tokens_total"] is None or \
                    usage["input_tokens_total"] > limit:
                reasons.append("rate card input-length scope exceeded or unknown")
        for bucket, count in usage["buckets"].items():
            if count is None or count == 0:
                continue
            rate = rate_card["rates"].get(bucket)
            if rate is None:
                reasons.append(f"missing rate: {bucket}")
                continue
            try:
                if isinstance(rate, bool):
                    raise InvalidOperation
                price = Decimal(str(rate))
                if not price.is_finite() or price < 0:
                    raise InvalidOperation
            except (InvalidOperation, ValueError):
                raise ValueError(f"invalid rate for {bucket}") from None
            components[bucket] = price * count / Decimal(1_000_000)
    cost = None if reasons else sum(components.values(), Decimal(0))
    return {**usage, "cost_usd": str(cost) if cost is not None else None,
            "cost_status": "unknown" if reasons else "estimated",
            "reasons": reasons,
            "rate_card": rate_card,
            "scope": "token charges only; excludes tool, search, storage and other fees"}


def build_report(records, *, corpus, harness_revision, tokenizer=None,
                 rate_card=None, outcomes=None, evaluation_settings=None,
                 expected_reported_models=None):
    """Records contain provider, requested_model, response and optional task_id.

    Corpus hashes the same raw UTF-8 bytes across tokenizers. Tokenizer labels
    are caller declarations; a response model string is not proof of an
    immutable weight snapshot. Outcomes are per TASK, never per API call.
    An optional exact allowlist checks response identities, not requested
    aliases. A fallback/missing identity blocks the comparison metric, while
    retaining token costs for the model actually billed. A reported-name
    match does not verify the provider's underlying weights or thinking mode.
    """
    if not isinstance(records, list) or not records:
        raise ValueError("need at least one response record")
    if not isinstance(corpus, bytes):
        raise ValueError("corpus must be raw UTF-8 bytes")
    corpus.decode("utf-8")
    if not harness_revision:
        raise ValueError("harness_revision is required")
    if evaluation_settings is not None and not isinstance(evaluation_settings, dict):
        raise ValueError("evaluation_settings must be an object")
    if expected_reported_models is not None and (
            not isinstance(expected_reported_models, list) or
            not expected_reported_models or any(
                not isinstance(m, str) or not m.strip()
                for m in expected_reported_models)):
        raise ValueError("expected_reported_models must be a non-empty list of model ids")
    rows = []
    for record in records:
        if not isinstance(record, dict) or "response" not in record or \
                "provider" not in record:
            raise ValueError("each record needs provider and response")
        response = record["response"]
        row = estimate_cost(response, record["provider"], rate_card)
        rows.append({**row, "provider": record["provider"],
                     "requested_model": record.get("requested_model"),
                     "reported_model": response.get("model"),
                     "system_fingerprint": response.get("system_fingerprint"),
                     "service_tier": response.get("service_tier"),
                     "response_created": response.get("created"),
                     "task_id": record.get("task_id")})
    known = all(r["cost_usd"] is not None for r in rows)
    total = sum((Decimal(r["cost_usd"]) for r in rows), Decimal(0)) if known else None
    mismatched = [i for i, row in enumerate(rows)
                  if expected_reported_models is not None and
                  row["reported_model"] not in expected_reported_models]
    identity_status = ("unverified" if expected_reported_models is None
                       else "mismatch" if mismatched else "matched")
    successes = None
    if outcomes is not None:
        if not isinstance(outcomes, dict):
            raise ValueError("outcomes must be an object")
        tasks = {r["task_id"] for r in rows}
        if None in tasks or set(outcomes) != tasks or any(
                type(v) is not bool for v in outcomes.values()):
            raise ValueError("outcomes must give exactly one boolean per recorded task")
        successes = sum(outcomes.values())
    return {"schema": "offline-api-accounting/1", "verification_tier": "unverified",
            "analyzed_at": datetime.now(timezone.utc).isoformat(),
            "provenance": {"harness_revision": harness_revision,
                           "corpus_sha256": hashlib.sha256(corpus).hexdigest(),
                           "corpus_utf8_bytes": len(corpus),
                           "tokenizer": tokenizer,
                           "evaluation_settings": evaluation_settings,
                           "tokenizer_verified": False, "snapshot_verified": False},
            "n_calls": len(rows), "n_successful_tasks": successes,
            "model_identity": {"status": identity_status,
                               "expected_reported_models": (list(expected_reported_models)
                                   if expected_reported_models is not None else None),
                               "mismatched_call_indices": mismatched},
            "cost_usd": str(total) if total is not None else None,
            "cost_per_correct_usd": str(total / successes)
            if total is not None and successes and not mismatched else None,
            "note": "Offline estimate from supplied counters and rates; not a live benchmark.",
            "calls": rows}
