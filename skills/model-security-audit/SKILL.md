---
name: model-security-audit
type: atomic
description: >-
  Model-security probes for the HeliosLM serving stack. Use when auditing
  a HeliosLM deployment for weight fingerprinting, membership-inference
  leakage through scalar scores, distillation leakage through logit
  exposure, or cross-tenant prefix-cache isolation. Do not use for
  adversarial-input robustness (jailbreaks/prompt injection) — out of
  scope.
guard: '{"require_authorization": true, "forbidden": ["probing third-party models or deployments without written authorization", "publishing fingerprints that identify a deployed model without the operator consent"]}'
---

# Model security audit: fingerprint, membership, distillation, cache isolation

## CORE RULE

The attack surface is defined by **observability**, not by attack names.
What an API returns per query decides what an attacker can learn:

| Observable per query | Enables | HeliosLM surface |
|---|---|---|
| Full logits [V] | fast distillation/extraction | training API only — never serve |
| Scalar NLL/ppl | membership inference (slow distillation) | `eval/score_stream.py` |
| Output tokens + hit/miss timing | prefix-cache side channels | `src/inference/prefix_pool.py` |
| Logit fingerprint vector | model identification / clone detection | any logits endpoint |

## Intent classifier

| User goal | Path (Procedure step) |
|---|---|
| Verify a deployment runs the claimed weights | 1 (fingerprint) |
| Quantify training-data memorization leakage | 2 (membership) |
| Decide whether an endpoint may expose logits | 3 (distillation budget) |
| Audit multi-tenant prefix-cache isolation | 4 (tenant isolation) |
| Audit the scorer's per-query information budget | 5 (leak-budget schema) |

Input inventory: a loadable model (or serving endpoint), a fixed probe
input set, and — for path 2 — one known member sequence. Missing the
member sequence degrades path 2 to a demonstration on synthetic data.

## When to use / when not to use

Use when: reviewing a HeliosLM serving deployment before exposure,
regression-testing that a refactor did not widen an observable, or
calibrating the trust gate against measured leakage.

Do not use for: non-HeliosLM models (the probes are written against
`EvastarV5.forward`'s `(logits, hidden, past)` contract), jailbreak or
prompt-injection robustness (behavioral, not observability, surface), or
any target you are not authorized to probe (see `guard`).

## Tool preparation

- Python 3.11+ with the repo's pinned torch CPU wheel:
  `pip install torch --index-url https://download.pytorch.org/whl/cpu`
  Verify: `python -c "import torch; print(torch.__version__)"`.
- Repo checkout on `PYTHONPATH` (run from the repo root). Verify:
  `python -c "from evastar.src.model_v5 import EvastarV5"`.

## Procedure

Every path is an executable test in `evastar/tests/case_security.py`;
run individually or through the suite runner.

1. **Fingerprint** — `test_model_fingerprint`: probe-set last-position
   logits form a fingerprint vector; identical weights give a bitwise
   identical fingerprint, any weight perturbation moves it, a
   different-seed model is far away in cosine distance.
2. **Membership** — `test_membership_inference_signal`: overfit a lite
   model on one member sequence, then confirm scalar NLL alone separates
   member from non-member with a wide margin. Takeaway: even aggregated
   scalars leak membership; do not expose scoring against private
   reference corpora without authorization.
3. **Distillation budget** — `test_distillation_leak_budget`: a student
   trained on full soft logits approaches the teacher measurably faster
   (lower final KL) than one trained on argmax hard labels at the same
   query budget. Takeaway: full-logit endpoints are the extraction
   surface; serving must return tokens or scalar scores only.
4. **Tenant isolation** — `test_prefix_pool_tenant_isolation`: two pools
   with different fingerprints over the same model never serve each
   other's KV (hits stay zero cross-tenant) while outputs remain bitwise
   equal to from-scratch generation.
5. **Leak-budget schema** — `test_score_stream_leak_budget`:
   `score_record` emits exactly four rounded scalars
   (`sum_logprob`, `n_tokens`, `nll_per_token`, `ppl`) — no per-token
   logprob vector. The rounding is part of the budget, not cosmetics.

Run all five:

```bash
python -c "
from evastar.tests import case_security as cs
for t in (cs.test_model_fingerprint,
          cs.test_membership_inference_signal,
          cs.test_distillation_leak_budget,
          cs.test_prefix_pool_tenant_isolation,
          cs.test_score_stream_leak_budget):
    t()
"
```

## Cross-domain

Consumed by the serving-hardening workflow once [[<gateway-name>]] lands;
the trust gate's quant-drift calibration (v5.43) is the numerical
complement — this skill audits the *observability* side.

## Pitfalls

- **Random-weight teachers distill too easily**: an untrained teacher's
  logits are near-uniform, so both soft and hard students converge fast
  and the gap shrinks. Sharpen the teacher distribution (low temperature)
  or use a trained checkpoint before drawing quantitative conclusions;
  the test asserts the *ordering*, not absolute KL values.
- **Membership margins collapse on short sequences**: NLL differences
  need enough tokens to dominate noise; below ~8 continuation tokens the
  member/non-member gap is unreliable on a lite model. Overfit longer
  members or raise the margin threshold.
- **Fingerprint vectors are seed-sensitive, not architecture-sensitive**:
  two configs sharing a state_dict produce identical fingerprints — that
  is the point (weights identity), but it means the probe cannot tell a
  2-layer from a relabeled 2-layer deployment. Pair with a config hash
  when architecture provenance matters.
- **Pool hit/miss timing is NOT covered**: the tenant-isolation test
  proves no *state* flows cross-fingerprint; it says nothing about
  latency leakage of hit/miss status, which `stats()` exposes by design
  to the pool owner only. Do not share one pool object across tenants —
  attach one pool per (model, tenant fingerprint).
