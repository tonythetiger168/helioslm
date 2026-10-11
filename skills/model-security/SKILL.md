---
name: model-security
type: gateway
description: >-
  Use when a task concerns the security posture of a HeliosLM model or
  deployment — fingerprinting, membership inference, distillation
  leakage, tenant isolation, or output-channel budgeting. This is the
  model-security domain gateway: it owns the end-to-end workflow and
  selects the atomic skill for each goal.
guard: '{"require_authorization": true, "forbidden": ["probing third-party models or deployments without written authorization", "publishing fingerprints that identify a deployed model without the operator consent"]}'
---

# Model Security

## Full workflow

1. **Scope and authorization.** Take the authorization level from
   [[agent-entry]]. Active probes require `owned` or `eval-benchmark`;
   `external-unknown` permits read-only inspection only.
2. **Baseline observability.** Enumerate the channels the deployment
   legitimately exposes (scores, cached-prefix behavior, reload path)
   and pin the observability budget per channel. The domain's core
   rule: probes are observability-driven — measure only what the
   deployment already reveals, never add side channels.
3. **Run the atomic audit.** Dispatch per the selection tree below;
   today every goal routes to [[model-security-audit]].
4. **Quantify and report.** Express each finding as a leak budget with
   deterministic reproduction seeds, and state explicitly which
   exposures are accepted versus fixed.

## Which atomic skill when

| goal | skill / path |
|---|---|
| verify checkpoint identity, detect silent weight swaps | [[model-security-audit]] — fingerprint path |
| test whether a sample was in the training set | [[model-security-audit]] — membership-inference path |
| measure what soft labels leak to a distillation student | [[model-security-audit]] — distillation leak-budget path |
| verify prefix-cache tenant isolation | [[model-security-audit]] — tenant-isolation path |
| audit the score channel's information budget | [[model-security-audit]] — score-stream path |

New atomic skills join this table as the domain grows; a goal that fits
none of them is a gap — report it, do not stretch an existing skill.

## Cross-domain

Compound tasks routed by [[agent-entry]] that touch both model quality
and security (e.g. a numerical regression that also changes the
fingerprint) run the quality workflow first, then re-enter this gateway
to re-baseline the security probes.

## Pitfalls

- Running active probes before the authorization level is established —
  the gateway inherits the guard from entry; skipping it violates the
  domain's precondition.
- Confusing the runtime trust gate with authorization — the trust gate
  guards numerical confidence; it says nothing about whether probing
  this target is permitted.
- Treating one probe's result as conclusive — fingerprint, membership,
  and leak-budget signals are each one observable; report them as a
  budget, not a verdict.
- Letting probe outputs (fingerprints, membership scores) leak into
  public artifacts — they are themselves identifying information and
  fall under the guard's `forbidden` list.
