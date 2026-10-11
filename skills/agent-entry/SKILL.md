---
name: agent-entry
type: entry
description: >-
  Use when any task is routed into the HeliosLM skills system. This is
  the single entry point: it probes the task, establishes the
  authorization level, and dispatches to a domain gateway. It never
  performs domain work itself.
---

# Agent Entry

## When to use / when not to use

Use this skill as the first step of every task that lands in `skills/`:
it is the only place that establishes the authorization level and the
only place allowed to dispatch across domains. Do not use it to perform
domain work — once the domain is known and the authorization level is
established, the gateway owns the task.

## Procedure

1. **Probe the task.** Restate the user's goal in one sentence and take
   an input inventory: which assets already exist (checkpoint,
   deployment, logs, dataset) and in what form. If the goal or the
   target artifact is ambiguous, ask before dispatching — never guess.
2. **Classify the domain.** Map the task to exactly one gateway by the
   target artifact and the requested operation, not by surface keywords.
3. **Establish the authorization level.** Exactly one of:
   - `owned` — the user owns the model, weights, and deployment;
   - `eval-benchmark` — a public benchmark or evaluation artifact used
     for its documented purpose;
   - `external-unknown` — anything third-party: ask the user about
     ownership first, and allow read-only analysis only until clarified.
4. **Check the target skill's `guard`.** When `require_authorization`
   is true, dispatch only after step 3 produced a level, carry the level
   into the dispatch context, and refuse anything on the `forbidden`
   list regardless of level.
5. **Dispatch to the gateway.** Currently registered:
   - [[model-security]] — security posture of a model or deployment.
   Hand over the probe summary, the input inventory, and the
   authorization level.
6. **Stop.** The entry skill's job ends at dispatch; the gateway runs
   the workflow from there.

## Pitfalls

- Doing domain work in the entry skill "to save a step" — it bypasses
  the gateway's selection tree and breaks the audit trail; entry only
  routes.
- Dispatching to a guarded skill without an established authorization
  level — `require_authorization` is a hard precondition, not a hint.
- Classifying on a single keyword — "attack" in a debugging context is
  not a security task; probe the target artifact first.
- Naming a gateway that does not exist — the validator rejects dead
  `[[<gateway-name>]]` targets; if no gateway fits the task, say so
  instead of inventing one.
