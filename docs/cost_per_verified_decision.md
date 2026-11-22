# Cost per verified decision

Status: measured baseline, 2026-11-23. Method and numbers below are
produced by `eval/cost_per_task.py` on a committed evidence artifact;
the USD figure uses an explicit $1/worker-hour normalization, not a
market price.

## Definition

A task is a **verified decision** only when an oracle backs it: the
source eval marked it `correct` AND its `final` answer still equals the
`expected` answer (`load_completion_records` re-checks the equality —
a stale or hand-edited `correct` flag never becomes a cost denominator).
Cost per verified decision is then:

```
usd_per_completed_task = worker_seconds_total / n_completed
                         * (worker_hour_usd / 3600)
```

Worker-seconds come from the disaggregated serving simulator
(`agent/disagg.py`), swept over prefill/decode worker configs and
reduced to the latency-vs-cost Pareto front (`pareto_front`).

## Measured baseline

Evidence artifact: `checkpoints/echo_completion_evidence_2026-11.json`
— 24 char-level echo tasks run through the toy checkpoint after
memorization training on 16 of them (generation recipe at the bottom).
Result: **12/24 completed** (12/16 trained, 0/8 unseen — the model
memorizes; it does not generalize). The completion rate, not the token
count, dominates the cost.

Pareto front (`cost_front`, $1/worker-hour):

| Config | Workers | Makespan (s) | Worker-s / verified | USD / verified |
| --- | --- | --- | --- | --- |
| 1P+1D | 2 | 0.636 | 0.0875 | 2.43e-05 |
| 1P+2D | 3 | 0.498 | 0.0875 | 2.43e-05 |

Reading: adding a decode worker cuts makespan 22% at identical
worker-seconds — latency is buyable, cost per verified decision is not,
because every config burns the same total work on the same 24 requests
and only 12 verify. **The cheapest way to cut cost per verified decision
at this scale is raising the completion rate, not tuning the fleet.**

These are toy-scale mechanics numbers (char-level checkpoint, prompts of
~10 tokens): the shape of the metric is the deliverable, not the
absolute dollars. Do not quote the USD figure as a product cost.

## Reproduce

```bash
python3 docs/cost_per_verified_decision_probe.py
```

The probe loads the committed evidence through the same loud schema
validation the tests use and prints the measured front as JSON. To
regenerate the evidence itself: train `_lite_model(seed=123)` on
`full = task + expected` for 16 echo tasks (AdamW lr 3e-3, 200 steps,
minibatch 4, `random.Random(20261123)` task stream), then greedy-decode
all 24 with `max_new_tokens=4` and record `final` vs `expected`.

## Honesty rules baked into the module

- Zero completed tasks → `CostSchemaError` (cost per verified decision
  is undefined, not infinite and not zero).
- Missing/typo'd fields, non-bool `correct`, empty prompt/final → loud
  schema error at load time.
- `worker_hour_usd <= 0` → loud error; a cost claim needs an explicit
  price normalization.
- A front with fewer than two points → loud error; a single config is a
  measurement, not a trade-off.
