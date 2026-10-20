# PR #16 Python 3.11 CI investigation

Reviewed on 2026-10-18 against main `a0f09332f1fc6633615d48397dda77e629ac6cf6`.

## Decision

Close [PR #16](https://github.com/tonythetiger168/helioslm/pull/16) without
merging and delete `debug/ci-py311-probe`. Its bisect is complete, the useful
thread limit is already on main, and subsequent full Python 3.11 CI is green.
No runtime repair remains necessary for this investigation.

## Evidence

- Before the thread limit, [PR #15 run #349, Python 3.11 log](https://github.com/tonythetiger168/helioslm/actions/runs/38337599562/job/115101114004)
  reports `AssertionError: distill loss did not drop: 5.2185 -> 4.6427` in
  `test_indexer_distill_learned_sweep`, then exits with code 1. This observed
  failure is a numerical test assertion; it does not establish an OOM or
  thread-creation failure.
- The latest debug run is [#353](https://github.com/tonythetiger168/helioslm/actions/runs/38343271452),
  at `e78f428f4fe4a30b4e893950bf043fdafcbff436` (probe 4). Unlike the initial
  stub and half-suite probes, it restores the full 90-test runner and sets
  `torch.set_num_threads(2)`. Both Python versions pass. The [Python 3.11 log](https://github.com/tonythetiger168/helioslm/actions/runs/38343271452/job/115118810356)
  confirms 90/90 runner tests, 263 pytest tests, and 9/9 integration tests.
- [PR #15](https://github.com/tonythetiger168/helioslm/pull/15) includes
  `7cc9d49` and `e981ff9`, preserving main's test registration order and adding
  the two-thread guard only when `GITHUB_ACTIONS == "true"`. Its final
  [run #356](https://github.com/tonythetiger168/helioslm/actions/runs/38362172207)
  passes both Python 3.11 and 3.12. It merged as `fc98e5b`.
- [PR #34](https://github.com/tonythetiger168/helioslm/pull/34) merged as
  `9f42c8b`, adding `requirements-ci.txt`: torch 2.14.1, pytest 9.1.1,
  numpy 2.4.6 for Python <3.12 and 2.5.4 for Python >=3.12. The recorded
  reason for the numpy split is the absence of CPython 3.11 wheels in numpy
  2.5. Its [main run #403](https://github.com/tonythetiger168/helioslm/actions/runs/38921096405)
  passes installation and all test stages on both versions.
- The latest main [run #408](https://github.com/tonythetiger168/helioslm/actions/runs/39098632198)
  passes both versions. Its [Python 3.11 log](https://github.com/tonythetiger168/helioslm/actions/runs/39098632198/job/117590147771)
  confirms installation of the cp311 numpy 2.4.6 wheel, torch 2.14.1+cpu,
  and pytest 9.1.1, followed by 93/93 runner tests, 418 pytest tests, and
  9/9 integration tests. The workflow still runs the full runner, pytest,
  and integration stages; main retains the two-thread CI guard.

## Interpretation and cleanup safety

The full-suite success with the thread limit, its successful incorporation
into PR #15, and subsequent green main runs are sufficient operational
evidence that the original CI problem is resolved. They do not prove the
precise low-level cause. In particular, debug run #353 already installed
numpy 2.4.6 and torch 2.14.1+cpu before dependencies were pinned: PR #34
stabilizes dependency resolution rather than demonstrating a separate fix
for the original assertion.

The debug branch diverges from main; its commits are not all ancestors of
main. That is expected after squash merging PR #15. Its six feature commits
are included in PR #15; its remaining four commits are the stub, two
half-suite probes, and the final unconditional thread pin. Main carries the
production CI-only version of that pin. Merging the old debug branch would
therefore be inappropriate. Closing it and deleting its branch does not
remove the fixes from main.

Validation for this documentation/comment-only follow-up consists of the
linked GitHub Actions jobs, source/history comparison, and `git diff --check`.
No new test execution or dependency changes are needed.
