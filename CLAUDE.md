# Working in this repository

This file is for coding agents (and humans) making changes here. It is
public; keep it to engineering rules.

## Pull requests

- One item per PR. Keep PRs small and focused.
- Merge with squash, and only when CI is green on the PR's exact head.
- Before starting, check open PRs and the recent `main` log so work is
  not done twice.

## Version and CHANGELOG

- The version lives only in `helioslm_v5/version.py` (`__version__`).
  `pyproject.toml` reads it; do not write a version anywhere else.
- Every PR that adds an entry to `helioslm_v5/CHANGELOG.md` bumps
  `__version__` to match the new `## vX.Y` heading in the same PR.
  `case_harness_repair.test_version_matches_changelog` enforces this.
- If several PRs race, the first to merge takes the next number; rebase
  and renumber the others.

## Tests

- Put tests under `helioslm_v5/tests/`. CI runs
  `python -m pytest helioslm_v5/tests/ -q -n 4 --ignore=helioslm_v5/tests/test_v5.py`,
  so a test anywhere else (for example a top-level `tests/`) is never run
  by CI and a green check says nothing about it.
- Every change needs a test; optimizations of core paths need an
  equivalence oracle (see `CONTRIBUTING.md`).
- Keep the CPU path the default; guard CUDA-only code.

## Public text (README, docs, model cards, PR and issue text)

- Keep the README neutral and factual: describe what the code does and
  what was measured, with the command that reproduces it.
- Do not state or imply that the project aims to beat, surpass or replace
  any other model, product or company.
- Do not name companies in prose. Third-party model or dataset ids appear
  only where code or a config needs the exact id; in prose say
  "Taiwanese reference models" or "reference models".
- Never commit credentials or tokens; secrets come from environment
  variables.

## Files not to edit

- `docs/TW_DATASETS_2026Q4.md`
- `docs/V5.4_FIXES_REPORT.md`
