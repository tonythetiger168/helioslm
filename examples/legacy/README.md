# Archived scripts (#54)

Superseded versions, kept so the committed evidence that cites them can
still be reproduced. They are not maintained or tested beyond their
imports. Run them from the repository root, as before:
`python examples/legacy/eval_mid_agent_v4.py`.

## Supported versions

| What | Use | Archived |
|---|---|---|
| Mid-model agent eval | `examples/eval_mid_agent_v5.py` (v5.34: mid + GroundingGate) | `legacy/eval_mid_agent.py` (v1), `_v2`, `_v3`, `_v4` (v5.33) |
| Harness web UI | `helioslm_v5/harness/web_ui_v2.py` | `helioslm_v5/harness/legacy/web_ui_v1.py` (v1.36) |

## What each archived version is

- `eval_mid_agent.py` (v1): AgentLoop tool format. Scored 0/12; that was
  later traced to a format shift: the mid model was trained on the
  ChatSession format. Writes `checkpoints/mid_agent_eval.json`.
- `eval_mid_agent_v2.py`: the same tasks in the model's own (ChatSession)
  format. Writes `checkpoints/mid_agent_eval_v2.json`.
- `eval_mid_agent_v3.py`: prompt-diff oracle. It checks the eval
  harness's ChatSession prompt against the training-format prompt byte
  for byte (v2 scored 0/12 while training eval scored 20/20). Writes
  `checkpoints/mid_agent_eval_v3.json`.
- `eval_mid_agent_v4.py`: agency eval with full evidence capture. Writes
  `checkpoints/mid_agent_eval_v4.json` (0/12 correct, the baseline that
  v5's GroundingGate is compared against).

The JSON outputs stay in `checkpoints/` unchanged, and the changelog keeps
the old paths: they are historical records.

## Migration

- Mid eval: `python examples/eval_mid_agent_v5.py`. It needs the same
  `checkpoints/mid_sft_v5.33.*` files (`HELIOS_CKPT_DIR` to point
  elsewhere).
- Web UI: `python helioslm_v5/harness/web_ui_v2.py`. It binds 127.0.0.1,
  checks Host / Origin, and keeps sessions on disk.
- `helioslm_v5/harness/web_ui.py` is now a shim: `from harness.web_ui
  import ...` and `python helioslm_v5/harness/web_ui.py` still load v1,
  with a `DeprecationWarning`.

## Fixed while archiving

- **v1 web UI path traversal.** `/video/<name>` and `/audio/<name>`
  joined the raw URL onto `~/.helios_media`. A request such as
  `/video/../../etc/hostname` returned the file whenever that directory
  existed, and v1 binds `0.0.0.0`. The routes now resolve the name inside
  the media directory and return 404 otherwise (the same check as v2);
  `test_web_ui.py` covers it.
- **v1 entrypoint.** `python helioslm_v5/harness/web_ui.py` failed with
  `ModuleNotFoundError: harness.harness_core`: the module lives in the
  top-level `harness/`, which only pytest put on the path. v1 now adds the
  repository root itself.
