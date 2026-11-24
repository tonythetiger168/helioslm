# Terminal-Bench Phase 2 feasibility — Harbor-native adapter, not a `term_bench` translation

Audit date: 2026-11-25  
Status: **feasibility complete; no adapter implemented and no upstream task executed**

## Decision

- **GO:** build a Harbor-native adapter for a pinned Terminal-Bench 2.1 task
  set, beginning with read-only discovery and inventory. The pinned 2.1 tasks
  use a narrow runtime subset that Harbor can preserve directly.
- **NO-GO:** do not translate upstream tasks into the existing
  `term_bench.task["verify"](root)` protocol and call the result
  Terminal-Bench. That would replace the environment, filesystem layout,
  command limits, verifier lifecycle and reward contract.
- **DEFER:** a no-docker compatibility mode is not an acceptable default or
  acceptance path. Under the conservative static gate below, **0/89** tasks
  are automatically eligible. Any future manually audited exception is a
  separately named custom subset, never a full-split Terminal-Bench score.

## Scope and non-events

This audit downloaded and parsed task definitions as data. It did **not** run
`solution/solve.sh`, environment setup scripts, `tests/test.sh`, pytest,
Dockerfiles, generated code, or any benchmark sample. It did not pull or run a
task image, start a model rollout, rent an endpoint, start SFT, or incur a
teacher API charge. There is no new Terminal-Bench score in this report.

Terminal-Bench 2.0 and 2.1 are different task sets. Terminal-Bench 2.1 fixes
28 of the 89 tasks in 2.0, so their scores and task-set hashes must not be
mixed. This audit selects 2.1; a future 2.0 run requires its own pin and
report.

## Pinned input and verification

| Identity | Value |
|---|---|
| Source | `harbor-framework/terminal-bench-2-1` |
| Source commit | `12b7565679ce5006230b820666353673a133db93` |
| Git `tasks/` tree | `2f0f5fdc68f0befd9b4745386eb8698264b00d8a` |
| Downloaded archive SHA-256 | `4d7ed7555a8c23985ddbfc9129525c3a6b4219df40512ca49878c7a8c2a85fcd` |
| Parsed tasks | 89 |
| Task files in canonical manifest | 1,037 |
| Canonical task-set SHA-256 | `0c84eb916c0ff15d0570753ee149d626d77b4d3260d43cec6581c84a062ca1ac` |

The archive's `tasks/` tree was independently reconstructed as Git blob/tree
objects and matched the source tree above. The canonical task-set manifest is
a sorted JSON list of `{path, mode, sha256}` records for every regular file or
symlink below `tasks/`; paths are relative to `tasks/`. Its serialization uses
UTF-8, sorted object keys and compact JSON separators. The task-set SHA-256 is
the SHA-256 of those canonical bytes. This definition is deterministic across
archive timestamps and compression changes and is the identity to carry into
Agentic Index artifacts.

## Static capability matrix

The audit parsed the actual task files rather than assuming that the current
Harbor documentation schema describes every published task.

| Capability | Pinned Terminal-Bench 2.1 result | Adapter consequence |
|---|---:|---|
| `task.toml` parsed | 89/89 | Discovery can cover the full pinned set. |
| `schema_version = "1.1"` | 89/89 | Do not claim that these tasks use the newer schema shown in current generic Harbor examples. |
| Linux | 89/89 | No Windows path or verifier branch is needed for this pin. |
| Prebuilt `environment.docker_image` | 89/89 | Preserve the image; do not substitute the host Python environment. |
| `environment/Dockerfile` also present | 89/89 | Both build and prebuilt provenance exist; Harbor normally prefers the prebuilt reference. |
| Compose / multi-container | 0/89 | The first native adapter can reject compose rather than silently flatten services. |
| `[[steps]]` / multi-step | 0/89 | No step scheduler is needed for this pin. |
| Separate verifier or `tests/Dockerfile` | 0/89 | Shared-verifier semantics apply to this pin; rejection logic is still required for other sets. |
| Declared artifacts | 0/89 | No separate-verifier artifact handoff is needed for this pin. |
| GPU or TPU request | 0/89 (`gpus = 0`) | CPU task declarations only; model serving resources are a separate concern. |
| MCP servers | 0/89 (all empty) | No MCP registration is needed for this pin. |
| `allow_internet = true` | 89/89 | Network access is part of the declared environment; offline execution is not equivalent. |
| Environment shell setup files | 4/89 | Static discovery must retain them; direct host execution is forbidden. |
| `tests/test.sh` | 89/89 | Keep Harbor's verifier entrypoint and lifecycle. |
| pytest with CTRF output | 89/89 | This pin is pytest-based, but the adapter must still execute `test.sh`, not invent a pytest-only generic contract. |
| `/logs/verifier/reward.txt` | 89/89 | Score from Harbor's raw reward, not a replacement Python callable. |
| Verifier installs test dependencies | 89/89 | Verifier setup may use the network and must remain inside the isolated task lifecycle. |
| Explicit `/app` path in inspected instruction/environment/verifier/solution text | 88/89 | A temporary host cwd does not reproduce absolute task paths and can leak access outside the sandbox. |
| Image reference pinned by `@sha256` digest | 0/89 | Record the resolved image digest at run time; a source commit alone does not freeze mutable tags or downloaded dependencies. |

Declared resources are modest but exceed the existing miniature's limits:
83 tasks request 1 CPU, 3 request 2 and 3 request 4; memory is 2 GiB for 68,
4 GiB for 13 and 8 GiB for 8; all request 10 GiB storage. Agent timeouts range
from 600 to 12,000 seconds and the environment build timeout is 600 seconds
for every task. These are episode/environment budgets, not per-command
limits, but they demonstrate why the existing 10-second command timeout cannot
be imposed on a comparable run.

## Why direct `term_bench` translation is not comparable

| Surface | Existing in-house `term_bench` | Pinned upstream requirement |
|---|---|---|
| Task definition | 50 built-in Python fixtures and callables | Versioned Harbor directories and TOML |
| Environment | Host commands with cwd/HOME and rlimits | Prebuilt image or Dockerfile with declared resources |
| Absolute paths | Temporary root is not a filesystem boundary | `/app`, `/tests`, `/logs/verifier` and image state |
| Execution | One bash reply per step; default 10 steps, 10-second command timeout and 2,000-character output | Long builds, services and episode-scale agent/verifier timeouts |
| Verification | `task["verify"](root)` after DONE | `tests/test.sh` in the task lifecycle, CTRF plus reward |
| Isolation | Explicit warning that commands can escape the temporary root | Untrusted task and model code must remain in an isolated runtime |
| Result identity | In-house task-set name | Source/tree/task-set hashes plus Harbor and image identities |

Making an upstream task importable or running its tests on the host would only
prove that a modified harness produced an outcome. It would not establish
Terminal-Bench compatibility.

## Recommended implementation

### Phase 2A — read-only discovery and inventory

Implement `helioslm_v5/eval/tbench_adapter.py` first as a parser/classifier:

1. Require an explicit task-set name, version and source/task-set hash.
2. Parse TOML and enumerate files without importing or executing task content.
3. Validate required files and declared schema/features.
4. Emit one machine-readable row per task, including every exclusion reason.
5. `list` must print every skipped task ID and reason; a summary count alone
   is insufficient.
6. CI uses small synthetic fixtures. It must not execute upstream tasks or
   require Docker, network access or the 2.1 archive.

### Phase 2B — Harbor-native execution bridge

Use Harbor's environment and verifier rather than recreating them in
`term_bench`:

1. Resolve the pinned task through Harbor and preserve its prebuilt image or
   build definition, resources, network declaration and timeouts.
2. Bridge the HeliosLM candidate's command/observation loop to Harbor's agent
   or `BaseEnvironment` interface after checking the then-current Harbor API.
3. On DONE or budget exhaustion, invoke Harbor's verifier and retain the raw
   reward and CTRF/verifier logs. A parser failure is not a passed task.
4. Start with a small container-only oracle/no-op smoke pair, then run the
   pinned full split only when the isolated runtime is available. Oracle
   `solve.sh` is untrusted code and receives the same isolation as model
   commands.
5. Keep native Harbor full-split results separate from any compatibility
   subset and from the existing 50-task in-house terminal score.

The pinned 2.1 declarations allow the first native runner to reject Windows,
compose, multi-step, separate verifier, artifacts, accelerators and MCPs with
explicit reasons. Those features remain part of the general Harbor format and
must not be assumed absent from other datasets or future versions.

### Phase 2C — decontamination and result integration

Before a candidate rollout, apply training-mixture decontamination to task
instructions and record excluded IDs. A contaminated task cannot silently
become a zero or disappear from the denominator. Convert only native result
rows into the terminal component descriptor, recomputing the score from all
eligible task rows and preserving the exact task-set identity.

Each run artifact must include:

- task-set name/version, source commit/tree and canonical task-set SHA-256;
- Harbor version and runtime/backend identity;
- resolved image digest per task, plus pull/build failure rows;
- model/checkpoint identity, sampling/decode settings and protocol limits;
- per-task reward, verifier status, timeout/failure category and raw-artifact
  hashes;
- full-split or exact subset ID list, skipped IDs and reasons, and
  decontamination exclusions.

Vendor-reported leaderboard numbers, HeliosLM runs of external models, and
HeliosLM candidate reproductions are three different evidence classes. They
must be labeled separately and never merged into one score table without that
distinction.

## Skip-reason taxonomy

The initial classifier should use stable machine-readable reasons, including:

- `invalid_task`, `unsupported_schema`, `missing_required_file`;
- `unsupported_os`, `multi_container`, `multi_step`;
- `separate_verifier`, `artifact_handoff`, `accelerator_requested`;
- `mcp_requested`, `network_policy_unsupported`;
- `native_runtime_unavailable`, `image_pull_or_build_failed`;
- `absolute_path_isolation_required`, `verifier_dependencies_unavailable`;
- `contaminated`.

A static inventory row is not a runtime success. Image availability, service
health, verifier behavior and actual task outcomes remain unmeasured until an
isolated run completes.

## Security and reproducibility boundaries

- Discovery treats all upstream text and files as untrusted data.
- Execution requires an existing disposable container/pod and an isolated
  Harbor runtime; it must not expose host credentials, host paths, or the
  host Docker socket to the agent task container.
- Do not weaken a declared network policy merely to make a task run. If a
  policy is unsupported, skip with a reason.
- No automatic endpoint rental or training is part of this work. API keys, if
  a later authorized model run needs them, come only from environment
  variables and are never committed; stored artifacts retain hashes, not
  secrets.
- Resolve image tags to digests and retain verifier dependency/log evidence.
  The pinned source does not make mutable image tags or network-installed
  packages bit-for-bit reproducible.

## Acceptance revision

The old acceptance — run an automatically selected docker-free subset through
`term_bench` — is withdrawn. The new acceptance is:

1. Read-only `list`/inventory covers every task in a pinned set and reports
   all skipped IDs/reasons with synthetic-fixture unit tests.
2. Native execution preserves Harbor environment, verifier and reward
   semantics inside the required isolated runtime.
3. A reported full-split 2.1 result covers all 89 tasks, including failures;
   any subset is named with its version, runtime and complete ID list.
4. A no-docker result, if ever manually justified, is labeled
   `Terminal-Bench (subset, no-docker)` and is never compared as a full-split
   or native-Harbor result.
5. No score is published from parsing, imports, static eligibility, an
   incomplete run, or a different task-set version.

## Sources

- Terminal-Bench 2.1 release and 28-task revision:
  <https://www.tbench.ai/news/terminal-bench-2-1>
- Pinned official dataset repository:
  <https://github.com/harbor-framework/terminal-bench-2-1/tree/12b7565679ce5006230b820666353673a133db93>
- Harbor task configuration:
  <https://docs.harborframework.com/tasks/configuration>
- Harbor environments:
  <https://docs.harborframework.com/tasks/environment>
- Harbor verifier and reward contract:
  <https://docs.harborframework.com/tasks/verifier>
- Separate verifier and artifacts:
  <https://docs.harborframework.com/tasks/separate-verifier>
  and <https://docs.harborframework.com/tasks/artifacts>
- Multi-step and multi-container:
  <https://docs.harborframework.com/tasks/multi-step>
  and <https://docs.harborframework.com/tasks/multi-container>
- Network policies:
  <https://docs.harborframework.com/tasks/network-policies>
