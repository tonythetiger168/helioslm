"""Read-only Harbor task-set identity, discovery and capability inventory.

This module deliberately has no execution command.  Task TOML and text are
untrusted data: discovery never imports Harbor, runs a task script, builds an
image, opens a socket, or reads host environment variables.  Native rollout
and verification are a separate, container-only phase.

Example (all identity pins are mandatory)::

    python -m helioslm_v5.eval.tbench_adapter list \
      --root /path/to/terminal-bench-2-1/tasks \
      --task-set terminal-bench-2.1 \
      --source-commit 12b7565679ce5006230b820666353673a133db93 \
      --expected-tasks-tree 2f0f5fdc68f0befd9b4745386eb8698264b00d8a \
      --expected-task-set-sha256 \
        0c84eb916c0ff15d0570753ee149d626d77b4d3260d43cec6581c84a062ca1ac \
      --mode native --out inventory.json
"""

import argparse
import hashlib
import json
import os
import re
import stat
import sys
import tomllib
from pathlib import Path


SCHEMA = "tbench-inventory/1"
SUPPORTED_TASK_SCHEMA = "1.1"
MAX_CONFIG_BYTES = 1024 * 1024
MAX_INSPECTION_BYTES = 4 * 1024 * 1024
_TASK_ID = re.compile(r"[A-Za-z0-9][A-Za-z0-9_.-]{0,127}\Z")
_HEX40 = re.compile(r"[0-9a-f]{40}\Z")
_HEX64 = re.compile(r"[0-9a-f]{64}\Z")
_ABSOLUTE_APP = re.compile(r"/app(?:/|\b)")
_VERIFIER_INSTALL = re.compile(r"\b(?:pip|pip3|uv|uvx)\b")
_RESOURCE_KEYS = ("cpus", "memory_mb", "storage_mb", "gpus", "gpu_types", "tpu")
_INSPECTION_FILES = (
    "instruction.md", "environment/Dockerfile", "tests/test.sh",
    "solution/solve.sh",
)


class TaskSetError(ValueError):
    """Invalid identity, unsafe file shape, or unreadable task metadata."""


def _blob_sha(data):
    return hashlib.sha1(b"blob " + str(len(data)).encode() + b"\0" + data).hexdigest()


def _regular_hashes(path):
    """Hash a regular file without following a final-component symlink."""
    flags = os.O_RDONLY | getattr(os, "O_NOFOLLOW", 0)
    fd = os.open(path, flags)
    with os.fdopen(fd, "rb", closefd=True) as handle:
        info = os.fstat(handle.fileno())
        if not stat.S_ISREG(info.st_mode):
            raise TaskSetError(f"not a regular file: {path}")
        sha256 = hashlib.sha256()
        git = hashlib.sha1(b"blob " + str(info.st_size).encode() + b"\0")
        while True:
            block = handle.read(1024 * 1024)
            if not block:
                break
            sha256.update(block)
            git.update(block)
        mode = "100755" if info.st_mode & 0o111 else "100644"
        return mode, sha256.hexdigest(), git.hexdigest()


def _tree_sha(entries):
    """Reconstruct Git's tasks tree from (relative path, mode, blob SHA)."""
    root = {}
    for relative, mode, blob in entries:
        parts = relative.split("/")
        node = root
        for part in parts[:-1]:
            child = node.get(part)
            if not isinstance(child, dict):
                child = {}
                node[part] = child
            node = child
        node[parts[-1]] = (mode, blob)

    def digest(node):
        encoded = []
        for name, child in node.items():
            is_dir = isinstance(child, dict)
            mode = "40000" if is_dir else child[0]
            blob = digest(child) if is_dir else child[1]
            key = name.encode("utf-8") + (b"/" if is_dir else b"")
            encoded.append((key, mode, name, blob))
        body = b"".join(
            mode.encode() + b" " + name.encode("utf-8") + b"\0" + bytes.fromhex(blob)
            for _, mode, name, blob in sorted(encoded, key=lambda row: row[0]))
        return hashlib.sha1(b"tree " + str(len(body)).encode() + b"\0" + body).hexdigest()

    return digest(root)


def task_set_identity(root):
    """Return canonical manifest, its SHA-256, and the Git tasks-tree SHA-1.

    Canonical bytes are the UTF-8 compact JSON list of sorted
    ``{path, mode, sha256}`` records.  Paths are relative to ``root`` and use
    POSIX separators.  Symlinks contribute their link text and are never
    followed.  Empty directories and non-Git special files are not portable:
    the former are absent from Git, while the latter are rejected.
    """
    root = Path(root)
    if root.is_symlink() or not root.is_dir():
        raise TaskSetError(f"tasks root is not a real directory: {root}")
    manifest = []
    git_entries = []

    def walk(directory, prefix):
        with os.scandir(directory) as iterator:
            children = sorted(iterator, key=lambda entry: os.fsencode(entry.name))
        for entry in children:
            relative = prefix + entry.name
            try:
                relative.encode("utf-8")
            except UnicodeEncodeError as exc:
                raise TaskSetError("task paths must be valid UTF-8") from exc
            path = Path(entry.path)
            if entry.is_symlink():
                data = os.fsencode(os.readlink(path))
                manifest.append({"path": relative, "mode": "120000",
                                 "sha256": hashlib.sha256(data).hexdigest()})
                git_entries.append((relative, "120000", _blob_sha(data)))
            elif entry.is_dir(follow_symlinks=False):
                walk(path, relative + "/")
            elif entry.is_file(follow_symlinks=False):
                mode, sha256, blob = _regular_hashes(path)
                manifest.append({"path": relative, "mode": mode, "sha256": sha256})
                git_entries.append((relative, mode, blob))
            else:
                raise TaskSetError(f"unsupported special file in task set: {relative!r}")

    walk(root, "")
    manifest.sort(key=lambda row: row["path"])
    canonical = json.dumps(manifest, ensure_ascii=False, sort_keys=True,
                           separators=(",", ":")).encode("utf-8")
    return manifest, hashlib.sha256(canonical).hexdigest(), _tree_sha(git_entries)


def _read_limited(root, relative, limit):
    path = Path(root) / relative
    flags = os.O_RDONLY | getattr(os, "O_NOFOLLOW", 0)
    fd = os.open(path, flags)
    with os.fdopen(fd, "rb", closefd=True) as handle:
        info = os.fstat(handle.fileno())
        if not stat.S_ISREG(info.st_mode):
            raise TaskSetError(f"required task file is not regular: {relative!r}")
        data = handle.read(limit + 1)
    if len(data) > limit:
        raise TaskSetError(f"task file exceeds {limit} bytes: {relative!r}")
    return data


def _inspection_text(root, task_id, relative):
    try:
        return _read_limited(root, f"{task_id}/{relative}",
                             MAX_INSPECTION_BYTES).decode("utf-8", "replace")
    except (OSError, TaskSetError):
        return ""


def _section(config, key):
    value = config.get(key, {})
    if not isinstance(value, dict):
        raise TaskSetError(f"task section [{key}] must be a table")
    return value


def _accelerator_requested(environment):
    gpus = environment.get("gpus", 0)
    if isinstance(gpus, bool) or not isinstance(gpus, (int, float)) or gpus != 0:
        return True
    if environment.get("gpu_types"):
        return True
    return bool(environment.get("tpu"))


def _classify(root, task_id, config, files, mode):
    environment = _section(config, "environment")
    agent = _section(config, "agent")
    verifier = _section(config, "verifier")
    schema = config.get("schema_version")
    os_name = environment.get("os", "linux")
    image = environment.get("docker_image")
    compose = ("environment/docker-compose.yaml" in files or
               "environment/docker-compose.yml" in files)
    dockerfile = "environment/Dockerfile" in files
    texts = {name: _inspection_text(root, task_id, name)
             for name in _INSPECTION_FILES if name in files}
    test_text = texts.get("tests/test.sh", "")
    absolute_app = any(_ABSOLUTE_APP.search(text) for text in texts.values())
    verifier_install = bool(_VERIFIER_INSTALL.search(test_text))
    shell_files = sorted(path for path in files
                         if path.startswith("environment/") and path.endswith(".sh"))
    row = {
        "id": task_id,
        "schema_version": str(schema) if schema is not None else None,
        "os": str(os_name),
        "docker_image": image if isinstance(image, str) else None,
        "dockerfile": dockerfile,
        "compose": compose,
        "environment_shell_files": shell_files,
        "steps": len(config.get("steps", [])) if isinstance(config.get("steps", []), list) else -1,
        "artifacts_declared": bool(config.get("artifacts")),
        "verifier_mode": verifier.get("environment_mode", "shared"),
        "resources": {key: environment[key] for key in _RESOURCE_KEYS if key in environment},
        "allow_internet": environment.get("allow_internet"),
        "absolute_app_path": absolute_app,
        "verifier_installs_dependencies": verifier_install,
    }
    reasons = []
    if str(schema) != SUPPORTED_TASK_SCHEMA:
        reasons.append("unsupported_schema")
    if os_name != "linux":
        reasons.append("unsupported_os")
    if compose:
        reasons.append("multi_container")
    if config.get("steps"):
        reasons.append("multi_step")
    if (verifier.get("environment_mode", "shared") != "shared" or
            "environment" in verifier or "tests/Dockerfile" in files or
            "tests/docker-compose.yaml" in files or "tests/docker-compose.yml" in files):
        reasons.append("separate_verifier")
    if config.get("artifacts"):
        reasons.append("artifact_handoff")
    if _accelerator_requested(environment):
        reasons.append("accelerator_requested")
    if environment.get("mcp_servers"):
        reasons.append("mcp_requested")
    for section in (environment, agent, verifier):
        if section.get("network_mode") == "allowlist" or section.get("allowed_hosts"):
            reasons.append("network_policy_unsupported")
            break
    required = {"instruction.md", "task.toml", "tests/test.sh"}
    if not required.issubset(files):
        reasons.append("missing_required_file")
    if not image and not dockerfile and not compose:
        reasons.append("missing_required_file")
    if mode == "no-docker":
        # A temporary cwd is not an isolation boundary.  Never auto-admit a
        # task to host execution, even when static parsing found no other gap.
        reasons.append("native_runtime_unavailable")
        if absolute_app:
            reasons.append("absolute_path_isolation_required")
        if verifier_install:
            reasons.append("verifier_dependencies_unavailable")
    row["skipped_reasons"] = sorted(set(reasons))
    row["status"] = "skipped" if reasons else "eligible_static"
    return row


def inventory(root, task_set, source_commit, expected_task_set_sha256,
              expected_tasks_tree=None, mode="native"):
    """Verify task-set identity and return a deterministic, score-free report."""
    if mode not in ("native", "no-docker"):
        raise TaskSetError("mode must be native or no-docker")
    if not task_set or not task_set.strip():
        raise TaskSetError("task_set must be non-empty")
    if not _HEX40.fullmatch(source_commit):
        raise TaskSetError("source_commit must be 40 lowercase hexadecimal characters")
    if not _HEX64.fullmatch(expected_task_set_sha256):
        raise TaskSetError("expected task-set SHA must be 64 lowercase hexadecimal characters")
    if expected_tasks_tree is not None and not _HEX40.fullmatch(expected_tasks_tree):
        raise TaskSetError("expected tasks tree must be 40 lowercase hexadecimal characters")
    root = Path(root)
    manifest, task_set_sha256, tasks_tree = task_set_identity(root)
    if task_set_sha256 != expected_task_set_sha256:
        raise TaskSetError("task-set SHA-256 mismatch; refusing to relabel task content")
    if expected_tasks_tree is not None and tasks_tree != expected_tasks_tree:
        raise TaskSetError("Git tasks-tree mismatch; refusing to relabel task content")
    by_task = {}
    for item in manifest:
        parts = item["path"].split("/", 1)
        if len(parts) == 2:
            by_task.setdefault(parts[0], set()).add(parts[1])
    if not by_task:
        raise TaskSetError("no task directories found")
    rows = []
    for task_id in sorted(by_task):
        files = by_task[task_id]
        if not _TASK_ID.fullmatch(task_id):
            rows.append({"id": task_id, "status": "skipped",
                         "skipped_reasons": ["invalid_task"]})
            continue
        try:
            raw = _read_limited(root, f"{task_id}/task.toml", MAX_CONFIG_BYTES)
            config = tomllib.loads(raw.decode("utf-8"))
            rows.append(_classify(root, task_id, config, files, mode))
        except (OSError, UnicodeError, tomllib.TOMLDecodeError, TaskSetError) as exc:
            rows.append({"id": task_id, "status": "skipped",
                         "skipped_reasons": ["invalid_task"],
                         "error": str(exc)[:500]})
    eligible = sum(row["status"] == "eligible_static" for row in rows)
    return {
        "schema": SCHEMA,
        "mode": mode,
        "task_set": task_set,
        "source_commit": source_commit,
        "tasks_tree": tasks_tree,
        "task_set_sha256": task_set_sha256,
        "task_files": len(manifest),
        "summary": {"tasks": len(rows), "eligible_static": eligible,
                    "skipped": len(rows) - eligible},
        "rows": rows,
    }


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    listing = commands.add_parser("list", help="read-only discovery and capability inventory")
    listing.add_argument("--root", required=True, help="upstream tasks/ directory")
    listing.add_argument("--task-set", required=True)
    listing.add_argument("--source-commit", required=True)
    listing.add_argument("--expected-task-set-sha256", required=True)
    listing.add_argument("--expected-tasks-tree")
    listing.add_argument("--mode", choices=("native", "no-docker"), default="native")
    listing.add_argument("--out", help="optional JSON report outside the task root")
    args = parser.parse_args(argv)
    try:
        if args.out:
            output = Path(args.out).resolve()
            root = Path(args.root).resolve()
            if output == root or root in output.parents:
                raise TaskSetError("--out must be outside the read-only task root")
        report = inventory(args.root, args.task_set, args.source_commit,
                           args.expected_task_set_sha256, args.expected_tasks_tree,
                           args.mode)
    except (OSError, TaskSetError) as exc:
        parser.exit(2, f"error: {exc}\n")
    for row in report["rows"]:
        task_id = json.dumps(row["id"], ensure_ascii=True)
        reasons = ",".join(row["skipped_reasons"]) or "-"
        print(f"task={task_id} status={row['status']} reasons={reasons}")
    print(json.dumps(report["summary"], sort_keys=True))
    if args.out:
        output = Path(args.out)
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text(json.dumps(report, ensure_ascii=False, sort_keys=True,
                                    indent=2) + "\n", encoding="utf-8", newline="\n")
    return 0


if __name__ == "__main__":
    sys.exit(main())
