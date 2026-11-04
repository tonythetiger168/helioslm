"""CPU pin for the checked-in serverless vLLM worker env.

Reads ``serving/worker.env.example`` and ``serving/README.md``. No socket,
no sleep, no network, so the daily gate folds this file (see
``test_worker_launch_env_suite``). ``test_openai_stream_smoke`` stays
exempt because it binds a port.

Allowlist checked against worker ``src/args_builder.py``
(``VALUE_FLAGS``, ``ENV_ALIASES``, ``RESERVED_ENV_VARS``) and
``docs/configuration.md`` at commit
e8b857df84a64871d3a51b22ea9e1f15d8c43c80. ``MAX_NUM_SEQS`` is a value
flag, so it does not need ``VLLM_EXTRA_ARGS``. ``HF_TOKEN`` is reserved:
the worker forwards it in the child environment and never as ``--hf-token``.
This file must not assign it.
"""

import json
import re
from pathlib import Path

_REPO = Path(__file__).resolve().parents[2]
_ENV_PATH = _REPO / "serving" / "worker.env.example"
_README_PATH = _REPO / "serving" / "README.md"

# UPPER_SNAKE of a VALUE_FLAGS entry (flag_to_env_name).
_VALUE_FLAGS = {
    "DTYPE",
    "LOAD_FORMAT",
    "MAX_MODEL_LEN",
    "MAX_NUM_SEQS",
    "SPECULATIVE_CONFIG",
    "TENSOR_PARALLEL_SIZE",
}
# Historical aliases that do not match a flag name.
_ALIASES = {
    "MODEL_NAME",
    "OPENAI_SERVED_MODEL_NAME_OVERRIDE",
}
_SPECULATIVE = (
    '{"method":"mtp","model":"google/gemma-4-26B-A4B-it-assistant",'
    '"num_speculative_tokens":4}'
)
_ASSISTANT_REPO = "google/gemma-4-26B-A4B-it-assistant"
_BANNED_VENDOR = re.compile(r"runpod|cloudflare|bitstarlabs", re.IGNORECASE)
_HF_TOKEN = re.compile(r"hf_[A-Za-z0-9]{8,}", re.IGNORECASE)
_SK_TOKEN = re.compile(r"(?:^|[^A-Za-z0-9])sk-[A-Za-z0-9_-]{8,}")
_HEX_VALUE = re.compile(r"^[0-9a-fA-F]{32,}$")
_B64_VALUE = re.compile(r"^[A-Za-z0-9+/]{40,}={0,2}$")
_QUANT_WORD = re.compile(
    r"fp8|int8|int4|awq|gptq|nvfp4|bitsandbytes|mxfp4",
    re.IGNORECASE,
)


def _parse_env(text):
    """KEY=VALUE assignments. A comment on the previous line is required."""
    env = {}
    lines = text.splitlines()
    for index, raw in enumerate(lines):
        line = raw.strip()
        if not line or line.startswith("#"):
            continue
        if raw != line:
            raise AssertionError(f"assignment is indented: {raw!r}")
        key, sep, value = line.partition("=")
        assert sep, f"not an assignment: {line!r}"
        assert key and key == key.strip(), key
        assert " " not in key, key
        assert "#" not in value, f"inline comment in {key}"
        prev = lines[index - 1].strip() if index else ""
        assert prev.startswith("#") and prev != "#", (
            f"{key} needs a one-line comment")
        assert key not in env, key
        env[key] = value
    return env


def _assert_clean_prose(text):
    assert not _BANNED_VENDOR.search(text), "vendor name in serving prose"
    assert not _HF_TOKEN.search(text)
    assert not _SK_TOKEN.search(text)
    stripped = text.replace(_ASSISTANT_REPO, "")
    assert "google" not in stripped.lower()


def test_worker_launch_env():
    text = _ENV_PATH.read_text(encoding="utf-8")
    readme = _README_PATH.read_text(encoding="utf-8")
    assert "\r" not in text and "\r" not in readme
    _assert_clean_prose(text)
    _assert_clean_prose(readme)
    assert len(readme.splitlines()) <= 60

    env = _parse_env(text)
    assert set(env) == _VALUE_FLAGS | _ALIASES
    assert "VLLM_EXTRA_ARGS" not in env
    assert "QUANTIZATION" not in env
    assert "KV_CACHE_DTYPE" not in env

    assert env["MODEL_NAME"] == "chienhsinlin/helioslm-chat-g26b"
    assert env["OPENAI_SERVED_MODEL_NAME_OVERRIDE"] == "helioslm"
    assert env["DTYPE"] == "bfloat16"
    assert env["LOAD_FORMAT"] == "safetensors"
    assert env["TENSOR_PARALLEL_SIZE"] == "1"
    assert env["MAX_MODEL_LEN"] == "16384"
    assert env["MAX_NUM_SEQS"] == "16"
    assert env["SPECULATIVE_CONFIG"] == _SPECULATIVE

    spec = json.loads(env["SPECULATIVE_CONFIG"])
    assert spec["method"] == "mtp"
    assert spec["num_speculative_tokens"] == 4
    assert spec["model"] == _ASSISTANT_REPO
    assert set(spec) == {"method", "model", "num_speculative_tokens"}
    assert "dflash" not in env["SPECULATIVE_CONFIG"].lower()
    assert "dflash" not in text.lower()

    for key, value in env.items():
        assert value != "", key
        assert not key.endswith(("_TOKEN", "_KEY", "_SECRET", "_PASSWORD"))
        assert "TOKEN" not in key and "KEY" not in key
        assert not _HF_TOKEN.search(value)
        assert not _SK_TOKEN.search(value)
        assert not _HEX_VALUE.match(value)
        assert not _B64_VALUE.match(value)
        assert not _QUANT_WORD.search(value), value

    assert "HF_TOKEN" in text
    assert "scale to zero" in text.lower()
    assert "VLLM_VERSION=v0.33.0" in text
    assert "v0.23.0" in text
