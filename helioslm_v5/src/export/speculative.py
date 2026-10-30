"""Speculative decoding with a draft model (v5.209).

llama.cpp's ``llama-server --spec-type <type> -md draft.gguf`` lets a
cheap drafter propose tokens that the main model checks in one batch; the
output is the main model's own, only faster. Two drafter kinds:

* ``draft-mtp``: a multi-token-prediction head trained for the main
  model (Gemma 4 ships one per size as ``google/gemma-4-*-it-assistant``,
  GGUF architecture ``gemma4-assistant``). It reuses the main model's KV
  cache and is the first choice: about 1.6x on Gemma 4 26B-A4B with ~0.7
  acceptance (llama.cpp since 2026-06).
* ``draft-simple``: any smaller model with the same vocabulary, e.g.
  Gemma 4 E2B for 26B-A4B or E4B. :func:`vocab_mismatch` repeats
  llama.cpp's ``common_speculative_are_compatible`` check (same tokenizer
  model and special tokens, vocab sizes within 128, token texts equal from
  id 5 on), so a pair is known to work before it ships.

A drafter only pays off when it is much cheaper than the main model, so
the E2B pilot (L8 / L16) gets none. ``--spec-type ngram-mod`` needs no
drafter at all (it reuses n-grams from the context) and suits code and HDL.

Measured 2026-10-30 (examples/spec_bench.py, A40, llama.cpp a34de86,
Gemma 4 26B-A4B-it Q4_K_M, 6 greedy prompts x 256 tokens): 95.9 tok/s
alone; MTP drafter Q8_0 152.9 (1.60x, acceptance 0.59); ngram-mod 147.8
(1.54x); E2B Q4_K_M as a simple drafter 68.1 (0.71x, slower), so tier
packages ship the MTP drafter, not E2B.
"""
from .gguf import read_gguf_metadata

# llama.cpp common/speculative.cpp
VOCAB_MAX_SIZE_DIFFERENCE = 128
VOCAB_CHECK_START_TOKEN_ID = 5
# drafted tokens per step (llama-server --spec-draft-n-max) per drafter
# kind: an MTP head drafts a few, a separate small model more
SPEC_TYPES = {"draft-mtp": 4, "draft-simple": 16}
DRAFT_MIN = 1
DRAFT_P_MIN = 0.75

_SPECIAL = ("tokenizer.ggml.model", "tokenizer.ggml.add_bos_token",
            "tokenizer.ggml.bos_token_id", "tokenizer.ggml.eos_token_id")


def draft_mismatch(target, draft, spec_type="draft-simple"):
    """Why ``draft`` cannot draft for ``target`` under ``spec_type`` (``None`` if it can).

    An MTP drafter must be an ``<arch>-assistant`` (or other ``<arch>-*``)
    GGUF of the main model's architecture; a simple drafter must share the
    vocabulary (:func:`vocab_mismatch`).
    """
    if spec_type not in SPEC_TYPES:
        raise ValueError(f"spec_type must be one of {list(SPEC_TYPES)}, got {spec_type!r}")
    if spec_type == "draft-simple":
        return vocab_mismatch(target, draft)
    t = target if isinstance(target, dict) else read_gguf_metadata(target)
    d = draft if isinstance(draft, dict) else read_gguf_metadata(draft)
    ta, da = t.get("general.architecture", ""), d.get("general.architecture", "")
    if not ta or not da.startswith(ta + "-"):
        return f"MTP drafter architecture {da!r} is not a {ta!r} head"
    return None


def vocab_mismatch(target, draft):
    """Why the draft cannot serve ``target`` (``None`` when it can).

    ``target`` / ``draft`` are GGUF metadata dicts or paths to GGUF files.
    """
    t = target if isinstance(target, dict) else read_gguf_metadata(target)
    d = draft if isinstance(draft, dict) else read_gguf_metadata(draft)
    tt, dt = t.get("tokenizer.ggml.tokens"), d.get("tokenizer.ggml.tokens")
    if not tt or not dt:
        return "no embedded vocabulary (tokenizer.ggml.tokens)"
    for key in _SPECIAL:
        if t.get(key) != d.get(key):
            return f"{key} differs: {t.get(key)!r} vs {d.get(key)!r}"
    if abs(len(tt) - len(dt)) > VOCAB_MAX_SIZE_DIFFERENCE:
        return f"vocab sizes {len(tt)} vs {len(dt)} differ by more than {VOCAB_MAX_SIZE_DIFFERENCE}"
    for i in range(VOCAB_CHECK_START_TOKEN_ID, min(len(tt), len(dt))):
        if tt[i] != dt[i]:
            return f"token {i} differs: {tt[i]!r} vs {dt[i]!r}"
    return None


def server_args(draft_file, spec_type="draft-simple", gpu_layers=99, n_max=None):
    """llama-server arguments that add ``draft_file`` as a ``spec_type`` drafter."""
    args = ["--spec-type", spec_type, "-md", str(draft_file),
            "--spec-draft-n-max", str(n_max or SPEC_TYPES[spec_type])]
    if spec_type == "draft-simple":
        args += ["--spec-draft-n-min", str(DRAFT_MIN), "--spec-draft-p-min", str(DRAFT_P_MIN)]
    if gpu_layers:
        args += ["-ngld", str(gpu_layers)]
    return args


def server_command(target_file, draft_file, ctx, gpu_layers=99, port=8080,
                   spec_type="draft-simple"):
    """``llama-server`` command line serving ``target_file`` with ``draft_file``."""
    ngl = f" -ngl {gpu_layers}" if gpu_layers else ""
    return (f"llama-server -m {target_file} -c {ctx}{ngl} "
            + " ".join(server_args(draft_file, spec_type, gpu_layers)) + f" --port {port}")
