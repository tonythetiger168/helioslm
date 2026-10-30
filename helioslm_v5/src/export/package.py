"""Local deployment packages per product tier (v5.148).

One directory per tier of the README *Product roadmap* local series
(L-Edge / L8 / L16 / L24 / L48+ / L128 / L256 / L512): the tier's llama.cpp GGUF files, an
Ollama ``Modelfile``, the llama.cpp commands and a ``manifest.json``
(sizes, sha256, architecture, whether the weights fit the tier's memory).

The same call packages a pilot checkpoint today and the real weights
later. The GGUF types are the ones the exporter writes: F16 stands in for
the table's BF16 and Q4_K for Q4_K_M / Q6_K (same Q4_K blocks llama.cpp's
Q4_K_M uses for most tensors).
"""
import hashlib
import json
import shutil
from dataclasses import dataclass
from pathlib import Path

from .llamacpp import KIMI_K3, export_llamacpp_gguf, llamacpp_arch


@dataclass(frozen=True)
class Tier:
    name: str
    memory_gb: float      # VRAM (system RAM for L-Edge)
    quants: tuple         # first = the tier's default file
    ctx: int
    gpu_layers: int       # llama.cpp -ngl (0 = CPU only)
    hardware: str


TIERS = {t.name: t for t in (
    Tier("L-Edge", 8, ("q4_k", "q8_0"), 8192, 0,
         "laptop CPU, Raspberry Pi 5, phone NPU"),
    Tier("L8", 8, ("q4_k", "q8_0"), 32768, 99, "RTX 4060, 3060 Ti"),
    Tier("L16", 16, ("f16", "q4_k"), 32768, 99,
         "RTX 3060 12GB, 4070, 4060 Ti / 5060 Ti 16GB"),
    Tier("L24", 24, ("q4_k",), 32768, 99, "RTX 3090, 4090, 5090"),
    Tier("L48+", 48, ("q8_0", "f16", "q4_k"), 131072, 99,
         "RTX PRO 6000 96GB, 2x24 GB, 64-96 GB unified memory"),
    # v5.190: unified-memory tiers. 128 GB runs v4.0 100B-A10B at Q4_K or
    # v3.0 30B-A3B at Q8_0 / BF16; 256 GB runs 100B-A10B at Q8_0 or v5.0
    # 300B-A20B at Q4_K; 512 GB runs 300B-A20B at Q8_0
    Tier("L128", 128, ("q4_k", "q8_0", "f16"), 131072, 99,
         "128 GB unified memory: NVIDIA DGX Spark, AMD Ryzen AI Max+ 395, "
         "Mac (M4 / M5 series incl. Max / Ultra) 128 GB"),
    Tier("L256", 256, ("q8_0", "q4_k"), 131072, 99,
         "Mac (M4 / M5 series incl. Ultra; Mac Studio M3 Ultra) 256 GB unified memory"),
    Tier("L512", 512, ("q8_0", "q4_k"), 131072, 99,
         "Mac (M4 / M5 series incl. Ultra; Mac Studio M3 Ultra) 512 GB unified memory"),
)}

# weights may use this share of the tier's memory; the rest is KV cache,
# activations and the desktop
FIT_SHARE = 0.75
QUANT_LABEL = {"f16": "F16", "q8_0": "Q8_0", "q4_k": "Q4_K"}
# quants an importance matrix changes (v5.208)
IMATRIX_QUANTS = ("q4_k",)


def _sha256(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for block in iter(lambda: f.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def _slug(tier):
    return tier.lower().replace("+", "plus")


def modelfile(gguf_name, tier, chat=False):
    """Ollama Modelfile text for ``gguf_name``: a base model takes the raw
    prompt; a chat model (``chat``) keeps the template embedded in the
    GGUF (``tokenizer.chat_template``), so no TEMPLATE line is written."""
    template = "" if chat else 'TEMPLATE """{{ .Prompt }}"""\n'
    return f"FROM ./{gguf_name}\n{template}PARAMETER num_ctx {tier.ctx}\n"


def _readme(name, tier, arch, files, has_tokenizer):
    first = files[0]["file"]
    ngl = f" -ngl {tier.gpu_layers}" if tier.gpu_layers else ""
    lines = [f"# {name} {tier.name}", "",
             f"Hardware: {tier.hardware} ({tier.memory_gb:g} GB).", "",
             "| file | type | size | fits |", "|---|---|---|---|"]
    lines += [f"| {f['file']} | {QUANT_LABEL[f['quant']]} | {f['bytes'] / 2**30:.2f} GiB "
              f"| {'yes' if f['fits'] else 'no'} |" for f in files]
    lines += ["", "llama.cpp:", "", "```",
              f"llama-cli -m {first} -c {tier.ctx}{ngl}",
              f"llama-server -m {first} -c {tier.ctx}{ngl} --port 8080", "```", ""]
    if not has_tokenizer:
        lines += ["No tokenizer is embedded: the files take token ids, not "
                  "text, and no Ollama Modelfile is written.", ""]
    elif arch == KIMI_K3:
        lines += [f"Ollama: `ollama create {_slug(name)}-{_slug(tier.name)} -f Modelfile`. "
                  "The kimi-k3 architecture needs an Ollama build whose bundled "
                  "llama.cpp has it; llama.cpp itself has it since f5027fb.", ""]
    else:
        lines += [f"Ollama: `ollama create {_slug(name)}-{_slug(tier.name)} -f Modelfile`, "
                  f"then `ollama run {_slug(name)}-{_slug(tier.name)}`.", ""]
    return "\n".join(lines)


def package_tier(model, config, tier, out_dir, *, name="helioslm", tokenizer=None,
                 quants=None, ctx=None):
    """Write the ``tier`` package for ``model`` into ``out_dir``.

    ``tier`` is a key of :data:`TIERS`; ``quants`` / ``ctx`` override the
    tier's defaults. ``tokenizer`` (a byte-level BPE ``tokenizer.json``) is
    embedded so llama.cpp / Ollama take text; without it there is no
    Modelfile. Raises ``ValueError`` for a model llama.cpp cannot run.
    Returns the manifest dict (also written as ``manifest.json``).
    """
    if tier not in TIERS:
        raise ValueError(f"tier must be one of {list(TIERS)}, got {tier!r}")
    arch = llamacpp_arch(config)
    if arch is None:
        raise ValueError("this config has no llama.cpp architecture "
                         "(see llamacpp_incompatibilities / kimi_k3_incompatibilities)")
    t = TIERS[tier]
    if ctx is not None:
        t = Tier(t.name, t.memory_gb, t.quants, int(ctx), t.gpu_layers, t.hardware)
    quants = tuple(quants or t.quants)
    bad = [q for q in quants if q not in QUANT_LABEL]
    if bad:
        raise ValueError(f"quants must be from {list(QUANT_LABEL)}, got {bad}")
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    files = []
    for q in quants:
        fname = f"{_slug(name)}-{_slug(t.name)}-{QUANT_LABEL[q]}.gguf"
        path = export_llamacpp_gguf(model, out / fname, config, dtype=q,
                                    name=f"{name} {t.name}", tokenizer=tokenizer)
        size = Path(path).stat().st_size
        files.append({"file": fname, "quant": q, "bytes": size, "sha256": _sha256(path),
                      "fits": size <= FIT_SHARE * t.memory_gb * 2**30})
    return _write_package(out, name, t, arch, files, tokenizer is not None)


def _write_package(out, name, t, arch, files, has_tokenizer, extra=None, chat=False):
    if has_tokenizer:
        (out / "Modelfile").write_text(modelfile(files[0]["file"], t, chat))
    readme = _readme(name, t, arch, files, has_tokenizer)
    im = (extra or {}).get("imatrix")
    if im:
        readme += ("\nThe " + ", ".join(QUANT_LABEL[q] for q in im["quants"])
                   + " files were quantized with an importance matrix (llama.cpp "
                   "imatrix): same size, lower quantization error.\n")
    (out / "README.md").write_text(readme)
    manifest = {"name": name, "tier": t.name, "arch": arch, "memory_gb": t.memory_gb,
                "ctx": t.ctx, "gpu_layers": t.gpu_layers, "hardware": t.hardware,
                "tokenizer": has_tokenizer, "files": files,
                "fits": files[0]["fits"], **(extra or {})}
    (out / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    return manifest


def package_gguf(ggufs, tier, out_dir, *, name, arch, ctx=None, base=None, quants=None,
                 chat=False, imatrix=None):
    """Package GGUF files made outside this repo (v5.196).

    For a continued-pretraining checkpoint of an open Hugging Face base
    (route B), llama.cpp's own ``convert_hf_to_gguf.py`` and
    ``llama-quantize`` make the GGUFs; this writes them into the same
    tier layout as :func:`package_tier` so ``publish_cloudflare.py``
    bundles them unchanged. ``ggufs`` maps a quant key of
    :data:`QUANT_LABEL` to a GGUF path; the tier's quants that are present
    are copied in the tier's order. ``arch`` is the llama.cpp architecture
    (e.g. ``gemma4``), ``base`` the base model id kept in the manifest.
    Converted HF checkpoints carry their tokenizer, so a Modelfile is
    always written. ``quants`` overrides the tier's quant list (order =
    file order, first = default), e.g. Q8_0 on L16 for a larger model.
    ``chat`` marks an instruction-tuned model: its Modelfile keeps the
    GGUF's own chat template. ``imatrix`` (v5.208) is the llama-imatrix
    file the k-quants were made with; its sha256 and the files it applies
    to go into the manifest.
    """
    if tier not in TIERS:
        raise ValueError(f"tier must be one of {list(TIERS)}, got {tier!r}")
    t = TIERS[tier]
    if ctx is not None:
        t = Tier(t.name, t.memory_gb, t.quants, int(ctx), t.gpu_layers, t.hardware)
    bad = [q for q in ggufs if q not in QUANT_LABEL]
    if bad:
        raise ValueError(f"quants must be from {list(QUANT_LABEL)}, got {bad}")
    quants = [q for q in (quants or t.quants) if q in ggufs]
    if not quants:
        raise ValueError(f"{t.name} needs one of {list(t.quants)}, got {list(ggufs)}")
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    files = []
    for q in quants:
        fname = f"{_slug(name)}-{_slug(t.name)}-{QUANT_LABEL[q]}.gguf"
        shutil.copyfile(ggufs[q], out / fname)
        size = (out / fname).stat().st_size
        files.append({"file": fname, "quant": q, "bytes": size,
                      "sha256": _sha256(out / fname),
                      "fits": size <= FIT_SHARE * t.memory_gb * 2**30})
    extra = {"chat": chat, **({"base": base} if base else {})}
    im_quants = [f["quant"] for f in files if f["quant"] in IMATRIX_QUANTS]
    if imatrix and im_quants:
        extra["imatrix"] = {"sha256": _sha256(imatrix), "quants": im_quants}
    return _write_package(out, name, t, arch, files, True, extra, chat)
