"""T142 — local deployment packages per product tier, v5.148.

``package_tier`` writes the tier's GGUF files (the llama.cpp exporter's
own output), a Modelfile when a tokenizer is embedded, the llama.cpp
commands and a manifest whose sizes / hashes / fit flags match the files.
``examples/package_tiers.py`` packages a pilot run directory. The files
were also loaded and run in llama.cpp f5027fb during development (see the
v5.148 CHANGELOG entry).
"""
import hashlib
import json
import subprocess
import sys
from pathlib import Path

import pytest
import torch

from helioslm_v5.configs.config_v5 import HeliosLMv5Config
from helioslm_v5.src.export import package
from helioslm_v5.src.export.gguf import read_gguf
from helioslm_v5.src.export.package import TIERS, package_tier
from helioslm_v5.src.model_v5 import HeliosLMv5
from helioslm_v5.src.training.pretrain import pilot_config

_REPO = Path(__file__).resolve().parents[2]


def _model(**kw):
    torch.manual_seed(0)
    config = HeliosLMv5Config(size="lite", num_hidden_layers=2, **kw)
    return HeliosLMv5(config).eval(), config


def _tokenizer(tmp_path, vocab):
    tokenizers = pytest.importorskip("tokenizers")
    from tokenizers import decoders, models, pre_tokenizers, trainers
    tok = tokenizers.Tokenizer(models.BPE(unk_token="[UNK]"))
    tok.pre_tokenizer = pre_tokenizers.ByteLevel(add_prefix_space=False)
    tok.decoder = decoders.ByteLevel()
    tok.train_from_iterator(["台灣的天氣今天很好。", "hello world, hello tiers"] * 20,
                            trainer=trainers.BpeTrainer(
                                vocab_size=min(vocab, 400), show_progress=False,
                                initial_alphabet=pre_tokenizers.ByteLevel.alphabet(),
                                special_tokens=["[UNK]", "<|endoftext|>"]))
    path = tmp_path / "tokenizer.json"
    tok.save(str(path))
    return path


def test_tiers_follow_the_readme_table():
    readme = (_REPO / "README.md").read_text(encoding="utf-8")
    for name in TIERS:
        assert f"| **{name}** |" in readme
    assert TIERS["L-Edge"].gpu_layers == 0
    assert all(t.gpu_layers > 0 for n, t in TIERS.items() if n != "L-Edge")


def test_package_without_tokenizer(tmp_path):
    model, config = _model()
    m = package_tier(model, config, "L8", tmp_path)
    assert m["arch"] == "deepseek2" and m["tier"] == "L8" and m["ctx"] == 32768
    assert [f["quant"] for f in m["files"]] == ["q4_k", "q8_0"]
    for f in m["files"]:
        path = tmp_path / f["file"]
        assert path.stat().st_size == f["bytes"]
        assert hashlib.sha256(path.read_bytes()).hexdigest() == f["sha256"]
        assert f["fits"]
        md = read_gguf(path).metadata
        assert md["general.architecture"] == "deepseek2"
        assert md["general.name"] == "helioslm L8"
    assert not (tmp_path / "Modelfile").exists()
    readme = (tmp_path / "README.md").read_text()
    assert "llama-cli -m helioslm-l8-Q4_K.gguf -c 32768 -ngl 99" in readme
    assert "token ids" in readme
    assert json.loads((tmp_path / "manifest.json").read_text()) == m


def test_package_with_tokenizer_writes_modelfile(tmp_path):
    model, config = _model()
    tok = _tokenizer(tmp_path, config.vocab_size)
    out = tmp_path / "pkg"
    m = package_tier(model, config, "L-Edge", out, name="Pilot", tokenizer=tok,
                     quants=["q8_0"], ctx=4096)
    assert m["tokenizer"] and m["ctx"] == 4096 and m["gpu_layers"] == 0
    mf = (out / "Modelfile").read_text()
    assert mf.startswith("FROM ./pilot-l-edge-Q8_0.gguf\n")
    assert "PARAMETER num_ctx 4096" in mf
    readme = (out / "README.md").read_text()
    assert "-ngl" not in readme and "ollama create pilot-l-edge" in readme
    md = read_gguf(out / "pilot-l-edge-Q8_0.gguf").metadata
    assert md["tokenizer.ggml.model"] == "gpt2"


def test_fit_flag(tmp_path, monkeypatch):
    model, config = _model()
    monkeypatch.setattr(package, "FIT_SHARE", 1e-6)
    m = package_tier(model, config, "L24", tmp_path)
    assert not m["fits"] and not m["files"][0]["fits"]
    assert "| no |" in (tmp_path / "README.md").read_text()


def test_kimi_k3_package(tmp_path):
    config = pilot_config("k3_llamacpp", vocab_size=512, hidden_size=128,
                          num_layers=4, intermediate_size=256)
    torch.manual_seed(0)
    model = HeliosLMv5(config).eval()
    m = package_tier(model, config, "L16", tmp_path)
    assert m["arch"] == "kimi-k3"
    assert [f["quant"] for f in m["files"]] == ["f16", "q4_k"]
    assert read_gguf(tmp_path / m["files"][0]["file"]).metadata[
        "general.architecture"] == "kimi-k3"


def test_refusals(tmp_path):
    model, config = _model()
    with pytest.raises(ValueError, match="tier must be one of"):
        package_tier(model, config, "L12", tmp_path)
    with pytest.raises(ValueError, match="quants must be from"):
        package_tier(model, config, "L8", tmp_path, quants=["q6_k"])
    config = pilot_config("mimo_hybrid", vocab_size=512, hidden_size=128,
                          num_layers=2, intermediate_size=256)
    with pytest.raises(ValueError, match="no llama.cpp architecture"):
        package_tier(HeliosLMv5(config), config, "L8", tmp_path)


def test_cli_packages_a_pilot_run(tmp_path):
    run = tmp_path / "run"
    run.mkdir()
    config = pilot_config("dense", vocab_size=512, hidden_size=128, num_layers=2,
                          intermediate_size=256, max_seq=128)
    torch.manual_seed(0)
    model = HeliosLMv5(config)
    torch.save({"model": model.state_dict(), "step": 0}, run / "ckpt.pt")
    (run / "run.json").write_text(json.dumps({
        "data": str(tmp_path), "arch": "dense", "hidden": 128, "layers": 2,
        "intermediate": 256, "seq": 128, "balance": "smebu",
        "moe_dispatch": "batched", "kda_chunk": 64, "ple_dim": 0}))
    out = tmp_path / "dist"
    r = subprocess.run([sys.executable, str(_REPO / "examples" / "package_tiers.py"),
                        "--run", str(run), "--out", str(out), "--no-tokenizer",
                        "--tiers", "L-Edge", "L48+"],
                       capture_output=True, text=True, cwd=_REPO)
    assert r.returncode == 0, r.stderr
    assert sorted(p.name for p in out.iterdir()) == ["L-Edge", "L48+"]
    m = json.loads((out / "L48+" / "manifest.json").read_text())
    assert [f["file"] for f in m["files"]] == [
        "helioslm-l48plus-Q8_0.gguf", "helioslm-l48plus-F16.gguf", "helioslm-l48plus-Q4_K.gguf"]
    r = subprocess.run([sys.executable, str(_REPO / "examples" / "package_tiers.py"),
                        "--run", str(run), "--out", str(out)],
                       capture_output=True, text=True, cwd=_REPO)
    assert r.returncode != 0 and "--no-tokenizer" in r.stderr


def test_unified_memory_tiers():
    """T164 (v5.190): L128 names the 128 GB unified-memory machines and
    fits 100B-A10B Q4_K_M (61 GB); L256 / L512 cover Mac 256 / 512 GB
    unified memory, and their largest estimated file (300B-A20B Q4_K_M
    183 GB, Q8_0 318 GB) fits 75%."""
    t = TIERS["L128"]
    assert t.memory_gb == 128 and t.quants[0] == "q4_k" and 61 <= package.FIT_SHARE * 128
    for name in ("DGX Spark", "Ryzen AI Max+ 395", "Ultra"):
        assert name in t.hardware
    for name, gb, biggest in (("L256", 256, 183), ("L512", 512, 318)):
        t = TIERS[name]
        assert t.memory_gb == gb and t.quants == ("q8_0", "q4_k") and t.gpu_layers == 99
        assert "M4 / M5" in t.hardware and biggest <= package.FIT_SHARE * gb
    readme = (_REPO / "README.md").read_text(encoding="utf-8")
    assert "| **L128** | 128 GB unified memory |" in readme
    assert "| **L256** | 256 GB unified memory |" in readme
    assert "| **L512** | 512 GB unified memory |" in readme


def test_package_gguf_external_files(tmp_path):
    """T166 (v5.196): GGUFs made by llama.cpp from a Hugging Face checkpoint
    go into the same tier layout, in the tier's quant order, with sizes /
    hashes matching the copies, a Modelfile and the base in the manifest;
    the script's conversion plan quantizes from one F16 file."""
    src = {}
    for q, n in (("f16", 3000), ("q8_0", 2000), ("q4_k", 1000)):
        src[q] = tmp_path / f"{q}.gguf"
        src[q].write_bytes(bytes([len(q)]) * n)
    m = package.package_gguf(src, "L16", tmp_path / "L16", name="tw-pilot",
                             arch="gemma4", base="google/gemma-4-E2B")
    assert [f["quant"] for f in m["files"]] == ["f16", "q4_k"]
    assert m["arch"] == "gemma4" and m["base"] == "google/gemma-4-E2B" and m["tokenizer"]
    for f in m["files"]:
        data = (tmp_path / "L16" / f["file"]).read_bytes()
        assert f["bytes"] == len(data) and f["sha256"] == hashlib.sha256(data).hexdigest()
    assert "tw-pilot-l16-F16.gguf" in (tmp_path / "L16" / "Modelfile").read_text()
    with pytest.raises(ValueError):
        package.package_gguf({"q8_0": src["q8_0"]}, "L16", tmp_path / "x",
                             name="n", arch="gemma4")
    sys.path.insert(0, str(_REPO / "examples"))
    import package_hf
    cmds, paths = package_hf.convert_commands("ckpt", "/llama", tmp_path, ["q4_k", "q8_0"])
    assert cmds[0][1].endswith("convert_hf_to_gguf.py") and cmds[0][-1] == str(paths["f16"])
    assert [c[-1] for c in cmds[1:]] == ["Q4_K_M", "Q8_0"]
    assert all(c[1] == str(paths["f16"]) for c in cmds[1:])
    r = subprocess.run([sys.executable, "examples/package_hf.py", "--out", str(tmp_path / "d"),
                        "--name", "n", "--arch", "gemma4", "--tiers", "L8",
                        "--gguf", f"q4_k={src['q4_k']}", f"q8_0={src['q8_0']}"],
                       capture_output=True, text=True, cwd=_REPO)
    assert r.returncode == 0, r.stderr
    assert json.loads((tmp_path / "d" / "L8" / "manifest.json").read_text())["files"][0]["quant"] == "q4_k"
    from helioslm_v5.src.export.cloudflare import bundle_package
    entry = bundle_package(tmp_path / "d" / "L8", tmp_path / "cf", version="b0")
    assert [b["quant"] for b in entry["bundles"]] == ["Q4_K", "Q8_0"]


def test_package_gguf_quant_override(tmp_path):
    """T166 (v5.201): a per-tier quant override picks other files, in its
    order (a larger model ships Q8_0 on L16 instead of F16); a chat model's
    Modelfile keeps the GGUF's own chat template."""
    src = {q: tmp_path / f"{q}.gguf" for q in ("f16", "q8_0", "q4_k")}
    for p in src.values():
        p.write_bytes(b"x" * 10)
    m = package.package_gguf(src, "L16", tmp_path / "L16", name="hdl", arch="gemma4",
                             quants=["q8_0"])
    assert [f["quant"] for f in m["files"]] == ["q8_0"]
    assert "TEMPLATE" in (tmp_path / "L16" / "Modelfile").read_text() and not m["chat"]
    c = package.package_gguf(src, "L8", tmp_path / "c", name="hdl", arch="gemma4", chat=True)
    assert c["chat"] and "TEMPLATE" not in (tmp_path / "c" / "Modelfile").read_text()
    r = subprocess.run([sys.executable, "examples/package_hf.py", "--out", str(tmp_path / "d"),
                        "--name", "n", "--arch", "gemma4", "--tiers", "L16", "L24",
                        "--quants", "L16=q8_0", "L24=f16",
                        "--gguf", *[f"{q}={p}" for q, p in src.items()]],
                       capture_output=True, text=True, cwd=_REPO)
    assert r.returncode == 0, r.stderr
    got = {t: [f["quant"] for f in json.loads((tmp_path / "d" / t / "manifest.json").read_text())["files"]]
           for t in ("L16", "L24")}
    assert got == {"L16": ["q8_0"], "L24": ["f16"]}


def test_package_hf_imatrix(tmp_path):
    """T166 (v5.208): with calibration text, the conversion plan runs
    llama-imatrix on the F16 file once and quantizes only the k-quants
    with it; the manifest records the matrix's hash and which files used
    it; imatrix_calib.py interleaves and deduplicates its sources."""
    sys.path.insert(0, str(_REPO / "examples"))
    import imatrix_calib
    import package_hf
    cmds, paths = package_hf.convert_commands("ckpt", "/llama", tmp_path, ["q4_k", "q8_0"],
                                              imatrix_data="calib.txt", chunks=50)
    assert cmds[1][0].endswith("llama-imatrix") and cmds[1][cmds[1].index("-m") + 1] == str(paths["f16"])
    assert cmds[1][-2:] == ["--chunks", "50"]
    dat = cmds[1][cmds[1].index("-o") + 1]
    q4, q8 = cmds[2], cmds[3]
    assert q4[1:3] == ["--imatrix", dat] and q4[-1] == "Q4_K_M"
    assert "--imatrix" not in q8 and q8[-1] == "Q8_0"
    # a reused matrix skips llama-imatrix; Q8_0 alone needs none
    cmds, _ = package_hf.convert_commands("c", "/l", tmp_path, ["q4_k"], imatrix="old.dat")
    assert len(cmds) == 2 and cmds[1][1:3] == ["--imatrix", "old.dat"]
    cmds, _ = package_hf.convert_commands("c", "/l", tmp_path, ["q8_0"], imatrix_data="c.txt")
    assert not any("llama-imatrix" in c[0] for c in cmds)

    src = {q: tmp_path / f"{q}.gguf" for q in ("f16", "q4_k", "q8_0")}
    for p in src.values():
        p.write_bytes(b"x" * 10)
    im = tmp_path / "imatrix.dat"
    im.write_bytes(b"imatrix")
    m = package.package_gguf(src, "L8", tmp_path / "L8", name="n", arch="gemma4", imatrix=im)
    assert m["imatrix"] == {"sha256": hashlib.sha256(b"imatrix").hexdigest(), "quants": ["q4_k"]}
    assert "importance matrix" in (tmp_path / "L8" / "README.md").read_text()
    m = package.package_gguf(src, "L16", tmp_path / "L16", name="n", arch="gemma4",
                             quants=["f16"], imatrix=im)
    assert "imatrix" not in m

    a = tmp_path / "a.txt"
    a.write_text("\n\n".join(["甲" * 300, "乙" * 300, "甲" * 300]), encoding="utf-8")
    b = tmp_path / "b.jsonl"
    b.write_text("\n".join(json.dumps(r, ensure_ascii=False) for r in (
        {"text": "x" * 300}, {"messages": [{"content": "問" * 150}, {"content": "答" * 150}]},
        {"text": "short"})) + "\n", encoding="utf-8")
    docs = imatrix_calib.build([a, b], max_chars=10_000).split("\n\n")
    assert [d[0] for d in docs] == ["甲", "x", "乙", "問"]
    assert len(imatrix_calib.build([a, b], max_chars=700)) < 700


def _vocab_gguf(path, tokens, **kv):
    from helioslm_v5.src.export.gguf import write_gguf
    meta = {"general.architecture": "gemma4", "tokenizer.ggml.model": "llama",
            "tokenizer.ggml.bos_token_id": 2, "tokenizer.ggml.eos_token_id": 1,
            "tokenizer.ggml.tokens": tokens, **kv}
    from helioslm_v5.src.export.gguf import _python_to_typed
    write_gguf(path, [(k, *_python_to_typed(v)) for k, v in meta.items()],
               [("w", torch.zeros(4))], "f32")
    return path


def test_speculative_draft(tmp_path):
    """T166 (v5.209): a draft GGUF is checked against the target's
    vocabulary the way llama.cpp does, copied into the tier package with
    its hash, its llama-server -md command goes into the README and the
    Cloudflare zip carries it; a mismatched vocabulary is refused."""
    from helioslm_v5.src.export.cloudflare import bundle_package
    from helioslm_v5.src.export.gguf import read_gguf_metadata
    from helioslm_v5.src.export.speculative import vocab_mismatch
    toks = [f"t{i}" for i in range(300)]
    target = _vocab_gguf(tmp_path / "t.gguf", toks)
    good = _vocab_gguf(tmp_path / "d.gguf", toks[:260])
    assert read_gguf_metadata(target)["tokenizer.ggml.tokens"] == toks
    assert vocab_mismatch(target, good) is None
    assert "differ by more than" in vocab_mismatch(target, _vocab_gguf(
        tmp_path / "s.gguf", toks[:100]))
    assert "token 7" in vocab_mismatch(target, _vocab_gguf(
        tmp_path / "x.gguf", toks[:7] + ["zz"] + toks[8:]))
    assert vocab_mismatch(target, _vocab_gguf(tmp_path / "y.gguf", ["a"] * 3 + toks[3:])) is None
    assert "eos" in vocab_mismatch(target, _vocab_gguf(
        tmp_path / "e.gguf", toks, **{"tokenizer.ggml.eos_token_id": 9}))

    m = package.package_gguf({"q4_k": target}, "L24", tmp_path / "L24", name="g26",
                             arch="gemma4", chat=True, draft=good,
                             draft_base="google/gemma-4-E2B-it", spec_type="draft-simple")
    d = m["draft"]
    assert d["file"] == "g26-l24-draft.gguf" and d["base"] == "google/gemma-4-E2B-it"
    assert d["spec_type"] == "draft-simple"
    data = (tmp_path / "L24" / d["file"]).read_bytes()
    assert d["bytes"] == len(data) and d["sha256"] == hashlib.sha256(data).hexdigest()
    readme = (tmp_path / "L24" / "README.md").read_text()
    assert "--spec-type draft-simple -md g26-l24-draft.gguf --spec-draft-n-max 16" in readme
    with pytest.raises(ValueError, match="draft model does not fit"):
        package.package_gguf({"q4_k": target}, "L24", tmp_path / "bad", name="g26",
                             arch="gemma4", draft=tmp_path / "s.gguf", spec_type="draft-simple")
    # MTP head (default kind): checked by architecture, not vocabulary
    mtp = _vocab_gguf(tmp_path / "mtp.gguf", ["m"] * 5,
                      **{"general.architecture": "gemma4-assistant"})
    m = package.package_gguf({"q4_k": target}, "L48+", tmp_path / "L48", name="g26",
                             arch="gemma4", draft=mtp)
    assert m["draft"]["spec_type"] == "draft-mtp"
    assert "--spec-type draft-mtp -md g26-l48plus-draft.gguf --spec-draft-n-max 4" in (
        tmp_path / "L48" / "README.md").read_text()
    with pytest.raises(ValueError, match="not a 'gemma4' head"):
        package.package_gguf({"q4_k": target}, "L24", tmp_path / "bad2", name="g26",
                             arch="gemma4", draft=good)
    import zipfile
    entry = bundle_package(tmp_path / "L24", tmp_path / "cf", version="v")
    with zipfile.ZipFile(tmp_path / "cf" / entry["bundles"][0]["zip"]) as z:
        assert "g26-l24-draft.gguf" in z.namelist()
        assert "TEMPLATE" not in z.read("Modelfile").decode()  # chat model keeps its template

    sys.path.insert(0, str(_REPO / "examples"))
    import spec_bench
    s = spec_bench.summarize([{"predicted_n": 100, "predicted_ms": 2000, "draft_n": 80,
                               "draft_n_accepted": 60},
                              {"predicted_n": 100, "predicted_ms": 2000}])
    assert s == {"tokens": 200, "tok_per_s": 50.0, "draft_tokens": 80, "acceptance": 0.75}


def test_kv_cache_options(tmp_path):
    """T166 (v5.210): packages tell how to quantize the KV cache for long
    context (llama.cpp flags with flash attention, Ollama variables) and
    list the options in the manifest; quant_bench.py reads perplexity and
    KV size from llama-perplexity logs."""
    assert package.kv_flags("f16") == ""
    assert package.kv_flags("q8_0") == "-fa on -ctk q8_0 -ctv q8_0"
    with pytest.raises(ValueError):
        package.kv_flags("q2_k")
    src = tmp_path / "q4.gguf"
    src.write_bytes(b"x" * 10)
    m = package.package_gguf({"q4_k": src}, "L24", tmp_path / "L24", name="n", arch="gemma4")
    assert m["kv_cache"] == {"recommended": "q8_0", "options": ["f16", "q8_0", "q4_0"]}
    readme = (tmp_path / "L24" / "README.md").read_text()
    assert "-c 32768 -ngl 99 -fa on -ctk q8_0 -ctv q8_0" in readme
    assert "OLLAMA_KV_CACHE_TYPE=q8_0" in readme
    sys.path.insert(0, str(_REPO / "examples"))
    import quant_bench
    # an iSWA model (Gemma) logs its full and sliding-window caches separately
    log = ("llama_kv_cache: size =  512.00 MiB ( 8192 cells, 5 layers), K (q8_0): 256.00 MiB\n"
           "llama_kv_cache: size =  128.00 MiB ( 1024 cells, 25 layers)\n"
           "[1]4.1,[2]4.3,\nFinal estimate: PPL = 7.1234 +/- 0.05678\n")
    assert quant_bench.parse(log) == (7.1234, 0.05678, 640.0)
    assert quant_bench.parse("nothing") == (None, None, None)


def test_moe_expert_offload_note(tmp_path):
    """T166 (v5.211): a mixture-of-experts GGUF (``<arch>.expert_count``)
    gets the llama.cpp --cpu-moe / --n-cpu-moe advice and its expert count
    in the manifest; a dense or unreadable file gets neither."""
    moe = _vocab_gguf(tmp_path / "moe.gguf", ["a"] * 10, **{"gemma4.expert_count": 128})
    dense = _vocab_gguf(tmp_path / "dense.gguf", ["a"] * 10)
    assert package.gguf_expert_count(moe) == 128 and package.gguf_expert_count(dense) == 0
    m = package.package_gguf({"q4_k": moe}, "L16", tmp_path / "L16", name="g26", arch="gemma4",
                             quants=["q4_k"])
    assert m["experts"] == 128
    readme = (tmp_path / "L16" / "README.md").read_text()
    assert "--cpu-moe" in readme and "--n-cpu-moe N" in readme
    m = package.package_gguf({"q4_k": dense}, "L8", tmp_path / "L8", name="d", arch="gemma4")
    assert "experts" not in m and "--cpu-moe" not in (tmp_path / "L8" / "README.md").read_text()
    raw = tmp_path / "raw.gguf"
    raw.write_bytes(b"x" * 10)
    assert package.gguf_expert_count(raw) == 0
