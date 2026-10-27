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
