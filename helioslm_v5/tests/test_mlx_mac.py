"""v5.245 — Mac (Apple Silicon) MLX download zips and page section.

``bundle_mlx`` zips an MLX model directory (config, safetensors,
tokenizer, chat template) into one folder plus a README; ``mac_section``
lists the builds by Mac memory; ``build_site(mac_entries=...)`` and
``add_mac_to_site`` put that section on the download page.
"""
import json
import subprocess
import sys
import zipfile
from pathlib import Path

import pytest

from helioslm_v5.src.export.cloudflare import build_site
from helioslm_v5.src.export.mlx import (
    MAC_ROWS, add_mac_to_site, bundle_mlx, fix_mlx_config, mac_ram, mac_section,
    mlx_config_fixes, quant_predicate)

_REPO = Path(__file__).resolve().parents[2]
BASE = "https://dl.example.com"


def _mlx_dir(root, bits):
    d = root / f"mlx{bits}"
    d.mkdir()
    (d / "config.json").write_text(json.dumps(
        {"model_type": "gemma4", "quantization": {"group_size": 64, "bits": bits}}))
    (d / "model.safetensors").write_bytes(b"\0" * 1000)
    (d / "tokenizer.json").write_text("{}")
    (d / "chat_template.jinja").write_text("{{ messages }}")
    (d / "notes.bin").write_bytes(b"x")  # not an MLX file: left out
    return d


def test_mac_rows_cover_both_models():
    assert mac_ram("e4b", 4) == "8 GB"
    assert mac_ram("26b-a4b", 8) == "64 GB 以上"
    assert {m for _, m, _ in MAC_ROWS} == {"e4b", "26b-a4b"}
    with pytest.raises(KeyError):
        mac_ram("e4b", 3)


def test_bundle_mlx_zip(tmp_path):
    e = bundle_mlx(_mlx_dir(tmp_path, 4), tmp_path / "r2", version="mac-1",
                   name="HeliosLM TW E4B", model="e4b", bits=4, base="google/gemma-4-E4B")
    assert e["zip"] == "helioslm-tw-e4b-mlx-4bit-mac-1.zip"
    assert e["key"] == "mac-1/" + e["zip"] and e["mac_ram"] == "8 GB"
    assert e["weights_bytes"] == 1000 and not e["chat"]
    with zipfile.ZipFile(tmp_path / "r2" / e["zip"]) as z:
        names = set(z.namelist())
        info = z.getinfo("helioslm-tw-e4b-mlx-4bit/model.safetensors")
        rd = z.read("helioslm-tw-e4b-mlx-4bit/README.md").decode()
    assert names == {f"helioslm-tw-e4b-mlx-4bit/{n}" for n in (
        "config.json", "model.safetensors", "tokenizer.json", "chat_template.jinja",
        "README.md")}
    assert info.compress_type == zipfile.ZIP_STORED
    assert "--ignore-chat-template" in rd and "mlx-lm" in rd


def test_bundle_mlx_checks_bits_and_weights(tmp_path):
    d = _mlx_dir(tmp_path, 8)
    with pytest.raises(ValueError):
        bundle_mlx(d, tmp_path / "r2", version="v", name="x", model="e4b", bits=4)
    (d / "model.safetensors").unlink()
    with pytest.raises(FileNotFoundError):
        bundle_mlx(d, tmp_path / "r2", version="v", name="x", model="e4b", bits=8)


def _entries(tmp_path):
    a = bundle_mlx(_mlx_dir(tmp_path, 8), tmp_path / "r2", version="mac-1",
                   name="helioslm-tw-26b-a4b", model="26b-a4b", bits=8, chat=True)
    b = bundle_mlx(_mlx_dir(tmp_path, 4), tmp_path / "r2", version="mac-1",
                   name="helioslm-tw-e4b", model="e4b", bits=4)
    return [a, b]


def test_mac_section_order_and_commands(tmp_path):
    es = _entries(tmp_path)
    s = mac_section(es, BASE)
    assert s.index("8 GB") < s.index("64 GB 以上")  # MAC_ROWS order
    assert f'{BASE}/mac-1/helioslm-tw-e4b-mlx-4bit-mac-1.zip' in s
    assert "mlx_lm.chat --model ./helioslm-tw-26b-a4b-mlx-8bit" in s
    assert "mlx_lm.generate --model ./helioslm-tw-e4b-mlx-4bit --ignore-chat-template" in s
    assert "Intel Mac" in s
    assert "會先思考再回答" in s  # chat entry without reasoning="off"
    off = [dict(x, reasoning="off") for x in es]
    assert "會先思考再回答" not in mac_section(off, BASE)
    assert mac_section([], BASE) == ""


def test_build_site_with_mac(tmp_path):
    doc = build_site([], tmp_path / "site", base_url=BASE, version="v",
                     mac_entries=_entries(tmp_path))
    page = (tmp_path / "site" / "index.html").read_text()
    assert "Mac（Apple Silicon）· MLX" in page
    assert len(doc["mac"]) == 2 and doc["mac"][0]["url"].startswith(BASE + "/mac-1/")
    plain = build_site([], tmp_path / "plain", base_url=BASE, version="v")
    assert "mac" not in plain
    assert "MLX" not in (tmp_path / "plain" / "index.html").read_text()


def test_add_mac_to_built_site_replaces_earlier_section(tmp_path):
    site = tmp_path / "site"
    site.mkdir()
    (site / "index.html").write_text(
        "<main><h2>Downloads</h2><table></table>\n<h2>如何下載並在本機執行 · Run</h2>"
        "<p>guide</p><h2>版本與功能</h2></main>")
    (site / "packages.json").write_text('{"tiers": []}')
    es = _entries(tmp_path)
    add_mac_to_site(tmp_path / "site", es[:1], BASE)
    doc = add_mac_to_site(tmp_path / "site", es, BASE + "/")
    page = (tmp_path / "site" / "index.html").read_text()
    assert page.count("<h2>Mac（Apple Silicon）") == 1
    assert page.index("<h2>Mac（Apple Silicon）") < page.index("<h2>如何下載並在本機執行")
    assert "helioslm-tw-e4b-mlx-4bit" in page and len(doc["mac"]) == 2
    assert json.loads((tmp_path / "site" / "packages.json").read_text())["mac"][1]["url"] \
        == f"{BASE}/mac-1/helioslm-tw-e4b-mlx-4bit-mac-1.zip"


def test_package_mlx_cli_with_ready_dirs(tmp_path):
    build_site([], tmp_path / "site", base_url=BASE, version="v")
    d4, d8 = _mlx_dir(tmp_path, 4), _mlx_dir(tmp_path, 8)
    r = subprocess.run([sys.executable, str(_REPO / "examples" / "package_mlx.py"),
                        "--mlx", f"4={d4}", f"8={d8}", "--model", "e4b",
                        "--name", "helioslm-tw-e4b", "--version", "mac-1",
                        "--out", str(tmp_path / "cf"), "--site", str(tmp_path / "site"),
                        "--base-url", BASE], capture_output=True, text=True)
    assert r.returncode == 0, r.stderr
    mac = json.loads((tmp_path / "cf" / "mac.json").read_text())
    assert [m["bits"] for m in mac] == [4, 8]
    assert (tmp_path / "cf" / "r2" / "mac-1" / "helioslm-tw-e4b-mlx-8bit-mac-1.zip").exists()
    assert "16–24 GB" in (tmp_path / "site" / "index.html").read_text()


def _g26_config():
    types = (["sliding_attention"] * 5 + ["full_attention"]) * 5
    per = {f"{i:02d}": {"head_dim": 512, "num_key_value_heads": 2}
           for i, t in enumerate(types) if t == "full_attention"}
    return {"model_type": "gemma4", "text_config": {
        "head_dim": 256, "num_key_value_heads": 8, "attention_k_eq_v": True,
        "layer_types": types, "per_layer_config": per}}


def test_mlx_config_fixes_from_per_layer_config(tmp_path):
    cfg = _g26_config()
    assert mlx_config_fixes(cfg) == {"global_head_dim": 512, "num_global_key_value_heads": 2}
    (tmp_path / "config.json").write_text(json.dumps(cfg))
    assert fix_mlx_config(tmp_path) == {"global_head_dim": 512,
                                        "num_global_key_value_heads": 2}
    tc = json.loads((tmp_path / "config.json").read_text())["text_config"]
    assert tc["global_head_dim"] == 512 and tc["num_global_key_value_heads"] == 2
    assert fix_mlx_config(tmp_path) == {}  # already there
    assert mlx_config_fixes({"text_config": {"head_dim": 256}}) == {}
    bad = _g26_config()
    bad["text_config"]["per_layer_config"]["11"]["head_dim"] = 256
    with pytest.raises(ValueError):
        mlx_config_fixes(bad)


def test_quant_recipes():
    small = quant_predicate("e4b-small")
    assert small("language_model.model.per_layer_model_projection") is False
    assert small("language_model.model.layers.3.per_layer_input_gate") is False
    assert small("language_model.model.embed_tokens_per_layer") is True
    assert small("language_model.model.layers.3.mlp.down_proj") == {"group_size": 64, "bits": 6}
    assert small("language_model.model.layers.3.router.proj")["bits"] == 8
    assert quant_predicate("plain")("x.layers.0.self_attn.q_proj") is True
    assert quant_predicate("emb8")("x.embed_tokens")["bits"] == 8
    assert quant_predicate("emb8")("x.layers.0.mlp.up_proj") is True
    with pytest.raises(ValueError):
        quant_predicate("q3")
