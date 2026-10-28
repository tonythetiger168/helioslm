"""T163 — Cloudflare download bundles per GPU-memory tier, v5.188.

``bundle_package`` turns a tier package (``package_tier`` output) into
one self-contained zip per quantization (GGUF + Modelfile + README +
manifest, stored); ``build_site`` lists every local tier on a static
page whose links point at the R2 download domain and refuses GitHub
hosts. ``examples/publish_cloudflare.py`` runs both over a ``dist/``
tree; ``scripts/cloudflare_publish.sh`` uploads (dry run here).
"""
import hashlib
import json
import os
import subprocess
import sys
import zipfile
from pathlib import Path

import pytest
import torch

from helioslm_v5.configs.config_v5 import HeliosLMv5Config
from helioslm_v5.src.export.cloudflare import (
    GENERATIONS, build_site, bundle_package, check_base_url)
from helioslm_v5.src.export.package import TIERS, package_tier
from helioslm_v5.src.model_v5 import HeliosLMv5

_REPO = Path(__file__).resolve().parents[2]
BASE = "https://dl.example.com"


@pytest.fixture(scope="module")
def dist(tmp_path_factory):
    torch.manual_seed(0)
    config = HeliosLMv5Config(size="lite", num_hidden_layers=2)
    model = HeliosLMv5(config).eval()
    root = tmp_path_factory.mktemp("dist")
    for tier in ("L-Edge", "L8"):
        package_tier(model, config, tier, root / tier)
    return root


def test_bundle_per_quant(dist, tmp_path):
    e = bundle_package(dist / "L8", tmp_path, version="v0-test")
    assert e["tier"] == "L8" and e["status"] == "preview"
    assert [b["quant"] for b in e["bundles"]] == ["Q4_K", "Q8_0"]
    m = json.loads((dist / "L8" / "manifest.json").read_text())
    for b, f in zip(e["bundles"], m["files"]):
        z = tmp_path / b["zip"]
        assert b["key"] == f"v0-test/{b['zip']}" and b["zip"].endswith("-v0-test.zip")
        assert z.stat().st_size == b["bytes"]
        assert hashlib.sha256(z.read_bytes()).hexdigest() == b["sha256"]
        with zipfile.ZipFile(z) as zf:
            names = set(zf.namelist())
            assert names == {f["file"], "README.md", "manifest.json"}  # no tokenizer
            assert all(i.compress_type == zipfile.ZIP_STORED for i in zf.infolist())
            assert hashlib.sha256(zf.read(f["file"])).hexdigest() == f["sha256"]


def test_modelfile_points_at_its_own_gguf(dist, tmp_path):
    pkg = tmp_path / "pkg"
    pkg.mkdir()
    for p in (dist / "L8").iterdir():
        (pkg / p.name).write_bytes(p.read_bytes())
    m = json.loads((pkg / "manifest.json").read_text())
    m["tokenizer"] = True
    (pkg / "manifest.json").write_text(json.dumps(m))
    e = bundle_package(pkg, tmp_path / "out", version="v1")
    for b in e["bundles"]:
        with zipfile.ZipFile(tmp_path / "out" / b["zip"]) as zf:
            assert zf.read("Modelfile").decode().startswith(f"FROM ./{b['gguf']}\n")


def test_site_lists_every_tier_and_links_r2(dist, tmp_path):
    e = bundle_package(dist / "L8", tmp_path / "r2", version="v1")
    doc = build_site([e], tmp_path / "site", base_url=BASE + "/", version="v1")
    assert [t["tier"] for t in doc["tiers"]] == list(TIERS)
    l8 = next(t for t in doc["tiers"] if t["tier"] == "L8")
    assert l8["bundles"][0]["url"] == f"{BASE}/v1/{e['bundles'][0]['zip']}"
    assert all(not t["bundles"] for t in doc["tiers"] if t["tier"] != "L8")
    page = (tmp_path / "site" / "index.html").read_text()
    assert f'href="{BASE}/v1/' in page and "github" not in page.lower()
    assert page.count("開發中") == len(TIERS) - 1 + len(GENERATIONS)
    assert [g["version"] for g in doc["generations"]] == [
        "v1.0", "v2.0", "v3.0", "v4.0", "v5.0", "v6.0", "HDL"]
    assert all(g["status"] == "開發中" and g["features"] for g in doc["generations"])
    assert {t["tier"]: t["source"] for t in doc["tiers"]}["L-Edge"] == "v1.0"
    assert json.loads((tmp_path / "site" / "packages.json").read_text()) == doc


@pytest.mark.parametrize("url", [
    "https://github.com/x/y/releases", "https://objects.githubusercontent.com/a",
    "https://x.github.io/dl", "http://dl.example.com", "dl.example.com"])
def test_base_url_refusals(url):
    with pytest.raises(ValueError):
        check_base_url(url)


def test_cli_and_dry_run_upload(dist, tmp_path):
    out = tmp_path / "cf"
    r = subprocess.run([sys.executable, str(_REPO / "examples" / "publish_cloudflare.py"),
                        "--packages", str(dist), "--out", str(out), "--base-url", BASE,
                        "--version", "v1"], capture_output=True, text=True)
    assert r.returncode == 0, r.stderr
    zips = sorted(p.name for p in (out / "r2" / "v1").iterdir())
    assert len(zips) == 4 and all(z.endswith("-v1.zip") for z in zips)
    assert "L16 no package, listed as in development" in r.stdout
    env = dict(os.environ, DRY_RUN="1", CLOUDFLARE_ACCOUNT_ID="acct", R2_BUCKET="bkt",
               CLOUDFLARE_API_TOKEN="x")
    r = subprocess.run(["bash", str(_REPO / "scripts" / "cloudflare_publish.sh"), str(out)],
                       capture_output=True, text=True, env=env)
    assert r.returncode == 0, r.stderr
    assert "s3://bkt/" in r.stdout and "acct.r2.cloudflarestorage.com" in r.stdout
    assert "pages deploy" in r.stdout and "github" not in r.stdout.lower()
