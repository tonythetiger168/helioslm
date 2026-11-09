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

try:
    import pytest
except ImportError:
    # v5.240: this file is also executed directly by the daily TESTS
    # suite (case_upstream_suites), where pytest is not installed. The
    # shim covers exactly the three pytest features used below: fixture
    # (a no-op in both spellings — bare @fixture and @fixture(...);
    # the wrapper calls the function directly), mark.parametrize
    # (collection-time only; a no-op when called with explicit
    # arguments) and raises (pytest semantics: re.search on
    # str(exception) when match is given).
    import contextlib as _contextlib
    import re as _re

    class _Mark:
        @staticmethod
        def parametrize(*_args, **_kwargs):
            return lambda f: f

    class _PytestShim:
        mark = _Mark()

        @staticmethod
        def fixture(func=None, **_kwargs):
            return (lambda f: f) if func is None else func

        @staticmethod
        @_contextlib.contextmanager
        def raises(exc, match=None):
            try:
                yield
            except exc as e:
                if match is not None and not _re.search(match, str(e)):
                    raise AssertionError(
                        f"raised {e!r}, which does not match /{match}/"
                    ) from None
                return
            raise AssertionError(f"{exc.__name__} was not raised")

    pytest = _PytestShim()

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
    m = json.loads((dist / "L8" / "manifest.json").read_text(encoding="utf-8"))
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
    m = json.loads((pkg / "manifest.json").read_text(encoding="utf-8"))
    m["tokenizer"] = True
    (pkg / "manifest.json").write_text(json.dumps(m), encoding="utf-8")
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
    page = (tmp_path / "site" / "index.html").read_text(encoding="utf-8")
    assert f'href="{BASE}/v1/' in page and "github" not in page.lower()
    assert page.count("開發中") == len(TIERS) - 1 + len(GENERATIONS)
    assert [g["version"] for g in doc["generations"]] == [
        "v1.0", "v2.0", "v3.0", "v4.0", "v5.0", "v6.0", "HDL"]
    assert all(g["status"] == "開發中" and g["features"] for g in doc["generations"])
    assert {t["tier"]: t["source"] for t in doc["tiers"]}["L-Edge"] == "v1.0"
    assert json.loads((tmp_path / "site" / "packages.json").read_text(encoding="utf-8")) == doc


@pytest.mark.parametrize("url", [
    "https://github.com/x/y/releases", "https://objects.githubusercontent.com/a",
    "https://x.github.io/dl", "http://dl.example.com", "dl.example.com"])
def test_base_url_refusals(url):
    with pytest.raises(ValueError):
        check_base_url(url)


def test_cli_and_dry_run_upload(dist, tmp_path):
    out = tmp_path / "cf"
    # encoding="utf-8" (v5.240): with bare text=True the subprocess
    # pipes are decoded in the locale encoding (cp950 on zh-TW Windows),
    # which dies on the UTF-8 zh-TW the CLI prints and leaves stdout
    # None. Pin UTF-8 on both sides.
    r = subprocess.run([sys.executable, str(_REPO / "examples" / "publish_cloudflare.py"),
                        "--packages", str(dist), "--out", str(out), "--base-url", BASE,
                        "--version", "v1"], capture_output=True, text=True,
                       encoding="utf-8")
    assert r.returncode == 0, r.stderr
    zips = sorted(p.name for p in (out / "r2" / "v1").iterdir())
    assert len(zips) == 4 and all(z.endswith("-v1.zip") for z in zips)
    assert "L16 no package, listed as in development" in r.stdout
    env = dict(os.environ, DRY_RUN="1", CLOUDFLARE_ACCOUNT_ID="acct", R2_BUCKET="bkt",
               CLOUDFLARE_API_TOKEN="x")
    r = subprocess.run(["bash", str(_REPO / "scripts" / "cloudflare_publish.sh"), str(out)],
                       capture_output=True, text=True, encoding="utf-8", env=env)
    assert r.returncode == 0, r.stderr
    assert "s3://bkt/" in r.stdout and "acct.r2.cloudflarestorage.com" in r.stdout
    assert "pages deploy" in r.stdout and "github" not in r.stdout.lower()


def test_site_has_local_run_guide(dist, tmp_path):
    """v5.237: the page explains how to download and run each package
    (which file per GPU / Mac, checksum + unzip per OS, llama.cpp / Ollama /
    LM Studio), with each tier's own GGUF name and context."""
    e = bundle_package(dist / "L8", tmp_path / "r2", version="v1")
    assert e["chat"] is False and e["reasoning"] is None
    chat = dict(e, tier="L24", chat=True, reasoning="off")
    build_site([e, chat], tmp_path / "site", base_url=BASE, version="v1")
    page = (tmp_path / "site" / "index.html").read_text(encoding="utf-8")
    gguf = e["bundles"][0]["gguf"]
    assert "如何下載並在本機執行" in page
    for s in ("Get-FileHash", "sha256sum", "shasum -a 256", "LM Studio",
              "ollama create", "--cpu-moe", "127.0.0.1:8080"):
        assert s in page, s
    assert f"llama-completion -m {gguf} -c {e['ctx']} -ngl 99" in page  # base model
    assert f"llama-server -m {gguf} -c {e['ctx']} -ngl 99 --reasoning-budget 0" in page
    assert page.count("開發中") == len(TIERS) - 2 + len(GENERATIONS)
    assert "github" not in page.lower()


def test_site_has_measured_speed_and_mtp_tip(dist, tmp_path):
    """v5.238: measured llama.cpp speeds per tier, the MTP flags for the
    26B tiers (current llama.cpp spelling) and the -t tip."""
    from helioslm_v5.src.export.cloudflare import SPEED
    e = bundle_package(dist / "L8", tmp_path / "r2", version="v1")
    build_site([e], tmp_path / "site", base_url=BASE, version="v1")
    page = (tmp_path / "site" / "index.html").read_text(encoding="utf-8")
    assert "速度參考" in page and "--spec-type draft-mtp --spec-draft-n-max 3" in page
    assert "--draft-max" not in page and "-t &lt;" in page
    assert "CUDA_Host buffer" in page  # v5.243: free system RAM note
    assert all(str(row[3]) in page for row in SPEED)
    assert all(row[4] is None or row[4] > row[3] for row in SPEED)  # MTP only where it helps


def test_site_offers_26b_on_8gb(dist, tmp_path):
    """v5.261: with an L24 chat bundle, the L8 row also links the 26B file
    and the guide gives the --n-cpu-moe command for 8 GB cards."""
    from helioslm_v5.src.export.cloudflare import OFFLOAD_8GB
    e = bundle_package(dist / "L8", tmp_path / "r2", version="v1")
    chat = dict(e, tier="L24", chat=True, reasoning="off")
    build_site([e, chat], tmp_path / "site", base_url=BASE, version="v1")
    page = (tmp_path / "site" / "index.html").read_text(encoding="utf-8")
    assert "26B 對話" in page and f"--n-cpu-moe {OFFLOAD_8GB[2]}" in page
    assert "8 GB 顯示卡也想用對話模型" in page
    build_site([e], tmp_path / "site2", base_url=BASE, version="v1")
    page = (tmp_path / "site2" / "index.html").read_text(encoding="utf-8")
    assert "26B 對話" not in page and "8 GB 顯示卡也想用對話模型" not in page
