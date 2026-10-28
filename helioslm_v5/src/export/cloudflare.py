"""Cloudflare download bundles per GPU-memory tier (v5.188).

Turns the tier packages that :func:`package.package_tier` writes
(``dist/<tier>/``) into one zip per (tier, quantization) for Cloudflare R2,
and a static download page (Cloudflare Pages) whose links point at those
zips on the R2 download domain, never at GitHub:

  dist/L8/  ->  r2/<version>/helioslm-l8-Q4_K-<version>.zip
                r2/<version>/helioslm-l8-Q8_0-<version>.zip
                site/index.html, site/packages.json

Each zip is self-contained: the GGUF, an Ollama ``Modelfile`` pointing at
that GGUF (when the package has a tokenizer), the tier README and the
manifest. GGUF data does not compress, so entries are stored, not
deflated. Tiers without a package, and every product generation not yet
released, are listed on the page as 開發中 (in development).

The upload itself is ``scripts/cloudflare_publish.sh`` (R2 through its S3
API, which uploads multi-GB files in parts; the page by ``wrangler pages
deploy``, a direct upload with no Git integration).
"""
import hashlib
import html
import json
import zipfile
from pathlib import Path
from urllib.parse import urlparse

from .package import QUANT_LABEL, TIERS, _slug, modelfile

# memory ranges of the README Product roadmap table (TIERS keeps the floor)
_MEMORY = {"L-Edge": "CPU, ≤ 8 GB RAM", "L8": "8 GB VRAM", "L16": "12–16 GB VRAM",
           "L24": "24–32 GB VRAM", "L48+": "48–96 GB", "L128": "128 GB 統一記憶體",
           "L256": "256 GB 統一記憶體",
           "L512": "512 GB 統一記憶體"}

# source generation of each local tier (README Product roadmap table)
_SOURCE = {"L-Edge": "v1.0", "L8": "v2.0", "L16": "v2.0 / v3.0", "L24": "v3.0",
           "L48+": "v3.0 / v4.0", "L128": "v3.0 / v4.0",
           "L256": "v4.0 / v5.0",
           "L512": "v5.0"}

# product generations (README "Six proposed product generations") and the
# HDL track; status is "開發中" (in development) until a generation passes
# its pre-registered evaluation and its packages are published
GENERATIONS = (
    {"version": "v1.0", "name": "HeliosLM Lite", "params": "0.5B dense",
     "features": "基礎版：可重現的訓練與推論、CLI、測試；筆電 CPU 也能跑（L-Edge）"},
    {"version": "v2.0", "name": "HeliosLM Base", "params": "3B dense",
     "features": "資料管線、SFT / LoRA 微調、評測、SDK / API；8–16 GB 顯卡（L8 / L16）"},
    {"version": "v3.0", "name": "HeliosLM MoE", "params": "30B（每 token 約 3B）",
     "features": "MoE 專家路由、量化部署、HeliosLM Code beta；24–32 GB 顯卡（L24），"
                 "16 GB 加專家卸載"},
    {"version": "v4.0", "name": "HeliosLM Reasoner", "params": "100B（約 10B 啟用）",
     "features": "推理、程式代理、雲端 API、計量與安全閘門；48–96 GB 顯卡（L48+）、"
                 "128 GB 統一記憶體（L128：DGX Spark、Ryzen AI Max+ 395、Mac 含 Ultra）"},
    {"version": "v5.0", "name": "HeliosLM Pro", "params": "300B（約 20B 啟用）",
     "features": "多模態與長上下文、企業試點、分散式部署（雲端 C8）；"
                 "本地 Mac 256 / 512 GB 統一記憶體（L256 / L512，M4 / M5 系列含 Ultra）"},
    {"version": "v6.0", "name": "HeliosLM Frontier", "params": "1T（約 40B 啟用）",
     "features": "前沿研究候選，在既定評測上具競爭力（雲端 CX）"},
    {"version": "HDL", "name": "HeliosLM HDL", "params": "晶片設計專用版",
     "features": "Verilog / SystemVerilog / VHDL 產生、UVM 測試平台；評測（VerilogEval、"
                 "RTLLM、CVDP、UVM）與訓練資料工具已完成，模型尚未發布"},
)
STATUS = "開發中"

# hosts the download links must not point at
_GITHUB_HOSTS = ("github.com", "githubusercontent.com", "github.io")


def _sha256(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for block in iter(lambda: f.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def check_base_url(base_url):
    """``base_url`` without a trailing slash; raises ``ValueError`` unless it
    is an https URL off GitHub."""
    u = urlparse(base_url)
    host = (u.hostname or "").lower()
    if u.scheme != "https" or not host:
        raise ValueError(f"base_url must be an https URL, got {base_url!r}")
    if any(host == g or host.endswith("." + g) for g in _GITHUB_HOSTS):
        raise ValueError(f"download links must not point at GitHub ({host})")
    return base_url.rstrip("/")


def bundle_package(pkg_dir, out_dir, *, version, status="preview"):
    """Zip every GGUF of the tier package in ``pkg_dir`` into ``out_dir``.

    Returns the tier entry for ``packages.json``: the manifest fields plus
    one ``bundles`` item (zip name, R2 key, bytes, sha256, quant, fits) per
    GGUF, in the manifest's order (the first is the tier's default).
    """
    pkg = Path(pkg_dir)
    m = json.loads((pkg / "manifest.json").read_text())
    tier = TIERS[m["tier"]]
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    readme = (pkg / "README.md").read_text() if (pkg / "README.md").exists() else ""
    bundles = []
    for f in m["files"]:
        label = QUANT_LABEL[f["quant"]]
        zname = f"{_slug(m['name'])}-{_slug(tier.name)}-{label}-{version}.zip"
        zpath = out / zname
        with zipfile.ZipFile(zpath, "w", compression=zipfile.ZIP_STORED) as z:
            z.write(pkg / f["file"], f["file"])
            if m.get("tokenizer"):
                z.writestr("Modelfile", modelfile(f["file"], tier))
            if readme:
                z.writestr("README.md", readme)
            z.writestr("manifest.json", json.dumps(m, indent=2) + "\n")
        bundles.append({"zip": zname, "key": f"{version}/{zname}", "quant": label,
                        "gguf": f["file"], "gguf_sha256": f["sha256"],
                        "bytes": zpath.stat().st_size, "sha256": _sha256(zpath),
                        "fits": f["fits"]})
    return {"tier": tier.name, "memory_gb": tier.memory_gb, "hardware": tier.hardware,
            "ctx": m["ctx"], "arch": m["arch"], "status": status, "bundles": bundles}


def _size(n):
    return f"{n / 2**30:.2f} GB" if n >= 2**30 else f"{n / 2**20:.1f} MB"


def _page(name, version, tiers, base, generations):
    rows = []
    for t in tiers:
        mem = _MEMORY.get(t["tier"], f"{t['memory_gb']:g} GB VRAM")
        head = (f"<td><b>{html.escape(t['tier'])}</b></td><td>{mem}</td>"
                f"<td>{html.escape(t['hardware'])}</td>")
        if not t.get("bundles"):
            rows.append(f"<tr>{head}<td colspan=3><span class=dev>{STATUS}</span>"
                        f" <span class=dim>（來源 {_SOURCE.get(t['tier'], '')}）</span>"
                        "</td></tr>")
            continue
        links = "<br>".join(
            f'<a href="{html.escape(base + "/" + b["key"])}">{html.escape(b["quant"])}</a>'
            f' · {_size(b["bytes"])}{"" if b["fits"] else " (exceeds tier)"}'
            for b in t["bundles"])
        sums = "<br>".join(f"<code>{b['sha256'][:16]}…</code>" for b in t["bundles"])
        tag = " <span class=tag>preview</span>" if t["status"] == "preview" else ""
        rows.append(f"<tr>{head}<td>{links}{tag}</td><td>{t['ctx'] // 1024}K</td>"
                    f"<td>{sums}</td></tr>")
    gens = "\n".join(
        f"<tr><td><b>{html.escape(g['version'])}</b></td><td>{html.escape(g['name'])}"
        f"<br><span class=dim>{html.escape(g['params'])}</span></td>"
        f"<td>{html.escape(g['features'])}</td>"
        f"<td><span class=dev>{html.escape(g['status'])}</span></td></tr>"
        for g in generations)
    return f"""<!doctype html>
<html lang="zh-Hant"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>{html.escape(name)} downloads</title>
<style>
:root {{ --bg:#fff; --fg:#1b1b1f; --dim:#6b6b76; --line:#e3e3e8; --acc:#b4530a; }}
@media (prefers-color-scheme: dark) {{ :root {{ --bg:#141417; --fg:#ececf1;
  --dim:#9a9aa6; --line:#2c2c33; --acc:#f0a35a; }} }}
body {{ margin:0; background:var(--bg); color:var(--fg);
  font:15px/1.5 system-ui, sans-serif; }}
main {{ max-width:960px; margin:0 auto; padding:24px 16px; }}
table {{ width:100%; border-collapse:collapse; }}
th, td {{ text-align:left; padding:8px 6px; border-bottom:1px solid var(--line);
  vertical-align:top; }}
.wrap {{ overflow-x:auto; }} td:first-child {{ white-space:nowrap; }} a {{ color:var(--acc); }} .dim {{ color:var(--dim); }}
.dev {{ white-space:nowrap; font-size:13px; background:var(--line); border-radius:4px; padding:1px 6px; }}
h2 {{ font-size:18px; margin-top:32px; }}
.tag {{ font-size:12px; border:1px solid var(--acc); color:var(--acc);
  border-radius:4px; padding:0 4px; }}
code {{ font-size:12px; }}
</style></head><body><main>
<h1>{html.escape(name)} {html.escape(version)}</h1>
<p>依顯示卡記憶體選擇套件。每個 zip 含 GGUF、Ollama Modelfile、README 和 manifest。
Pick the package for your GPU memory; each zip holds the GGUF, an Ollama
Modelfile, a README and the manifest. Full SHA-256 sums are in
<a href="packages.json">packages.json</a>.</p>
<h2>依顯示卡記憶體下載 · Downloads by GPU memory</h2>
<div class=wrap><table>
<tr><th>Tier</th><th>Memory</th><th>Hardware</th><th>Download</th><th>Context</th><th>SHA-256</th></tr>
{chr(10).join(rows)}
</table></div>
<p class=dim>Run: <code>llama-cli -m &lt;file&gt;.gguf</code> or
<code>ollama create &lt;name&gt; -f Modelfile</code>.</p>
<h2>版本與功能 · Versions</h2>
<div class=wrap><table>
<tr><th>版本</th><th>模型</th><th>功能</th><th>狀態</th></tr>
{gens}
</table></div>
</main></body></html>
"""


def build_site(tier_entries, out_dir, *, base_url, version, name="HeliosLM"):
    """Write ``index.html`` and ``packages.json`` into ``out_dir``.

    ``tier_entries`` are :func:`bundle_package` results; every local tier
    of :data:`TIERS` is listed, the ones without an entry as in development
(開發中), followed by the product generations and their features.
    Links are ``base_url/<key>`` (the R2 download domain). Returns the
    ``packages.json`` dict.
    """
    base = check_base_url(base_url)
    by_tier = {e["tier"]: e for e in tier_entries}
    tiers = []
    for t in TIERS.values():
        e = by_tier.get(t.name) or {"tier": t.name, "memory_gb": t.memory_gb,
                                     "hardware": t.hardware, "ctx": t.ctx,
                                     "status": "in_development", "bundles": []}
        e = dict(e, source=_SOURCE[t.name],
                 bundles=[dict(b, url=f"{base}/{b['key']}") for b in e["bundles"]])
        tiers.append(e)
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    generations = [dict(g, status=STATUS) for g in GENERATIONS]
    doc = {"name": name, "version": version, "base_url": base, "tiers": tiers,
           "generations": generations}
    (out / "packages.json").write_text(json.dumps(doc, indent=2, ensure_ascii=False) + "\n")
    (out / "index.html").write_text(_page(name, version, tiers, base, generations))
    return doc
