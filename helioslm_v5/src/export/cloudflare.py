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

from .mlx import mac_section
from .package import QUANT_LABEL, TIERS, _slug, modelfile

# memory ranges of the README Product roadmap table (TIERS keeps the floor)
_MEMORY = {"L-Edge": "CPU, ≤ 8 GB RAM", "L8": "8 GB VRAM", "L16": "12–16 GB VRAM",
           "L24": "24–32 GB VRAM 或 32 GB 統一記憶體",
           "L48+": "48–96 GB VRAM 或 64–96 GB 統一記憶體", "L128": "128 GB 統一記憶體",
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
    GGUF, in the manifest's order (the first is the tier's default). A
    speculative-decoding draft model (v5.209) goes into every zip of the
    tier, and a chat model's Modelfile keeps the GGUF's chat template.
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
            if m.get("draft"):
                z.write(pkg / m["draft"]["file"], m["draft"]["file"])
            if m.get("tokenizer"):
                z.writestr("Modelfile", modelfile(f["file"], tier, m.get("chat", False)))
            if readme:
                z.writestr("README.md", readme)
            z.writestr("manifest.json", json.dumps(m, indent=2) + "\n")
        bundles.append({"zip": zname, "key": f"{version}/{zname}", "quant": label,
                        "gguf": f["file"], "gguf_sha256": f["sha256"],
                        "bytes": zpath.stat().st_size, "sha256": _sha256(zpath),
                        "fits": f["fits"]})
    return {"tier": tier.name, "memory_gb": tier.memory_gb, "hardware": tier.hardware,
            "ctx": m["ctx"], "arch": m["arch"], "status": status,
            "chat": bool(m.get("chat")), "reasoning": m.get("reasoning"),
            "bundles": bundles}


def _size(n):
    return f"{n / 2**30:.2f} GB" if n >= 2**30 else f"{n / 2**20:.1f} MB"


# which tier a Mac takes, by unified memory (the GPU may use about 2/3-3/4)
_MAC_TIERS = (("8 GB", "L-Edge"), ("16 GB", "L8"), ("24 GB", "L16"),
              ("32–48 GB", "L24"), ("64–96 GB", "L48+"), ("128 GB 以上", "L128"))


# measured 2026-11-04 (v5.238): llama.cpp build 12098, llama-server -ngl 99
# -c 8192 -fa on, one request, 512 new tokens, median of 8 chat prompts;
# (tier, file, GPU, decode tok/s, with MTP or None, VRAM)
SPEED = (("L8", "E4B Q4_K", "RTX 5090", 241, None, "3.7 GB"),
         ("L16", "E4B Q8_0", "RTX 5090", 196, None, "5.6 GB"),
         ("L24", "26B-A4B Q4_K", "RTX 5090", 254, 349, "17.2 GB"),
         ("L24", "26B-A4B Q4_K", "L40S 48 GB", 161, None, "17.1 GB"),
         ("L48+", "26B-A4B Q8_0", "L40S 48 GB", 121, None, "26.7 GB"),
         ("L48+", "26B-A4B Q8_0", "A40 48 GB", 85, 119, "26.5 GB"))
# Gemma 4 26B-A4B assistant (MTP) head used for the measurement above
MTP_HEAD_26B = "AtomicChat/gemma-4-26B-A4B-it-assistant-GGUF"


def _speed_html():
    e = html.escape
    rows = "\n".join(
        f"<tr><td><b>{e(t)}</b></td><td>{e(f)}</td><td>{e(g)}</td><td>{d}</td>"
        f"<td>{m if m else '—'}</td><td>{e(v)}</td></tr>"
        for t, f, g, d, m, v in SPEED)
    return f"""<h3>6. 速度參考（實測）</h3>
<p>llama.cpp、單一請求、產生 512 個 token 的中位數（2026-11-04 實測）。
單位 token/秒，越高越快；首字延遲約 0.04 秒（E4B）到 0.2 秒（26B）。
其他顯示卡沒有實測，請自行比較。</p>
<div class=wrap><table>
<tr><th>Tier</th><th>檔案</th><th>顯示卡</th><th>速度</th><th>開 MTP</th><th>VRAM</th></tr>
{rows}
</table></div>
<ul>
<li><b>MTP 加速（只建議 26B，L24 起）</b>：另外下載 Gemma 4 26B-A4B 的 assistant（MTP）GGUF
（例如 Hugging Face 上的 <code>{e(MTP_HEAD_26B)}</code>，Q8_0），在 <code>llama-server</code> 指令加上
<code>-md &lt;assistant 檔&gt;.gguf -ngld 99 --spec-type draft-mtp --spec-draft-n-max 3</code>，
實測快 37–40%，多用約 0.5 GB VRAM。E4B 層級（L-Edge、L8、L16）開 MTP 沒有幫助，L8 反而變慢。</li>
<li><b>CPU 執行緒</b>：加 <code>-t &lt;實體核心數&gt;</code>。在核心很多的機器上用預設值可能反而大幅變慢。</li>
</ul>
"""


def _cmd(t, b):
    """The llama.cpp command a tier's README gives, for the guide."""
    ngl = " -ngl 99" if t["tier"] != "L-Edge" else ""
    if not t.get("chat"):
        return (f'llama-completion -m {b["gguf"]} -c {t["ctx"]}{ngl} -n 256 '
                f'-p "台灣的夜市文化"')
    rb = " --reasoning-budget 0" if t.get("reasoning") == "off" else ""
    return f'llama-server -m {b["gguf"]} -c {t["ctx"]}{ngl}{rb} --port 8080'


def run_guide(tiers):
    """HTML section on downloading and running a package locally (v5.237;
    speed table and MTP tip v5.238): which file per GPU or Mac, download + checksum + unzip on Windows /
    Linux / macOS, and the llama.cpp, Ollama and LM Studio steps. Commands
    use each tier's own GGUF name, context and model kind (a base model
    continues text; a chat model answers)."""
    e = html.escape
    rows, first = [], None
    for t in tiers:
        if not t.get("bundles"):
            continue
        b = t["bundles"][0]
        first = first or (t, b)
        kind = "對話模型" if t.get("chat") else "基底模型（文字續寫）"
        name = b["gguf"].rsplit("-", 1)[0]
        rows.append(f"<tr><td><b>{e(t['tier'])}</b></td><td>{e(kind)}</td>"
                    f"<td><code>{e(b['gguf'])}</code></td>"
                    f"<td><code>{e(_cmd(t, b))}</code><br>"
                    f"<code>ollama create {e(name)} -f Modelfile</code></td></tr>")
    if not rows:
        return ""
    chat = next((t for t in tiers if t.get("bundles") and t.get("chat")), None)
    base = next((t for t in tiers if t.get("bundles") and not t.get("chat")), None)
    t0, b0 = first
    zipname = e(b0["zip"])
    mac = "、".join(f"{m} → {k}" for m, k in _MAC_TIERS)
    kinds = ""
    if chat and base:
        kinds = (f"<p><b>模型差別</b>：{e(base['tier'])} 等標「基底模型」的層級只做過續訓、"
                 "沒有對話微調：給它一段開頭，它會接著寫，不適合一問一答。"
                 f"想要對話請用標「對話模型」的層級（{e(chat['tier'])} 起）。"
                 "12–16 GB 顯示卡也能跑 L24，加上 <code>--cpu-moe</code>，"
                 "專家權重放在系統記憶體（建議 32 GB RAM 以上，速度較慢）。</p>")
    return f"""<h2>如何下載並在本機執行 · Run it on your computer</h2>
<p>三個步驟：選檔、下載解壓、用 llama.cpp、Ollama 或 LM Studio 執行。
模型完全在你的電腦上跑，不需要網路，對話內容不會傳出去。</p>
<h3>1. 選哪個檔</h3>
<ul>
<li><b>NVIDIA / AMD 顯示卡</b>：看顯示卡記憶體（VRAM），對照上表的 Memory 欄。
Windows：工作管理員 → 效能 → GPU →「專用 GPU 記憶體」；Linux：<code>nvidia-smi</code>。</li>
<li><b>Mac（Apple M 系列）</b>：看「關於這台 Mac」的記憶體：{e(mac)}。</li>
<li><b>Intel Mac、沒有獨立顯示卡的電腦</b>：L-Edge，只用 CPU（至少 8 GB RAM）。</li>
</ul>
{kinds}
<div class=wrap><table>
<tr><th>Tier</th><th>類型</th><th>zip 內的模型檔</th><th>執行指令（在解壓後的資料夾內）</th></tr>
{chr(10).join(rows)}
</table></div>
<h3>2. 下載、驗證、解壓</h3>
<p>在上表按右鍵複製下載連結。硬碟要留約兩倍檔案大小的空間，解壓後可刪掉 zip。
算出的 SHA-256 要和 <a href="packages.json">packages.json</a> 的 <code>sha256</code> 相同。</p>
<p><b>Windows</b>（PowerShell）：</p>
<pre><code>curl.exe -L -O &lt;下載連結&gt;
Get-FileHash .\\{zipname} -Algorithm SHA256
tar -xf .\\{zipname}</code></pre>
<p><b>Linux</b>：</p>
<pre><code>curl -L -O &lt;下載連結&gt;
sha256sum {zipname}
unzip {zipname}</code></pre>
<p><b>macOS</b>：</p>
<pre><code>curl -L -O &lt;下載連結&gt;
shasum -a 256 {zipname}
unzip {zipname}</code></pre>
<h3>3. 安裝執行工具（三選一）</h3>
<ul>
<li><b>LM Studio</b>（有圖形介面，最簡單）：從 lmstudio.ai 下載安裝，用「匯入模型」選 zip 裡的
<code>.gguf</code>（或命令列 <code>lms import &lt;模型檔&gt;.gguf</code>），載入時把 GPU Offload 拉到最大。</li>
<li><b>Ollama</b>：從 ollama.com 下載安裝（Windows / macOS）；Linux：
<code>curl -fsSL https://ollama.com/install.sh | sh</code>。</li>
<li><b>llama.cpp</b>：Windows <code>winget install llama.cpp</code>；macOS / Linux
<code>brew install llama.cpp</code>；也可下載官方預編譯版（NVIDIA 選 CUDA 版，
AMD / Intel 顯示卡選 Vulkan 版，Mac 已內建 Metal）。</li>
</ul>
<h3>4. 執行</h3>
<p><b>llama.cpp</b>：在解壓後的資料夾執行上表的指令。<code>llama-server</code>
啟動後用瀏覽器開 <code>http://127.0.0.1:8080</code> 就能對話，
其他程式可用 OpenAI 相容 API <code>http://127.0.0.1:8080/v1</code>。
基底模型用 <code>llama-completion</code>，把 <code>-p</code> 換成你的開頭。</p>
<p><b>Ollama</b>：在解壓後的資料夾執行上表的 <code>ollama create</code>，再
<code>ollama run &lt;名稱&gt;</code>（名稱同 create 那一行）。</p>
<h3>5. 記憶體不夠時</h3>
<ul>
<li>把 <code>-c</code> 調小（例如 <code>-c 8192</code>），上下文越長越吃記憶體。</li>
<li>KV 快取量化：加 <code>-fa on -ctk q8_0 -ctv q8_0</code>（Ollama：啟動前設
<code>OLLAMA_FLASH_ATTENTION=1</code> 和 <code>OLLAMA_KV_CACHE_TYPE=q8_0</code>）。</li>
<li>26B-A4B（L24 起）是混合專家模型：加 <code>--cpu-moe</code> 把專家權重放到系統記憶體，
或用 <code>--n-cpu-moe N</code> 只放前 N 層；統一記憶體的電腦不需要。</li>
<li>每個 zip 裡的 README.md 有該層級的完整指令。</li>
</ul>
{_speed_html()}"""


def _page(name, version, tiers, base, generations, mac=()):
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
pre {{ overflow-x:auto; background:var(--line); border-radius:6px; padding:8px 10px; }}
h3 {{ font-size:16px; margin-top:20px; }}
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
{mac_section(list(mac), base)}{run_guide(tiers)}<h2>版本與功能 · Versions</h2>
<div class=wrap><table>
<tr><th>版本</th><th>模型</th><th>功能</th><th>狀態</th></tr>
{gens}
</table></div>
</main></body></html>
"""


def build_site(tier_entries, out_dir, *, base_url, version, name="HeliosLM",
               mac_entries=()):
    """Write ``index.html`` and ``packages.json`` into ``out_dir``.

    ``tier_entries`` are :func:`bundle_package` results; every local tier
    of :data:`TIERS` is listed, the ones without an entry as in development
(開發中), followed by the product generations and their features.
    Links are ``base_url/<key>`` (the R2 download domain). ``mac_entries``
    (:func:`.mlx.bundle_mlx` results, v5.242) add the Mac MLX section and a
    ``mac`` list to ``packages.json``. Returns the ``packages.json`` dict.
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
    mac = [dict(m, url=f"{base}/{m['key']}") for m in mac_entries]
    doc = {"name": name, "version": version, "base_url": base, "tiers": tiers,
           "generations": generations}
    if mac:
        doc["mac"] = mac
    (out / "packages.json").write_text(json.dumps(doc, indent=2, ensure_ascii=False) + "\n")
    (out / "index.html").write_text(_page(name, version, tiers, base, generations, mac))
    return doc
