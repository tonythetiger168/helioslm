"""Mac (Apple Silicon) downloads in MLX format (v5.242).

On an Apple M-series Mac the GGUF packages already run through llama.cpp
with Metal, but Apple's own MLX runtime (``mlx-lm``) is usually faster on
the same machine and is what LM Studio's MLX engine loads. This module
turns a quantized MLX model directory (``mlx_lm.convert -q --q-bits N``
output: ``config.json``, ``model*.safetensors``, tokenizer files and the
chat template) into one zip for the R2 download host, and writes the
"Mac" section of the download page.

Macs are picked by unified memory, not VRAM. macOS lets the GPU wire
about 2/3 of RAM on machines up to 36 GB and about 3/4 above, so:

  8 GB          E4B 4-bit   (about 5 GB of weights)
  16–24 GB      E4B 8-bit   (about 8.5 GB)
  32–48 GB      26B-A4B 4-bit  (about 15 GB)
  64 GB and up  26B-A4B 8-bit  (about 28 GB)

Intel Macs have no MLX; they take the L-Edge GGUF on the CPU.
"""
import hashlib
import html
import json
import re
import zipfile
from pathlib import Path


# (Mac unified memory, model key, bits): the row order of the page
MAC_ROWS = (("8 GB", "e4b", 4), ("16–24 GB", "e4b", 8),
            ("32–48 GB", "26b-a4b", 4), ("64 GB 以上", "26b-a4b", 8))

# files of an MLX model directory that go into the zip
_KEEP = ("*.safetensors", "*.json", "tokenizer*", "*.jinja", "*.model", "*.txt")


def _sha256(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for block in iter(lambda: f.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def _slug(name):
    return re.sub(r"[^a-z0-9.+]+", "-", name.lower()).strip("-")


def mac_ram(model, bits):
    """The page's Mac memory label for an MLX build of ``model`` (``e4b`` or
    ``26b-a4b``) at ``bits``; raises ``KeyError`` for a build the page has no
    row for."""
    for ram, m, b in MAC_ROWS:
        if m == model and b == bits:
            return ram
    raise KeyError(f"no Mac row for {model} {bits}-bit")


def _commands(dirname, chat):
    """mlx-lm commands for one build (run in the folder the zip unpacked
    into). A base model continues text, so it skips the chat template."""
    d = dirname
    if chat:
        return [f"mlx_lm.chat --model ./{d} --max-tokens 1024",
                f"mlx_lm.server --model ./{d} --port 8080"]
    return [f'mlx_lm.generate --model ./{d} --ignore-chat-template '
            f'--max-tokens 256 --prompt "台灣的夜市文化"']


def readme(name, model, bits, chat, base):
    dirname = f"{_slug(name)}-mlx-{bits}bit"
    kind = ("對話模型 (chat)" if chat else
            "基底模型（只做過續訓，會接著寫，不適合一問一答） (base model)")
    cmds = "\n".join(_commands(dirname, chat))
    return f"""# {name} MLX {bits}-bit（Mac Apple Silicon）

- 模型 / Model: {model}, {kind}
- 基於 / Based on: {base}
- 適用 / For: Apple M 系列 Mac，{mac_ram(model, bits)} 統一記憶體

## 執行 / Run

```
python3 -m venv ~/mlx && source ~/mlx/bin/activate
pip install -U mlx-lm
{cmds}
```

`mlx_lm.server` 提供 OpenAI 相容 API：`http://127.0.0.1:8080/v1`。
LM Studio：把 `{dirname}` 資料夾放進 `~/.lmstudio/models/helioslm/`，選 MLX 引擎載入。
"""


def bundle_mlx(model_dir, out_dir, *, version, name, model, bits, chat=False,
               reasoning=None, ctx=32768, base="", status="preview"):
    """Zip the MLX model directory ``model_dir`` into ``out_dir``.

    The zip holds one folder ``<name>-mlx-<bits>bit/`` with the weights,
    config, tokenizer and chat template, plus a README with the mlx-lm and
    LM Studio steps (stored, not deflated: safetensors do not compress).
    Returns the ``packages.json`` entry of the build.
    """
    src = Path(model_dir)
    if not (src / "config.json").exists():
        raise FileNotFoundError(f"{src} has no config.json (not an MLX model directory)")
    cfg = json.loads((src / "config.json").read_text())
    q = cfg.get("quantization") or cfg.get("quantization_config") or {}
    if q.get("bits") not in (None, bits):
        raise ValueError(f"{src} is {q.get('bits')}-bit, not {bits}-bit")
    files = sorted({p for pat in _KEEP for p in src.glob(pat) if p.is_file()})
    if not any(p.suffix == ".safetensors" for p in files):
        raise FileNotFoundError(f"{src} has no .safetensors weights")
    dirname = f"{_slug(name)}-mlx-{bits}bit"
    zname = f"{dirname}-{version}.zip"
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    zpath = out / zname
    with zipfile.ZipFile(zpath, "w", compression=zipfile.ZIP_STORED) as z:
        for p in files:
            z.write(p, f"{dirname}/{p.name}")
        z.writestr(f"{dirname}/README.md", readme(name, model, bits, chat, base))
    weights = sum(p.stat().st_size for p in files if p.suffix == ".safetensors")
    return {"mac_ram": mac_ram(model, bits), "model": model, "bits": bits,
            "format": "mlx", "dir": dirname, "zip": zname, "key": f"{version}/{zname}",
            "bytes": zpath.stat().st_size, "sha256": _sha256(zpath),
            "weights_bytes": weights, "ctx": ctx, "chat": bool(chat),
            "reasoning": reasoning, "status": status}


def _size(n):
    return f"{n / 2**30:.2f} GB" if n >= 2**30 else f"{n / 2**20:.1f} MB"


_MODEL_LABEL = {"e4b": "E4B", "26b-a4b": "26B-A4B"}


def mac_section(entries, base):
    """HTML section "Mac（Apple Silicon）" of the download page for the MLX
    ``entries`` (:func:`bundle_mlx` results), in :data:`MAC_ROWS` order;
    links are ``base/<key>``. Empty string when there are none."""
    if not entries:
        return ""
    e = html.escape
    order = {(m, b): i for i, (_, m, b) in enumerate(MAC_ROWS)}
    entries = sorted(entries, key=lambda x: order.get((x["model"], x["bits"]), 99))
    rows = []
    for x in entries:
        kind = "對話模型" if x["chat"] else "基底模型（文字續寫）"
        tag = " <span class=tag>preview</span>" if x.get("status") == "preview" else ""
        cmd = "<br>".join(f"<code>{e(c)}</code>" for c in _commands(x["dir"], x["chat"]))
        rows.append(
            f"<tr><td><b>{e(x['mac_ram'])}</b></td>"
            f"<td>{e(_MODEL_LABEL.get(x['model'], x['model']))} MLX {x['bits']}-bit"
            f"<br><span class=dim>{e(kind)}</span></td>"
            f'<td><a href="{e(base + "/" + x["key"])}">{e(x["zip"])}</a>{tag}'
            f" · {_size(x['bytes'])}<br><code>{x['sha256'][:16]}…</code></td>"
            f"<td>{cmd}</td></tr>")
    first = entries[0]
    return f"""<h2>Mac（Apple Silicon）· MLX</h2>
<p>Apple M 系列 Mac（M1 起）建議用這裡的 MLX 版：Apple 自家的 MLX 執行環境，
在 Mac 上通常比 GGUF 快。上面的 GGUF 套件在 Mac 上也能用（llama.cpp、Ollama、LM Studio 都內建 Metal）。
依「關於這台 Mac」顯示的記憶體選：</p>
<div class=wrap><table>
<tr><th>Mac 記憶體</th><th>模型</th><th>下載 · SHA-256</th><th>執行指令（在解壓後的資料夾內）</th></tr>
{chr(10).join(rows)}
</table></div>
<p><b>安裝 mlx-lm</b>（終端機，需要 Python 3.10 以上；macOS 內建的 python3 太舊時先
<code>brew install python</code>）：</p>
<pre><code>python3 -m venv ~/mlx &amp;&amp; source ~/mlx/bin/activate
pip install -U mlx-lm
curl -L -O &lt;下載連結&gt;
shasum -a 256 {e(first['zip'])}
unzip {e(first['zip'])}</code></pre>
<ul>
<li><code>mlx_lm.chat</code> 在終端機對話；<code>mlx_lm.server</code> 啟動 OpenAI 相容 API
<code>http://127.0.0.1:8080/v1</code>。</li>
<li><b>LM Studio</b>：把解壓出來的資料夾放進 <code>~/.lmstudio/models/helioslm/</code>，
在 LM Studio 選它（MLX 引擎）載入。</li>
<li>記憶體剛好卡在邊界時，先關掉其他大型程式；E4B 4-bit 在 8 GB Mac 上只適合短對話。</li>
<li><b>Intel Mac</b> 沒有 MLX：用上表的 L-Edge GGUF，llama.cpp 只用 CPU。</li>
</ul>
"""


_GUIDE_HEAD = "<h2>如何下載並在本機執行"


def add_mac_to_site(site_dir, entries, base):
    """Add the Mac MLX section to an already built site (``index.html`` +
    ``packages.json`` in ``site_dir``), so a Mac release does not rebuild or
    re-upload the GGUF bundles. The section goes in front of the run guide
    (or before ``</main>``); an earlier Mac section is replaced. Returns the
    updated ``packages.json`` dict."""
    site = Path(site_dir)
    base = base.rstrip("/")
    page = (site / "index.html").read_text()
    start = page.find("<h2>Mac（Apple Silicon）")
    if start >= 0:
        end = page.find("<h2>", start + 4)
        if end < 0:
            end = page.find("</main>", start)
        page = page[:start] + page[end:]
    sec = mac_section(list(entries), base)
    at = page.find(_GUIDE_HEAD)
    if at < 0:
        at = page.find("</main>")
    if at < 0:
        raise ValueError(f"{site / 'index.html'}: no run guide or </main> to insert at")
    page = page[:at] + sec + page[at:]
    (site / "index.html").write_text(page)
    doc = json.loads((site / "packages.json").read_text())
    doc["mac"] = [dict(m, url=f"{base}/{m['key']}") for m in entries]
    (site / "packages.json").write_text(json.dumps(doc, indent=2, ensure_ascii=False) + "\n")
    return doc


def mlx_config_fixes(config):
    """Keys to add to a Gemma 4 ``config.json`` so mlx-lm builds the right
    attention shapes. Checkpoints saved by newer transformers describe the
    full-attention layers in ``text_config.per_layer_config`` (e.g. 26B-A4B:
    ``{"05": {"head_dim": 512, "num_key_value_heads": 2}, ...}``), while
    mlx-lm 0.32 reads ``global_head_dim`` / ``num_global_key_value_heads``
    and otherwise expects 8 KV heads there (shape error on ``k_proj``).
    Returns ``{}`` when nothing is missing."""
    tc = config.get("text_config", config)
    per = tc.get("per_layer_config") or {}
    types = tc.get("layer_types") or []
    full = [v for k, v in per.items()
            if k.isdigit() and int(k) < len(types) and types[int(k)] == "full_attention"]
    fixes = {}
    for src, dst in (("head_dim", "global_head_dim"),
                     ("num_key_value_heads", "num_global_key_value_heads")):
        vals = {v[src] for v in full if src in v}
        if len(vals) > 1:
            raise ValueError(f"full-attention layers disagree on {src}: {sorted(vals)}")
        if vals and dst not in tc:
            fixes[dst] = vals.pop()
    return fixes


def fix_mlx_config(hf_dir):
    """Apply :func:`mlx_config_fixes` to ``hf_dir/config.json`` in place
    (into ``text_config`` when there is one); returns the keys added."""
    p = Path(hf_dir) / "config.json"
    cfg = json.loads(p.read_text())
    fixes = mlx_config_fixes(cfg)
    if fixes:
        cfg.get("text_config", cfg).update(fixes)
        p.write_text(json.dumps(cfg, indent=2) + "\n")
    return fixes
