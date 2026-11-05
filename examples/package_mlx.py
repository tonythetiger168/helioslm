"""package_mlx.py — Mac (Apple Silicon) MLX download zips (v5.245).

  pip install mlx-lm            # on Linux: pip install "mlx[cpu]" mlx-lm
  python examples/package_mlx.py --hf ckpt/e4b --model e4b --bits 4 8 --recipe 4=e4b-small \\
      --name helioslm-tw-e4b --base google/gemma-4-E4B \\
      --version mac-preview-1 --out cf/
  python examples/package_mlx.py --hf ckpt/g26 --model 26b-a4b --bits 4 8 \\
      --name helioslm-tw-26b-a4b --base google/gemma-4-26B-A4B-it --chat \\
      --reasoning off --version mac-preview-1 --out cf/ --site cf/site

Converts the Hugging Face checkpoint --hf with ``mlx_lm.convert -q
--q-bits N`` for every --bits (unless --mlx gives ready MLX directories),
zips each build (``mlx.bundle_mlx``) into <out>/r2/<version>/ and appends
the entries to <out>/mac.json. With --site, the Mac section is added to
that built site (``mlx.add_mac_to_site``) from every entry in mac.json.
Upload the zips to R2 like the GGUF ones.
"""
import argparse
import json
import subprocess
import sys
from pathlib import Path

_REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_REPO))

from helioslm_v5.src.export.cloudflare import check_base_url  # noqa: E402
from helioslm_v5.src.export.mlx import (  # noqa: E402
    RECIPES, add_mac_to_site, bundle_mlx, fix_mlx_config, mac_ram, quant_predicate)


def convert(hf, out, bits, group=64, recipe="plain"):
    """``mlx_lm.convert`` of ``hf`` to ``out`` at ``bits`` (``recipe``: see
    ``mlx.RECIPES``; non-plain recipes go through the Python API)."""
    if recipe != "plain":
        from mlx_lm.convert import convert as mlx_convert
        print("mlx_lm.convert", hf, out, bits, recipe, flush=True)
        mlx_convert(str(hf), str(out), quantize=True, q_bits=bits, q_group_size=group,
                    quant_predicate=quant_predicate(recipe), dtype="bfloat16")
        return out
    cmd = [sys.executable, "-m", "mlx_lm", "convert", "--hf-path", str(hf),
           "--mlx-path", str(out), "-q", "--q-bits", str(bits),
           "--q-group-size", str(group)]
    print(" ".join(cmd), flush=True)
    subprocess.run(cmd, check=True)
    return out


def main():
    ap = argparse.ArgumentParser()
    src = ap.add_mutually_exclusive_group(required=True)
    src.add_argument("--hf", help="Hugging Face checkpoint directory or repo id")
    src.add_argument("--mlx", nargs="+", metavar="BITS=DIR",
                     help="ready MLX directories, e.g. 4=mlx4 8=mlx8")
    ap.add_argument("--model", required=True, choices=["e4b", "26b-a4b"])
    ap.add_argument("--bits", nargs="+", type=int, default=[4, 8])
    ap.add_argument("--recipe", nargs="+", metavar="BITS=RECIPE", default=[],
                    help=f"quantization recipe per bits, e.g. 4=e4b-small; one of {RECIPES}")
    ap.add_argument("--name", required=True)
    ap.add_argument("--base", default="", help="upstream base model id, for the README")
    ap.add_argument("--chat", action="store_true", help="instruction-tuned (chat) model")
    ap.add_argument("--reasoning", default=None)
    ap.add_argument("--ctx", type=int, default=32768)
    ap.add_argument("--version", required=True)
    ap.add_argument("--status", choices=["preview", "release"], default="preview")
    ap.add_argument("--out", required=True)
    ap.add_argument("--site", help="built site directory to add the Mac section to")
    ap.add_argument("--base-url", default="https://dl.bitstarlabs.ai")
    args = ap.parse_args()
    base_url = check_base_url(args.base_url)
    out = Path(args.out)
    if args.mlx:
        dirs = {int(b): Path(d) for b, d in (x.split("=", 1) for x in args.mlx)}
    else:
        dirs = {}
        if Path(args.hf, "config.json").exists():
            fixes = fix_mlx_config(args.hf)
            if fixes:
                print("config.json: added for mlx-lm", fixes)
        for b in args.bits:
            mac_ram(args.model, b)  # fail before a long conversion
            recipes = dict(x.split("=", 1) for x in args.recipe)
            dirs[b] = convert(args.hf, out / "mlx" / f"{args.model}-{b}bit", b,
                              recipe=recipes.get(str(b), "plain"))
    index = out / "mac.json"
    entries = json.loads(index.read_text()) if index.exists() else []
    for b, d in sorted(dirs.items()):
        e = bundle_mlx(d, out / "r2" / args.version, version=args.version,
                       name=args.name, model=args.model, bits=b, chat=args.chat,
                       reasoning=args.reasoning, ctx=args.ctx, base=args.base,
                       status=args.status)
        entries = [x for x in entries if x["zip"] != e["zip"]] + [e]
        print(e["mac_ram"], e["zip"], e["bytes"], e["sha256"])
    index.write_text(json.dumps(entries, indent=2, ensure_ascii=False) + "\n")
    if args.site:
        add_mac_to_site(args.site, entries, base_url)
        print("site:", args.site)


if __name__ == "__main__":
    main()
