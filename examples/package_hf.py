"""package_hf.py — tier packages from a Hugging Face checkpoint (v5.195).

For route B (continued pretraining of an open base, e.g. the pilot B0
checkpoint): llama.cpp converts and quantizes, then each tier gets the
same layout as examples/package_tiers.py (GGUF, Modelfile, README,
manifest), ready for examples/publish_cloudflare.py.

  python examples/package_hf.py --hf runs/b0/final --llama-cpp ~/llama.cpp \\
      --out dist/ --tiers L8 L16 --name helioslm-tw-e2b-pilot \\
      --arch gemma4 --base google/gemma-4-E2B

--dry-run prints the llama.cpp commands only. --gguf q4_k=<file> ...
skips conversion and packages existing files.
"""
import argparse
import shlex
import subprocess
import sys
from pathlib import Path

_REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_REPO))

from helioslm_v5.src.export.package import TIERS, package_gguf  # noqa: E402

# quant key -> llama-quantize type (f16 is convert_hf_to_gguf's output)
_QTYPE = {"q8_0": "Q8_0", "q4_k": "Q4_K_M"}


def convert_commands(hf, llama_cpp, work, quants):
    """(commands, {quant: gguf path}) that make ``quants`` from ``hf``."""
    llama_cpp, work = Path(llama_cpp), Path(work)
    f16 = work / "model-F16.gguf"
    cmds = [[sys.executable, str(llama_cpp / "convert_hf_to_gguf.py"), str(hf),
             "--outtype", "f16", "--outfile", str(f16)]]
    paths = {"f16": f16}
    for q in quants:
        if q in _QTYPE:
            paths[q] = work / f"model-{_QTYPE[q]}.gguf"
            cmds.append([str(llama_cpp / "build" / "bin" / "llama-quantize"),
                         str(f16), str(paths[q]), _QTYPE[q]])
    return cmds, {q: paths[q] for q in ["f16", *quants] if q in paths}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--hf", help="Hugging Face checkpoint directory")
    ap.add_argument("--llama-cpp", help="llama.cpp checkout with build/bin/llama-quantize")
    ap.add_argument("--gguf", nargs="+", default=[], metavar="QUANT=FILE",
                    help="existing GGUFs instead of converting")
    ap.add_argument("--out", required=True)
    ap.add_argument("--tiers", nargs="+", choices=list(TIERS), default=["L8", "L16"])
    ap.add_argument("--name", required=True)
    ap.add_argument("--arch", required=True, help="llama.cpp architecture, e.g. gemma4")
    ap.add_argument("--base", help="base model id for the manifest")
    ap.add_argument("--ctx", type=int)
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args()
    out = Path(args.out)
    if args.gguf:
        ggufs = dict(g.split("=", 1) for g in args.gguf)
    else:
        if not (args.hf and args.llama_cpp):
            ap.error("--hf and --llama-cpp are needed unless --gguf is given")
        quants = sorted({q for t in args.tiers for q in TIERS[t].quants} - {"f16"})
        cmds, ggufs = convert_commands(args.hf, args.llama_cpp, out / "_gguf", quants)
        for c in cmds:
            print("+", shlex.join(c), flush=True)
            if not args.dry_run:
                (out / "_gguf").mkdir(parents=True, exist_ok=True)
                subprocess.run(c, check=True)
        if args.dry_run:
            return
    for tier in args.tiers:
        m = package_gguf(ggufs, tier, out / tier, name=args.name, arch=args.arch,
                         ctx=args.ctx, base=args.base)
        print(tier, [(f["file"], f["bytes"], f["fits"]) for f in m["files"]])


if __name__ == "__main__":
    main()
