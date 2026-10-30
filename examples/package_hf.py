"""package_hf.py — tier packages from a Hugging Face checkpoint (v5.196).

For route B (continued pretraining of an open base, e.g. the pilot B0
checkpoint): llama.cpp converts and quantizes, then each tier gets the
same layout as examples/package_tiers.py (GGUF, Modelfile, README,
manifest), ready for examples/publish_cloudflare.py.

  python examples/package_hf.py --hf runs/b0/final --llama-cpp ~/llama.cpp \\
      --out dist/ --tiers L8 L16 --name helioslm-tw-e2b-pilot \\
      --arch gemma4 --base google/gemma-4-E2B

--dry-run prints the llama.cpp commands only. --gguf q4_k=<file> ...
skips conversion and packages existing files.

--imatrix-data calib.txt (v5.208) runs llama-imatrix on the F16 file
first and quantizes the k-quants (Q4_K_M) with that importance matrix:
same file size, lower quantization error. --imatrix imatrix.dat reuses
one made earlier. examples/imatrix_calib.py builds the calibration text.

--draft L24=mtp-gemma-4-26B-A4B-it-Q8_0.gguf (v5.209) adds a drafter for
speculative decoding to that tier: --spec-type draft-mtp (default) for the
model's MTP head, draft-simple for a smaller same-vocabulary model such
as Gemma 4 E2B. examples/spec_bench.py measures the speed-up.
"""
import argparse
import shlex
import subprocess
import sys
from pathlib import Path

_REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_REPO))

from helioslm_v5.src.export.package import IMATRIX_QUANTS, TIERS, package_gguf  # noqa: E402
from helioslm_v5.src.export.speculative import SPEC_TYPES  # noqa: E402

# quant key -> llama-quantize type (f16 is convert_hf_to_gguf's output)
_QTYPE = {"q8_0": "Q8_0", "q4_k": "Q4_K_M"}


def convert_commands(hf, llama_cpp, work, quants, imatrix_data=None, imatrix=None,
                     chunks=None):
    """(commands, {quant: gguf path}) that make ``quants`` from ``hf``.

    ``imatrix_data`` (a calibration text file) adds a ``llama-imatrix``
    pass over the F16 file, written to ``work/imatrix.dat``; ``imatrix``
    reuses an existing matrix instead. Either way the quants in
    :data:`IMATRIX_QUANTS` are made with ``--imatrix``. ``chunks`` caps the
    calibration at that many context-sized chunks.
    """
    llama_cpp, work = Path(llama_cpp), Path(work)
    bin_ = llama_cpp / "build" / "bin"
    f16 = work / "model-F16.gguf"
    cmds = [[sys.executable, str(llama_cpp / "convert_hf_to_gguf.py"), str(hf),
             "--outtype", "f16", "--outfile", str(f16)]]
    paths = {"f16": f16}
    if imatrix_data and not imatrix and set(IMATRIX_QUANTS) & set(quants):
        imatrix = work / "imatrix.dat"
        cmd = [str(bin_ / "llama-imatrix"), "-m", str(f16), "-f", str(imatrix_data),
               "-o", str(imatrix), "-ngl", "99"]
        if chunks:
            cmd += ["--chunks", str(chunks)]
        cmds.append(cmd)
    for q in quants:
        if q in _QTYPE:
            paths[q] = work / f"model-{_QTYPE[q]}.gguf"
            im = ["--imatrix", str(imatrix)] if imatrix and q in IMATRIX_QUANTS else []
            cmds.append([str(bin_ / "llama-quantize"), *im,
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
    ap.add_argument("--chat", action="store_true",
                    help="instruction-tuned model: Modelfile keeps the GGUF chat template")
    ap.add_argument("--quants", nargs="+", metavar="TIER=Q[,Q]", default=[],
                    help="per-tier quant override, e.g. L16=q8_0 L24=f16")
    ap.add_argument("--imatrix-data", metavar="TXT",
                    help="calibration text: make an importance matrix for the k-quants")
    ap.add_argument("--imatrix", metavar="DAT", help="existing llama-imatrix output to reuse")
    ap.add_argument("--imatrix-chunks", type=int, default=200,
                    help="calibration chunks for llama-imatrix (default 200)")
    ap.add_argument("--draft", nargs="+", default=[], metavar="TIER=GGUF",
                    help="draft model for speculative decoding, e.g. L24=e2b-q4.gguf")
    ap.add_argument("--draft-base", help="draft model id for the manifest")
    ap.add_argument("--spec-type", default="draft-mtp", choices=list(SPEC_TYPES),
                    help="drafter kind (default draft-mtp)")
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args()
    out = Path(args.out)
    override = {k: v.split(",") for k, v in (o.split("=", 1) for o in args.quants)}
    drafts = dict(d.split("=", 1) for d in args.draft)
    if args.gguf:
        ggufs = dict(g.split("=", 1) for g in args.gguf)
    else:
        if not (args.hf and args.llama_cpp):
            ap.error("--hf and --llama-cpp are needed unless --gguf is given")
        quants = sorted({q for t in args.tiers for q in override.get(t, TIERS[t].quants)}
                        - {"f16"})
        cmds, ggufs = convert_commands(args.hf, args.llama_cpp, out / "_gguf", quants,
                                       args.imatrix_data, args.imatrix, args.imatrix_chunks)
        for c in cmds:
            print("+", shlex.join(c), flush=True)
            if not args.dry_run:
                (out / "_gguf").mkdir(parents=True, exist_ok=True)
                subprocess.run(c, check=True)
        if args.dry_run:
            return
    imatrix = args.imatrix or (out / "_gguf" / "imatrix.dat" if args.imatrix_data else None)
    if imatrix and not Path(imatrix).is_file():
        imatrix = None  # no k-quant needed one
    for tier in args.tiers:
        m = package_gguf(ggufs, tier, out / tier, name=args.name, arch=args.arch,
                         ctx=args.ctx, base=args.base, quants=override.get(tier),
                         chat=args.chat, imatrix=imatrix, draft=drafts.get(tier),
                         draft_base=args.draft_base, spec_type=args.spec_type)
        print(tier, [(f["file"], f["bytes"], f["fits"]) for f in m["files"]])


if __name__ == "__main__":
    main()
