"""quant_bench.py — perplexity and KV-cache memory across GGUFs and KV types (v5.210).

Runs llama.cpp's llama-perplexity for every GGUF x KV-cache type on one
text file and reports the final perplexity and the KV cache size llama.cpp
allocates. Compares Q4_K_M with and without an importance matrix
(v5.208) and F16 / Q8_0 / Q4_0 KV caches at long context.

  python examples/quant_bench.py --bin ~/llama.cpp/build/bin \\
      --gguf q4=model-Q4_K_M.gguf q4_imatrix=model-Q4_K_M-imat.gguf \\
      --text eval.txt --ctx 8192 --kv f16 q8_0 q4_0 --out quant_bench.json
"""
import argparse
import json
import re
import subprocess
import sys
from pathlib import Path

_REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_REPO))

from helioslm_v5.src.export.package import KV_CACHE, kv_flags  # noqa: E402

_PPL = re.compile(r"Final estimate: PPL = ([0-9.]+) \+/- ([0-9.]+)")
_KV = re.compile(r"(?:KV self size|llama_kv_cache\w*: +size) += +([0-9.]+) MiB")


def parse(log):
    """(ppl, ppl_err, kv_mib) from a llama-perplexity log; None if absent."""
    m = _PPL.search(log)
    kv = [float(x) for x in _KV.findall(log)]
    return (float(m.group(1)) if m else None, float(m.group(2)) if m else None,
            round(sum(kv), 2) if kv else None)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--bin", required=True, help="llama.cpp build/bin")
    ap.add_argument("--gguf", nargs="+", required=True, metavar="LABEL=FILE")
    ap.add_argument("--text", required=True)
    ap.add_argument("--ctx", type=int, default=4096)
    ap.add_argument("--chunks", type=int, default=40)
    ap.add_argument("--kv", nargs="+", default=["f16"], choices=list(KV_CACHE))
    ap.add_argument("--ngl", type=int, default=99)
    ap.add_argument("--out", default="quant_bench.json")
    args = ap.parse_args()
    rows = []
    for label, path in (g.split("=", 1) for g in args.gguf):
        for kv in args.kv:
            cmd = [str(Path(args.bin) / "llama-perplexity"), "-m", path, "-f", args.text,
                   "-c", str(args.ctx), "--chunks", str(args.chunks), "-ngl", str(args.ngl),
                   *kv_flags(kv).split()]
            print("+", " ".join(cmd), flush=True)
            r = subprocess.run(cmd, capture_output=True, text=True)
            ppl, err, kv_mib = parse(r.stdout + r.stderr)
            rows.append({"model": label, "gguf": path, "kv": kv, "ctx": args.ctx,
                         "ppl": ppl, "ppl_err": err, "kv_mib": kv_mib,
                         "bytes": Path(path).stat().st_size, "exit": r.returncode})
            print(json.dumps(rows[-1]), flush=True)
    Path(args.out).write_text(json.dumps(rows, indent=2) + "\n")


if __name__ == "__main__":
    main()
