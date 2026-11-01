"""Probe backends, run the numeric oracle and time generation (#175).

    python examples/device_bench.py --device cpu --dtype float32
    python examples/device_bench.py --device cuda --dtype bfloat16 \
        --usd-per-hour 0.59 --out bench.json

Prints / writes one device-bench/1 record. A failed oracle exits 1 and
writes no timing. --strict refuses features that fall back on the device.
"""
import argparse
import json
import os
import subprocess
import sys

import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from helioslm_v5.eval import device_matrix as dm  # noqa: E402
from helioslm_v5.src.model_v5 import HeliosLMv5  # noqa: E402

DTYPES = {"float32": torch.float32, "bfloat16": torch.bfloat16,
          "float16": torch.float16}


def _commit():
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "--short", "HEAD"], text=True,
            cwd=os.path.dirname(os.path.abspath(__file__))).strip()
    except Exception:
        return "unknown"


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--device", default="cpu")
    ap.add_argument("--dtype", default="float32", choices=list(DTYPES))
    ap.add_argument("--batch", type=int, default=1)
    ap.add_argument("--prompt-len", type=int, default=32)
    ap.add_argument("--new-tokens", type=int, default=16)
    ap.add_argument("--repeats", type=int, default=3)
    ap.add_argument("--usd-per-hour", type=float, default=None,
                    help="hourly price of the machine; no price = no cost")
    ap.add_argument("--strict", action="store_true",
                    help="fail if a model feature falls back on this backend")
    ap.add_argument("--moe", action="store_true",
                    help="lite with MoE layers (float32 oracle only: bf16 "
                         "routing flips, see oracle_config)")
    ap.add_argument("--out", default=None)
    a = ap.parse_args(argv)

    backend = torch.device(a.device).type
    probe = dm.probe_backends().get(backend)
    if probe is None or not probe["available"]:
        print(json.dumps({"backend": backend, "probe": probe}, indent=2))
        return 2
    for feat in ("mla_attention", "kv_cache_decode"):
        dm.resolve(feat, backend, strict=a.strict)

    torch.manual_seed(0)
    model = HeliosLMv5(dm.oracle_config(dense=not a.moe)).eval()
    oracle = dm.forward_oracle(model, a.device, DTYPES[a.dtype])
    if not oracle["passed"]:
        print(json.dumps({"oracle": oracle}, indent=2))
        return 1
    run = model.to(device=a.device, dtype=DTYPES[a.dtype])
    timing = dm.measure_generation(run, a.device, a.batch, a.prompt_len,
                                   a.new_tokens, a.repeats)
    rec = dm.make_record(
        usd_per_hour=a.usd_per_hour, backend=backend,
        device_name=probe["device_name"], dtype=a.dtype, batch=a.batch,
        prompt_len=a.prompt_len, new_tokens=a.new_tokens,
        model="lite-moe" if a.moe else "lite-dense",
        torch_version=torch.__version__, commit=_commit(), oracle=oracle,
        timing=timing)
    text = json.dumps(rec, indent=2)
    print(text)
    if a.out:
        with open(a.out, "w") as f:
            f.write(text + "\n")
    return 0


if __name__ == "__main__":
    sys.exit(main())
