"""package_tiers.py — local deployment packages from a pilot run (v5.148).

  python examples/package_tiers.py --run runs/dense --out dist/
  python examples/package_tiers.py --run runs/dense --out dist/ --tiers L-Edge L8

Writes dist/<tier>/ with the tier's GGUF files, an Ollama Modelfile, the
llama.cpp commands and manifest.json (see helioslm_v5/src/export/package.py).
The tokenizer is the run's data directory tokenizer.json unless --tokenizer
is given; --no-tokenizer writes id-only files.
"""
import argparse
import json
import sys
from pathlib import Path

import torch

_REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_REPO))

from helioslm_v5.src.export.package import TIERS, package_tier  # noqa: E402
from helioslm_v5.src.training.pretrain import HeliosLMv5, pilot_config  # noqa: E402


def load_run(run_dir):
    """(model, config, run) from a pilot_100m.py output directory."""
    run_dir = Path(run_dir)
    run = json.loads((run_dir / "run.json").read_text())
    state = torch.load(run_dir / "ckpt.pt", map_location="cpu", weights_only=False)["model"]
    cfg = pilot_config(run["arch"], vocab_size=state["embed_tokens.weight"].shape[0],
                       hidden_size=run["hidden"], num_layers=run["layers"],
                       intermediate_size=run["intermediate"], max_seq=run["seq"],
                       balance=run["balance"], moe_dispatch=run["moe_dispatch"],
                       kda_chunk=run["kda_chunk"], ple_dim=run.get("ple_dim", 0))
    model = HeliosLMv5(cfg)
    model.load_state_dict(state)
    return model.float().eval(), cfg, run


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", required=True, help="pilot_100m.py --out directory")
    ap.add_argument("--out", required=True)
    ap.add_argument("--tiers", nargs="+", choices=list(TIERS), default=list(TIERS))
    ap.add_argument("--name", default="helioslm")
    ap.add_argument("--tokenizer", help="tokenizer.json to embed")
    ap.add_argument("--no-tokenizer", action="store_true")
    ap.add_argument("--ctx", type=int, help="context length for every tier")
    args = ap.parse_args()
    model, cfg, run = load_run(args.run)
    tok = None
    if not args.no_tokenizer:
        tok = Path(args.tokenizer or Path(run["data"]) / "tokenizer.json")
        if not tok.exists():
            raise SystemExit(f"{tok} not found: pass --tokenizer or --no-tokenizer")
    for tier in args.tiers:
        m = package_tier(model, cfg, tier, Path(args.out) / tier, name=args.name,
                         tokenizer=tok, ctx=args.ctx)
        print(tier, m["arch"], [(f["file"], f["bytes"], f["fits"]) for f in m["files"]])


if __name__ == "__main__":
    main()
