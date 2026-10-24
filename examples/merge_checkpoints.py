"""merge_checkpoints.py — LB7 checkpoint merging (v5.124).

  # average the last checkpoints of a WSD decay phase (model soup)
  python3 examples/merge_checkpoints.py --method linear \\
      --models runs/a/ckpt_900.pt runs/a/ckpt_1000.pt --out runs/soup/ckpt.pt
  # fold domain / RL specialists back into one model
  python3 examples/merge_checkpoints.py --method ties --base runs/sft/ckpt.pt \\
      --models runs/rl_math/ckpt.pt runs/rl_code/ckpt.pt --density 0.2 \\
      --out runs/merged/ckpt.pt

Inputs are pretrain() / pilot_100m.py checkpoints (``{"model": ...}``) or
bare state dicts. The output is ``{"model": merged, "merge": recipe}``,
which ``pretrain(init_from=...)`` and the eval scripts load. A run.json
next to the first model is copied beside the output, so tools that read
the architecture from it keep working.
"""
import argparse
import json
import shutil
import sys
from pathlib import Path

import torch

_REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_REPO))

from evastar.src.training.merge import (  # noqa: E402
    METHODS, load_state, merge_states)


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--models", nargs="+", required=True)
    ap.add_argument("--weights", nargs="*", type=float)
    ap.add_argument("--method", choices=METHODS, default="linear")
    ap.add_argument("--base")
    ap.add_argument("--scale", type=float, default=1.0)
    ap.add_argument("--density", type=float, default=0.2)
    ap.add_argument("--out", required=True)
    a = ap.parse_args(argv)
    merged = merge_states([load_state(p) for p in a.models], a.method,
                          a.weights, load_state(a.base) if a.base else None,
                          a.scale, a.density)
    recipe = {"method": a.method, "models": a.models, "weights": a.weights,
              "base": a.base, "scale": a.scale, "density": a.density}
    out = Path(a.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    torch.save({"model": merged, "merge": recipe}, out)
    run = Path(a.models[0]).parent / "run.json"
    if run.exists() and run.resolve() != (out.parent / "run.json").resolve():
        shutil.copy(run, out.parent / "run.json")
    print(json.dumps(recipe))
    return recipe


if __name__ == "__main__":
    main()
