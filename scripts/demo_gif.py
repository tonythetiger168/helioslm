"""Demo script for the HeliosLM README GIF.

Runs a tiny HeliosLM generation on CPU and streams the result to the
terminal with a typewriter effect, so it records well as a GIF.

Record with asciinema + agg:
    asciinema rec demo.cast --command "python demo_gif.py"
    agg demo.cast docs/demo.gif   # https://github.com/asciinema/agg

Or use a screen recorder (macOS: Cmd+Shift+5, Linux: peek/obs, Windows:
ShareX). Keep the clip under ~15 seconds, 800x450px or so, loop-friendly.
"""

import sys
import time

from helioslm_v5.configs.config_v5 import HeliosLMv5Config
from helioslm_v5.src.model_v5 import HeliosLMv5

PROMPT_IDS = [11, 7, 42, 19]   # untrained model: pseudo-vocab ids, demo only
MAX_NEW_TOKENS = 16


def type_out(text: str, delay: float = 0.015) -> None:
    """Print text with a typewriter effect (looks great in GIFs)."""
    for ch in text:
        sys.stdout.write(ch)
        sys.stdout.flush()
        time.sleep(delay)
    sys.stdout.write("\n")


def main() -> None:
    type_out("$ python demo_gif.py", delay=0.012)
    time.sleep(0.3)

    cfg = HeliosLMv5Config(size="lite")
    model = HeliosLMv5(cfg)
    model.eval()
    type_out(
        f"[load] HeliosLM {cfg.model_name} ({cfg.size}) ready — "
        f"{cfg.num_hidden_layers} layers, dim {cfg.hidden_size}, on CPU",
        delay=0.006,
    )
    time.sleep(0.25)
    type_out(f"[gen ] prompt ids: {PROMPT_IDS}  (untrained model, tokens are illustrative)",
             delay=0.006)
    time.sleep(0.25)

    # Single greedy generation pass, then stream the tokens for the recording.
    import torch
    t0 = time.time()
    with torch.no_grad():
        out = model.generate(torch.tensor([list(PROMPT_IDS)]), max_new_tokens=MAX_NEW_TOKENS,
                             temperature=0)
    dt = time.time() - t0
    new_tokens = out[0][len(PROMPT_IDS):]

    sys.stdout.write("[gen ] ")
    for tok in new_tokens:
        sys.stdout.write(f"{tok} ")
        sys.stdout.flush()
        time.sleep(0.09)
    sys.stdout.write("\n")

    type_out(f"[done] {len(new_tokens)} tokens in {dt:.1f}s on CPU — no GPU required",
             delay=0.006)
    type_out("$ █", delay=0.01)


if __name__ == "__main__":
    main()
