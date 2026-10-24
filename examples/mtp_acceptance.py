"""mtp_acceptance.py — roadmap Q5: MTP speculative-decoding acceptance
and a greedy-equivalence gate.

For each model and prompt, greedy speculative decoding (MTPDecoder) is
run next to plain greedy ``model.generate``:

- GATE: the two token sequences must be identical (speculative decoding
  is lossless at temperature 0). A failing prompt fails the run.
- acceptance rate (accepted drafts / drafted), mean accepted drafts per
  round, new tokens per main-model verification round, and wall-clock
  tokens/s for both decoders (CPU, informational only).

Models:
- ``toy_v5.13``: the committed toy checkpoint (lite, MTP head trained by
  the aux loss) — the acceptance numbers;
- ``qwen_next_lite_untrained``: ``qwen_next_config()`` at random init —
  only the gate is meaningful (acceptance of an untrained head is noise);
  it exercises rollback with GDA recurrent state AND the Q1 conv state.

Usage:
  python examples/mtp_acceptance.py --output benchmarks/mtp_acceptance_2026-10-16.json
"""
import argparse
import hashlib
import json
import sys
import time
from pathlib import Path

import torch

_REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_REPO))

from evastar.configs.config_v5 import (  # noqa: E402
    EvastarV5Config, qwen_next_config)
from evastar.src.inference.mtp import MTPDecoder  # noqa: E402
from evastar.src.model_v5 import EvastarV5  # noqa: E402

PROMPTS = ["HeliosLM", "The model", "import torch", "def forward",
           "# v5.16", "Attention(", "KV cache", "speculative"]
CKPT = _REPO / "checkpoints" / "toy_v5.13.pt"


def encode(text, vocab):
    return torch.tensor([[b % vocab for b in text.encode()]], dtype=torch.long)


def load_models():
    out = []
    if CKPT.exists():
        ckpt = torch.load(CKPT, map_location="cpu", weights_only=False)
        cfg = EvastarV5Config(size="lite")
        m = EvastarV5(cfg)
        m.load_state_dict(ckpt["state_dict"])
        out.append(("toy_v5.13", m.eval(), cfg))
    torch.manual_seed(0)
    cfg = qwen_next_config()
    out.append(("qwen_next_lite_untrained", EvastarV5(cfg).eval(), cfg))
    return out


@torch.no_grad()
def measure(name, model, cfg, max_new):
    rows, fails = [], []
    for p in PROMPTS:
        ids = encode(p, cfg.vocab_size)
        t0 = time.perf_counter()
        plain = model.generate(ids, max_new_tokens=max_new, temperature=0)
        t_plain = time.perf_counter() - t0
        t0 = time.perf_counter()
        r = MTPDecoder(model, model.mtp_modules, cfg).generate(
            ids, max_new_tokens=max_new, temperature=0)
        t_spec = time.perf_counter() - t0
        same = torch.equal(plain[0], r.sequences[0][:plain.shape[1]]) \
            and r.sequences.shape[1] >= plain.shape[1]
        if not same:
            fails.append(p)
        new = plain.shape[1] - ids.shape[1]
        rows.append({"prompt": p, "greedy_equal": same,
                     "drafted": r.num_drafted, "accepted": r.num_accepted,
                     "rounds": r.num_rounds,
                     "tok_s_plain": round(new / t_plain, 1),
                     "tok_s_spec": round(new / t_spec, 1)})
    drafted = sum(r["drafted"] for r in rows)
    accepted = sum(r["accepted"] for r in rows)
    rounds = sum(r["rounds"] for r in rows)
    return {
        "model": name,
        "mtp_modules": len(model.mtp_modules),
        "gate_greedy_equal": not fails,
        "acceptance_rate": round(accepted / drafted, 4) if drafted else None,
        "accepted_per_round": round(accepted / rounds, 4) if rounds else None,
        "tokens_per_round": round(len(PROMPTS) * max_new / rounds, 4)
        if rounds else None,
        "per_prompt": rows,
    }, fails


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--max-new", type=int, default=32)
    ap.add_argument("--output")
    args = ap.parse_args()
    torch.set_num_threads(1)
    results, failed = [], {}
    for name, model, cfg in load_models():
        res, fails = measure(name, model, cfg, args.max_new)
        results.append(res)
        if fails:
            failed[name] = fails
        print(json.dumps({k: v for k, v in res.items() if k != "per_prompt"}))
    if failed:
        sys.exit(f"greedy-equivalence gate FAILED: {failed}")
    # wall-clock fields are telemetry: excluded from the hash
    canon = [{**r, "per_prompt": [{k: v for k, v in p.items()
                                   if not k.startswith("tok_s")}
                                  for p in r["per_prompt"]]} for r in results]
    out = {"experiment": "Q5 MTP acceptance + greedy-equivalence gate",
           "max_new": args.max_new, "prompts": PROMPTS,
           "results": results,
           "results_sha256": hashlib.sha256(
               json.dumps(canon, sort_keys=True).encode()).hexdigest()}
    if args.output:
        Path(args.output).write_text(json.dumps(out, indent=2) + "\n")
        print("wrote", args.output)


if __name__ == "__main__":
    main()
