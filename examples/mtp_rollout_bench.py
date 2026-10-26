"""mtp_rollout_bench.py — OS5: do MTP speculative rollouts speed up GRPO
sampling without changing what is sampled?

GRPO samples at temperature 1, top_p 1 (grpo.py ``_sample_response``).
For each model this runs the same rollout batch (``--groups`` prompts x
``--group-size`` samples, ``--max-new`` tokens) through plain
``model.generate`` and through ``use_mtp=True`` and reports:

- tokens/s for both, and the speed-up (wall clock; CPU numbers are
  telemetry, the GPU run is the one that counts);
- MTP acceptance rate and tokens per main-model round;
- GATE 1: greedy outputs identical (speculative decoding is lossless at
  temperature 0);
- GATE 2: the distribution of the first ``--tv-tokens`` sampled tokens at
  temperature 1 matches between the two samplers: total-variation
  distance between the empirical distributions below ``--tv-max``
  (sampling noise alone gives roughly sqrt(K / N)).

Models: the committed toy checkpoint when present (the acceptance number
that means something) and an untrained lite model (gates only).

Usage:
  python examples/mtp_rollout_bench.py --device cuda --output benchmarks/mtp_rollout_<date>.json
"""
import argparse
import collections
import json
import sys
import time
from pathlib import Path

import torch

_REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_REPO))

from helioslm_v5.configs.config_v5 import HeliosLMv5Config  # noqa: E402
from helioslm_v5.src.inference.mtp import MTPDecoder  # noqa: E402
from helioslm_v5.src.model_v5 import HeliosLMv5  # noqa: E402

PROMPTS = ["What is 12 + 7?", "def forward", "HeliosLM", "KV cache",
           "speculative", "import torch", "The model", "Attention("]
CKPT = _REPO / "checkpoints" / "toy_v5.13.pt"


def encode(text, vocab, device):
    return torch.tensor([[b % vocab for b in text.encode()]],
                        dtype=torch.long, device=device)


def load_models(device, tiny_vocab=None):
    out = []
    if CKPT.exists() and tiny_vocab is None:
        ckpt = torch.load(CKPT, map_location="cpu", weights_only=False)
        cfg = HeliosLMv5Config(size="lite")
        m = HeliosLMv5(cfg)
        m.load_state_dict(ckpt["state_dict"])
        out.append(("toy_v5.13", m.to(device).eval(), cfg))
    torch.manual_seed(0)
    kw = {"vocab_size": tiny_vocab} if tiny_vocab else {}
    cfg = HeliosLMv5Config(size="lite", **kw)
    out.append(("lite_untrained", HeliosLMv5(cfg).to(device).eval(), cfg))
    return out


def sync(device):
    if device.startswith("cuda"):
        torch.cuda.synchronize()


@torch.no_grad()
def rollouts(model, ids, n, max_new, use_mtp):
    seqs = []
    for _ in range(n):
        seqs.append(model.generate(ids, max_new_tokens=max_new,
                                   temperature=1.0, top_p=1.0,
                                   use_mtp=use_mtp)[0, ids.shape[1]:])
    return seqs


def tv_distance(a, b):
    ca, cb = collections.Counter(a), collections.Counter(b)
    keys = set(ca) | set(cb)
    return 0.5 * sum(abs(ca[k] / len(a) - cb[k] / len(b)) for k in keys)


@torch.no_grad()
def measure(name, model, cfg, args):
    dev = args.device
    res = {"model": name, "mtp_modules": len(model.mtp_modules)}
    # throughput on a GRPO-shaped batch
    for use_mtp in (False, True):
        torch.manual_seed(1)
        sync(dev)
        t0 = time.perf_counter()
        n_tok = 0
        for p in PROMPTS[:args.groups]:
            ids = encode(p, cfg.vocab_size, dev)
            n_tok += sum(len(s) for s in rollouts(model, ids, args.group_size,
                                                 args.max_new, use_mtp))
        sync(dev)
        res[f"tok_s_{'mtp' if use_mtp else 'plain'}"] = round(
            n_tok / (time.perf_counter() - t0), 1)
    res["speedup"] = round(res["tok_s_mtp"] / res["tok_s_plain"], 3)
    # acceptance at the GRPO sampling temperature
    drafted = accepted = rounds = new = 0
    for p in PROMPTS[:args.groups]:
        ids = encode(p, cfg.vocab_size, dev)
        r = MTPDecoder(model, model.mtp_modules, cfg).generate(
            ids, max_new_tokens=args.max_new, temperature=1.0, top_p=1.0)
        drafted += r.num_drafted
        accepted += r.num_accepted
        rounds += r.num_rounds
        new += r.sequences.shape[1] - ids.shape[1]
    res["acceptance_rate"] = round(accepted / drafted, 4) if drafted else None
    res["tokens_per_round"] = round(new / rounds, 3) if rounds else None
    # gate 1: greedy equality
    fails = []
    for p in PROMPTS:
        ids = encode(p, cfg.vocab_size, dev)
        a = model.generate(ids, max_new_tokens=args.max_new, temperature=0)
        b = model.generate(ids, max_new_tokens=args.max_new, temperature=0,
                           use_mtp=True)
        if not torch.equal(a[0], b[0][:a.shape[1]]):
            fails.append(p)
    res["gate_greedy_equal"] = not fails
    # gate 2: sampled distribution of the first tv_tokens tokens
    ids = encode(PROMPTS[0], cfg.vocab_size, dev)
    torch.manual_seed(2)
    a = [tuple(s[:args.tv_tokens].tolist())
         for s in rollouts(model, ids, args.tv_samples, args.tv_tokens, False)]
    torch.manual_seed(3)
    b = [tuple(s[:args.tv_tokens].tolist())
         for s in rollouts(model, ids, args.tv_samples, args.tv_tokens, True)]
    res["tv_distance"] = round(tv_distance(a, b), 4)
    res["tv_support"] = len(set(a) | set(b))
    res["gate_distribution"] = res["tv_distance"] <= args.tv_max
    return res


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--device", default="cpu")
    ap.add_argument("--groups", type=int, default=4)
    ap.add_argument("--group-size", type=int, default=8)
    ap.add_argument("--max-new", type=int, default=32)
    ap.add_argument("--tv-tokens", type=int, default=1)
    ap.add_argument("--tv-samples", type=int, default=2000)
    ap.add_argument("--tv-max", type=float, default=0.1)
    ap.add_argument("--tiny-vocab", type=int, default=None,
                    help="untrained model with this vocab only (quick check)")
    ap.add_argument("--output")
    args = ap.parse_args()
    if args.device == "cpu":
        torch.set_num_threads(1)
    results = [measure(n, m, c, args)
               for n, m, c in load_models(args.device, args.tiny_vocab)]
    for r in results:
        print(json.dumps(r))
    out = {"experiment": "OS5 MTP speculative GRPO rollouts",
           "device": args.device,
           "gpu": torch.cuda.get_device_name() if args.device.startswith("cuda") else None,
           "args": vars(args), "results": results}
    if args.output:
        Path(args.output).write_text(json.dumps(out, indent=2) + "\n")
        print("wrote", args.output)
    if not all(r["gate_greedy_equal"] and r["gate_distribution"] for r in results):
        sys.exit("OS5 gate FAILED")


if __name__ == "__main__":
    main()
