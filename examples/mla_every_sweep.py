"""mla_every_sweep.py — B1-5: sweep ``full_attention_every`` over {2, 4, 8}
with a HarnessEvolver-style bitwise gate. A bitwise-reproducible
architecture-density A/B at toy scale.

WHAT THIS MEASURES
------------------
The hybrid interleave rule (v5.5) makes layer i an MLA (full attention,
KV cache grows with sequence) iff ``i == 0 or i % every == every - 1``,
and a GatedDeltaAttention (linear attention, FIXED recurrent state)
otherwise. ``every`` is therefore the MLA-density knob: at 8 layers,
every=2 puts MLA at {0,1,3,5,7} (5/8), every=4 at {0,3,7} (3/8),
every=8 at {0,7} (2/8). A dense-MLA reference arm (hybrid disabled)
closes the quality scale.

Per arm the script reports three quantities:
  1. STRUCTURE (exact): layer map, parameter count, analytic cache bytes
     per token at a fixed sequence length (reusing bench_cpu's
     structural counters — no estimates).
  2. BITWISE GATE (HarnessEvolver discipline): the same config + seed
     built and greedy-decoded TWICE must produce token-identical output;
     an arm that fails is excluded from the recommendation. This is the
     temp-0 invariance oracle of harness_evolver.py applied to an
     architecture knob instead of a serving knob.
  3. QUALITY (trained): short char-level training on the repo corpus
     (same data pipeline as train_toy_checkpoint.py, identical seed and
     data order across arms), reporting final train loss and held-out
     val loss. The auto-search picks the min-val-loss arm among
     gate-passing arms; the cache-byte tradeoff is reported alongside
     so the quality-vs-memory frontier is the reader's call.

DELIBERATE SCOPE LIMITS (recorded, not hidden)
----------------------------------------------
- Lite WIDTH (hidden 256) at configurable DEPTH (default 8 layers, vs
  the stock lite's 2 — at 2 layers the interleave rule is degenerate:
  every=2 makes both layers MLA). Findings are toy-scale architecture
  signals, not production guidance.
- MTP modules are built (same param count across arms) but excluded
  from the loss and unused in decoding, isolating the attention
  interleave as the only moving part.
- Training determinism holds for a fixed machine + torch build +
  thread count (same convention as hca_ab_passkey.py); the bitwise
  gate verifies it in-process for decoding.

BITWISE DISCIPLINE
------------------
All randomness flows through torch.manual_seed / torch.Generator with
fixed seeds; ``results_sha256`` covers the canonical JSON of the
results table ONLY (wall-clock telemetry is excluded, same convention
as the world-model smoke runner).

Usage:
  python examples/mla_every_sweep.py                     # full sweep
  python examples/mla_every_sweep.py --mla-every 2 8     # subset
  python examples/mla_every_sweep.py --steps 0           # analytic + gate only
  python examples/mla_every_sweep.py --output benchmarks/mla_every_2026-10-12.json
"""
import argparse
import hashlib
import json
import sys
import time
from pathlib import Path

import torch

_SCRIPT_DIR = Path(__file__).resolve().parent
_REPO_ROOT = _SCRIPT_DIR.parent
sys.path.insert(0, str(_REPO_ROOT))
sys.path.insert(0, str(_SCRIPT_DIR))

from evastar.configs.config_v5 import (EvastarV5Config,  # noqa: E402
                                           HybridAttentionConfig)
from evastar.src.model_v5 import EvastarV5, _is_full_attention_layer  # noqa: E402
from benchmarks.bench_cpu import cache_bytes  # noqa: E402

EXPERIMENT = "B1-5 mla-every sweep"
SCORING_TAG = "toy-char-lm-val-loss + HarnessEvolver-style bitwise gate"
PROBE_VERSION = 1


def arm_config(every, layers):
    """Lite width at ``layers`` depth; ``every=None`` = dense-MLA reference
    (hybrid disabled)."""
    if every is None:
        return EvastarV5Config(size="lite", num_hidden_layers=layers)
    return EvastarV5Config(
        size="lite", num_hidden_layers=layers,
        hybrid_attention=HybridAttentionConfig(enabled=True,
                                               full_attention_every=every))


def layer_map(cfg):
    return [_is_full_attention_layer(cfg, i)
            for i in range(cfg.num_hidden_layers)]


@torch.no_grad()
def build_and_generate(cfg, seed, prompts, decode_tokens):
    """Build a fresh model under ``seed`` and greedy-decode every prompt.
    Two calls with identical arguments must return identical token lists
    (the bitwise gate)."""
    torch.manual_seed(seed)
    model = EvastarV5(cfg).eval()
    outs = []
    for p in prompts:
        out = model.generate(p, max_new_tokens=decode_tokens, temperature=0)
        outs.append(out[0].tolist())
    return outs


def fixed_prompts(cfg, seed, n_prompts, prompt_len):
    g = torch.Generator().manual_seed(seed + 1234)
    return [torch.randint(0, cfg.vocab_size, (1, prompt_len), generator=g)
            for _ in range(n_prompts)]


def train_arm(cfg, seed, steps, batch, seq, lr, ids, val_windows):
    """Short char-LM training, mirroring train_toy_checkpoint.py's loop
    (Adam + cosine, no MTP loss — see module docstring). Returns
    (final_train_loss10, val_loss, probe_ids)."""
    torch.manual_seed(seed)
    model = EvastarV5(cfg)
    model.train()
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=max(steps, 1))
    gen = torch.Generator().manual_seed(seed)

    def draw():
        ix = torch.randint(0, len(ids) - seq - 2, (batch,), generator=gen)
        return torch.stack([ids[i:i + seq + 1] for i in ix])

    losses = []
    for _step in range(steps):
        data = draw()
        x, y = data[:, :-1], data[:, 1:]
        logits, _hidden, _ = model(x)
        loss = torch.nn.functional.cross_entropy(
            logits.reshape(-1, logits.shape[-1]), y.reshape(-1))
        opt.zero_grad()
        loss.backward()
        opt.step()
        sched.step()
        losses.append(loss.item())
    tail = losses[-10:] if len(losses) >= 10 else losses
    final10 = sum(tail) / len(tail)

    model.eval()
    with torch.no_grad():
        vx = val_windows[:, :-1]
        vy = val_windows[:, 1:]
        vlogits, _, _ = model(vx)
        val_loss = torch.nn.functional.cross_entropy(
            vlogits.reshape(-1, vlogits.shape[-1]), vy.reshape(-1)).item()
        probe = model.generate(
            val_windows[:1, :8].clone(), max_new_tokens=24,
            temperature=0)[0].tolist()
    return final10, val_loss, probe


def run_sweep(args):
    from train_toy_checkpoint import build_corpus, encode

    reference = arm_config(None, args.layers)
    arms = [("dense_mla", None)] + [(f"mla_every={e}", e)
                                    for e in args.mla_every]

    # Corpus + fixed val windows are shared across arms (same bytes, same
    # windows) so the data side never confounds the architecture side.
    ids = None
    val_windows = None
    if args.steps > 0:
        ids = torch.tensor(encode(build_corpus(), reference.vocab_size),
                           dtype=torch.long)
        vgen = torch.Generator().manual_seed(args.seed + 999)
        starts = torch.randint(0, len(ids) - 257, (2,), generator=vgen)
        val_windows = torch.stack([ids[i:i + 257] for i in starts])

    results_arms = []
    telemetry = {}
    for label, every in arms:
        t0 = time.perf_counter()
        cfg = arm_config(every, args.layers)
        lm = layer_map(cfg)
        n_mla = sum(lm)
        prompts = fixed_prompts(cfg, args.seed, args.prompts,
                                args.prompt_len)

        # --- bitwise gate: same config + seed, twice, token-identical ---
        outs1 = build_and_generate(cfg, args.seed, prompts,
                                   args.decode_tokens)
        outs2 = build_and_generate(cfg, args.seed, prompts,
                                   args.decode_tokens)
        gate_pass = outs1 == outs2
        gate_sha = hashlib.sha256(
            json.dumps(outs1).encode()).hexdigest()

        torch.manual_seed(args.seed)  # identical param count basis
        n_params = sum(p.numel() for p in EvastarV5(cfg).parameters())

        arm = {
            "label": label,
            "full_attention_every": every,
            "layers": cfg.num_hidden_layers,
            "layer_map_is_mla": lm,
            "n_mla_layers": n_mla,
            "n_gda_layers": cfg.num_hidden_layers - n_mla,
            "params": n_params,
            "cache_bytes": cache_bytes(cfg, args.cache_seq),
            "bitwise_gate": {"pass": gate_pass,
                             "outputs_sha256": gate_sha,
                             "prompts": args.prompts,
                             "decode_tokens": args.decode_tokens},
        }

        if args.steps > 0:
            final10, val_loss, probe = train_arm(
                cfg, args.seed, args.steps, args.batch, args.seq,
                args.lr, ids, val_windows)
            arm["training"] = {
                "steps": args.steps,
                "final_train_loss10": final10,
                "val_loss": val_loss,
                "probe_ids": probe,
                "probe_sha256": hashlib.sha256(
                    json.dumps(probe).encode()).hexdigest(),
            }
        results_arms.append(arm)
        telemetry[label] = {"wall_s": round(time.perf_counter() - t0, 2)}
        print(f"[arm {label}] MLA {n_mla}/{cfg.num_hidden_layers}  "
              f"params {n_params/1e6:.2f}M  gate={'PASS' if gate_pass else 'FAIL'}"
              + (f"  val_loss={arm['training']['val_loss']:.4f}"
                 if args.steps > 0 else ""), flush=True)

    # --- auto-search: min val loss among gate-passing trained arms ---
    trained = [a for a in results_arms
               if a["bitwise_gate"]["pass"] and "training" in a]
    if trained:
        best = min(trained, key=lambda a: a["training"]["val_loss"])
        recommendation = {
            "criterion": "min val_loss among bitwise-gate-passing arms",
            "winner": best["label"],
            "val_loss": best["training"]["val_loss"],
            "note": ("quality-only pick; the cache-byte tradeoff per arm "
                     "is reported in cache_bytes and is the reader's call"),
        }
    else:
        recommendation = {
            "criterion": None,
            "reason": ("no trained gate-passing arms "
                       "(--steps 0 or gate failure)"),
        }

    results = {
        "config": {
            "layers": args.layers, "width": "lite", "steps": args.steps,
            "batch": args.batch, "seq": args.seq, "lr": args.lr,
            "seed": args.seed, "cache_seq": args.cache_seq,
            "mla_every": args.mla_every,
            "mtp": "built but excluded from loss and decoding",
        },
        "arms": results_arms,
        "recommendation": recommendation,
    }
    return results, telemetry


def main():
    ap = argparse.ArgumentParser(
        description="B1-5: sweep full_attention_every over {2,4,8} with a "
                    "bitwise gate — bitwise-reproducible toy-scale A/B")
    ap.add_argument("--mla-every", type=int, nargs="+", choices=[2, 4, 8],
                    default=[2, 4, 8],
                    help="interleave values to sweep (dense-MLA reference "
                         "arm is always included)")
    ap.add_argument("--layers", type=int, default=8,
                    help="model depth (stock lite=2 is degenerate for the "
                         "interleave rule)")
    ap.add_argument("--steps", type=int, default=120,
                    help="training steps per arm; 0 = analytic + gate only")
    ap.add_argument("--batch", type=int, default=4)
    ap.add_argument("--seq", type=int, default=64)
    ap.add_argument("--lr", type=float, default=3e-3)
    ap.add_argument("--seed", type=int, default=5)
    ap.add_argument("--prompts", type=int, default=4,
                    help="bitwise-gate prompts per arm")
    ap.add_argument("--prompt-len", type=int, default=12)
    ap.add_argument("--decode-tokens", type=int, default=16)
    ap.add_argument("--cache-seq", type=int, default=2048,
                    help="sequence length for the analytic cache accounting")
    ap.add_argument("--output", default=None)
    args = ap.parse_args()

    results, telemetry = run_sweep(args)
    payload = {
        "experiment": EXPERIMENT,
        "scoring": SCORING_TAG,
        "probe_version": PROBE_VERSION,
        "results": results,
        "telemetry": telemetry,
    }
    canon = json.dumps(results, sort_keys=True).encode()
    payload["results_sha256"] = hashlib.sha256(canon).hexdigest()

    print(f"\n{'label':<14}{'MLA':>6}{'params':>9}{'cache@seq':>12}"
          + (f"{'val_loss':>10}" if args.steps > 0 else "") + "  gate")
    for a in results["arms"]:
        row = (f"{a['label']:<14}{a['n_mla_layers']:>6}"
               f"{a['params']/1e6:>8.2f}M"
               f"{a['cache_bytes']['helioslm_bytes']/1e6:>11.2f}M")
        if args.steps > 0:
            row += f"{a['training']['val_loss']:>10.4f}"
        row += f"  {'PASS' if a['bitwise_gate']['pass'] else 'FAIL'}"
        print(row)
    print(f"recommendation: {results['recommendation']}")
    print(f"results sha256: {payload['results_sha256']}")

    if args.output:
        out = Path(args.output)
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(json.dumps(payload, indent=2) + "\n",
                       encoding="utf-8")
        print(f"saved -> {out}")


if __name__ == "__main__":
    main()
