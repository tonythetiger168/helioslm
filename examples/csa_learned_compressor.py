"""csa_learned_compressor.py — roadmap Phase B: can a LEARNED compressor
give CSA back the exact recall that mean-pooling destroys?

Follow-up to benchmarks/csa_ab_2026-10-15.md, where mean-pooled CSA sat
at chance although its indexer picked the right chunk 86–100% of the
time. Here the same identity-wired CSA gets ``compressor="learned"``
(an MLP softmax-pooler, zero-initialised = mean pooling) and ONLY the
compressor is trained; q/k/v/out/indexer stay identity and frozen.

Training: end-to-end task loss (cross-entropy of the passkey readout
over the candidate values), back-propagated through the full CSA
forward — the compressor is never told which token is planted.
Training trials come from a separate seed stream at a SHORTER length
(L=128); evaluation reuses the A/B probe's seed-0 stream at L=512/1024,
so evaluation sequences are both unseen and longer than training ones.

The trained compressor weights are stored in the artifact (small MLP),
so the evaluation is reproducible without retraining.

Run from repo root:
    python examples/csa_learned_compressor.py --output benchmarks/csa_learned_2026-10-15.json
    python examples/csa_learned_compressor.py --check
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
sys.path.insert(0, str(_REPO / "examples"))

from csa_ab_passkey import _wire_csa, selection_hit_rate  # noqa: E402
from hca_ab_passkey import SCORING_TAG, _trial, probe_accuracy  # noqa: E402
from helioslm_v5.src.attention.kv_compress import (  # noqa: E402
    CompressedSparseAttention)

DEFAULT_OUT = "benchmarks/csa_learned_2026-10-15.json"
V1_SELECTION = ("logit scaling 1/amp, lr and steps chosen on training loss "
                "and a separate validation stream (seed 2000, L=128, 150 "
                "trials, ratio 4): lr 3e-3/1200 steps 0.487, 1e-2/1200 "
                "0.533, 1e-2/2400 0.567; ratio 16 stayed at 0.167 for both "
                "step counts. The evaluation stream was not used for any "
                "choice.")
EVAL_KEYS = ("dim", "heads", "hidden", "pairs", "amp", "boost", "trials",
             "seq_lens", "topks", "eval_seed", "act")


def oracle_pool_weights(self, x, attention_mask, B, C, m):
    """MEASURING STICK, not a method: all weight on each chunk's
    largest-norm real token. Gives the accuracy a perfect compressor
    would reach on this probe (the planted tokens are the high-norm
    ones by construction)."""
    n = x.norm(dim=-1).masked_fill(attention_mask == 0, -1.0).view(B, C, m)
    w = torch.nn.functional.one_hot(n.argmax(-1), m).to(x.dtype)
    return w.view(B, C, m, 1, 1).expand(B, C, m, self.n_heads, 1)


def oracle_ceiling(args, ratio):
    def make():
        m = _wire_csa(CompressedSparseAttention(
            args.dim, n_heads=args.heads, ratio=ratio, topk=args.topks[0]),
            args.dim)
        m._pool_weights = oracle_pool_weights.__get__(m)
        return m
    return {f"L={L}": probe_accuracy(make, L, args.pairs, args.dim,
                                     args.amp, args.boost, args.trials,
                                     args.eval_seed)
            for L in args.seq_lens}
COMPRESSOR_KEYS = ("compressor_mlp.", "compressor_pos")


def make_csa(args, ratio, topk, state=None):
    m = _wire_csa(CompressedSparseAttention(
        args.dim, n_heads=args.heads, ratio=ratio, topk=topk,
        compressor="learned", compressor_hidden=args.hidden,
        compressor_act=getattr(args, "act", "gelu")), args.dim)
    if state is not None:
        m.load_state_dict(state, strict=False)
    return m.eval()


def compressor_state(m):
    return {k: v.detach().clone() for k, v in m.state_dict().items()
            if k.startswith(COMPRESSOR_KEYS)}


def transfer_state(state, ratio, heads):
    """A compressor state for another ratio: the MLP scores each token on
    its own, so it carries over unchanged; the per-position bias belongs
    to the old chunk size and restarts at zero."""
    out = dict(state)
    out["compressor_pos"] = torch.zeros(ratio, heads)
    return out


def train(args, ratio, init_state=None):
    """Train the compressor only, end-to-end on the readout loss.
    ``init_state`` (v5.78): start from another ratio's trained compressor
    (see transfer_state) instead of the zero init (= mean pooling)."""
    torch.manual_seed(args.seed)
    m = make_csa(args, ratio, args.train_topk).train()
    if init_state is not None:
        m.load_state_dict(transfer_state(init_state, ratio, args.heads),
                          strict=False)
    params = [p for n, p in m.named_parameters()
              if n.startswith(COMPRESSOR_KEYS)]
    for n, p in m.named_parameters():
        p.requires_grad_(n.startswith(COMPRESSOR_KEYS))
    opt = torch.optim.Adam(params, lr=args.lr)
    g = torch.Generator().manual_seed(args.train_seed)
    losses = []
    for _ in range(args.steps):
        xs, targets, cands = [], [], []
        for _ in range(args.batch):
            x, qi, wvals = _trial(g, args.train_len, args.pairs, args.dim,
                                  args.amp, args.boost)
            xs.append(x)
            targets.append(qi)
            cands.append(wvals)
        out = m(torch.cat(xs))[:, -1]                          # (B, D)
        # raw scores are O(amp^2) and saturate the cross-entropy; divide
        # by amp (chosen on TRAINING loss, not evaluation accuracy)
        logits = torch.einsum("bpd,bd->bp", torch.stack(cands), out) \
            / args.amp
        loss = torch.nn.functional.cross_entropy(logits,
                                                 torch.tensor(targets))
        opt.zero_grad()
        loss.backward()
        opt.step()
        losses.append(loss.item())
    return compressor_state(m), losses


def evaluate(args, ratio, state):
    rows = {}
    for L in args.seq_lens:
        for k in args.topks:
            make = (lambda kk=k: make_csa(args, ratio, kk, state))
            rows[f"L={L}/k{k}"] = {
                "accuracy": probe_accuracy(make, L, args.pairs, args.dim,
                                           args.amp, args.boost,
                                           args.trials, args.eval_seed),
                "selection_hit_rate": selection_hit_rate(
                    make, ratio, L, _EvalArgs(args)),
            }
    return rows


class _EvalArgs:
    """selection_hit_rate reads trials/pairs/dim/amp/boost/seed."""
    def __init__(self, a):
        self.trials, self.pairs, self.dim = a.trials, a.pairs, a.dim
        self.amp, self.boost, self.seed = a.amp, a.boost, a.eval_seed


def _to_json(state):
    return {k: v.tolist() for k, v in state.items()}


def _from_json(blob):
    return {k: torch.tensor(v, dtype=torch.float32) for k, v in blob.items()}


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--ratios", type=int, nargs="+", default=[4, 16])
    ap.add_argument("--topks", type=int, nargs="+", default=[8, 32])
    ap.add_argument("--seq-lens", type=int, nargs="+", default=[512, 1024])
    ap.add_argument("--trials", type=int, default=200)
    ap.add_argument("--pairs", type=int, default=8)
    ap.add_argument("--dim", type=int, default=64)
    ap.add_argument("--heads", type=int, default=2)
    ap.add_argument("--amp", type=float, default=24.0)
    ap.add_argument("--boost", type=float, default=1.6)
    ap.add_argument("--hidden", type=int, default=16)
    ap.add_argument("--act", choices=["gelu", "relu2"], default="gelu")
    ap.add_argument("--steps", type=int, default=2400)
    ap.add_argument("--batch", type=int, default=16)
    ap.add_argument("--lr", type=float, default=1e-2)
    ap.add_argument("--train-len", type=int, default=128)
    ap.add_argument("--train-topk", type=int, default=8)
    ap.add_argument("--train-seed", type=int, default=1000)
    ap.add_argument("--eval-seed", type=int, default=0)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--date", default="2026-10-15")
    ap.add_argument("--output", default=None)
    ap.add_argument("--check", action="store_true",
                    help="re-evaluate the committed weights and compare")
    ap.add_argument("--artifact", default=DEFAULT_OUT,
                    help="artifact that --check re-evaluates")
    ap.add_argument("--oracle-ceiling", action="store_true",
                    help="also record the oracle-pooling ceiling")
    ap.add_argument("--selection-note", default=V1_SELECTION,
                    help="how the training settings were chosen")
    ap.add_argument("--init-from", default=None,
                    help="v5.78 curriculum: start every ratio from the "
                         "ratio-4 compressor stored in this artifact")
    args = ap.parse_args()

    if args.check:
        art = json.loads((_REPO / args.artifact).read_text())
        for key in EVAL_KEYS:          # evaluate with the artifact's own
            # sizes, not the CLI's; artifacts before v5.61 have no "act"
            setattr(args, key, art["params"].get(key, "gelu" if key == "act"
                                                  else None))
        for r in art["results"]:
            got = evaluate(args, r["ratio"], _from_json(r["compressor_state"]))
            if got != r["eval"]:
                sys.exit(f"ratio {r['ratio']}: evaluation differs from "
                         f"the committed artifact")
        print("learned-compressor evaluation matches the committed artifact")
        return

    init = None
    if args.init_from:
        src = json.loads((_REPO / args.init_from).read_text())
        init = _from_json(next(r for r in src["results"]
                               if r["ratio"] == 4)["compressor_state"])
        if src["params"]["hidden"] != args.hidden \
                or src["params"].get("act", "gelu") != args.act:
            sys.exit("--init-from artifact has a different hidden / act")
    results = []
    for ratio in args.ratios:
        t0 = time.time()
        zero_shot = None
        if init is not None:
            zero_shot = evaluate(args, ratio,
                                 transfer_state(init, ratio, args.heads))
        state, losses = train(args, ratio, init)
        ev = evaluate(args, ratio, state)
        print(f"ratio {ratio}: loss {losses[0]:.3f} -> {losses[-1]:.3f} "
              f"({time.time() - t0:.0f}s)")
        for key, row in ev.items():
            print(f"  {key}: acc {row['accuracy']:.3f} "
                  f"hit {row['selection_hit_rate']:.3f}")
        if zero_shot is not None:
            for key, row in zero_shot.items():
                print(f"  zero-shot {key}: acc {row['accuracy']:.3f}")
        results.append({"ratio": ratio,
                        **({"zero_shot_eval": zero_shot} if zero_shot else {}),
                        "loss_first": round(losses[0], 4),
                        "loss_last": round(losses[-1], 4),
                        "eval": ev,
                        "compressor_state": _to_json(state)})
    params = {k: v for k, v in vars(args).items()
              if k not in ("output", "check", "artifact", "oracle_ceiling",
                           "selection_note", "init_from")}
    if args.init_from:
        params["init_from"] = args.init_from
    blob = json.dumps([{k: r[k] for k in ("ratio", "eval")}
                       for r in results], sort_keys=True)
    payload = {
        "experiment": "Phase B v4_csa: learned compressor on the passkey probe",
        "scoring": SCORING_TAG,
        "verification_tier": "measured",
        "torch": torch.__version__,
        "params": params,
        "chance": 1.0 / args.pairs,
        "baseline": "benchmarks/csa_ab_2026-10-15.json (mean-pooled CSA)",
        "model_selection": args.selection_note,
        "note": "only the compressor is trained (end-to-end readout loss, "
                "train seed stream and L=128); q/k/v/out/indexer stay "
                "identity-wired and frozen; evaluated on the A/B seed-0 "
                "stream at longer L",
        "results": results,
        "results_sha256": hashlib.sha256(blob.encode()).hexdigest(),
    }
    if args.oracle_ceiling:
        payload["oracle_ceiling"] = {
            "note": "measuring stick, not a method: each chunk pooled onto "
                    "its largest-norm token",
            **{f"ratio={r}": oracle_ceiling(args, r) for r in args.ratios}}
    if args.output:
        Path(args.output).write_text(json.dumps(payload, indent=1) + "\n",
                                     encoding="utf-8")
        print(f"wrote {args.output}")


if __name__ == "__main__":
    main()
