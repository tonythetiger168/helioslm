"""alignbench_run.py - v5.38: run OUR RLCD stack on RLCDAlignBench.

Dataset: sumleo/RLCDAlignBench (HF, GATED -- request access first;
paper: 'Just Ask Jev', ICLR 2027 under review). 7,193 instances across
44 benchmarks, 10 alignment-failure classes, binary labels.

We do NOT have a Jev API key. Two legitimate legs:
  inspect    dump the first instance's fields (field mapping is
             finalized AFTER the gated data lands -- do this first)
  head       SUPERVISED DecisionHead per benchmark (the honest analog
             of the paper's TF-IDF LR baseline): split per benchmark,
             train on train half, AUROC on held-out half
  jev        recompute Jev's cached per-instance responses into AUROC
             offline (no key needed; the cache ships in the dataset)

Honest positioning (recorded): the paper's headline is ZERO-SHOT Jev
at 0.886 median AUROC; our head leg is supervised-at-toy-encoder-scale
and answers a different question -- 'does an open RLCD readout
pipeline run on the canonical benchmark' -- not 'beat Jev'.

Usage:
  python examples/alignbench_run.py inspect --data data/all.jsonl
  python examples/alignbench_run.py head --data data/all.jsonl
  python examples/alignbench_run.py jev --data data/all.jsonl
"""
import argparse
import json
import math
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "helioslm_v5" / "agent"))


def load(path):
    with open(path, encoding="utf-8") as f:
        return [json.loads(l) for l in f if l.strip()]


def auroc(scores, labels):
    """Mann-Whitney AUROC; returns None if a class is empty."""
    pos = [s for s, y in zip(scores, labels) if y == 1]
    neg = [s for s, y in zip(scores, labels) if y == 0]
    if not pos or not neg:
        return None
    wins, total = 0.0, len(pos) * len(neg)
    for ps in pos:
        for ns in neg:
            wins += 0.5 if ps == ns else (1.0 if ps > ns else 0.0)
    return wins / total


def render_state(item):
    """Generic state: prompt + response (the paper's blind main
    variant). Field names finalized by `inspect`."""
    prompt = item.get("prompt") or item.get("question") or ""
    response = item.get("response") or item.get("output") or ""
    return f"[request] {prompt}\n[response] {response}"


def cmd_inspect(args):
    items = load(args.data)
    print(f"instances: {len(items)}")
    srcs = {}
    for it in items:
        srcs[it.get("source", "?")] = srcs.get(it.get("source", "?"), 0) + 1
    print(f"benchmarks: {len(srcs)}")
    for k in sorted(srcs)[:10]:
        print(f"  {k}: {srcs[k]}")
    print("\nfirst instance keys:", sorted(items[0].keys()))
    for k in sorted(items[0].keys()):
        v = str(items[0][k])[:100]
        print(f"  {k}: {v}")


def cmd_head(args):
    import random
    import torch
    from decision_head import DecisionHead
    from trajectory import text_to_ids

    items = load(args.data)
    by_src = {}
    for it in items:
        by_src.setdefault(it.get("source", "?"), []).append(it)
    rng = random.Random(0)
    results = {}
    for src, group in sorted(by_src.items()):
        labels = [1 if it.get("label_correct") == 0 else 0
                  for it in group]   # 1 = FAILURE present (finalize via inspect)
        if len(set(labels)) < 2:
            continue
        idx = list(range(len(group)))
        rng.shuffle(idx)
        half = len(idx) // 2
        train = [{"ids": torch.tensor(text_to_ids(render_state(group[i])[:700])),
                  "answers": {"trust": "yes" if labels[i] == 1 else "no",
                              "trust__target_conf": float(labels[i])}}
                 for i in idx[:half]]
        test = idx[half:]
        torch.manual_seed(0)
        head = DecisionHead({"trust": ("noul", None)}, brier_lambda=1.0)
        head.fit(train, epochs=1500, lr=1e-1)
        head.eval()
        scores = []
        with torch.no_grad():
            for i in test:
                p, _ = head.decide(
                    "trust", torch.tensor(
                        text_to_ids(render_state(group[i])[:700])))
                scores.append(p)
        a = auroc(scores, [labels[i] for i in test])
        if a is not None:
            results[src] = a
    for k, v in sorted(results.items(), key=lambda kv: -kv[1]):
        print(f"  {k}: AUROC {v:.3f} (n held-out)")
    vals = sorted(results.values())
    if vals:
        print(f"\nmedian AUROC: {vals[len(vals)//2]:.3f} over "
              f"{len(results)} benchmarks")


def cmd_jev(args):
    print("offline Jev-cache recompute: pending dataset access "
          "(cached response field names come from `inspect`)")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("mode", choices=["inspect", "head", "jev"])
    ap.add_argument("--data", default="data/all.jsonl")
    cmd_inspect(ap.parse_args()) if False else None
    args = ap.parse_args()
    {"inspect": cmd_inspect, "head": cmd_head, "jev": cmd_jev}[args.mode](args)
