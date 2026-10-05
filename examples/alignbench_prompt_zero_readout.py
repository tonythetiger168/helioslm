"""Readout for the prompt-zero leg (sandbox side).

Input: prompt_zero_scores.json  {scores, labels, bench}
Output: per-benchmark AUROC + median, appended to the comparison table.
No training at all -- the scores ARE the detector output.
"""
import json, sys

def auroc(scores, labels):
    scores = [float(s) for s in scores]
    labels = [int(l) for l in labels]
    pos = [s for s, l in zip(scores, labels) if l == 1]
    neg = [s for s, l in zip(scores, labels) if l == 0]
    if not pos or not neg:
        return None
    order = sorted(range(len(scores)), key=lambda i: scores[i])
    ranks = [0.0] * len(scores)
    i = 0
    while i < len(order):
        j = i
        while j + 1 < len(order) and scores[order[j + 1]] == scores[order[i]]:
            j += 1
        avg_rank = (i + 1 + j + 1) / 2.0
        for k in range(i, j + 1):
            ranks[order[k]] = avg_rank
        i = j + 1
    r_pos = sum(ranks[i] for i, l in enumerate(labels) if l == 1)
    return (r_pos - len(pos) * (len(pos) + 1) / 2.0) / (len(pos) * len(neg))

def main(path):
    d = json.load(open(path))
    scores, labels, bench = d["scores"], d["labels"], d["bench"]
    by_src = {}
    for s, l, b in zip(scores, labels, bench):
        by_src.setdefault(b, []).append((s, l))
    results = {}
    for b, pairs in sorted(by_src.items()):
        ls = [p[1] for p in pairs]
        if len(set(ls)) < 2:
            continue
        a = auroc([p[0] for p in pairs], ls)
        if a is not None:
            results[b] = {"auroc": a, "n": len(pairs)}
    vals = sorted(v["auroc"] for v in results.values())
    med = vals[len(vals) // 2]
    print(f"PROMPT-ZERO median AUROC: {med:.3f}  (n={len(results)})")
    for b in sorted(results, key=lambda k: -results[k]["auroc"])[:5]:
        print(f"  {b}: {results[b]['auroc']:.3f}")
    out = {"median": med, "per_benchmark": results,
           "protocol": "zero-shot, fixed template, P(yes) from full "
                       "distribution, no training"}
    json.dump(out, open("prompt_zero_readout.json", "w"), indent=1)

if __name__ == "__main__":
    main(sys.argv[1] if len(sys.argv) > 1 else "prompt_zero_scores.json")
