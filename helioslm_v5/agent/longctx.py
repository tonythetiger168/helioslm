import random

FILLER = "the quick brown fox jumps over a lazy dog near an old river bank "


def make_needle_corpus(rng: random.Random, n_tokens: int,
                       needle: str, depth: float):
    words = FILLER.split()
    filler = [rng.choice(words) for _ in range(n_tokens)]
    pos = int(depth * (n_tokens - 1))
    fact = f"the magic number is {needle}"
    return (" ".join(filler[:pos] + fact.split() + filler[pos:]),
            "what is the magic number?")


def probe_report(model_fn, n_tokens: int = 128_000,
                 depths=(0.1, 0.5, 0.9), trials=5, seed=16) -> dict:
    out = {}
    for d in depths:
        hits = 0
        for t in range(trials):
            rng = random.Random(f"{seed}:{int(d * 100)}:{t}")
            needle = str(rng.randint(1000, 9999))
            corpus, q = make_needle_corpus(rng, n_tokens, needle, d)
            hits += needle in model_fn(corpus, q)
        out[f"depth_{d}"] = hits / trials
    dvals = [v for k, v in out.items() if k.startswith("depth")]
    out["mean"] = sum(dvals) / len(dvals)
    return out
