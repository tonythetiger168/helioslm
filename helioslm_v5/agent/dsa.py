import math

CERT_BITS = 40
REL_TOL = 1e-9


def dense_decode(scores: list, V: list) -> list:
    m = max(scores)
    exps = [math.exp(s - m) for s in scores]
    denom = sum(exps)
    return [sum(e * V[i][j] for i, e in enumerate(exps)) / denom
            for j in range(len(V[0]))]


def exact_topk(scores: list, k: int) -> list:
    if not 1 <= k <= len(scores):
        raise ValueError("k out of range")
    return sorted(range(len(scores)), key=lambda i: (-scores[i], i))[:k]


def sparse_decode(scores: list, V: list, k: int):
    idx = exact_topk(scores, k)
    if k == len(scores):
        return dense_decode(scores, V), True
    sel = [scores[i] for i in idx]
    sel_set = set(idx)
    unsel_best = max(scores[i] for i in range(len(scores))
                     if i not in sel_set)
    m = max(scores)
    denom_sel = sum(math.exp(s - m) for s in sel)
    bound = (len(scores) - k) * math.exp(unsel_best - m)
    cert = (bound / (denom_sel + bound)) < 2.0 ** -CERT_BITS
    exps = [math.exp(s - m) for s in sel]
    denom = sum(exps)
    return ([sum(e * V[i][j] for i, e in zip(idx, exps)) / denom
             for j in range(len(V[0]))], cert)


def dsa_gate(sparse_out: list, dense_out: list, certificate_ok: bool,
             k: int, n: int) -> None:
    if not certificate_ok:
        return
    worst = max(abs(a - b) / max(1.0, abs(b))
                for a, b in zip(sparse_out, dense_out))
    if worst > REL_TOL:
        raise AssertionError(
            f"DSA GATE VIOLATION (k={k}, n={n}): certified but fp64 gap "
            f"{worst:.3e} > {REL_TOL} — this is a bug, not a trade-off")
