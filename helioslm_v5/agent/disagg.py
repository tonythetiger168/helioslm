import hashlib
from dataclasses import dataclass, field

PF_A, PF_B = 0.05, 0.001
DC_A, DC_B = 0.02, 0.001
HIT_FRAC = 0.3


@dataclass(frozen=True)
class Request:
    rid: int
    prefill_tokens: int
    decode_tokens: int
    cache_key: str | None = None


@dataclass(frozen=True)
class DisaggConfig:
    n_prefill: int
    n_decode: int
    policy: str = "cache_aware"

    def fingerprint(self) -> tuple:
        return (self.n_prefill, self.n_decode, self.policy,
                PF_A, PF_B, DC_A, DC_B, HIT_FRAC)


def workload_fingerprint(workload: list) -> str:
    h = hashlib.sha256()
    for r in workload:
        h.update(repr(r).encode())
    return h.hexdigest()[:16]


@dataclass
class _Worker:
    busy_until: float = 0.0
    cache: set = field(default_factory=set)


def evaluate(workload: list, cfg: DisaggConfig) -> tuple:
    pf = [_Worker() for _ in range(cfg.n_prefill)]
    dc = [_Worker() for _ in range(cfg.n_decode)]
    worker_seconds = 0.0

    def assign(req, pool, phase):
        if cfg.policy == "round_robin":
            return pool[req.rid % len(pool)]
        if phase == "prefill" and req.cache_key is not None:
            holders = [w for w in pool if req.cache_key in w.cache]
            if holders:
                return min(holders, key=lambda w: w.busy_until)
        return min(pool, key=lambda w: w.busy_until)

    for req in workload:
        w = assign(req, pf, "prefill")
        hit = req.cache_key is not None and req.cache_key in w.cache
        dur = (PF_A + PF_B * req.prefill_tokens) * (HIT_FRAC if hit else 1.0)
        w.busy_until += dur
        if req.cache_key is not None:
            w.cache.add(req.cache_key)
        worker_seconds += dur
        w2 = assign(req, dc, "decode")
        dur2 = DC_A + DC_B * req.decode_tokens
        w2.busy_until = max(w2.busy_until, w.busy_until) + dur2
        worker_seconds += dur2

    return max(w.busy_until for w in pf + dc), worker_seconds


def monotonicity_gate(workload: list, configs: list) -> None:
    wf = workload_fingerprint(workload)
    results = [(c, evaluate(workload, c)[0]) for c in configs]
    for c0, m0 in results:
        for c1, m1 in results:
            if (c1.n_prefill + c1.n_decode) > (c0.n_prefill + c0.n_decode) \
                    and m1 > m0 + 1e-9:
                raise AssertionError(
                    f"DISAGG MONOTONICITY GATE VIOLATED (wf={wf}): "
                    f"{c0} -> {c1} raised makespan {m0:.4f} -> {m1:.4f} — "
                    f"model bug, not a trade-off")
