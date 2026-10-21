"""Multi-seed runs for the A/B scripts (#55).

Every ``examples/*_ab*.py`` script compares arms under one ``--seed``.
``add_seed_args`` adds ``--seeds S1 S2 ...`` beside it, and ``run`` calls
the script's experiment once per seed:

- precedence: ``--seed`` and ``--seeds`` are mutually exclusive. With
  neither or with ``--seed`` the script runs exactly as before and writes
  the same payload (same keys, same ``results_sha256``); ``--seeds``
  switches to the multi-seed payload below. ``--seeds 0`` is a one-seed
  multi-seed run, not the legacy payload.
- pairing: within a seed every arm runs with that seed (each script
  already seeds all its arms from ``args.seed``), so the arms of a seed
  are paired and ``per_seed`` lets a reader take paired differences.
- payload: ``seeds``; ``per_seed`` (each seed's complete legacy payload);
  ``aggregate`` (the ``results`` of all seeds merged field by field: a
  value equal in every seed is kept as is, numbers that differ become
  ``{"mean", "std", "min", "max", "n"}`` with the sample std, dicts and
  equal-length lists are merged recursively, anything else is listed per
  seed); ``environment``; ``per_seed_results_sha256``.

Aggregates are descriptive. A script's committed single-seed claims are
not multi-seed claims until a ``--seeds`` artifact is committed.
"""
import argparse
import copy
import hashlib
import json
import math
import platform
import sys


def _seed_list(text):
    try:
        v = int(text)
    except ValueError:
        raise argparse.ArgumentTypeError(f"seed must be an integer, got {text!r}")
    if v < 0:
        raise argparse.ArgumentTypeError(f"seed must be >= 0, got {v}")
    return v


def add_seed_args(ap, default=0):
    """Add ``--seed`` (legacy, default ``default``) and ``--seeds``."""
    g = ap.add_mutually_exclusive_group()
    g.add_argument("--seed", type=_seed_list, default=default,
                   help=f"single seed, legacy payload (default {default})")
    g.add_argument("--seeds", type=_seed_list, nargs="+", default=None,
                   help="several distinct seeds: run every arm once per "
                        "seed and write per-seed results plus aggregates")
    return ap


def resolve_seeds(args):
    """The seeds to run, or None for the legacy single-seed payload."""
    seeds = getattr(args, "seeds", None)
    if seeds is None:
        return None
    if len(set(seeds)) != len(seeds):
        raise SystemExit(f"--seeds must be distinct, got {seeds}")
    return list(seeds)


def _is_num(v):
    return isinstance(v, (int, float)) and not isinstance(v, bool)


def aggregate(values):
    """Merge one field's values across seeds (see the module docstring)."""
    first = values[0]
    if all(v == first for v in values[1:]):
        return first
    if all(_is_num(v) for v in values):
        vals = [float(v) for v in values]
        n = len(vals)
        mean = sum(vals) / n
        std = math.sqrt(sum((v - mean) ** 2 for v in vals) / (n - 1)) if n > 1 else 0.0
        return {"mean": mean, "std": std, "min": min(vals), "max": max(vals), "n": n}
    if all(isinstance(v, dict) for v in values):
        keys = list(first)
        if all(list(v) == keys for v in values):
            return {k: aggregate([v[k] for v in values]) for k in keys}
    if all(isinstance(v, list) for v in values):
        if all(len(v) == len(first) for v in values):
            return [aggregate([v[i] for v in values]) for i in range(len(first))]
    return {"per_seed": values}


def environment():
    env = {"python": sys.version.split()[0], "platform": platform.platform(),
           "argv": sys.argv[1:]}
    try:
        import torch
        env["torch"] = torch.__version__
    except ImportError:
        pass
    return env


def run(args, experiment):
    """``experiment(args) -> payload`` once per seed (legacy: once)."""
    seeds = resolve_seeds(args)
    if seeds is None:
        return experiment(args)
    per_seed = []
    for seed in seeds:
        ns = copy.copy(args)
        ns.seed = seed
        ns.seeds = None
        payload = experiment(ns)
        per_seed.append({"seed": seed, **payload})
    results = [p["results"] for p in per_seed]
    return {"experiment": per_seed[0].get("experiment"),
            "seeds": seeds,
            "per_seed": per_seed,
            "aggregate": aggregate(results),
            "environment": environment(),
            "per_seed_results_sha256": hashlib.sha256(
                json.dumps(results, sort_keys=True).encode()).hexdigest()}
