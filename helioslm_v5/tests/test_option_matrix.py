"""T110 — tech-debt Phase 2: option compatibility matrix and config
defaults in one place.

Pins:

1. every pair of feature options (and every single one) either
   - is rejected with a message that comes from OPTION_RULES (the one
     table of cross-option constraints), or
   - builds a model whose cached decode equals its full forward;
2. every OPTION_RULES rule is hit by at least one pair or one listed
   single config, and a rule's message never appears for a valid config;
3. the core model modules read config fields directly (no
   ``getattr(config..., default)`` fallbacks that can drift from the
   dataclass defaults), and every field they read exists on the config
   dataclasses.

Run from repo root: python3 helioslm_v5/tests/test_option_matrix.py
"""
import ast
import dataclasses
import itertools
import sys
from pathlib import Path

import torch

_REPO = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(_REPO))

from helioslm_v5.configs.config_v5 import (  # noqa: E402
    OPTION_RULES, AttentionConfig, OptionConflictError, HeliosLMv5Config, HybridAttentionConfig,
    MoEConfig, MTPConfig)
from helioslm_v5.src.model_v5 import HeliosLMv5  # noqa: E402

# Each feature turns one option (or a minimal coherent group) on.
# Sections: "attention" / "moe" / "hybrid" / "mtp" kwargs, "top" kwargs.
FEATURES = {
    "expanded": {"attention": dict(use_absorption=False)},
    "nope": {"attention": dict(nope=True)},
    "nope_temp": {"attention": dict(nope=True, nope_attn_temperature=True,
                                    nope_temp_floor_scale=4)},
    "qk_norm": {"attention": dict(qk_norm=True)},
    "soft_cap": {"attention": dict(logit_soft_cap=30.0)},
    "output_gate": {"attention": dict(output_gate=True)},
    "window": {"attention": dict(sliding_window=4)},
    "window_sink": {"attention": dict(sliding_window=4, sliding_window_sink=1)},
    "interleave": {"attention": dict(sliding_window=4, global_attention_every=2)},
    "learnable_sink": {"attention": dict(learnable_sink=True)},
    "fp8_cache": {"attention": dict(kv_cache_dtype="fp8")},
    # top_k >= sequence length: the sparse code path runs but selects
    # every key, so decode == full is a valid oracle.
    "sparse": {"attention": dict(sparse_top_k=16)},
    "sparse_learned": {"attention": dict(sparse_top_k=16, sparse_indexer="learned",
                                         sparse_prefill=True)},
    "yarn": {"attention": dict(rope_scaling={"type": "yarn", "factor": 2.0})},
    "p_rope": {"attention": dict(global_rope_partial=0.25)},
    "k_eq_v": {"attention": dict(global_k_eq_v=True, v_head_dim=24)},
    "kv_share": {"attention": dict(kv_share_last_n=1)},
    "hybrid": {"hybrid": dict(enabled=True, full_attention_every=2)},
    # lite defaults: attention residuals off, so HC needs nothing else
    "hyper_conn": {"top": dict(use_hyper_connections=True)},
    "hc_sinkhorn": {"top": dict(use_hyper_connections=True,
                                hyper_connection_sinkhorn=True)},
    "attn_res": {"top": dict(use_attention_residuals=True)},
    "first_k_dense": {"moe": dict(first_k_dense=1)},
    "clamped_swiglu": {"moe": dict(swiglu_limit=7.0, swiglu_alpha=1.702,
                                   swiglu_up_offset=1.0)},
    "zero_centered": {"top": dict(zero_centered_norm=True)},
    "final_soft_cap": {"top": dict(final_logit_soft_cap=30.0)},
    "mtp_light": {"mtp": dict(style="light")},
    "sandwich_norm": {"top": dict(sandwich_norm=True)},
    "smebu": {"moe": dict(balance_strategy="smebu")},
}

# Single configs that trigger the rules no feature pair reaches (a
# feature is a coherent "on" setting, so it never violates a rule alone).
RULE_PROBES = {
    "global_attention_every_needs_window": {"attention": dict(global_attention_every=2)},
    "window_sink_needs_window": {"attention": dict(sliding_window_sink=1)},
    "k_eq_v_needs_equal_dims": {"attention": dict(global_k_eq_v=True, v_head_dim=32)},
    "nope_temperature_needs_nope": {"attention": dict(nope_attn_temperature=True)},
    "learned_indexer_needs_sparse": {"attention": dict(sparse_indexer="learned")},
    "indexer_share_needs_learned": {"attention": dict(sparse_top_k=16,
                                                      indexer_share_every=2)},
    "sparse_prefill_needs_sparse": {"attention": dict(sparse_prefill=True)},
    "sinkhorn_needs_hyper_connections": {"top": dict(hyper_connection_sinkhorn=True)},
}

_LAYERS = 4      # 2-way interleave: global 0, 1, 3; windowed 2


def _merge(*feats):
    out = {"attention": {}, "moe": {}, "hybrid": {"enabled": False},
           "mtp": {}, "top": {}}
    for f in feats:
        for sec, kw in f.items():
            out[sec].update(kw)
    return out


def _config(spec):
    return HeliosLMv5Config(
        size="lite", num_hidden_layers=_LAYERS,
        attention=AttentionConfig(**spec["attention"]),
        moe=MoEConfig(**spec["moe"]),
        hybrid_attention=HybridAttentionConfig(**spec["hybrid"]),
        mtp=MTPConfig(**spec["mtp"]), **spec["top"])


def _try(names):
    spec = _merge(*(FEATURES[n] for n in names))
    try:
        cfg = _config(spec)
    except OptionConflictError as e:
        return "rejected", e.rule, str(e)
    except ValueError as e:
        return "rejected", None, str(e)
    torch.manual_seed(0)
    try:
        model = HeliosLMv5(cfg).eval()
    except ValueError as e:
        return "rejected_at_build", [], str(e)
    ids = torch.randint(3, cfg.vocab_size, (1, 10),
                        generator=torch.Generator().manual_seed(1))
    with torch.no_grad():
        full = model(ids)[0]
        lg, _, past = model(ids[:, :4], use_cache=True)
        outs = [lg]
        for t in range(4, 10):
            lg, _, past = model(ids[:, t:t + 1], past_key_values=past,
                                use_cache=True)
            outs.append(lg)
    err = (torch.cat(outs, 1) - full).abs().max().item()
    return "ok", err, None


def _run_matrix():
    names = sorted(FEATURES)
    combos = [(n,) for n in names] + list(itertools.combinations(names, 2))
    hit_rules, bad, ok = set(), [], 0
    for combo in combos:
        status, info, msg = _try(combo)
        if status == "ok":
            ok += 1
            # fp8 cache: a latent sitting on an E4M3 rounding boundary can
            # round differently in a 10-token prefill and a 1-token decode
            # (float noise ~1e-7 in the projection flips it), so those
            # combos get the fp8 error scale (test_fp8_kv_cache measures
            # 2.7e-2 vs fp32) instead of the exact-path tolerance.
            tol = 1e-2 if "fp8_cache" in combo else 1e-4
            if not info < tol:
                bad.append((combo, f"decode != full ({info:.2e})"))
        elif status == "rejected":
            if not info:
                bad.append((combo, f"rejection not from OPTION_RULES: {msg}"))
            hit_rules.add(info)
        else:
            bad.append((combo, f"rejected only at model build: {msg}"))
    assert not bad, "\n".join(f"{c}: {m}" for c, m in bad)
    return combos, ok, hit_rules


_HITS = None


def test_pairwise_matrix():
    global _HITS
    combos, ok, _HITS = _run_matrix()
    print(f"PASS test_pairwise_matrix ({len(combos)} combos, {ok} built, "
          f"{len(combos) - ok} rejected by {len(_HITS)} rules)")


def test_every_rule_is_reachable():
    hit_rules = set(_HITS) if _HITS is not None else set()
    if _HITS is None:      # run alone: config-level pass only (no models)
        for combo in itertools.combinations(sorted(FEATURES), 2):
            try:
                _config(_merge(*(FEATURES[n] for n in combo)))
            except OptionConflictError as e:
                hit_rules.add(e.rule)
    for name, probe in RULE_PROBES.items():
        try:
            _config(_merge(probe))
        except OptionConflictError as e:
            assert e.rule == name, (name, e.rule, str(e))
            hit_rules.add(name)
        else:
            raise AssertionError(f"probe for {name} was accepted")
    missing = {r.name for r in OPTION_RULES} - hit_rules
    assert not missing, f"rules never triggered: {sorted(missing)}"
    assert len({r.name for r in OPTION_RULES}) == len(OPTION_RULES)
    assert all(r.kind in ("requires", "excludes") for r in OPTION_RULES)
    print(f"PASS test_every_rule_is_reachable ({len(OPTION_RULES)} rules)")


_CORE = ["helioslm_v5/src/attention/mla.py", "helioslm_v5/src/model_v5.py",
         "helioslm_v5/src/moe/sigmoid_moe.py",
         "helioslm_v5/src/attention/linear_attention.py",
         "helioslm_v5/src/vision/navit.py"]
def _fields(cls):
    names = {f.name for f in dataclasses.fields(cls)}
    names |= {n for n, v in vars(cls).items() if isinstance(v, property)}
    return names


def test_core_modules_read_config_directly():
    top = _fields(HeliosLMv5Config)
    lite = HeliosLMv5Config(size="lite")
    section_cls = {f.name: type(getattr(lite, f.name))
                   for f in dataclasses.fields(HeliosLMv5Config)
                   if dataclasses.is_dataclass(getattr(lite, f.name))}
    problems = []
    for rel in _CORE:
        tree = ast.parse((_REPO / rel).read_text(encoding="utf-8"))
        for node in ast.walk(tree):
            if (isinstance(node, ast.Call) and isinstance(node.func, ast.Name)
                    and node.func.id == "getattr" and node.args
                    and "config" in ast.unparse(node.args[0])):
                problems.append(f"{rel}:{node.lineno} getattr fallback "
                                f"{ast.unparse(node)}")
            if not isinstance(node, ast.Attribute):
                continue
            chain = []
            cur = node
            while isinstance(cur, ast.Attribute):
                chain.append(cur.attr)
                cur = cur.value
            chain.reverse()
            if isinstance(cur, ast.Name) and cur.id == "config":
                path = chain
            elif (isinstance(cur, ast.Name) and cur.id == "self"
                  and chain[:1] == ["config"]):
                path = chain[1:]
            else:
                continue
            if not path or node is not _outermost(tree, node):
                continue
            head = path[0]
            if head not in top:
                problems.append(f"{rel}:{node.lineno} config.{head} unknown")
            elif head in section_cls and len(path) > 1 \
                    and path[1] not in _fields(section_cls[head]):
                problems.append(f"{rel}:{node.lineno} config.{head}.{path[1]} unknown")
    assert not problems, "\n".join(problems)
    print(f"PASS test_core_modules_read_config_directly ({len(_CORE)} modules)")


_PARENTS = {}


def _outermost(tree, node):
    if tree not in _PARENTS:
        parents = {}
        for p in ast.walk(tree):
            for ch in ast.iter_child_nodes(p):
                parents[ch] = p
        _PARENTS[tree] = parents
    parents = _PARENTS[tree]
    while isinstance(parents.get(node), ast.Attribute):
        node = parents[node]
    return node


if __name__ == "__main__":
    test_pairwise_matrix()
    test_every_rule_is_reachable()
    test_core_modules_read_config_directly()
