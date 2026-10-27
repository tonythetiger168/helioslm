"""MoE configuration table for #42 (30B-A3B), OS6 (v5.102).

REFERENCE holds the MoE fields of the five models in the 2026-10-22
open-model review (plus MiMo-V2.5-Pro), read from each repository's
config.json on 2026-10-22. ``moe_ffn_params`` counts routed + shared
expert weights (three matrices per SwiGLU expert); on the references it
lands within 2-3% of each vendor's TOTAL parameter count, which pins
the fields as read (attention, embeddings and dense layers make up the
rest). Active counts are lower than the vendors' because attention and
embeddings are left out; ``model_params`` adds a GQA attention and
embedding estimate for candidate configs.

``candidates`` enumerates HeliosLM MoEConfig-shaped configs near a
(total, active) target. It is a sizing aid for choosing what to sweep at
small scale; it does not say which config trains best.
"""
from dataclasses import dataclass
from typing import Dict, List, Optional


@dataclass(frozen=True)
class MoESpec:
    layers: int
    hidden: int
    experts: int          # routed
    shared: int
    top_k: int
    first_k_dense: int
    expert_inter: int     # moe_intermediate_size
    dense_inter: Optional[int] = None
    scoring: str = "sigmoid"
    claimed_total_b: Optional[float] = None
    claimed_active_b: Optional[float] = None
    note: str = ""


REFERENCE: Dict[str, MoESpec] = {
    "tencent/Hy3": MoESpec(80, 4096, 192, 1, 8, 1, 1536, 13312, "sigmoid",
                           295, 21, "router bias, route_norm, scale 2.826"),
    "XiaomiMiMo/MiMo-V2.5": MoESpec(48, 4096, 256, 0, 8, 1, 2048, None,
                                    "sigmoid", 310, 15, "no shared expert"),
    "XiaomiMiMo/MiMo-V2.5-Pro": MoESpec(70, 6144, 384, 0, 8, 1, 2048, None,
                                        "sigmoid", None, None,
                                        "no shared expert"),
    "deepseek-ai/DeepSeek-V4-Flash": MoESpec(
        43, 4096, 256, 1, 6, 0, 2048, None, "sqrtsoftplus", 284, 13,
        "first 3 layers hash-routed (num_hash_layers=3), scale 1.5"),
    "MiniMaxAI/MiniMax-M3": MoESpec(60, 6144, 128, 1, 4, 3, 3072, 12288,
                                    "sigmoid", 428, 23,
                                    "routing bias, scale 2.0"),
    "zai-org/GLM-5.2": MoESpec(78, 6144, 256, 1, 8, 3, 2048, 12288,
                               "sigmoid", 753, None,
                               "noaux_tc, scale 2.5"),
}


def moe_ffn_params(s: MoESpec):
    """(total, active) routed + shared expert weights, MoE layers only."""
    per = 3 * s.hidden * s.expert_inter
    moe_layers = s.layers - s.first_k_dense
    return (moe_layers * (s.experts + s.shared) * per,
            moe_layers * (s.top_k + s.shared) * per)


def model_params(s: MoESpec, q_heads: int, kv_heads: int, head_dim: int,
                 vocab: int, tied: bool = False):
    """(total, active) estimate: experts + dense FFN layers + GQA
    attention (q, k, v, o) + embeddings (input and output unless tied).
    Norms, routers and MTP heads are ignored (< 1%)."""
    ffn_t, ffn_a = moe_ffn_params(s)
    dense = s.first_k_dense * 3 * s.hidden * (s.dense_inter or 4 * s.hidden)
    attn = s.layers * s.hidden * head_dim * (2 * q_heads + 2 * kv_heads)
    emb = vocab * s.hidden * (1 if tied else 2)
    rest = dense + attn + emb
    return ffn_t + rest, ffn_a + rest


def ratios(s: MoESpec) -> Dict[str, float]:
    total, active = moe_ffn_params(s)
    return {"top_k/experts": s.top_k / s.experts,
            "active_ffn_fraction": active / total,
            "expert_inter/hidden": s.expert_inter / s.hidden}


def candidates(target_total=30e9, target_active=3e9, tol=0.12,
               vocab=128_000, head_dim=128) -> List[dict]:
    """Configs within ``tol`` of both targets over a small grid."""
    out = []
    for hidden, layers in ((2048, 48), (2560, 40), (2048, 40), (3072, 32)):
        q = hidden // head_dim
        kv = max(q // 8, 2)
        for experts, top_k, inter in ((128, 8, 768), (256, 8, 512),
                                      (64, 4, 1536), (192, 8, 640),
                                      (128, 4, 1024), (256, 8, 384)):
            for shared in (0, 1):
                s = MoESpec(layers, hidden, experts, shared, top_k, 1, inter,
                            4 * hidden)
                t, a = model_params(s, q, kv, head_dim, vocab)
                if abs(t / target_total - 1) <= tol and \
                        abs(a / target_active - 1) <= tol:
                    out.append({"spec": s, "total": t, "active": a,
                                "q_heads": q, "kv_heads": kv})
    return sorted(out, key=lambda c: abs(c["total"] / target_total - 1)
                  + abs(c["active"] / target_active - 1))
