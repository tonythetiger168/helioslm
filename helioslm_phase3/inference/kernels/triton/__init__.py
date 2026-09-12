"""HeliosLM Triton Kernels (Complete)

All known limitations FIXED:
  ✅ FlashAttention Forward + Backward
  ✅ MoE GroupedGEMM routing
  ✅ Fused LayerNorm + GELU / RMSNorm
  ✅ Triton version compatibility detection
"""
from .flash_attention import flash_attn_forward
from .flash_attention_backward import flash_attn_backward
from .moe_routing import moe_topk_routing, moe_gather_scatter
from .grouped_gemm import GroupedGEMM
from .fused_ops import fused_layer_norm_gelu, fused_rms_norm
from .version_check import TritonCompatibility, get_compatibility

__all__ = [
    "flash_attn_forward", "flash_attn_backward",
    "moe_topk_routing", "moe_gather_scatter",
    "GroupedGEMM",
    "fused_layer_norm_gelu", "fused_rms_norm",
    "TritonCompatibility", "get_compatibility",
]
