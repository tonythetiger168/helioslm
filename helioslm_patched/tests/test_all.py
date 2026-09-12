"""K3+ v4.1 Test Suite - Production-Ready"""
import sys
from pathlib import Path
ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT))

import torch
from configs import ultra_config, pro_config, lite_config, nano_config
from src.model import KimiK3Plus
from src.attention import KimiDeltaAttention, RMSNorm, RotaryPositionalEmbedding
from src.moe import StableLatentMoE
from src.speculative_decoding import DraftModel, SpeculativeDecoder
from src.rag import RAGModule
from src.cot_compiler import CoTCompiler
from src.reasoning_budget import ReasoningBudgetController
from src.agentic import AgenticLayer
from src.memory import LongTermMemory
from src.multimodal import MultimodalFusion, VisionEncoder, AudioEncoder
from src.adaptive_reasoning_v2 import AdaptiveReasoningController, MCTSReasoningSearch
from src.online_learning import UserLoRA, OnlineLearningManager
from src.safety_alignment import SafetyAlignmentLayer, HarmfulnessClassifier
from src.compression import CompressionManager, KVCacheCompressor, DynamicQuantizer


def test_sizes():
    for name, cfg in [("ultra", ultra_config), ("pro", pro_config), ("lite", lite_config), ("nano", nano_config)]:
        m = KimiK3Plus(cfg, size=name)
        params = sum(p.numel() for p in m.parameters())
        print(f"  {name.upper()}: {params:,} params")
    print("✅ 4 sizes OK")


def test_forward():
    m = KimiK3Plus(lite_config, size="lite")
    ids = torch.randint(0, lite_config.vocab_size, (2, 128))
    logits, aux, metadata = m(ids)
    assert logits.shape == (2, 128, lite_config.vocab_size)
    assert aux >= 0
    assert "rag" in metadata
    assert "safety" in metadata
    print("✅ Forward OK (with metadata)")


def test_forward_with_cache():
    m = KimiK3Plus(lite_config, size="lite")
    ids = torch.randint(0, lite_config.vocab_size, (1, 10))
    logits, aux, past_kvs, metadata = m(ids, use_cache=True)
    assert past_kvs is not None
    assert len(past_kvs) == lite_config.num_hidden_layers
    assert past_kvs[0][0].shape[2] == 10  # KV cache length
    print("✅ Forward with KV-Cache OK")


def test_kv_cache_generation():
    m = KimiK3Plus(lite_config, size="lite")
    m.eval()
    ids = torch.randint(0, lite_config.vocab_size, (1, 10))

    # First forward
    logits, aux, past_kvs, _ = m(ids, use_cache=True)

    # Second forward with single token
    next_ids = torch.randint(0, lite_config.vocab_size, (1, 1))
    logits2, aux2, past_kvs2, _ = m(next_ids, past_key_values=past_kvs, use_cache=True)

    assert past_kvs2[0][0].shape[2] == 11  # Cache grew by 1
    print("✅ KV-Cache incremental generation OK")


def test_speculative():
    m = KimiK3Plus(lite_config, size="lite")
    draft = DraftModel(lite_config)
    dec = SpeculativeDecoder(m, draft, lite_config)
    ids = torch.randint(0, lite_config.vocab_size, (1, 10))
    out, rate, speed = dec.generate(ids, 20)
    assert out.shape[1] > ids.shape[1]
    print(f"✅ Speculative OK (rate:{rate:.2f}, speed:{speed:.1f})")


def test_rag():
    rag = RAGModule(lite_config)
    h = torch.randn(2, 10, lite_config.hidden_size)
    out = rag(h)
    assert "confidence" in out
    assert "knowledge_emb" in out
    assert out["hidden_states"].shape == h.shape  # Fused output
    print("✅ RAG OK (with fusion)")


def test_cot():
    cot = CoTCompiler(lite_config)
    h = torch.randn(1, 10, lite_config.hidden_size)
    out = cot.compile_reasoning(h)
    assert "max_steps" in out
    assert "should_continue" in out
    assert "should_answer" in out
    print("✅ CoT OK (with step control)")


def test_budget():
    rb = ReasoningBudgetController(lite_config)
    h = torch.randn(1, 10, lite_config.hidden_size)
    budget, diff, conf = rb.get_budget(h, 4096)
    assert budget > 0
    assert diff in ["simple", "medium", "hard"]
    print(f"✅ Budget OK (budget:{budget}, diff:{diff}, conf:{conf:.2f})")


def test_agentic():
    a = AgenticLayer(lite_config)
    h = torch.randn(1, 10, lite_config.hidden_size)
    out = a(h)
    assert "tools" in out
    assert "confidence" in out["tools"]
    print("✅ Agentic OK (with confidence)")


def test_memory():
    mem = LongTermMemory(lite_config)
    k = torch.randn(lite_config.hidden_size)
    v = torch.randn(100, lite_config.hidden_size)
    mem.add(k, v, 1.5)
    r = mem.retrieve(k, 3)
    assert len(r) > 0
    print("✅ Memory OK")


def test_multimodal():
    mm = MultimodalFusion(lite_config)
    h = torch.randn(1, 10, lite_config.hidden_size)
    out = mm(h)
    assert out.shape == h.shape
    print("✅ Multimodal OK")


def test_multimodal_vision():
    mm = MultimodalFusion(lite_config)
    images = torch.randn(1, 3, 224, 224)
    vision_feat = mm.encode_images(images)
    assert vision_feat is not None
    assert vision_feat.shape[0] == 1
    print(f"✅ Vision Encoder OK (shape: {vision_feat.shape})")


def test_adaptive_reasoning():
    ar = AdaptiveReasoningController(lite_config)
    h = torch.randn(1, 10, lite_config.hidden_size)
    budget, level, level_idx, conf = ar.get_budget(h)
    assert budget > 0
    assert level in ["minimal", "low", "medium", "high", "max"]
    print(f"✅ Adaptive Reasoning OK (level:{level}, budget:{budget}, conf:{conf:.2f})")


def test_online_learning():
    ol = OnlineLearningManager(lite_config)
    adapter = ol.get_user_adapter("test_user")
    assert adapter is not None
    h = torch.randn(1, 10, lite_config.hidden_size)
    adapted = adapter(h)
    assert adapted.shape == h.shape
    print("✅ Online Learning OK")


def test_safety():
    safety = SafetyAlignmentLayer(lite_config)
    h = torch.randn(1, 10, lite_config.hidden_size)
    out = safety(h)
    assert "safety" in out
    assert "values" in out
    assert "is_harmful" in out["safety"]
    print("✅ Safety Alignment OK (with intervention check)")


def test_compression():
    comp = CompressionManager(lite_config)
    k = torch.randn(1, 4, 100, lite_config.hidden_size)
    v = torch.randn(1, 4, 100, lite_config.hidden_size)
    ck, cv = comp.compress_kv_cache(k, v, 100)
    assert ck.shape[2] < k.shape[2]
    print(f"✅ Compression OK ({k.shape[2]} -> {ck.shape[2]} tokens)")


def test_flash_attention():
    from src.attention import KimiDeltaAttention
    attn = KimiDeltaAttention(lite_config)
    h = torch.randn(1, 128, lite_config.hidden_size)
    out, residual, present_kv = attn(h, 0, use_cache=False)
    assert out.shape == h.shape
    print("✅ FlashAttention OK")


def test_rope():
    from src.attention import RotaryPositionalEmbedding, apply_rotary_pos_emb
    rope = RotaryPositionalEmbedding(64, max_seq_len=2048)
    q = torch.randn(1, 8, 128, 64)
    k = torch.randn(1, 8, 128, 64)
    cos, sin = rope(q, seq_len=128)
    q_embed, k_embed = apply_rotary_pos_emb(q, k, cos, sin)
    assert q_embed.shape == q.shape
    assert k_embed.shape == k.shape
    print("✅ RoPE OK")


if __name__ == "__main__":
    print("🧪 K3+ v4.1 Production Test Suite")
    print("=" * 50)

    tests = [
        test_sizes,
        test_forward,
        test_forward_with_cache,
        test_kv_cache_generation,
        test_flash_attention,
        test_rope,
        test_speculative,
        test_rag,
        test_cot,
        test_budget,
        test_agentic,
        test_memory,
        test_multimodal,
        test_multimodal_vision,
        test_adaptive_reasoning,
        test_online_learning,
        test_safety,
        test_compression,
    ]

    passed = 0
    failed = 0

    for test in tests:
        try:
            test()
            passed += 1
        except Exception as e:
            print(f"❌ {test.__name__} FAILED: {e}")
            failed += 1

    print("=" * 50)
    print(f"🎉 {passed}/{passed+failed} tests passed!")
    if failed > 0:
        print(f"⚠️  {failed} tests failed")
