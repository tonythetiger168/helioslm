"""Kimi K3+ v2.0 完整测试套件"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

import torch
from configs import ultra_config, pro_config, lite_config, nano_config
from model import KimiK3Plus
from attention import KimiDeltaAttention
from moe import StableLatentMoE
from speculative_decoding import DraftModel, SpeculativeDecoder
from multimodal import UnifiedMultimodalEncoder
from agentic import AgenticLayer
from memory import LongTermMemory


def test_model_sizes():
    """测试四个尺寸模型初始化"""
    configs = {
        "ultra": ultra_config,
        "pro": pro_config,
        "lite": lite_config,
        "nano": nano_config,
    }

    for name, config in configs.items():
        model = KimiK3Plus(config, size=name)
        total = sum(p.numel() for p in model.parameters())
        print(f"  {name.upper()}: {total:,} parameters")
        assert model is not None
    print("✅ 四尺寸模型初始化测试通过")


def test_forward_pass():
    """测试前向传播"""
    model = KimiK3Plus(lite_config, size="lite")
    input_ids = torch.randint(0, lite_config.vocab_size, (2, 128))
    logits, aux_loss = model(input_ids)
    assert logits.shape == (2, 128, lite_config.vocab_size)
    assert aux_loss >= 0
    print("✅ 前向传播测试通过")


def test_attention():
    """测试 KDA 注意力"""
    attn = KimiDeltaAttention(lite_config)
    hidden = torch.randn(2, 128, lite_config.hidden_size)
    output, residual = attn(hidden, layer_idx=0)
    assert output.shape == hidden.shape
    print("✅ KDA 注意力测试通过")


def test_moe_dynamic_sparsity():
    """测试动态稀疏度 MoE"""
    moe = StableLatentMoE(lite_config)
    hidden = torch.randn(2, 64, lite_config.hidden_size)
    output, aux_loss = moe(hidden)
    assert output.shape == hidden.shape
    assert aux_loss >= 0
    print("✅ 动态稀疏度 MoE 测试通过")


def test_speculative_decoding():
    """测试推测解码"""
    model = KimiK3Plus(lite_config, size="lite")
    draft_model = DraftModel(lite_config)
    decoder = SpeculativeDecoder(model, draft_model, lite_config)

    input_ids = torch.randint(0, lite_config.vocab_size, (1, 10))
    output, rate = decoder.generate(input_ids, max_new_tokens=20)
    assert output.shape[1] > input_ids.shape[1]
    print(f"✅ 推测解码测试通过 (接受率: {rate:.2f})")


def test_multimodal():
    """测试多模态编码器"""
    encoder = UnifiedMultimodalEncoder(lite_config)

    # 文本
    text_ids = torch.randint(0, lite_config.vocab_size, (2, 32))
    out = encoder(input_ids=text_ids)
    assert out is not None

    # 图像
    images = torch.randn(2, 3, 448, 448)
    out = encoder(images=images)
    assert out is not None
    print("✅ 多模态编码器测试通过")


def test_agentic():
    """测试 Agentic 层"""
    agentic = AgenticLayer(lite_config)
    hidden = torch.randn(1, 10, lite_config.hidden_size)
    outputs = agentic(hidden, task_goal=hidden)
    assert "plan" in outputs
    print("✅ Agentic 层测试通过")


def test_memory():
    """测试长期记忆"""
    memory = LongTermMemory(lite_config)
    key = torch.randn(lite_config.hidden_size)
    value = torch.randn(100, lite_config.hidden_size)
    memory.add(key, value, importance=1.5)

    retrieved = memory.retrieve(key, top_k=3)
    assert len(retrieved) > 0
    print("✅ 长期记忆测试通过")


def test_quantization_configs():
    """测试量化配置"""
    q = lite_config.quantization
    assert q.cloud_weight_bits == 4
    assert q.edge_weight_bits == 4
    assert q.mobile_weight_bits == 3
    print("✅ 量化配置测试通过")


def main():
    print("🧪 Kimi K3+ v2.0 测试套件")
    print("=" * 50)

    test_model_sizes()
    test_forward_pass()
    test_attention()
    test_moe_dynamic_sparsity()
    test_speculative_decoding()
    test_multimodal()
    test_agentic()
    test_memory()
    test_quantization_configs()

    print("\n" + "=" * 50)
    print("🎉 所有测试通过！")


if __name__ == "__main__":
    main()
