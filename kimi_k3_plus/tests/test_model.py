"""单元测试"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

import torch
from configs.model_config import default_config
from model import KimiK3Plus
from attention import KimiDeltaAttention
from moe import StableLatentMoE


def test_model_initialization():
    model = KimiK3Plus(default_config)
    assert model is not None
    print("模型初始化测试通过")


def test_forward_pass():
    model = KimiK3Plus(default_config)
    input_ids = torch.randint(0, default_config.vocab_size, (2, 128))
    logits, aux_loss = model(input_ids)
    assert logits.shape == (2, 128, default_config.vocab_size)
    assert aux_loss >= 0
    print("前向传播测试通过")


def test_attention():
    attn = KimiDeltaAttention(default_config)
    hidden = torch.randn(2, 128, default_config.hidden_size)
    output, residual = attn(hidden, layer_idx=0)
    assert output.shape == hidden.shape
    print("注意力模块测试通过")


def test_moe():
    moe = StableLatentMoE(default_config)
    hidden = torch.randn(2, 128, default_config.hidden_size)
    output, aux_loss = moe(hidden)
    assert output.shape == hidden.shape
    assert aux_loss >= 0
    print("MoE 模块测试通过")


if __name__ == "__main__":
    test_model_initialization()
    test_forward_pass()
    test_attention()
    test_moe()
    print("所有测试通过！")
