"""HeliosLM v1.0 Tests - P0+P1+P2+P3 Production Modules"""

import sys
import os
import json as json_mod
import tempfile
import torch
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from configs.base_config import HeliosLMConfig
from src import HeliosLM, HeliosAttention, StableLatentMoE
from src.speculative_decoding import DraftModel, SpeculativeDecoder
from src.rag import RAGModule
from src.cot_compiler import CoTCompiler
from src.agentic import AgenticLayer
from src.memory import LongTermMemory
from src.tokenizer import HeliosTokenizer
from src.data_pipeline import PretrainDataset, SFTDataset, create_dataloader


# Tiny test config to avoid OOM in test environment
test_config = HeliosLMConfig(
    model_name="HeliosLM-Test",
    hidden_size=256,
    num_hidden_layers=2,
    max_position_embeddings=1024,
    intermediate_size=512,
)
test_config.moe.num_experts = 4
test_config.moe.num_activated_experts = 2
test_config.moe.expert_hidden_size = 512
test_config.moe.num_shared_experts = 1
test_config.attention.num_attention_heads = 4
test_config.attention.num_key_value_heads = 2
test_config.speculative.enabled = False


def test_attention():
    """Test P0: HeliosAttention with FlashAttention-compatible interface."""
    attn = HeliosAttention(test_config)
    x = torch.randn(2, 16, test_config.hidden_size)
    out, residual = attn(x, layer_idx=0)
    assert out.shape == x.shape
    print("  HeliosAttention: PASSED")


def test_moe():
    """Test P0+P2: StableLatentMoE with dynamic sparsity."""
    moe = StableLatentMoE(test_config)
    x = torch.randn(2, 16, test_config.hidden_size)
    out, aux_loss = moe(x, task_type="programming")
    assert out.shape == x.shape
    assert aux_loss.item() >= 0
    print("  StableLatentMoE: PASSED")


def test_rag():
    """Test P1: RAGModule with confidence calibration."""
    rag = RAGModule(test_config)
    x = torch.randn(1, 16, test_config.hidden_size)
    result = rag(x)
    assert "confidence" in result
    assert "uncertain" in result
    print("  RAGModule: PASSED")


def test_cot_compiler():
    """Test P2: CoTCompiler with structured reasoning."""
    cot = CoTCompiler(test_config)
    x = torch.randn(1, 16, test_config.hidden_size)
    result = cot.compile_reasoning(x, task_difficulty="hard")
    assert "step_type" in result
    assert "max_steps" in result
    print("  CoTCompiler: PASSED")


def test_agentic():
    """Test P3: AgenticLayer with MCP and reflection."""
    agent = AgenticLayer(test_config)
    x = torch.randn(1, 16, test_config.hidden_size)
    result = agent(x)
    assert "tools" in result
    print("  AgenticLayer: PASSED")


def test_memory():
    """Test P3: LongTermMemory with FAISS."""
    memory = LongTermMemory(test_config)
    key = torch.randn(test_config.hidden_size)
    memory.add(key, "test memory")
    results = memory.retrieve(key, top_k=1)
    assert len(results) > 0
    print("  LongTermMemory: PASSED")


def test_tokenizer():
    """Test HeliosTokenizer."""
    tokenizer = HeliosTokenizer(vocab_size=160000)
    text = "Hello world"
    tokens = tokenizer.encode(text)
    decoded = tokenizer.decode(tokens)
    assert len(tokens) > 0
    
    batch = tokenizer.batch_encode(["Hello", "World"], padding=True)
    assert "input_ids" in batch
    assert "attention_mask" in batch
    print("  HeliosTokenizer: PASSED")


def test_data_pipeline():
    """Test data pipeline components."""
    with tempfile.TemporaryDirectory() as tmpdir:
        pretrain_file = os.path.join(tmpdir, "train_0000.jsonl")
        with open(pretrain_file, "w") as f:
            for i in range(10):
                f.write(json_mod.dumps({"text": f"This is sample text number {i}. " * 50}) + "\n")
        
        dataset = PretrainDataset(tmpdir, seq_len=512)
        sample = next(iter(dataset))
        assert "input_ids" in sample
        assert "labels" in sample
        assert sample["input_ids"].shape[0] == 512
        
        sft_file = os.path.join(tmpdir, "sft.jsonl")
        with open(sft_file, "w") as f:
            for i in range(10):
                f.write(json_mod.dumps({"prompt": f"Question {i}", "response": f"Answer {i}"}) + "\n")
        
        sft_dataset = SFTDataset(sft_file, seq_len=512)
        sft_sample = next(iter(sft_dataset))
        assert "input_ids" in sft_sample
    
    print("  Data Pipeline: PASSED")


def test_full_model():
    """Test full HeliosLM forward pass with tiny config."""
    model = HeliosLM(test_config, size="test")
    input_ids = torch.randint(0, test_config.vocab_size, (1, 16))
    output = model(input_ids)
    assert "logits" in output
    assert "aux_loss" in output
    print("  HeliosLM Full Model: PASSED")


def test_model_generation():
    """Test model generation with tiny config."""
    model = HeliosLM(test_config, size="test")
    input_ids = torch.randint(0, test_config.vocab_size, (1, 8))
    output = model.generate(input_ids, max_new_tokens=5, temperature=0.7)
    assert output.shape[1] > input_ids.shape[1]
    print("  Model Generation: PASSED")


def run_all_tests():
    print("=" * 60)
    print("HeliosLM v1.0 Production Test Suite")
    print("=" * 60)
    
    tests = [
        test_attention,
        test_moe,
        test_rag,
        test_cot_compiler,
        test_agentic,
        test_memory,
        test_tokenizer,
        test_data_pipeline,
        test_full_model,
        test_model_generation,
    ]
    
    passed = 0
    failed = 0
    
    for test in tests:
        try:
            test()
            passed += 1
        except Exception as e:
            print(f"  {test.__name__}: FAILED - {e}")
            import traceback
            traceback.print_exc()
            failed += 1
    
    print("=" * 60)
    print(f"Results: {passed} passed, {failed} failed")
    print("=" * 60)
    return failed == 0


if __name__ == "__main__":
    success = run_all_tests()
    sys.exit(0 if success else 1)
