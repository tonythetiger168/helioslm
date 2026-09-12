"""Integration tests for Phase 2 core modules.

Tests:
  1. Expert Parallelism: verify All-to-All dispatch/combine
  2. Ring Attention: verify blockwise attention correctness
  3. Quality Classifier: verify scoring pipeline
  4. End-to-end: data pipeline → model forward
"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

import torch
import pytest


class TestExpertParallelism:
    """Test expert parallelism components."""

    def test_all_to_all_single(self):
        """Test basic all-to-all communication."""
        from training.expert_parallelism.alltoall_router import all_to_all
        # Mock test (no distributed env): should return input unchanged
        x = torch.randn(4, 8)
        result = all_to_all(x)
        assert result.shape == x.shape

    def test_load_balance_loss(self):
        """Test load balance loss computation."""
        from training.expert_parallelism.load_balancer import compute_load_balance_loss

        router_probs = torch.softmax(torch.randn(100, 8), dim=-1)
        expert_indices = torch.topk(router_probs, 2, dim=-1).indices

        loss = compute_load_balance_loss(router_probs, expert_indices, num_experts=8)
        assert loss.item() >= 0
        assert loss.item() < 1.0

    def test_expert_assignment(self):
        """Test expert assignment logic."""
        from training.expert_parallelism.expert_parallel_group import get_expert_assignment

        experts = get_expert_assignment(256, 8, 3)
        assert len(experts) == 32  # 256 / 8
        assert experts[0] == 96    # 3 * 32
        assert experts[-1] == 127


class TestRingAttention:
    """Test Ring Attention components."""

    def test_sequence_split_gather(self):
        """Test sequence splitting and gathering."""
        from training.ring_attention.sequence_parallel import split_sequence, gather_sequence
        # Mock (no distributed): should be identity
        x = torch.randn(2, 16, 4, 64)
        split = split_sequence(x, dim=1)
        # With cp_size=1, should return full tensor
        assert split.shape == x.shape

    def test_blockwise_attention(self):
        """Test that blockwise attention produces same result as standard."""
        from training.ring_attention.ring_attention import _process_kv_block

        q = torch.randn(1, 8, 2, 64)
        k = torch.randn(1, 8, 2, 64)
        v = torch.randn(1, 8, 2, 64)

        # Standard attention
        scores = torch.einsum('bqhd,bkhd->bhqk', q, k) / 8.0
        attn = torch.softmax(scores, dim=-1)
        expected = torch.einsum('bhqk,bkhd->bqhd', attn, v)

        # Blockwise (single block, no causal)
        output = torch.zeros_like(q)
        m = torch.full((1, 8, 2, 1), float('-inf'))
        l = torch.zeros((1, 8, 2, 1))

        _process_kv_block(q, k, v, output, m, l, 1/8.0, False, 0, 0, 1, 8)
        result = output / l

        assert torch.allclose(result, expected, atol=1e-4)


class TestQualityClassifier:
    """Test quality classifier components."""

    def test_model_forward(self):
        """Test classifier forward pass."""
        from data_pipeline.quality_classifier.classifier_model import QualityClassifier

        model = QualityClassifier(vocab_size=1000, hidden_size=256, num_layers=2, num_heads=4)
        input_ids = torch.randint(0, 1000, (2, 32))
        attention_mask = torch.ones(2, 32)

        scores = model(input_ids, attention_mask)

        assert "grammar" in scores
        assert "composite" in scores
        assert scores["composite"].shape == (2,)
        assert torch.all((scores["composite"] >= 0) & (scores["composite"] <= 1))

    def test_auto_labeling(self):
        """Test automatic labeling."""
        from data_pipeline.quality_classifier.data_labeling import auto_label_documents

        text = "The quick brown fox jumps over the lazy dog. " * 10
        labels = auto_label_documents(text)

        assert "grammar" in labels
        assert "composite" in labels
        assert 0 <= labels["composite"] <= 1

    def test_quality_dimensions(self):
        """Test QualityDimensions dataclass."""
        from data_pipeline.quality_classifier.classifier_model import QualityDimensions

        dims = QualityDimensions(grammar=0.9, knowledge=0.8, coherence=0.7, toxicity=0.9, diversity=0.6)
        composite = dims.composite
        expected = 0.25*0.9 + 0.25*0.8 + 0.20*0.7 + 0.15*0.9 + 0.15*0.6
        assert abs(composite - expected) < 1e-6


class TestEndToEnd:
    """End-to-end integration tests."""

    def test_data_pipeline_flow(self):
        """Test full data pipeline: ingestion → filter → dedup → score."""
        from data_pipeline.filtering import LanguageFilter, CompositeFilter
        from data_pipeline.deduplication import ExactDeduplicator

        filter_fn = CompositeFilter([LanguageFilter()])
        dedup_fn = ExactDeduplicator()

        docs = [
            {"text": "This is a high quality document with proper grammar and structure." * 5},
            {"text": "Short."},  # Should be filtered
            {"text": "This is a high quality document with proper grammar and structure." * 5},  # Duplicate
            {"text": "Another unique document about machine learning and artificial intelligence." * 5},
        ]

        results = []
        for doc in docs:
            if not filter_fn(doc):
                continue
            if not dedup_fn(doc):
                continue
            results.append(doc)

        assert len(results) == 2  # First and last

    def test_moe_with_ep(self):
        """Test MoE layer with expert parallelism (mock)."""
        from src.moe import StableLatentMoE
        from configs.base_config import HeliosLMConfig

        config = HeliosLMConfig()
        config.hidden_size = 256
        config.moe.num_experts = 8
        config.moe.num_activated_experts = 2
        config.moe.expert_hidden_size = 512
        config.moe.num_shared_experts = 1
        config.moe.min_experts = 1
        config.moe.max_experts = 4
        config.moe.load_balance_loss_coef = 0.01
        config.moe.capacity_factor = 1.25
        config.moe.domain_grouping = False

        moe = StableLatentMoE(config)
        x = torch.randn(2, 16, 256)

        out, aux_loss = moe(x)
        assert out.shape == x.shape
        assert aux_loss.item() >= 0


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
