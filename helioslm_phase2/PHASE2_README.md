# HeliosLM Phase 2 — Pre-training Infrastructure

## Overview

Production-grade infrastructure for 15-20T token pre-training on 256x H100.

## Modules

### 1. Data Pipeline (`data_pipeline/`)

| Component | Purpose |
|-----------|---------|
| `ingestion.py` | Multi-source data ingestion (CC, GitHub, arXiv, Books) |
| `filtering.py` | Quality filtering (language, toxicity, perplexity) |
| `deduplication.py` | MinHash LSH fuzzy dedup + exact dedup |
| `tokenizer_wrapper.py` | SentencePiece/HF tokenizer interface |
| `mixer.py` | Temperature-based data blending |
| `streaming_dataset.py` | Streaming IterableDataset with sequence packing |
| `quality_classifier/` | 0.5B param model for data quality scoring |

### 2. Training (`training/`)

| Component | Purpose |
|-----------|---------|
| `deepspeed_trainer.py` | ZeRO-3 distributed training engine |
| `checkpoint_manager.py` | Auto-rotation + best tracking + cloud sync |
| `lr_scheduler.py` | Cosine / Warmup-Stable-Decay schedulers |
| `long_context.py` | Progressive 4K→1M context via YaRN |
| `expert_parallelism/` | EP for MoE: All-to-All + load balancing |
| `ring_attention/` | Blockwise attention for 1M+ context |

### 3. Alignment (`alignment/`)

| Component | Purpose |
|-----------|---------|
| `sft_trainer.py` | Supervised fine-tuning with prompt masking |
| `dpo_trainer.py` | Direct Preference Optimization |
| `reward_model.py` | Bradley-Terry reward model (RLHF backup) |

### 4. Evaluation (`evaluation/`)

| Component | Purpose |
|-----------|---------|
| `benchmarks.py` | Perplexity, HellaSwag, MMLU, GSM8K, HumanEval |
| `eval_runner.py` | Automated evaluation pipeline |

## Quick Start

```bash
# 1. Train quality classifier
python scripts/train_quality_classifier.py \
    --data data/labeled.jsonl \
    --output checkpoints/quality_classifier.pt

# 2. Score and filter data
python scripts/score_data.py \
    --input data/raw \
    --output data/filtered \
    --classifier checkpoints/quality_classifier.pt \
    --top-percent 30

# 3. Pre-train with DeepSpeed + EP
 deepspeed --num_gpus 256 scripts/pretrain.py \
    --deepspeed_config configs/deepspeed_zero3_ep.json \
    --output_dir checkpoints/pretrain

# 4. Evaluate checkpoint
python scripts/evaluate.py \
    --checkpoint checkpoints/pretrain/best/model.pt \
    --output eval_results.json
```

## Architecture Diagram

```
Data Pipeline                    Training                          Alignment
┌─────────────┐               ┌─────────────┐                  ┌─────────────┐
│ Ingestion   │──→│ Filtering   │──→│ Deduplication │──→│ Tokenization  │
└─────────────┘               └─────────────┘                  └─────────────┘
       │                              │                                │
       ↓                              ↓                                ↓
┌─────────────┐               ┌─────────────┐                  ┌─────────────┐
│ Quality     │               │ DeepSpeed   │                  │ SFT         │
│ Classifier  │──→│ ZeRO-3 + EP │──→│ DPO         │
└─────────────┘               └─────────────┘                  └─────────────┘
       │                              │                                │
       ↓                              ↓                                ↓
┌─────────────┐               ┌─────────────┐                  ┌─────────────┐
│ Data Mixer  │──→│ Ring Attn   │──→│ Eval        │
└─────────────┘               └─────────────┘                  └─────────────┘
```

## Cost Breakdown

| Phase | Resources | Duration | Cost |
|-------|-----------|----------|------|
| Data prep | 16x CPU | 2 weeks | $50K |
| Pre-train | 256x H100 | 4 months | $3.5M |
| Long context | 256x H100 | 1 month | $0.8M |
| SFT | 64x H100 | 2 weeks | $0.2M |
| DPO | 64x H100 | 1 week | $0.1M |
| Eval + buffer | 64x H100 | 2 weeks | $0.3M |
| **Total** | | **~6 months** | **~$5M** |
