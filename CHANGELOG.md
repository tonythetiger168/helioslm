# HeliosLM v4.1 Changelog

## v4.1.0 (2026-09-08) - Production Release

### Fixed (from v4.0)
- **GQA Bug**: FlashAttention now properly repeats KV heads to match Q heads
- **KV-Cache**: Full incremental generation support, O(1) per-step
- **Module Fusion**: P1-P7 outputs now fused into hidden states via gating
- **MoE Routing**: Vectorized expert routing, eliminated Python loops
- **Dynamic Sparsity**: Actually uses difficulty-based k adjustment
- **Multimodal Position**: Fusion moved to embedding layer (LLaVA standard)
- **Safety Intervention**: Generation loop checks is_harmful and redirects
- **Tool Loop**: Generation detects tool_scores and triggers execution
- **Compression**: INT4 packed storage, real KV compression, YaRN context extension
- **Tokenizer**: Real chat template, special tokens, batch encoding

### Added
- **RoPE**: Rotary Position Embeddings with dynamic scaling
- **SDPA**: torch.nn.functional.scaled_dot_product_attention
- **Top-p Sampling**: Nucleus sampling in generation
- **generate_from_text**: String-based generation API
- **Streaming Callback**: Token-by-token streaming support
- **DeepSpeed Config**: ZeRO-1/2/3 with CPU offload
- **FSDP Config**: FULL_SHARD / SHARD_GRAD_OP strategies
- **Docker**: CUDA 12.1 production container
- **K8s**: Deployment + Service + HPA autoscaling
- **OpenAI API**: /v1/chat/completions, /v1/completions, /v1/embeddings
- **Rate Limiting**: Token bucket per API key
- **Authentication**: Bearer token validation

### Total Files
- 99 Python modules
- 4 model configurations
- 3 deployment manifests
- 18 integration tests
