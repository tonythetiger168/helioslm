"""Phase 3 Inference Configuration"""

# Model serving
MODEL_CONFIG = {
    "model_path": "checkpoints/dpo_final.pt",
    "model_size": "ultra",
    "max_batch_size": 32,
    "max_seq_len": 32768,
    "max_num_seqs": 256,
    "block_size": 16,
    "num_blocks": 1024,
}

# Performance
PERFORMANCE_CONFIG = {
    "tensor_parallel": 8,
    "pipeline_parallel": 1,
    "speculative_decoding": True,
    "max_draft_tokens": 5,
    "prefix_cache": True,
    "max_cache_size": 100,
}

# Quantization
QUANTIZATION_CONFIG = {
    "enabled": False,
    "method": "fp8",  # fp8, int8, awq, gptq, dynamic
    "calibration_data": None,
}

# API
API_CONFIG = {
    "rest_port": 8000,
    "grpc_port": 50051,
    "metrics_port": 9090,
    "host": "0.0.0.0",
}

# Safety
SAFETY_CONFIG = {
    "enable_moderation": True,
    "enable_audit_log": True,
    "max_prompt_length": 100000,
    "max_tokens_limit": 8192,
    "rate_limit_rpm": 60,
    "rate_limit_tpm": 100000,
}

# Monitoring
MONITORING_CONFIG = {
    "prometheus_enabled": True,
    "grafana_enabled": True,
    "log_level": "INFO",
}
