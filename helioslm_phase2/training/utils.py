"""Training utilities for distributed setup and memory estimation."""
import os
import torch
import torch.distributed as dist


def setup_distributed():
    """Initialize distributed training."""
    if "RANK" in os.environ and "WORLD_SIZE" in os.environ:
        rank = int(os.environ["RANK"])
        world_size = int(os.environ["WORLD_SIZE"])
        local_rank = int(os.environ.get("LOCAL_RANK", 0))

        torch.cuda.set_device(local_rank)
        dist.init_process_group("nccl", rank=rank, world_size=world_size)

        print(f"🌍 Distributed initialized: rank={rank}/{world_size}, local_rank={local_rank}")
        return rank, world_size, local_rank
    else:
        return 0, 1, 0


def get_model_size(model: torch.nn.Module) -> Dict[str, float]:
    """Get model parameter counts in billions."""
    total = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)

    return {
        "total_params_B": total / 1e9,
        "trainable_params_B": trainable / 1e9,
        "total_params": total,
        "trainable_params": trainable,
    }


def estimate_memory(
    total_params: int,
    batch_size: int,
    seq_len: int,
    dtype_bytes: int = 2,  # BF16
    activation_checkpointing: bool = True,
    zero_stage: int = 3,
) ) -> Dict[str, float]:
    """
    Estimate GPU memory requirements.

    Rough formula (for ZeRO-3):
      Model params: total_params * dtype_bytes / world_size
      Optimizer states: 2 * total_params * 4 / world_size (Adam 32-bit copies)
      Gradients: total_params * dtype_bytes / world_size
      Activations: batch_size * seq_len * hidden_size * layers * 4 (if no checkpointing)
    """
    # Model weights (sharded across GPUs)
    model_mem = total_params * dtype_bytes

    # Optimizer states (full precision copies)
    optimizer_mem = total_params * 4 * 2  # 2 copies (momentum + variance)

    # Gradients
    grad_mem = total_params * dtype_bytes

    # Activations (rough estimate)
    if activation_checkpointing:
        activation_mem = batch_size * seq_len * 4096 * 4  # Simplified
    else:
        activation_mem = batch_size * seq_len * 4096 * 60 * 4

    total_mem = model_mem + optimizer_mem + grad_mem + activation_mem

    return {
        "model_GB": model_mem / 1e9,
        "optimizer_GB": optimizer_mem / 1e9,
        "gradients_GB": grad_mem / 1e9,
        "activations_GB": activation_mem / 1e9,
        "total_GB": total_mem / 1e9,
        "per_gpu_GB": total_mem / 1e9 / 256,  # Assuming 256 GPUs
    }


def print_model_info(model: torch.nn.Module):
    """Print formatted model information."""
    sizes = get_model_size(model)
    print("=" * 60)
    print("Model Information")
    print("=" * 60)
    print(f"Total parameters:     {sizes['total_params_B']:.2f}B")
    print(f"Trainable parameters: {sizes['trainable_params_B']:.2f}B")
    print("=" * 60)
