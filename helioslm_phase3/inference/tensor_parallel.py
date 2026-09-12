"""Tensor parallelism for multi-GPU inference serving.

Splits model layers across multiple GPUs for higher throughput.
"""
import torch
import torch.nn as nn
import torch.distributed as dist
from typing import Optional


class TensorParallelEngine:
    """
    Tensor parallelism wrapper for model serving.

    Splits linear layers column-wise or row-wise across GPUs.
    """

    def __init__(self, model: nn.Module, tp_size: int = 2):
        self.model = model
        self.tp_size = tp_size
        self.tp_rank = dist.get_rank() if dist.is_initialized() else 0

        if tp_size > 1:
            self._shard_model()

    def _shard_model(self):
        """Shard model weights across TP group."""
        for name, module in self.model.named_modules():
            if isinstance(module, nn.Linear):
                # Column-wise split for output layers
                if "q_proj" in name or "k_proj" in name or "v_proj" in name or "gate_proj" in name:
                    self._column_parallel_linear(module)
                # Row-wise split for input layers
                elif "o_proj" in name or "down_proj" in name:
                    self._row_parallel_linear(module)

    def _column_parallel_linear(self, linear: nn.Linear):
        """Split linear layer column-wise."""
        output_size_per_partition = linear.out_features // self.tp_size
        start = self.tp_rank * output_size_per_partition
        end = start + output_size_per_partition

        linear.weight = nn.Parameter(linear.weight[start:end, :].contiguous())
        if linear.bias is not None:
            linear.bias = nn.Parameter(linear.bias[start:end].contiguous())
        linear.out_features = output_size_per_partition

    def _row_parallel_linear(self, linear: nn.Linear):
        """Split linear layer row-wise."""
        input_size_per_partition = linear.in_features // self.tp_size
        start = self.tp_rank * input_size_per_partition
        end = start + input_size_per_partition

        linear.weight = nn.Parameter(linear.weight[:, start:end].contiguous())
        linear.in_features = input_size_per_partition

    def all_gather_output(self, output: torch.Tensor) -> torch.Tensor:
        """All-gather outputs from all TP ranks."""
        if self.tp_size == 1:
            return output

        world_size = dist.get_world_size()
        tensor_list = [torch.empty_like(output) for _ in range(world_size)]
        dist.all_gather(tensor_list, output)
        return torch.cat(tensor_list, dim=-1)
