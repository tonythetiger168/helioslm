"""Distributed Training - DeepSpeed + FSDP + 3D Parallelism"""
import os
import json
from typing import Dict, Optional, List


class DeepSpeedConfigBuilder:
    """Build DeepSpeed configuration for ZeRO stages."""

    ZERO_STAGES = {
        0: "Disabled (no optimization)",
        1: "Optimizer state partitioning",
        2: "Optimizer + gradient partitioning",
        3: "Optimizer + gradient + parameter partitioning",
    }

    def __init__(self, model_size: str = "lite", zero_stage: int = 2, 
                 offload_optimizer: bool = False, offload_param: bool = False):
        self.model_size = model_size
        self.zero_stage = zero_stage
        self.offload_optimizer = offload_optimizer
        self.offload_param = offload_param

    def build(self) -> Dict:
        """Build DeepSpeed config dict."""
        config = {
            "train_batch_size": "auto",
            "train_micro_batch_size_per_gpu": "auto",
            "gradient_accumulation_steps": "auto",
            "gradient_clipping": 1.0,
            "optimizer": {
                "type": "AdamW",
                "params": {
                    "lr": 1.5e-4,
                    "betas": [0.9, 0.95],
                    "eps": 1e-8,
                    "weight_decay": 0.1,
                }
            },
            "scheduler": {
                "type": "WarmupDecayLR",
                "params": {
                    "warmup_min_lr": 0,
                    "warmup_max_lr": 1.5e-4,
                    "warmup_num_steps": 2000,
                    "total_num_steps": 100000,
                }
            },
            "zero_optimization": {
                "stage": self.zero_stage,
                "offload_optimizer": {
                    "device": "cpu" if self.offload_optimizer else "none",
                    "pin_memory": True,
                } if self.offload_optimizer else None,
                "offload_param": {
                    "device": "cpu" if self.offload_param else "none",
                    "pin_memory": True,
                } if self.offload_param else None,
                "overlap_comm": True,
                "contiguous_gradients": True,
                "sub_group_size": 1e9,
                "reduce_bucket_size": "auto",
                "stage3_prefetch_bucket_size": "auto",
                "stage3_param_persistence_threshold": "auto",
                "stage3_max_live_parameters": 1e9,
                "stage3_max_reuse_distance": 1e9,
            },
            "fp16": {
                "enabled": False,
            },
            "bf16": {
                "enabled": True,
            },
            "activation_checkpointing": {
                "partition_activations": True,
                "cpu_checkpointing": False,
                "contiguous_memory_optimization": False,
                "number_checkpoints": None,
                "synchronize_checkpoint_boundary": False,
                "profile": False,
            },
            "flops_profiler": {
                "enabled": False,
                "profile_step": 10,
                "detailed": True,
            },
            "wall_clock_breakdown": False,
        }

        # Remove None values
        config["zero_optimization"] = {k: v for k, v in config["zero_optimization"].items() if v is not None}

        return config

    def save(self, path: str):
        """Save config to JSON."""
        with open(path, "w") as f:
            json.dump(self.build(), f, indent=2)


class FSDPConfigBuilder:
    """Build PyTorch FSDP configuration."""

    def __init__(self, sharding_strategy: str = "FULL_SHARD", 
                 backward_prefetch: str = "BACKWARD_PRE",
                 cpu_offload: bool = False,
                 mixed_precision: str = "bf16"):
        self.sharding_strategy = sharding_strategy
        self.backward_prefetch = backward_prefetch
        self.cpu_offload = cpu_offload
        self.mixed_precision = mixed_precision

    def build(self) -> Dict:
        """Build FSDP config."""
        from torch.distributed.fsdp import ShardingStrategy, BackwardPrefetch

        strategy_map = {
            "FULL_SHARD": ShardingStrategy.FULL_SHARD,
            "SHARD_GRAD_OP": ShardingStrategy.SHARD_GRAD_OP,
            "NO_SHARD": ShardingStrategy.NO_SHARD,
        }

        prefetch_map = {
            "BACKWARD_PRE": BackwardPrefetch.BACKWARD_PRE,
            "BACKWARD_POST": BackwardPrefetch.BACKWARD_POST,
        }

        mp_policy = None
        if self.mixed_precision == "bf16":
            from torch.distributed.fsdp import MixedPrecision
            mp_policy = MixedPrecision(
                param_dtype=torch.bfloat16,
                reduce_dtype=torch.bfloat16,
                buffer_dtype=torch.bfloat16,
            )

        return {
            "sharding_strategy": strategy_map.get(self.sharding_strategy, ShardingStrategy.FULL_SHARD),
            "backward_prefetch": prefetch_map.get(self.backward_prefetch, BackwardPrefetch.BACKWARD_PRE),
            "cpu_offload": self.cpu_offload,
            "mixed_precision": mp_policy,
            "device_id": "auto",
            "limit_all_gathers": True,
            "use_orig_params": True,
        }


class ParallelismConfig:
    """3D Parallelism configuration (Data + Tensor + Pipeline)."""

    def __init__(self, 
                 data_parallel_size: int = 1,
                 tensor_parallel_size: int = 1,
                 pipeline_parallel_size: int = 1):
        self.dp = data_parallel_size
        self.tp = tensor_parallel_size
        self.pp = pipeline_parallel_size
        self.world_size = dp * tp * pp

    def get_device_mesh(self):
        """Get device mesh for 3D parallelism."""
        return {
            "data_parallel": list(range(0, self.world_size, self.tp * self.pp)),
            "tensor_parallel": list(range(self.pp)),
            "pipeline_parallel": list(range(self.pp)),
        }


class TrainingLauncher:
    """Unified training launcher for DeepSpeed/FSDP."""

    def __init__(self, config, model, dataloader, strategy: str = "deepspeed"):
        self.config = config
        self.model = model
        self.dataloader = dataloader
        self.strategy = strategy

    def launch(self, num_gpus: int = 1, num_nodes: int = 1):
        """Launch distributed training."""
        if self.strategy == "deepspeed":
            return self._launch_deepspeed(num_gpus, num_nodes)
        elif self.strategy == "fsdp":
            return self._launch_fsdp(num_gpus, num_nodes)
        else:
            raise ValueError(f"Unknown strategy: {self.strategy}")

    def _launch_deepspeed(self, num_gpus, num_nodes):
        """Launch with DeepSpeed."""
        import deepspeed

        ds_config = DeepSpeedConfigBuilder(
            model_size=self.config.model_name,
            zero_stage=2 if num_gpus <= 8 else 3,
        ).build()

        model_engine, optimizer, _, _ = deepspeed.initialize(
            model=self.model,
            config=ds_config,
        )

        return model_engine, optimizer

    def _launch_fsdp(self, num_gpus, num_nodes):
        """Launch with FSDP."""
        import torch.distributed as dist
        from torch.distributed.fsdp import FullyShardedDataParallel as FSDP

        dist.init_process_group("nccl")

        fsdp_config = FSDPConfigBuilder().build()

        model = FSDP(self.model, **fsdp_config)
        optimizer = torch.optim.AdamW(model.parameters(), lr=1.5e-4)

        return model, optimizer
