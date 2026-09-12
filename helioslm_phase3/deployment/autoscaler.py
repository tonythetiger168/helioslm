"""Auto-scaling strategies for HeliosLM deployments."""
import time
from typing import Dict, Optional
from dataclasses import dataclass


@dataclass
class ScalingDecision:
    action: str  # "scale_up", "scale_down", "maintain"
    target_replicas: int
    reason: str


class GPUAutoscaler:
    """
    Scale based on GPU utilization metrics.

    Scale up when GPU util > threshold for sustained period.
    Scale down when GPU util < threshold for sustained period.
    """

    def __init__(
        self,
        target_utilization: float = 0.7,
        scale_up_threshold: float = 0.8,
        scale_down_threshold: float = 0.3,
        scale_up_cooldown: int = 60,
        scale_down_cooldown: int = 300,
        scale_up_step: int = 1,
        scale_down_step: int = 1,
        min_replicas: int = 1,
        max_replicas: int = 10,
    ):
        self.target_utilization = target_utilization
        self.scale_up_threshold = scale_up_threshold
        self.scale_down_threshold = scale_down_threshold
        self.scale_up_cooldown = scale_up_cooldown
        self.scale_down_cooldown = scale_down_cooldown
        self.scale_up_step = scale_up_step
        self.scale_down_step = scale_down_step
        self.min_replicas = min_replicas
        self.max_replicas = max_replicas

        self.last_scale_up = 0
        self.last_scale_down = 0
        self.utilization_history: list = []

    def evaluate(self, current_replicas: int, gpu_utils: list) -> ScalingDecision:
        """Evaluate scaling need based on GPU utilization."""
        avg_util = sum(gpu_utils) / max(len(gpu_utils), 1)
        self.utilization_history.append(avg_util)
        self.utilization_history = self.utilization_history[-20:]  # Keep last 20

        now = time.time()

        # Check scale up
        if avg_util > self.scale_up_threshold:
            if now - self.last_scale_up > self.scale_up_cooldown:
                if current_replicas < self.max_replicas:
                    target = min(current_replicas + self.scale_up_step, self.max_replicas)
                    self.last_scale_up = now
                    return ScalingDecision(
                        action="scale_up",
                        target_replicas=target,
                        reason=f"GPU util {avg_util:.1%} > threshold {self.scale_up_threshold:.1%}",
                    )

        # Check scale down
        if avg_util < self.scale_down_threshold:
            if now - self.last_scale_down > self.scale_down_cooldown:
                if current_replicas > self.min_replicas:
                    target = max(current_replicas - self.scale_down_step, self.min_replicas)
                    self.last_scale_down = now
                    return ScalingDecision(
                        action="scale_down",
                        target_replicas=target,
                        reason=f"GPU util {avg_util:.1%} < threshold {self.scale_down_threshold:.1%}",
                    )

        return ScalingDecision(
            action="maintain",
            target_replicas=current_replicas,
            reason=f"GPU util {avg_util:.1%} within normal range",
        )


class RequestQueueAutoscaler:
    """
    Scale based on request queue depth.

    More responsive than GPU-based scaling for bursty traffic.
    """

    def __init__(
        self,
        target_queue_depth: int = 10,
        max_queue_depth: int = 50,
        min_queue_depth: int = 2,
        scale_up_step: int = 2,
        scale_down_step: int = 1,
        min_replicas: int = 1,
        max_replicas: int = 10,
    ):
        self.target_queue_depth = target_queue_depth
        self.max_queue_depth = max_queue_depth
        self.min_queue_depth = min_queue_depth
        self.scale_up_step = scale_up_step
        self.scale_down_step = scale_down_step
        self.min_replicas = min_replicas
        self.max_replicas = max_replicas

    def evaluate(self, current_replicas: int, queue_depth: int) -> ScalingDecision:
        """Evaluate scaling need based on queue depth."""

        if queue_depth > self.max_queue_depth:
            # Urgent scale up
            target = min(current_replicas + self.scale_up_step, self.max_replicas)
            return ScalingDecision(
                action="scale_up",
                target_replicas=target,
                reason=f"Queue depth {queue_depth} > max {self.max_queue_depth}",
            )

        if queue_depth > self.target_queue_depth:
            # Gradual scale up
            target = min(current_replicas + 1, self.max_replicas)
            return ScalingDecision(
                action="scale_up",
                target_replicas=target,
                reason=f"Queue depth {queue_depth} > target {self.target_queue_depth}",
            )

        if queue_depth < self.min_queue_depth and current_replicas > self.min_replicas:
            # Scale down
            target = max(current_replicas - self.scale_down_step, self.min_replicas)
            return ScalingDecision(
                action="scale_down",
                target_replicas=target,
                reason=f"Queue depth {queue_depth} < min {self.min_queue_depth}",
            )

        return ScalingDecision(
            action="maintain",
            target_replicas=current_replicas,
            reason=f"Queue depth {queue_depth} within target range",
        )
