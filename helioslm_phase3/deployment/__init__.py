"""HeliosLM Phase 3 — Deployment Infrastructure

Kubernetes-native deployment:
  - Custom Resource Definition (CRD) for HeliosLM models
  - Auto-scaling based on request queue depth and GPU utilization
  - Rolling updates with zero-downtime
  - Multi-zone deployment for high availability
"""
from .k8s_operator import HeliosLMOperator, HeliosLMModelSpec
from .autoscaler import GPUAutoscaler, RequestQueueAutoscaler
from .rolling_update import RollingUpdateManager

__all__ = [
    "HeliosLMOperator", "HeliosLMModelSpec",
    "GPUAutoscaler", "RequestQueueAutoscaler",
    "RollingUpdateManager",
]
