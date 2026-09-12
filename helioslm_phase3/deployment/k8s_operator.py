"""Kubernetes Operator for HeliosLM model serving.

Manages model lifecycle:
  - Deployment creation and scaling
  - Model loading and warm-up
  - Health checks and readiness probes
  - Configuration updates
"""
from dataclasses import dataclass
from typing import Dict, List, Optional


@dataclass
class HeliosLMModelSpec:
    """Specification for a HeliosLM model deployment."""
    name: str
    model_path: str
    model_size: str = "ultra"  # ultra, pro, lite, nano
    replicas: int = 1
    gpu_per_replica: int = 8
    memory_per_gpu: str = "80Gi"
    cpu_per_replica: str = "32"
    max_batch_size: int = 32
    max_seq_len: int = 32768
    quantization: Optional[str] = None  # fp8, int8, awq, gptq
    tensor_parallel: int = 8
    pipeline_parallel: int = 1

    # Auto-scaling
    min_replicas: int = 1
    max_replicas: int = 10
    target_gpu_utilization: float = 0.7
    target_queue_depth: int = 10


class HeliosLMOperator:
    """
    Kubernetes operator for managing HeliosLM deployments.

    In production, this would use kopf or the Kubernetes Python client
    to watch for CRD changes and manage Deployments/Services.
    """

    def __init__(self, namespace: str = "helioslm"):
        self.namespace = namespace
        self.deployments: Dict[str, HeliosLMModelSpec] = {}

    def create_deployment(self, spec: HeliosLMModelSpec) -> str:
        """Create a new model deployment."""
        deployment_name = f"helioslm-{spec.name}"

        # Generate Kubernetes manifests
        manifests = self._generate_manifests(spec)

        # In real implementation, apply to cluster
        # kubectl apply -f manifests

        self.deployments[spec.name] = spec

        print(f"🚀 Created deployment: {deployment_name}")
        print(f"   Replicas: {spec.replicas}")
        print(f"   GPUs: {spec.replicas * spec.gpu_per_replica}x {spec.memory_per_gpu}")
        print(f"   Model: {spec.model_path}")

        return deployment_name

    def _generate_manifests(self, spec: HeliosLMModelSpec) -> Dict:
        """Generate Kubernetes Deployment and Service manifests."""

        deployment = {
            "apiVersion": "apps/v1",
            "kind": "Deployment",
            "metadata": {
                "name": f"helioslm-{spec.name}",
                "namespace": self.namespace,
                "labels": {"app": "helioslm", "model": spec.name},
            },
            "spec": {
                "replicas": spec.replicas,
                "selector": {"matchLabels": {"app": "helioslm", "model": spec.name}},
                "template": {
                    "metadata": {"labels": {"app": "helioslm", "model": spec.name}},
                    "spec": {
                        "containers": [{
                            "name": "helioslm",
                            "image": "helioslm/inference:v1.0.0",
                            "command": ["python", "-m", "api.rest_server"],
                            "args": [
                                f"--model-path={spec.model_path}",
                                f"--model-size={spec.model_size}",
                                f"--tensor-parallel={spec.tensor_parallel}",
                                f"--max-batch-size={spec.max_batch_size}",
                            ],
                            "resources": {
                                "limits": {
                                    "nvidia.com/gpu": str(spec.gpu_per_replica),
                                    "memory": spec.memory_per_gpu,
                                    "cpu": spec.cpu_per_replica,
                                },
                                "requests": {
                                    "nvidia.com/gpu": str(spec.gpu_per_replica),
                                    "memory": spec.memory_per_gpu,
                                    "cpu": spec.cpu_per_replica,
                                },
                            },
                            "ports": [{"containerPort": 8000, "name": "http"}],
                            "livenessProbe": {
                                "httpGet": {"path": "/health", "port": 8000},
                                "initialDelaySeconds": 60,
                                "periodSeconds": 10,
                            },
                            "readinessProbe": {
                                "httpGet": {"path": "/health", "port": 8000},
                                "initialDelaySeconds": 30,
                                "periodSeconds": 5,
                            },
                            "env": [
                                {"name": "CUDA_VISIBLE_DEVICES", "value": ",".join(map(str, range(spec.gpu_per_replica)))},
                                {"name": "NCCL_DEBUG", "value": "WARN"},
                            ],
                        }],
                        "nodeSelector": {
                            "node-type": "gpu",
                        },
                        "tolerations": [{
                            "key": "nvidia.com/gpu",
                            "operator": "Exists",
                            "effect": "NoSchedule",
                        }],
                    },
                },
            },
        }

        service = {
            "apiVersion": "v1",
            "kind": "Service",
            "metadata": {
                "name": f"helioslm-{spec.name}",
                "namespace": self.namespace,
            },
            "spec": {
                "selector": {"app": "helioslm", "model": spec.name},
                "ports": [{"port": 80, "targetPort": 8000}],
                "type": "ClusterIP",
            },
        }

        hpa = {
            "apiVersion": "autoscaling/v2",
            "kind": "HorizontalPodAutoscaler",
            "metadata": {
                "name": f"helioslm-{spec.name}-hpa",
                "namespace": self.namespace,
            },
            "spec": {
                "scaleTargetRef": {
                    "apiVersion": "apps/v1",
                    "kind": "Deployment",
                    "name": f"helioslm-{spec.name}",
                },
                "minReplicas": spec.min_replicas,
                "maxReplicas": spec.max_replicas,
                "metrics": [
                    {
                        "type": "Pods",
                        "pods": {
                            "metric": {"name": "gpu_utilization"},
                            "target": {"type": "AverageValue", "averageValue": f"{int(spec.target_gpu_utilization * 100)}"},
                        },
                    },
                    {
                        "type": "Pods",
                        "pods": {
                            "metric": {"name": "request_queue_depth"},
                            "target": {"type": "AverageValue", "averageValue": str(spec.target_queue_depth)},
                        },
                    },
                ],
            },
        }

        return {"deployment": deployment, "service": service, "hpa": hpa}

    def update_deployment(self, name: str, new_spec: HeliosLMModelSpec):
        """Update an existing deployment."""
        if name not in self.deployments:
            raise ValueError(f"Deployment {name} not found")

        # Rolling update
        old_spec = self.deployments[name]

        # Strategy: scale up new version, then scale down old
        print(f"🔄 Rolling update: {name}")
        print(f"   {old_spec.model_path} -> {new_spec.model_path}")

        self.deployments[name] = new_spec

    def delete_deployment(self, name: str):
        """Delete a deployment."""
        if name in self.deployments:
            del self.deployments[name]
            print(f"🗑️  Deleted deployment: {name}")

    def list_deployments(self) -> List[str]:
        """List all deployments."""
        return list(self.deployments.keys())
