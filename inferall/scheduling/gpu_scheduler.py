"""
GPU Scheduler
--------------
GPU-aware request routing for multi-GPU systems.

Tracks which models are loaded on which GPUs and routes requests
to the least-loaded instance. Supports model replication across
multiple GPUs for parallel serving.
"""

import logging
from dataclasses import dataclass, field
from typing import Dict, List, Optional

logger = logging.getLogger(__name__)


@dataclass
class ModelInstance:
    """A single loaded instance of a model on a specific GPU."""

    model_id: str
    gpu_id: int
    active_requests: int = 0
    total_served: int = 0


class GPUScheduler:
    """
    Routes requests to the least-loaded model instance across GPUs.

    Supports:
    - Single model on one GPU (default)
    - Same model replicated on multiple GPUs
    - Routing to least-loaded instance
    """

    def __init__(self):
        # model_id → list of ModelInstance
        self._instances: Dict[str, List[ModelInstance]] = {}

    def register_instance(self, model_id: str, gpu_id: int) -> None:
        """Register a model instance on a GPU."""
        if model_id not in self._instances:
            self._instances[model_id] = []

        # Don't duplicate
        for inst in self._instances[model_id]:
            if inst.gpu_id == gpu_id:
                return

        self._instances[model_id].append(ModelInstance(model_id=model_id, gpu_id=gpu_id))
        logger.info("Registered instance: %s on GPU %d", model_id, gpu_id)

    def unregister_instance(self, model_id: str, gpu_id: int = None) -> None:
        """Remove a model instance. If gpu_id is None, removes all instances."""
        if model_id not in self._instances:
            return
        if gpu_id is None:
            del self._instances[model_id]
        else:
            self._instances[model_id] = [
                i for i in self._instances[model_id] if i.gpu_id != gpu_id
            ]
            if not self._instances[model_id]:
                del self._instances[model_id]

    def get_best_instance(self, model_id: str) -> Optional[ModelInstance]:
        """
        Get the least-loaded instance of a model.

        Returns None if no instances are registered.
        Picks the instance with fewest active requests,
        breaking ties by total served (prefer less-used).
        """
        instances = self._instances.get(model_id, [])
        if not instances:
            return None

        return min(instances, key=lambda i: (i.active_requests, i.total_served))

    def acquire(self, model_id: str) -> Optional[int]:
        """
        Acquire a slot on the best instance. Returns the GPU ID.
        Increments active_requests. Returns None if no instances.
        """
        inst = self.get_best_instance(model_id)
        if inst is None:
            return None
        inst.active_requests += 1
        return inst.gpu_id

    def release(self, model_id: str, gpu_id: int) -> None:
        """Release a slot on a specific instance."""
        for inst in self._instances.get(model_id, []):
            if inst.gpu_id == gpu_id:
                inst.active_requests = max(0, inst.active_requests - 1)
                inst.total_served += 1
                return

    def get_instance_count(self, model_id: str) -> int:
        """Get the number of instances for a model."""
        return len(self._instances.get(model_id, []))

    def get_all_instances(self) -> Dict[str, List[dict]]:
        """Get all instances for monitoring."""
        return {
            mid: [
                {
                    "gpu_id": i.gpu_id,
                    "active_requests": i.active_requests,
                    "total_served": i.total_served,
                }
                for i in instances
            ]
            for mid, instances in self._instances.items()
        }
