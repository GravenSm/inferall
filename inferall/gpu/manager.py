"""
GPU Resource Manager
--------------------
Manages GPU enumeration, live VRAM tracking, monitoring, and allocation tracking.
Refactored from the original gpu_manager.py.

Key changes from original:
- stdlib logging instead of RAGLogger
- pynvml for true free memory (accounts for other processes)
- Removed @log_operation decorators
- Keeps singleton pattern, GPUStats, monitoring thread
"""

import logging
import threading
import time
import traceback
from contextlib import contextmanager
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

import torch

logger = logging.getLogger(__name__)


# =============================================================================
# Data Structures
# =============================================================================

@dataclass
class GPUStats:
    """Statistics for a single GPU device."""

    device_id: int
    name: str
    total_memory: int         # bytes
    used_memory: int          # bytes (from pynvml — includes all processes)
    free_memory: int          # bytes (from pynvml — true free)
    torch_allocated: int      # bytes (PyTorch tensor allocations only)
    torch_reserved: int       # bytes (PyTorch caching allocator reserved)
    temperature: Optional[int] = None       # Celsius
    utilization: Optional[int] = None       # GPU compute utilization %
    power_usage: Optional[float] = None     # Watts


@dataclass
class GPUAllocation:
    """Tracks a model's GPU allocation."""

    device_id: int
    model_name: str
    allocated_memory: int   # bytes (estimated)
    timestamp: float


# =============================================================================
# GPU Manager
# =============================================================================

class GPUManager:
    """
    Singleton GPU resource manager.

    Uses pynvml for accurate free memory queries (sees all processes,
    not just PyTorch). Falls back to PyTorch CUDA APIs if pynvml
    is unavailable.
    """

    _instance = None
    _lock = threading.Lock()

    def __new__(cls):
        with cls._lock:
            if cls._instance is None:
                cls._instance = super().__new__(cls)
                cls._instance._initialized = False
            return cls._instance

    def __init__(self):
        if self._initialized:
            return

        self.n_gpus = torch.cuda.device_count()
        self.gpu_assignments: Dict[str, GPUAllocation] = {}

        # Try to initialize pynvml
        self._nvml_available = False
        try:
            import pynvml
            pynvml.nvmlInit()
            self._nvml_available = True
            logger.debug("pynvml initialized successfully")
        except Exception:
            logger.debug("pynvml not available, falling back to PyTorch CUDA APIs")

        self._initialized = True
        self._monitoring_active = False

        if self.n_gpus > 0:
            self._log_gpu_info()

    @classmethod
    def reset(cls):
        """Reset the singleton (for testing only)."""
        with cls._lock:
            if cls._instance is not None:
                cls._instance.stop_monitoring()
            cls._instance = None

    # -------------------------------------------------------------------------
    # GPU Information
    # -------------------------------------------------------------------------

    def _log_gpu_info(self):
        """Log details about all available GPUs."""
        for i in range(self.n_gpus):
            try:
                props = torch.cuda.get_device_properties(i)
                logger.info(
                    "GPU %d: %s — %.2f GB, compute %d.%d, %d SMs",
                    i, props.name, props.total_memory / 1024**3,
                    props.major, props.minor, props.multi_processor_count,
                )
            except Exception:
                logger.warning("Could not get properties for GPU %d", i, exc_info=True)

    def get_gpu_stats(self, device_id: int) -> GPUStats:
        """
        Get comprehensive statistics for a specific GPU.

        Uses pynvml for memory queries (accurate — sees all processes).
        Falls back to PyTorch APIs if pynvml is unavailable.
        """
        if not 0 <= device_id < self.n_gpus:
            raise ValueError(f"Invalid GPU device ID: {device_id} (have {self.n_gpus} GPUs)")

        props = torch.cuda.get_device_properties(device_id)
        total_memory = props.total_memory
        name = props.name

        # PyTorch allocations (for telemetry)
        torch_allocated = torch.cuda.memory_allocated(device_id)
        torch_reserved = torch.cuda.memory_reserved(device_id)

        # True memory from pynvml (or fallback)
        temperature = None
        utilization = None
        power_usage = None

        if self._nvml_available:
            try:
                import pynvml
                handle = pynvml.nvmlDeviceGetHandleByIndex(device_id)
                mem_info = pynvml.nvmlDeviceGetMemoryInfo(handle)
                used_memory = mem_info.used
                free_memory = mem_info.free

                try:
                    temperature = pynvml.nvmlDeviceGetTemperature(
                        handle, pynvml.NVML_TEMPERATURE_GPU
                    )
                except Exception:
                    pass
                try:
                    utilization = pynvml.nvmlDeviceGetUtilizationRates(handle).gpu
                except Exception:
                    pass
                try:
                    power_usage = pynvml.nvmlDeviceGetPowerUsage(handle) / 1000.0
                except Exception:
                    pass
            except Exception:
                logger.debug("pynvml query failed for GPU %d, falling back", device_id)
                used_memory = torch_allocated
                free_memory = total_memory - torch_allocated
        else:
            used_memory = torch_allocated
            free_memory = total_memory - torch_allocated

        return GPUStats(
            device_id=device_id,
            name=name,
            total_memory=total_memory,
            used_memory=used_memory,
            free_memory=free_memory,
            torch_allocated=torch_allocated,
            torch_reserved=torch_reserved,
            temperature=temperature,
            utilization=utilization,
            power_usage=power_usage,
        )

    def get_all_gpu_stats(self) -> List[GPUStats]:
        """Get stats for all GPUs."""
        return [self.get_gpu_stats(i) for i in range(self.n_gpus)]

    def get_free_memory(self, device_id: int) -> int:
        """Get free memory in bytes for a specific GPU (convenience method)."""
        return self.get_gpu_stats(device_id).free_memory

    # -------------------------------------------------------------------------
    # Monitoring
    # -------------------------------------------------------------------------

    def start_monitoring(self, interval: int = 30):
        """Start background monitoring thread."""
        if self._monitoring_active or self.n_gpus == 0:
            return

        self._monitoring_active = True
        self._monitoring_interval = interval
        self._monitoring_thread = threading.Thread(
            target=self._monitor_loop, daemon=True, name="gpu-monitor"
        )
        self._monitoring_thread.start()
        logger.debug("GPU monitoring started (interval=%ds)", interval)

    def stop_monitoring(self):
        """Stop the monitoring thread."""
        if not self._monitoring_active:
            return
        self._monitoring_active = False
        if hasattr(self, "_monitoring_thread"):
            self._monitoring_thread.join(timeout=2.0)
            logger.debug("GPU monitoring stopped")

    def _monitor_loop(self):
        """Background monitoring — logs warnings for critical usage."""
        while self._monitoring_active:
            try:
                for i in range(self.n_gpus):
                    stats = self.get_gpu_stats(i)
                    memory_pct = (stats.used_memory / stats.total_memory) * 100

                    if memory_pct > 95:
                        logger.warning(
                            "GPU %d memory critical: %.1f%% used (%.2f GB free)",
                            i, memory_pct, stats.free_memory / 1024**3,
                        )

                    if stats.temperature is not None and stats.temperature > 85:
                        logger.warning(
                            "GPU %d temperature high: %d°C", i, stats.temperature
                        )
            except Exception:
                logger.debug("Monitoring cycle error", exc_info=True)

            time.sleep(self._monitoring_interval)

    # -------------------------------------------------------------------------
    # Allocation Tracking
    # -------------------------------------------------------------------------

    def record_allocation(self, model_name: str, device_id: int, estimated_bytes: int):
        """Record that a model has been allocated to a GPU."""
        self.gpu_assignments[model_name] = GPUAllocation(
            device_id=device_id,
            model_name=model_name,
            allocated_memory=estimated_bytes,
            timestamp=time.time(),
        )
        logger.debug(
            "Recorded allocation: %s on GPU %d (%.2f GB estimated)",
            model_name, device_id, estimated_bytes / 1024**3,
        )

    def release_allocation(self, model_name: str):
        """Release a model's GPU allocation record and clear CUDA cache."""
        if model_name not in self.gpu_assignments:
            return

        allocation = self.gpu_assignments.pop(model_name)
        try:
            torch.cuda.set_device(allocation.device_id)
            torch.cuda.empty_cache()
        except Exception:
            logger.debug("Could not clear cache for GPU %d", allocation.device_id)

        logger.debug(
            "Released allocation: %s from GPU %d",
            model_name, allocation.device_id,
        )

    def get_assignments_for_gpu(self, device_id: int) -> List[GPUAllocation]:
        """Get all model assignments for a specific GPU."""
        return [a for a in self.gpu_assignments.values() if a.device_id == device_id]

    def get_gpu_model_count(self, device_id: int) -> int:
        """Count models loaded on a specific GPU."""
        return len(self.get_assignments_for_gpu(device_id))

    def get_least_loaded_gpu(self) -> Optional[int]:
        """
        Get the GPU with the fewest loaded models.
        Breaks ties by most free memory. Returns None if no GPUs available.
        """
        if self.n_gpus == 0:
            return None

        best_gpu = None
        best_count = float("inf")
        best_free = -1

        for i in range(self.n_gpus):
            count = self.get_gpu_model_count(i)
            try:
                free = self.get_free_memory(i)
            except Exception:
                free = 0

            if count < best_count or (count == best_count and free > best_free):
                best_gpu = i
                best_count = count
                best_free = free

        return best_gpu

    def get_gpu_utilization_summary(self) -> List[dict]:
        """Get a summary of GPU utilization for scheduling decisions."""
        summary = []
        for i in range(self.n_gpus):
            try:
                stats = self.get_gpu_stats(i)
                summary.append({
                    "gpu_id": i,
                    "name": stats.name,
                    "models_loaded": self.get_gpu_model_count(i),
                    "free_memory_gb": round(stats.free_memory / 1024**3, 2),
                    "used_memory_gb": round(stats.used_memory / 1024**3, 2),
                    "total_memory_gb": round(stats.total_memory / 1024**3, 2),
                    "utilization": stats.utilization,
                    "temperature": stats.temperature,
                })
            except Exception as e:
                summary.append({"gpu_id": i, "error": str(e)})
        return summary

    # -------------------------------------------------------------------------
    # Cleanup
    # -------------------------------------------------------------------------

    def cleanup(self):
        """Stop monitoring, release all allocations, clear CUDA cache."""
        self.stop_monitoring()

        for name in list(self.gpu_assignments.keys()):
            self.release_allocation(name)

        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            logger.debug("CUDA cache cleared")

    def __del__(self):
        try:
            self.cleanup()
        except Exception:
            pass
