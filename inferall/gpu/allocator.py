"""
GPU Allocator
-------------
Sits above GPUManager. Decides *how* to distribute a model across GPUs.

VRAM estimation formula:
  Total ≈ model_weights + kv_cache + activation_memory + cuda_overhead

Where:
  model_weights  = param_count × bytes_per_param
  kv_cache       = 2 × num_layers × num_kv_heads × head_dim × max_seq_len × 2 (fp16)
  activation_mem = ~15% of model_weights (batch_size=1 estimate)
  cuda_overhead  = 500 MB default (driver + context)

If config.json is available locally, we read architecture details for
precise KV cache estimation. Otherwise, fall back to param_count × bytes × 1.2.
"""

import json
import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional

from inferall.gpu.manager import GPUManager, GPUStats
from inferall.registry.metadata import ModelFormat, ModelRecord

logger = logging.getLogger(__name__)

# Bytes per parameter by format
_BYTES_PER_PARAM = {
    ModelFormat.TRANSFORMERS: 2.0,          # fp16
    ModelFormat.TRANSFORMERS_BNB_4BIT: 0.5, # 4-bit
    ModelFormat.TRANSFORMERS_BNB_8BIT: 1.0, # 8-bit
    ModelFormat.GPTQ: 0.5,                  # typically 4-bit
    ModelFormat.AWQ: 0.5,                   # typically 4-bit
    ModelFormat.GGUF: 0.56,                 # ~Q4_K_M average
    # Multi-modal formats (fp16 typical)
    ModelFormat.EMBEDDING: 2.0,
    ModelFormat.VISION_LANGUAGE: 2.0,
    ModelFormat.ASR: 2.0,
    ModelFormat.DIFFUSION: 2.0,
    ModelFormat.TTS: 2.0,
    ModelFormat.RERANK: 2.0,
    ModelFormat.IMAGE_TO_IMAGE: 2.0,
    ModelFormat.TEXT_TO_VIDEO: 3.0,         # video models have heavy temporal layers
    ModelFormat.SEQ2SEQ: 2.0,
    ModelFormat.CLASSIFICATION: 2.0,
    ModelFormat.OLLAMA_CLOUD: 0.0,         # No local VRAM — remote model
}

# Default CUDA overhead per GPU (driver + context)
_CUDA_OVERHEAD_BYTES = 500 * 1024**2  # 500 MB

# Activation memory as fraction of model weights
_ACTIVATION_FRACTION = 0.15


# =============================================================================
# Data Structures
# =============================================================================

@dataclass
class AllocationPlan:
    """
    Describes how to load a model onto GPU(s).

    For transformers backend: use max_memory + device_map.
    For llama.cpp backend: use tensor_split + n_gpu_layers.
    """

    # Transformers backend
    max_memory: Optional[Dict] = None              # e.g., {0: "10GiB", 1: "10GiB"}
    device_map: str = "auto"                       # "auto" or "cpu"

    # llama.cpp backend
    tensor_split: Optional[List[float]] = None     # e.g., [0.6, 0.4]
    n_gpu_layers: int = -1                         # -1 = all layers on GPU

    # Metadata
    estimated_vram_bytes: int = 0                  # Total estimated VRAM usage
    gpu_ids: List[int] = field(default_factory=list)  # Which GPUs are used
    offload_to_cpu: bool = False                   # Whether CPU offload is needed


@dataclass
class ModelArchInfo:
    """Architecture details parsed from config.json for KV cache estimation."""

    num_hidden_layers: int
    num_key_value_heads: int
    head_dim: int


# =============================================================================
# GPU Allocator
# =============================================================================

class GPUAllocator:
    """
    Computes allocation plans for models based on VRAM estimation
    and available GPU memory.
    """

    def __init__(self, gpu_manager: GPUManager, vram_buffer_mb: int = 512):
        self.gpu_manager = gpu_manager
        self.vram_buffer_bytes = vram_buffer_mb * 1024**2

    def compute_allocation(
        self,
        record: ModelRecord,
        max_seq_len: int = 4096,
    ) -> AllocationPlan:
        """
        Compute the best allocation plan for a model.

        Steps:
        1. Estimate total VRAM needed
        2. Query free memory on each GPU
        3. Try single GPU, then multi-GPU, then CPU fallback
        """
        estimated_vram = self._estimate_vram(record, max_seq_len)
        logger.info(
            "VRAM estimate for %s: %.2f GB",
            record.model_id, estimated_vram / 1024**3,
        )

        # No GPUs available
        if self.gpu_manager.n_gpus == 0:
            logger.warning("No GPUs available, using CPU for %s", record.model_id)
            return self._cpu_plan(estimated_vram)

        # Get available memory per GPU
        gpu_free = self._get_available_memory()

        if not gpu_free:
            logger.warning("Could not query GPU memory, using CPU for %s", record.model_id)
            return self._cpu_plan(estimated_vram)

        # Try single GPU
        plan = self._try_single_gpu(record, estimated_vram, gpu_free)
        if plan is not None:
            return plan

        # Try multi-GPU
        plan = self._try_multi_gpu(record, estimated_vram, gpu_free)
        if plan is not None:
            return plan

        # Fall back to CPU offload
        logger.warning(
            "Model %s (%.2f GB estimated) does not fit in available VRAM "
            "(total free: %.2f GB). Using CPU offload.",
            record.model_id,
            estimated_vram / 1024**3,
            sum(gpu_free.values()) / 1024**3,
        )
        return self._cpu_offload_plan(record, estimated_vram, gpu_free)

    # -------------------------------------------------------------------------
    # VRAM Estimation
    # -------------------------------------------------------------------------

    def _estimate_vram(self, record: ModelRecord, max_seq_len: int) -> int:
        """
        Estimate total VRAM needed in bytes.

        Uses architecture details from config.json if available,
        otherwise falls back to a simpler heuristic.
        """
        if record.param_count is None or record.param_count == 0:
            # No param count — use file size as a rough proxy
            logger.debug("No param_count for %s, using file size as estimate", record.model_id)
            return int(record.file_size_bytes * 1.2) + _CUDA_OVERHEAD_BYTES

        bytes_per_param = _BYTES_PER_PARAM.get(record.format, 2.0)
        model_weights = int(record.param_count * bytes_per_param)

        # Try precise KV cache estimation from config.json
        arch_info = self._read_arch_info(record.local_path)
        if arch_info is not None:
            kv_cache = self._estimate_kv_cache(arch_info, max_seq_len)
            logger.debug(
                "KV cache estimate for %s: %.2f GB (seq_len=%d)",
                record.model_id, kv_cache / 1024**3, max_seq_len,
            )
        else:
            # Fallback: 10% of model weights as KV cache estimate
            kv_cache = int(model_weights * 0.10)
            logger.debug(
                "No config.json for %s, using heuristic KV cache estimate",
                record.model_id,
            )

        activation_mem = int(model_weights * _ACTIVATION_FRACTION)
        total = model_weights + kv_cache + activation_mem + _CUDA_OVERHEAD_BYTES

        logger.debug(
            "VRAM breakdown for %s: weights=%.2fGB, kv=%.2fGB, "
            "activation=%.2fGB, overhead=%.2fGB, total=%.2fGB",
            record.model_id,
            model_weights / 1024**3,
            kv_cache / 1024**3,
            activation_mem / 1024**3,
            _CUDA_OVERHEAD_BYTES / 1024**3,
            total / 1024**3,
        )
        return total

    def _read_arch_info(self, local_path: Path) -> Optional[ModelArchInfo]:
        """Try to read architecture details from the model's config.json."""
        config_path = local_path / "config.json"
        if not config_path.exists():
            return None

        try:
            with open(config_path) as f:
                config = json.load(f)

            num_layers = config.get("num_hidden_layers")
            hidden_size = config.get("hidden_size")
            num_attention_heads = config.get("num_attention_heads")
            num_kv_heads = config.get("num_key_value_heads", num_attention_heads)

            if not all([num_layers, hidden_size, num_attention_heads]):
                return None

            head_dim = hidden_size // num_attention_heads

            return ModelArchInfo(
                num_hidden_layers=num_layers,
                num_key_value_heads=num_kv_heads,
                head_dim=head_dim,
            )
        except Exception:
            logger.debug("Could not parse config.json at %s", config_path, exc_info=True)
            return None

    def _estimate_kv_cache(self, arch: ModelArchInfo, max_seq_len: int) -> int:
        """
        Estimate KV cache size in bytes.

        KV cache = 2 (K+V) × num_layers × num_kv_heads × head_dim × seq_len × 2 (fp16 bytes)
        """
        return (
            2                           # K and V
            * arch.num_hidden_layers
            * arch.num_key_value_heads
            * arch.head_dim
            * max_seq_len
            * 2                         # fp16 = 2 bytes
        )

    # -------------------------------------------------------------------------
    # Allocation Strategies
    # -------------------------------------------------------------------------

    def _get_available_memory(self) -> Dict[int, int]:
        """
        Get available memory per GPU, after subtracting the VRAM buffer.
        Returns dict of gpu_id → available_bytes.
        """
        result = {}
        for i in range(self.gpu_manager.n_gpus):
            try:
                free = self.gpu_manager.get_free_memory(i)
                available = free - self.vram_buffer_bytes
                if available > 0:
                    result[i] = available
            except Exception:
                logger.debug("Could not query GPU %d", i, exc_info=True)
        return result

    def _try_single_gpu(
        self,
        record: ModelRecord,
        estimated_vram: int,
        gpu_free: Dict[int, int],
    ) -> Optional[AllocationPlan]:
        """Try to fit the model on a single GPU (the one with most free memory)."""
        # Sort by free memory descending
        sorted_gpus = sorted(gpu_free.items(), key=lambda x: x[1], reverse=True)

        for gpu_id, available in sorted_gpus:
            if available >= estimated_vram:
                max_memory_gib = available / 1024**3
                plan = AllocationPlan(
                    max_memory={gpu_id: f"{max_memory_gib:.1f}GiB"},
                    device_map="auto",
                    tensor_split=None,
                    n_gpu_layers=-1,
                    estimated_vram_bytes=estimated_vram,
                    gpu_ids=[gpu_id],
                )
                logger.info(
                    "Single GPU allocation for %s: GPU %d (%.2f GB available, "
                    "%.2f GB estimated)",
                    record.model_id, gpu_id,
                    available / 1024**3, estimated_vram / 1024**3,
                )
                return plan

        return None

    def _try_multi_gpu(
        self,
        record: ModelRecord,
        estimated_vram: int,
        gpu_free: Dict[int, int],
    ) -> Optional[AllocationPlan]:
        """
        Try to spread the model across multiple GPUs.

        Greedily adds GPUs (most-free first) until total available >= estimated VRAM.
        """
        if len(gpu_free) < 2:
            return None

        sorted_gpus = sorted(gpu_free.items(), key=lambda x: x[1], reverse=True)
        selected = []
        total_available = 0

        for gpu_id, available in sorted_gpus:
            selected.append((gpu_id, available))
            total_available += available
            if total_available >= estimated_vram:
                break

        if total_available < estimated_vram:
            return None

        # Build max_memory dict with proportional allocation
        gpu_ids = [g[0] for g in selected]
        max_memory = {}
        for gpu_id, available in selected:
            max_memory_gib = available / 1024**3
            max_memory[gpu_id] = f"{max_memory_gib:.1f}GiB"

        # Build tensor_split ratios (proportional to available memory)
        total = sum(a for _, a in selected)
        tensor_split = [a / total for _, a in selected]

        plan = AllocationPlan(
            max_memory=max_memory,
            device_map="auto",
            tensor_split=tensor_split,
            n_gpu_layers=-1,
            estimated_vram_bytes=estimated_vram,
            gpu_ids=gpu_ids,
        )

        logger.info(
            "Multi-GPU allocation for %s across GPUs %s "
            "(%.2f GB total available, %.2f GB estimated, split: %s)",
            record.model_id,
            gpu_ids,
            total_available / 1024**3,
            estimated_vram / 1024**3,
            [f"{r:.2f}" for r in tensor_split],
        )
        return plan

    def _cpu_plan(self, estimated_vram: int) -> AllocationPlan:
        """Pure CPU allocation (no GPUs available)."""
        return AllocationPlan(
            max_memory=None,
            device_map="cpu",
            tensor_split=None,
            n_gpu_layers=0,
            estimated_vram_bytes=estimated_vram,
            gpu_ids=[],
            offload_to_cpu=True,
        )

    def _cpu_offload_plan(
        self,
        record: ModelRecord,
        estimated_vram: int,
        gpu_free: Dict[int, int],
    ) -> AllocationPlan:
        """
        CPU offload plan — use available GPU memory but overflow to CPU/RAM.

        device_map="auto" with max_memory handles this natively in accelerate.
        """
        max_memory = {}
        gpu_ids = []
        for gpu_id, available in sorted(gpu_free.items()):
            if available > _CUDA_OVERHEAD_BYTES:
                max_memory[gpu_id] = f"{available / 1024**3:.1f}GiB"
                gpu_ids.append(gpu_id)

        plan = AllocationPlan(
            max_memory=max_memory if max_memory else None,
            device_map="auto" if max_memory else "cpu",
            tensor_split=None,
            n_gpu_layers=0,  # llama.cpp: no GPU layers when offloading
            estimated_vram_bytes=estimated_vram,
            gpu_ids=gpu_ids,
            offload_to_cpu=True,
        )

        logger.info(
            "CPU offload plan for %s: using GPUs %s with overflow to CPU",
            record.model_id, gpu_ids,
        )
        return plan
