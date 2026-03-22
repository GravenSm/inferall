"""Tests for inferall.gpu.allocator — VRAM estimation and allocation planning."""

import json
from datetime import datetime
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from inferall.gpu.allocator import (
    AllocationPlan,
    GPUAllocator,
    ModelArchInfo,
    _BYTES_PER_PARAM,
    _CUDA_OVERHEAD_BYTES,
)
from inferall.gpu.manager import GPUManager, GPUStats
from inferall.registry.metadata import ModelFormat, ModelRecord, ModelTask


def _make_record(
    model_id="test/model",
    fmt=ModelFormat.TRANSFORMERS,
    param_count=7_000_000_000,
    file_size_bytes=14_000_000_000,
    **kwargs,
):
    defaults = dict(
        model_id=model_id,
        revision="abc123",
        format=fmt,
        local_path=Path("/tmp/models/test"),
        file_size_bytes=file_size_bytes,
        param_count=param_count,
        gguf_variant=None,
        trust_remote_code=False,
        pipeline_tag="text-generation",
        pulled_at=datetime.now(),
        task=ModelTask.CHAT,
    )
    defaults.update(kwargs)
    return ModelRecord(**defaults)


def _mock_gpu_manager(n_gpus=1, free_memory=None):
    """Create a mock GPUManager with configurable free memory."""
    mgr = MagicMock(spec=GPUManager)
    mgr.n_gpus = n_gpus
    if free_memory is None:
        free_memory = {i: 24 * 1024**3 for i in range(n_gpus)}  # 24GB per GPU
    mgr.get_free_memory.side_effect = lambda i: free_memory.get(i, 0)
    return mgr


class TestAllocationPlan:
    def test_defaults(self):
        plan = AllocationPlan()
        assert plan.device_map == "auto"
        assert plan.gpu_ids == []
        assert plan.offload_to_cpu is False
        assert plan.estimated_vram_bytes == 0
        assert plan.n_gpu_layers == -1

    def test_custom_values(self):
        plan = AllocationPlan(
            max_memory={0: "10GiB"},
            gpu_ids=[0],
            estimated_vram_bytes=10 * 1024**3,
        )
        assert plan.gpu_ids == [0]


class TestModelArchInfo:
    def test_fields(self):
        info = ModelArchInfo(num_hidden_layers=32, num_key_value_heads=8, head_dim=128)
        assert info.num_hidden_layers == 32


class TestVRAMEstimation:
    def test_fp16_estimation(self):
        mgr = _mock_gpu_manager(n_gpus=0)
        allocator = GPUAllocator(mgr, vram_buffer_mb=512)
        record = _make_record(param_count=7_000_000_000)
        vram = allocator._estimate_vram(record, max_seq_len=4096)
        # 7B * 2 bytes = 14GB weights + KV cache + activation + overhead
        assert vram > 14 * 1024**3

    def test_4bit_estimation_smaller(self):
        mgr = _mock_gpu_manager(n_gpus=0)
        allocator = GPUAllocator(mgr, vram_buffer_mb=512)
        record_fp16 = _make_record(format=ModelFormat.TRANSFORMERS)
        record_4bit = _make_record(format=ModelFormat.TRANSFORMERS_BNB_4BIT)
        vram_fp16 = allocator._estimate_vram(record_fp16, 4096)
        vram_4bit = allocator._estimate_vram(record_4bit, 4096)
        assert vram_4bit < vram_fp16

    def test_no_param_count_uses_file_size(self):
        mgr = _mock_gpu_manager(n_gpus=0)
        allocator = GPUAllocator(mgr, vram_buffer_mb=512)
        record = _make_record(param_count=None, file_size_bytes=5_000_000_000)
        vram = allocator._estimate_vram(record, 4096)
        assert vram == int(5_000_000_000 * 1.2) + _CUDA_OVERHEAD_BYTES


class TestArchInfoParsing:
    def test_read_valid_config(self, tmp_path):
        config = {
            "num_hidden_layers": 32,
            "hidden_size": 4096,
            "num_attention_heads": 32,
            "num_key_value_heads": 8,
        }
        config_path = tmp_path / "config.json"
        config_path.write_text(json.dumps(config))

        mgr = _mock_gpu_manager(n_gpus=0)
        allocator = GPUAllocator(mgr)
        arch = allocator._read_arch_info(tmp_path)

        assert arch is not None
        assert arch.num_hidden_layers == 32
        assert arch.num_key_value_heads == 8
        assert arch.head_dim == 128  # 4096 / 32

    def test_read_missing_config(self, tmp_path):
        mgr = _mock_gpu_manager(n_gpus=0)
        allocator = GPUAllocator(mgr)
        assert allocator._read_arch_info(tmp_path) is None

    def test_read_incomplete_config(self, tmp_path):
        config = {"num_hidden_layers": 32}  # Missing hidden_size and num_attention_heads
        (tmp_path / "config.json").write_text(json.dumps(config))

        mgr = _mock_gpu_manager(n_gpus=0)
        allocator = GPUAllocator(mgr)
        assert allocator._read_arch_info(tmp_path) is None


class TestKVCacheEstimation:
    def test_kv_cache_formula(self):
        mgr = _mock_gpu_manager(n_gpus=0)
        allocator = GPUAllocator(mgr)
        arch = ModelArchInfo(num_hidden_layers=32, num_key_value_heads=8, head_dim=128)
        kv = allocator._estimate_kv_cache(arch, max_seq_len=4096)
        # 2 * 32 * 8 * 128 * 4096 * 2 = 536,870,912
        expected = 2 * 32 * 8 * 128 * 4096 * 2
        assert kv == expected


class TestAllocationStrategies:
    def test_single_gpu_fits(self):
        mgr = _mock_gpu_manager(n_gpus=1, free_memory={0: 24 * 1024**3})
        allocator = GPUAllocator(mgr, vram_buffer_mb=512)
        record = _make_record(param_count=3_000_000_000)  # ~6GB fp16 + overhead
        plan = allocator.compute_allocation(record)
        assert plan.gpu_ids == [0]
        assert plan.offload_to_cpu is False

    def test_no_gpu_uses_cpu(self):
        mgr = _mock_gpu_manager(n_gpus=0)
        allocator = GPUAllocator(mgr, vram_buffer_mb=512)
        record = _make_record()
        plan = allocator.compute_allocation(record)
        assert plan.device_map == "cpu"
        assert plan.gpu_ids == []
        assert plan.offload_to_cpu is True
        assert plan.n_gpu_layers == 0

    def test_multi_gpu_when_single_insufficient(self):
        # Each GPU has 10GB, model needs ~15GB
        mgr = _mock_gpu_manager(n_gpus=2, free_memory={0: 10 * 1024**3, 1: 10 * 1024**3})
        allocator = GPUAllocator(mgr, vram_buffer_mb=512)
        record = _make_record(param_count=7_000_000_000)
        plan = allocator.compute_allocation(record)
        assert len(plan.gpu_ids) >= 1

    def test_bytes_per_param_coverage(self):
        """Every ModelFormat should have a bytes_per_param entry."""
        for fmt in ModelFormat:
            assert fmt in _BYTES_PER_PARAM, f"ModelFormat.{fmt.name} missing from _BYTES_PER_PARAM"
