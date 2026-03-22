"""Tests for inferall.orchestrator — model lifecycle, locking, eviction."""

import threading
from datetime import datetime, timedelta
from unittest.mock import MagicMock, patch

import pytest

from inferall.backends.base import GenerationParams, GenerationResult, LoadedModel
from inferall.config import EngineConfig
from inferall.gpu.allocator import AllocationPlan, GPUAllocator
from inferall.gpu.manager import GPUManager
from inferall.orchestrator import (
    LoadedModelInfo,
    ModelNotFoundError,
    Orchestrator,
)
from inferall.registry.metadata import ModelFormat, ModelRecord, ModelTask
from inferall.registry.registry import ModelRegistry


def _make_loaded(model_id="test/model", backend_name="transformers", **kwargs):
    return LoadedModel(
        model_id=model_id,
        backend_name=backend_name,
        model=MagicMock(),
        tokenizer=MagicMock(),
        **kwargs,
    )


def _make_record(model_id="test/model", fmt=ModelFormat.TRANSFORMERS, **kwargs):
    defaults = dict(
        model_id=model_id,
        revision="abc123",
        format=fmt,
        local_path=MagicMock(),
        file_size_bytes=1_000_000_000,
        param_count=7_000_000_000,
        gguf_variant=None,
        trust_remote_code=False,
        pipeline_tag="text-generation",
        pulled_at=datetime.now(),
        task=ModelTask.CHAT,
    )
    defaults.update(kwargs)
    return ModelRecord(**defaults)


def _make_orchestrator(config=None, registry=None, gpu_manager=None, allocator=None):
    """Create an Orchestrator with mock dependencies."""
    if config is None:
        config = EngineConfig(idle_timeout=0)  # disable idle monitor for tests
    if registry is None:
        registry = MagicMock(spec=ModelRegistry)
    if gpu_manager is None:
        gpu_manager = MagicMock(spec=GPUManager)
        gpu_manager.gpu_assignments = {}
    if allocator is None:
        allocator = MagicMock(spec=GPUAllocator)
        allocator.compute_allocation.return_value = AllocationPlan(
            gpu_ids=[0], estimated_vram_bytes=1_000_000_000
        )
    return Orchestrator(config, registry, gpu_manager, allocator)


class TestOrchestratorGetOrLoad:
    def test_model_not_in_registry_raises(self):
        registry = MagicMock(spec=ModelRegistry)
        registry.get.return_value = None
        orch = _make_orchestrator(registry=registry)

        with pytest.raises(ModelNotFoundError):
            orch.get_or_load("unknown/model")

    def test_loads_model_and_increments_ref(self):
        record = _make_record()
        registry = MagicMock(spec=ModelRegistry)
        registry.get.return_value = record

        allocator = MagicMock(spec=GPUAllocator)
        allocator.compute_allocation.return_value = AllocationPlan(
            gpu_ids=[0], estimated_vram_bytes=1_000_000_000
        )

        gpu_manager = MagicMock(spec=GPUManager)
        gpu_manager.gpu_assignments = {}

        orch = _make_orchestrator(registry=registry, gpu_manager=gpu_manager, allocator=allocator)

        # Mock the backend load
        with patch.object(orch, '_get_backend') as mock_backend:
            mock_backend.return_value.load.return_value = _make_loaded()
            loaded = orch.get_or_load("test/model")

        assert loaded is not None
        assert orch._ref_counts["test/model"] == 1

    def test_fast_path_returns_cached(self):
        orch = _make_orchestrator()
        loaded = _make_loaded()
        orch.loaded_models["test/model"] = loaded
        orch._ref_counts["test/model"] = 0

        result = orch.get_or_load("test/model")
        assert result is loaded
        assert orch._ref_counts["test/model"] == 1


class TestOrchestratorRelease:
    def test_release_decrements_ref(self):
        orch = _make_orchestrator()
        orch.loaded_models["test/model"] = _make_loaded()
        orch._ref_counts["test/model"] = 2

        orch.release("test/model")
        assert orch._ref_counts["test/model"] == 1

    def test_release_does_not_go_negative(self):
        orch = _make_orchestrator()
        orch._ref_counts["test/model"] = 0
        orch.release("test/model")
        assert orch._ref_counts["test/model"] == 0


class TestOrchestratorUnload:
    def test_unload_removes_model(self):
        orch = _make_orchestrator()
        loaded = _make_loaded()
        orch.loaded_models["test/model"] = loaded
        orch._ref_counts["test/model"] = 0

        with patch.object(orch, '_get_backend') as mock_backend:
            orch.unload("test/model")

        assert "test/model" not in orch.loaded_models

    def test_unload_refuses_with_active_refs(self):
        orch = _make_orchestrator()
        orch.loaded_models["test/model"] = _make_loaded()
        orch._ref_counts["test/model"] = 1

        orch.unload("test/model")
        # Model should still be loaded
        assert "test/model" in orch.loaded_models

    def test_unload_nonexistent_is_noop(self):
        orch = _make_orchestrator()
        orch.unload("nonexistent/model")  # should not raise


class TestOrchestratorEviction:
    def test_evicts_lru_when_at_capacity(self):
        config = EngineConfig(max_loaded_models=2, idle_timeout=0)
        orch = _make_orchestrator(config=config)

        # Load 2 models at capacity
        for name in ["model/a", "model/b"]:
            loaded = _make_loaded(name)
            orch.loaded_models[name] = loaded
            orch._ref_counts[name] = 0

        # Make model/a older
        orch.loaded_models["model/a"].last_used_at = datetime.now() - timedelta(hours=1)
        orch.loaded_models["model/b"].last_used_at = datetime.now()

        with patch.object(orch, '_get_backend') as mock_backend:
            orch._evict_if_needed()

        assert "model/a" not in orch.loaded_models
        assert "model/b" in orch.loaded_models

    def test_does_not_evict_models_with_refs(self):
        config = EngineConfig(max_loaded_models=1, idle_timeout=0)
        orch = _make_orchestrator(config=config)

        loaded = _make_loaded("model/a")
        orch.loaded_models["model/a"] = loaded
        orch._ref_counts["model/a"] = 1  # active ref

        orch._evict_if_needed()
        assert "model/a" in orch.loaded_models  # not evicted


class TestOrchestratorListLoaded:
    def test_list_loaded_empty(self):
        orch = _make_orchestrator()
        assert orch.list_loaded() == []

    def test_list_loaded_returns_info(self):
        orch = _make_orchestrator()
        orch.loaded_models["test/model"] = _make_loaded()
        orch._ref_counts["test/model"] = 1

        infos = orch.list_loaded()
        assert len(infos) == 1
        assert infos[0].model_id == "test/model"
        assert infos[0].ref_count == 1


class TestOrchestratorShutdown:
    def test_shutdown_unloads_all(self):
        orch = _make_orchestrator()
        orch.loaded_models["model/a"] = _make_loaded("model/a")
        orch.loaded_models["model/b"] = _make_loaded("model/b")
        orch._ref_counts["model/a"] = 0
        orch._ref_counts["model/b"] = 0

        with patch.object(orch, '_get_backend') as mock_backend:
            orch.shutdown()

        assert len(orch.loaded_models) == 0


class TestOrchestratorBackendSelection:
    def test_transformers_format(self):
        orch = _make_orchestrator()
        backend = orch._get_backend(ModelFormat.TRANSFORMERS)
        assert backend.name == "transformers"

    def test_gguf_format(self):
        orch = _make_orchestrator()
        backend = orch._get_backend(ModelFormat.GGUF)
        assert backend.name == "llamacpp"

    def test_unsupported_format_raises(self):
        orch = _make_orchestrator()
        with pytest.raises(ValueError):
            orch._get_backend(MagicMock())  # bogus format


class TestFormatFromBackendName:
    def test_known_names(self):
        orch = _make_orchestrator()
        assert orch._format_from_backend_name("llamacpp") == ModelFormat.GGUF
        assert orch._format_from_backend_name("transformers") == ModelFormat.TRANSFORMERS
        assert orch._format_from_backend_name("embedding") == ModelFormat.EMBEDDING
        assert orch._format_from_backend_name("vlm") == ModelFormat.VISION_LANGUAGE
        assert orch._format_from_backend_name("asr") == ModelFormat.ASR
        assert orch._format_from_backend_name("diffusion") == ModelFormat.DIFFUSION
        assert orch._format_from_backend_name("tts") == ModelFormat.TTS

    def test_unknown_defaults_to_transformers(self):
        orch = _make_orchestrator()
        assert orch._format_from_backend_name("unknown") == ModelFormat.TRANSFORMERS
