"""Tests for unified classification support — backend, orchestrator, API endpoint."""

from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from inferall.backends.base import (
    ClassificationBackendABC,
    ClassificationParams,
    ClassificationResult,
    LoadedModel,
)
from inferall.registry.metadata import (
    FORMAT_TO_TASK,
    PIPELINE_TAG_TO_TASK,
    ModelFormat,
    ModelTask,
)


# =============================================================================
# Metadata / Enum Tests
# =============================================================================

class TestClassificationEnums:
    def test_model_task_exists(self):
        assert ModelTask.CLASSIFICATION.value == "classification"

    def test_model_format_exists(self):
        assert ModelFormat.CLASSIFICATION.value == "classification"

    def test_pipeline_tag_image_classification(self):
        assert PIPELINE_TAG_TO_TASK["image-classification"] == ModelTask.CLASSIFICATION

    def test_pipeline_tag_audio_classification(self):
        assert PIPELINE_TAG_TO_TASK["audio-classification"] == ModelTask.CLASSIFICATION

    def test_pipeline_tag_zero_shot(self):
        assert PIPELINE_TAG_TO_TASK["zero-shot-classification"] == ModelTask.CLASSIFICATION

    def test_pipeline_tag_zero_shot_image(self):
        assert PIPELINE_TAG_TO_TASK["zero-shot-image-classification"] == ModelTask.CLASSIFICATION

    def test_format_to_task_mapping(self):
        assert FORMAT_TO_TASK[ModelFormat.CLASSIFICATION] == ModelTask.CLASSIFICATION

    def test_not_in_blocked_tags(self):
        from inferall.registry.hf_resolver import _BLOCKED_TAGS
        for tag in ("image-classification", "audio-classification",
                     "zero-shot-classification", "zero-shot-image-classification"):
            assert tag not in _BLOCKED_TAGS


# =============================================================================
# Backend Property Tests
# =============================================================================

class TestClassificationBackendProperties:
    def test_name(self):
        from inferall.backends.classification_backend import TransformersClassificationBackend
        backend = TransformersClassificationBackend()
        assert backend.name == "classification"

    def test_resolve_device_gpu(self):
        from inferall.backends.classification_backend import TransformersClassificationBackend
        from inferall.gpu.allocator import AllocationPlan
        backend = TransformersClassificationBackend()
        assert backend._resolve_device(AllocationPlan(gpu_ids=[0])) == 0

    def test_resolve_device_cpu(self):
        from inferall.backends.classification_backend import TransformersClassificationBackend
        from inferall.gpu.allocator import AllocationPlan
        backend = TransformersClassificationBackend()
        assert backend._resolve_device(AllocationPlan(gpu_ids=[])) == -1


# =============================================================================
# Data Structure Tests
# =============================================================================

class TestClassificationParams:
    def test_defaults(self):
        p = ClassificationParams()
        assert p.top_k == 5
        assert p.candidate_labels is None
        assert p.image_b64 is None
        assert p.audio_b64 is None

    def test_custom_values(self):
        p = ClassificationParams(
            candidate_labels=["cat", "dog"],
            top_k=3,
            image_b64="base64data",
        )
        assert p.candidate_labels == ["cat", "dog"]


class TestClassificationResult:
    def test_fields(self):
        r = ClassificationResult(
            labels=[{"label": "cat", "score": 0.95}],
            model="test/vit",
            pipeline_tag="image-classification",
            total_time_ms=50.0,
        )
        assert r.labels[0]["label"] == "cat"
        assert r.pipeline_tag == "image-classification"


# =============================================================================
# Orchestrator Tests
# =============================================================================

class TestOrchestratorClassifyDispatch:
    def test_classify_dispatch(self):
        from inferall.config import EngineConfig
        from inferall.gpu.allocator import GPUAllocator
        from inferall.gpu.manager import GPUManager
        from inferall.orchestrator import Orchestrator
        from inferall.registry.registry import ModelRegistry

        config = EngineConfig(idle_timeout=0)
        orch = Orchestrator(
            config,
            MagicMock(spec=ModelRegistry),
            MagicMock(spec=GPUManager, gpu_assignments={}),
            MagicMock(spec=GPUAllocator),
        )

        loaded = LoadedModel(
            model_id="test/vit",
            backend_name="classification",
            model=MagicMock(),
            tokenizer="image-classification",
        )
        orch.loaded_models["test/vit"] = loaded
        orch._ref_counts["test/vit"] = 0

        expected = ClassificationResult(
            labels=[{"label": "cat", "score": 0.9}],
            model="test/vit",
        )

        with patch.object(orch, '_get_backend') as mock_get:
            mock_get.return_value.classify.return_value = expected
            result = orch.classify("test/vit", "", ClassificationParams())

        assert result.labels[0]["label"] == "cat"
        assert orch._ref_counts["test/vit"] == 0

    def test_backend_selection(self):
        from inferall.config import EngineConfig
        from inferall.gpu.allocator import GPUAllocator
        from inferall.gpu.manager import GPUManager
        from inferall.orchestrator import Orchestrator
        from inferall.registry.registry import ModelRegistry

        config = EngineConfig(idle_timeout=0)
        orch = Orchestrator(
            config,
            MagicMock(spec=ModelRegistry),
            MagicMock(spec=GPUManager, gpu_assignments={}),
            MagicMock(spec=GPUAllocator),
        )
        backend = orch._get_backend(ModelFormat.CLASSIFICATION)
        assert backend.name == "classification"

    def test_format_from_backend_name(self):
        from inferall.config import EngineConfig
        from inferall.gpu.allocator import GPUAllocator
        from inferall.gpu.manager import GPUManager
        from inferall.orchestrator import Orchestrator
        from inferall.registry.registry import ModelRegistry

        config = EngineConfig(idle_timeout=0)
        orch = Orchestrator(
            config,
            MagicMock(spec=ModelRegistry),
            MagicMock(spec=GPUManager, gpu_assignments={}),
            MagicMock(spec=GPUAllocator),
        )
        assert orch._format_from_backend_name("classification") == ModelFormat.CLASSIFICATION


# =============================================================================
# HF Resolver Tests
# =============================================================================

class TestHFResolverClassificationDetection:
    def test_image_classification_tag(self):
        from inferall.registry.hf_resolver import HFResolver
        resolver = HFResolver(models_dir=Path("/tmp/test"))
        info = MagicMock()
        info.pipeline_tag = "image-classification"
        info.tags = []
        info.siblings = []
        fmt, _ = resolver._detect_format("test/model", info, variant=None)
        assert fmt == ModelFormat.CLASSIFICATION

    def test_zero_shot_tag(self):
        from inferall.registry.hf_resolver import HFResolver
        resolver = HFResolver(models_dir=Path("/tmp/test"))
        info = MagicMock()
        info.pipeline_tag = "zero-shot-classification"
        info.tags = []
        info.siblings = []
        fmt, _ = resolver._detect_format("test/model", info, variant=None)
        assert fmt == ModelFormat.CLASSIFICATION

    def test_audio_classification_tag(self):
        from inferall.registry.hf_resolver import HFResolver
        resolver = HFResolver(models_dir=Path("/tmp/test"))
        info = MagicMock()
        info.pipeline_tag = "audio-classification"
        info.tags = []
        info.siblings = []
        fmt, _ = resolver._detect_format("test/model", info, variant=None)
        assert fmt == ModelFormat.CLASSIFICATION


# =============================================================================
# GPU Allocator Tests
# =============================================================================

class TestAllocatorClassificationFormat:
    def test_bytes_per_param_has_classification(self):
        from inferall.gpu.allocator import _BYTES_PER_PARAM
        assert ModelFormat.CLASSIFICATION in _BYTES_PER_PARAM


# =============================================================================
# API Endpoint Tests
# =============================================================================

class TestClassifyEndpoint:
    @pytest.fixture
    def mock_orchestrator(self):
        from inferall.orchestrator import Orchestrator
        orch = MagicMock(spec=Orchestrator)
        orch.list_loaded.return_value = []
        return orch

    @pytest.fixture
    def client(self, mock_orchestrator):
        from inferall.api.server import create_app
        from inferall.registry.registry import ModelRegistry
        from starlette.testclient import TestClient

        registry = MagicMock(spec=ModelRegistry)
        registry.list_all.return_value = []
        app = create_app(
            orchestrator=mock_orchestrator,
            registry=registry,
            api_key=None,
        )
        return TestClient(app)

    def test_classify_success(self, client, mock_orchestrator):
        mock_orchestrator.classify.return_value = ClassificationResult(
            labels=[
                {"label": "cat", "score": 0.95},
                {"label": "dog", "score": 0.04},
            ],
            model="test/vit",
            pipeline_tag="image-classification",
            total_time_ms=50.0,
        )

        resp = client.post("/v1/classify", json={
            "model": "test/vit",
            "image": "fake_base64_data",
        })

        assert resp.status_code == 200
        data = resp.json()
        assert len(data["labels"]) == 2
        assert data["labels"][0]["label"] == "cat"
        assert data["pipeline_tag"] == "image-classification"

    def test_classify_zero_shot(self, client, mock_orchestrator):
        mock_orchestrator.classify.return_value = ClassificationResult(
            labels=[{"label": "politics", "score": 0.8}],
            model="test/bart",
            pipeline_tag="zero-shot-classification",
        )

        resp = client.post("/v1/classify", json={
            "model": "test/bart",
            "text": "The president signed a new bill today",
            "candidate_labels": ["politics", "sports", "technology"],
        })
        assert resp.status_code == 200

    def test_classify_no_input_rejected(self, client):
        resp = client.post("/v1/classify", json={
            "model": "test/model",
        })
        assert resp.status_code == 400

    def test_classify_model_not_found(self, client, mock_orchestrator):
        from inferall.orchestrator import ModelNotFoundError
        mock_orchestrator.classify.side_effect = ModelNotFoundError("not found")

        resp = client.post("/v1/classify", json={
            "model": "missing/model",
            "text": "test",
        })
        assert resp.status_code == 404

    def test_health_includes_classification(self, client):
        resp = client.get("/health")
        assert resp.json()["capabilities"]["classification"] is True
