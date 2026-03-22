"""Tests for reranking support — backend, orchestrator dispatch, API endpoint."""

from datetime import datetime
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from inferall.backends.base import (
    LoadedModel,
    RerankBackend,
    RerankParams,
    RerankResult,
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

class TestRerankEnums:
    def test_model_task_rerank_exists(self):
        assert ModelTask.RERANK.value == "rerank"

    def test_model_format_rerank_exists(self):
        assert ModelFormat.RERANK.value == "rerank"

    def test_pipeline_tag_mapping(self):
        assert PIPELINE_TAG_TO_TASK["text-ranking"] == ModelTask.RERANK

    def test_format_to_task_mapping(self):
        assert FORMAT_TO_TASK[ModelFormat.RERANK] == ModelTask.RERANK


# =============================================================================
# Backend Property Tests
# =============================================================================

class TestCrossEncoderBackendProperties:
    def test_name(self):
        from inferall.backends.rerank_backend import CrossEncoderRerankerBackend
        backend = CrossEncoderRerankerBackend()
        assert backend.name == "rerank"

    def test_resolve_device_gpu(self):
        from inferall.backends.rerank_backend import CrossEncoderRerankerBackend
        from inferall.gpu.allocator import AllocationPlan
        backend = CrossEncoderRerankerBackend()
        plan = AllocationPlan(gpu_ids=[0])
        assert backend._resolve_device(plan) == "cuda:0"

    def test_resolve_device_cpu(self):
        from inferall.backends.rerank_backend import CrossEncoderRerankerBackend
        from inferall.gpu.allocator import AllocationPlan
        backend = CrossEncoderRerankerBackend()
        plan = AllocationPlan(gpu_ids=[])
        assert backend._resolve_device(plan) == "cpu"


# =============================================================================
# Build Result Tests
# =============================================================================

class TestBuildResult:
    def test_sorts_by_score_descending(self):
        from inferall.backends.rerank_backend import CrossEncoderRerankerBackend
        backend = CrossEncoderRerankerBackend()
        params = RerankParams()
        result = backend._build_result(
            scores=[0.1, 0.9, 0.5],
            documents=["doc_a", "doc_b", "doc_c"],
            params=params,
            model_id="test/model",
            prompt_tokens=100,
        )
        assert result.results[0]["relevance_score"] == 0.9
        assert result.results[0]["index"] == 1
        assert result.results[-1]["relevance_score"] == 0.1

    def test_top_n_limits_results(self):
        from inferall.backends.rerank_backend import CrossEncoderRerankerBackend
        backend = CrossEncoderRerankerBackend()
        params = RerankParams(top_n=2)
        result = backend._build_result(
            scores=[0.1, 0.9, 0.5, 0.3],
            documents=["a", "b", "c", "d"],
            params=params,
            model_id="test/model",
            prompt_tokens=100,
        )
        assert len(result.results) == 2

    def test_return_documents_includes_text(self):
        from inferall.backends.rerank_backend import CrossEncoderRerankerBackend
        backend = CrossEncoderRerankerBackend()
        params = RerankParams(return_documents=True)
        result = backend._build_result(
            scores=[0.8, 0.2],
            documents=["relevant doc", "irrelevant doc"],
            params=params,
            model_id="test/model",
            prompt_tokens=50,
        )
        assert "document" in result.results[0]
        assert result.results[0]["document"]["text"] == "relevant doc"

    def test_return_documents_false_excludes_text(self):
        from inferall.backends.rerank_backend import CrossEncoderRerankerBackend
        backend = CrossEncoderRerankerBackend()
        params = RerankParams(return_documents=False)
        result = backend._build_result(
            scores=[0.8],
            documents=["doc"],
            params=params,
            model_id="test/model",
            prompt_tokens=10,
        )
        assert "document" not in result.results[0]

    def test_empty_documents(self):
        from inferall.backends.rerank_backend import CrossEncoderRerankerBackend
        backend = CrossEncoderRerankerBackend()
        loaded = LoadedModel(
            model_id="test/model",
            backend_name="rerank",
            model=MagicMock(),
            tokenizer=None,
        )
        params = RerankParams()
        result = backend.rerank(loaded, "query", [], params)
        assert result.results == []
        assert result.usage["prompt_tokens"] == 0


# =============================================================================
# Orchestrator Integration Tests
# =============================================================================

class TestOrchestratorRerankDispatch:
    def test_rerank_dispatch(self):
        from inferall.config import EngineConfig
        from inferall.gpu.allocator import GPUAllocator
        from inferall.gpu.manager import GPUManager
        from inferall.orchestrator import Orchestrator
        from inferall.registry.registry import ModelRegistry

        config = EngineConfig(idle_timeout=0)
        registry = MagicMock(spec=ModelRegistry)
        gpu_manager = MagicMock(spec=GPUManager)
        gpu_manager.gpu_assignments = {}
        allocator = MagicMock(spec=GPUAllocator)

        orch = Orchestrator(config, registry, gpu_manager, allocator)

        loaded = LoadedModel(
            model_id="test/reranker",
            backend_name="rerank",
            model=MagicMock(),
            tokenizer=MagicMock(),
        )
        orch.loaded_models["test/reranker"] = loaded
        orch._ref_counts["test/reranker"] = 0

        expected = RerankResult(
            results=[{"index": 0, "relevance_score": 0.9}],
            model="test/reranker",
            usage={"prompt_tokens": 10},
        )

        with patch.object(orch, '_get_backend') as mock_get:
            mock_get.return_value.rerank.return_value = expected
            result = orch.rerank("test/reranker", "query", ["doc"], RerankParams())

        assert result.results[0]["relevance_score"] == 0.9
        assert orch._ref_counts["test/reranker"] == 0


class TestOrchestratorBackendSelection:
    def test_rerank_format_returns_backend(self):
        from inferall.config import EngineConfig
        from inferall.gpu.allocator import GPUAllocator
        from inferall.gpu.manager import GPUManager
        from inferall.orchestrator import Orchestrator
        from inferall.registry.registry import ModelRegistry

        config = EngineConfig(idle_timeout=0)
        registry = MagicMock(spec=ModelRegistry)
        gpu_manager = MagicMock(spec=GPUManager)
        gpu_manager.gpu_assignments = {}
        allocator = MagicMock(spec=GPUAllocator)

        orch = Orchestrator(config, registry, gpu_manager, allocator)
        backend = orch._get_backend(ModelFormat.RERANK)
        assert backend.name == "rerank"

    def test_format_from_backend_name(self):
        from inferall.config import EngineConfig
        from inferall.gpu.allocator import GPUAllocator
        from inferall.gpu.manager import GPUManager
        from inferall.orchestrator import Orchestrator
        from inferall.registry.registry import ModelRegistry

        config = EngineConfig(idle_timeout=0)
        registry = MagicMock(spec=ModelRegistry)
        gpu_manager = MagicMock(spec=GPUManager)
        gpu_manager.gpu_assignments = {}
        allocator = MagicMock(spec=GPUAllocator)

        orch = Orchestrator(config, registry, gpu_manager, allocator)
        assert orch._format_from_backend_name("rerank") == ModelFormat.RERANK


# =============================================================================
# HF Resolver Tests
# =============================================================================

class TestHFResolverRerankDetection:
    def test_text_ranking_pipeline_tag(self):
        from inferall.registry.hf_resolver import HFResolver

        resolver = HFResolver(models_dir=Path("/tmp/test"))
        info = MagicMock()
        info.pipeline_tag = "text-ranking"
        info.tags = []
        info.siblings = []

        fmt, gguf_file = resolver._detect_format("cross-encoder/ms-marco", info, variant=None)
        assert fmt == ModelFormat.RERANK
        assert gguf_file is None


# =============================================================================
# GPU Allocator Tests
# =============================================================================

class TestAllocatorRerankFormat:
    def test_bytes_per_param_has_rerank(self):
        from inferall.gpu.allocator import _BYTES_PER_PARAM
        assert ModelFormat.RERANK in _BYTES_PER_PARAM
        assert _BYTES_PER_PARAM[ModelFormat.RERANK] == 2.0


# =============================================================================
# API Endpoint Tests
# =============================================================================

class TestRerankEndpoint:
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

    def test_rerank_success(self, client, mock_orchestrator):
        mock_orchestrator.rerank.return_value = RerankResult(
            results=[
                {"index": 1, "relevance_score": 0.95},
                {"index": 0, "relevance_score": 0.2},
            ],
            model="test/reranker",
            usage={"prompt_tokens": 50},
        )

        resp = client.post("/v1/rerank", json={
            "model": "test/reranker",
            "query": "What is Python?",
            "documents": ["Python is a snake", "Python is a programming language"],
        })

        assert resp.status_code == 200
        data = resp.json()
        assert len(data["results"]) == 2
        assert data["results"][0]["relevance_score"] == 0.95
        assert data["meta"]["model"] == "test/reranker"

    def test_rerank_model_not_found(self, client, mock_orchestrator):
        from inferall.orchestrator import ModelNotFoundError
        mock_orchestrator.rerank.side_effect = ModelNotFoundError("not found")

        resp = client.post("/v1/rerank", json={
            "model": "missing/model",
            "query": "test",
            "documents": ["doc"],
        })
        assert resp.status_code == 404

    def test_rerank_with_top_n(self, client, mock_orchestrator):
        mock_orchestrator.rerank.return_value = RerankResult(
            results=[{"index": 0, "relevance_score": 0.9}],
            model="test/reranker",
            usage={"prompt_tokens": 20},
        )

        resp = client.post("/v1/rerank", json={
            "model": "test/reranker",
            "query": "query",
            "documents": ["a", "b", "c"],
            "top_n": 1,
        })

        assert resp.status_code == 200

    def test_health_includes_reranking(self, client):
        resp = client.get("/health")
        assert resp.json()["capabilities"]["reranking"] is True
