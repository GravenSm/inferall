"""Tests for seq2seq support — backend, orchestrator dispatch, API endpoint."""

from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from inferall.backends.base import (
    LoadedModel,
    Seq2SeqBackend,
    Seq2SeqParams,
    Seq2SeqResult,
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

class TestSeq2SeqEnums:
    def test_model_task_exists(self):
        assert ModelTask.SEQ2SEQ.value == "seq2seq"

    def test_model_format_exists(self):
        assert ModelFormat.SEQ2SEQ.value == "seq2seq"

    def test_pipeline_tag_translation(self):
        assert PIPELINE_TAG_TO_TASK["translation"] == ModelTask.SEQ2SEQ

    def test_pipeline_tag_summarization(self):
        assert PIPELINE_TAG_TO_TASK["summarization"] == ModelTask.SEQ2SEQ

    def test_pipeline_tag_text2text(self):
        assert PIPELINE_TAG_TO_TASK["text2text-generation"] == ModelTask.SEQ2SEQ

    def test_format_to_task_mapping(self):
        assert FORMAT_TO_TASK[ModelFormat.SEQ2SEQ] == ModelTask.SEQ2SEQ

    def test_not_in_blocked_tags(self):
        from inferall.registry.hf_resolver import _BLOCKED_TAGS
        assert "translation" not in _BLOCKED_TAGS
        assert "summarization" not in _BLOCKED_TAGS
        assert "text2text-generation" not in _BLOCKED_TAGS


# =============================================================================
# Backend Property Tests
# =============================================================================

class TestSeq2SeqBackendProperties:
    def test_name(self):
        from inferall.backends.seq2seq_backend import Seq2SeqTransformersBackend
        backend = Seq2SeqTransformersBackend()
        assert backend.name == "seq2seq"


# =============================================================================
# Data Structure Tests
# =============================================================================

class TestSeq2SeqParams:
    def test_defaults(self):
        p = Seq2SeqParams()
        assert p.max_tokens == 512
        assert p.temperature == 1.0
        assert p.num_beams == 4
        assert p.source_lang is None
        assert p.target_lang is None

    def test_custom_values(self):
        p = Seq2SeqParams(
            max_tokens=256, num_beams=8,
            source_lang="en", target_lang="fr",
        )
        assert p.num_beams == 8
        assert p.target_lang == "fr"


class TestSeq2SeqResult:
    def test_fields(self):
        r = Seq2SeqResult(
            text="Bonjour le monde",
            prompt_tokens=5,
            completion_tokens=4,
            total_time_ms=100.0,
            tokens_per_second=40.0,
        )
        assert r.text == "Bonjour le monde"
        assert r.tokens_per_second == 40.0


# =============================================================================
# Orchestrator Tests
# =============================================================================

class TestOrchestratorSeq2SeqDispatch:
    def test_seq2seq_generate_dispatch(self):
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
            model_id="test/t5",
            backend_name="seq2seq",
            model=MagicMock(),
            tokenizer=MagicMock(),
        )
        orch.loaded_models["test/t5"] = loaded
        orch._ref_counts["test/t5"] = 0

        expected = Seq2SeqResult(text="output", prompt_tokens=5, completion_tokens=3)

        with patch.object(orch, '_get_backend') as mock_get:
            mock_get.return_value.generate.return_value = expected
            result = orch.seq2seq_generate("test/t5", "input text", Seq2SeqParams())

        assert result.text == "output"
        assert orch._ref_counts["test/t5"] == 0

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
        backend = orch._get_backend(ModelFormat.SEQ2SEQ)
        assert backend.name == "seq2seq"

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
        assert orch._format_from_backend_name("seq2seq") == ModelFormat.SEQ2SEQ


# =============================================================================
# HF Resolver Tests
# =============================================================================

class TestHFResolverSeq2SeqDetection:
    def test_translation_tag(self):
        from inferall.registry.hf_resolver import HFResolver
        resolver = HFResolver(models_dir=Path("/tmp/test"))
        info = MagicMock()
        info.pipeline_tag = "translation"
        info.tags = []
        info.siblings = []
        fmt, _ = resolver._detect_format("test/model", info, variant=None)
        assert fmt == ModelFormat.SEQ2SEQ

    def test_summarization_tag(self):
        from inferall.registry.hf_resolver import HFResolver
        resolver = HFResolver(models_dir=Path("/tmp/test"))
        info = MagicMock()
        info.pipeline_tag = "summarization"
        info.tags = []
        info.siblings = []
        fmt, _ = resolver._detect_format("test/model", info, variant=None)
        assert fmt == ModelFormat.SEQ2SEQ

    def test_text2text_tag(self):
        from inferall.registry.hf_resolver import HFResolver
        resolver = HFResolver(models_dir=Path("/tmp/test"))
        info = MagicMock()
        info.pipeline_tag = "text2text-generation"
        info.tags = []
        info.siblings = []
        fmt, _ = resolver._detect_format("test/model", info, variant=None)
        assert fmt == ModelFormat.SEQ2SEQ


# =============================================================================
# GPU Allocator Tests
# =============================================================================

class TestAllocatorSeq2SeqFormat:
    def test_bytes_per_param_has_seq2seq(self):
        from inferall.gpu.allocator import _BYTES_PER_PARAM
        assert ModelFormat.SEQ2SEQ in _BYTES_PER_PARAM


# =============================================================================
# API Endpoint Tests
# =============================================================================

class TestSeq2SeqEndpoint:
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

    def test_text_generate_success(self, client, mock_orchestrator):
        mock_orchestrator.seq2seq_generate.return_value = Seq2SeqResult(
            text="Bonjour le monde",
            prompt_tokens=5,
            completion_tokens=4,
            total_time_ms=200.0,
            tokens_per_second=20.0,
        )

        resp = client.post("/v1/text/generate", json={
            "model": "test/t5",
            "input": "Translate to French: Hello world",
        })

        assert resp.status_code == 200
        data = resp.json()
        assert data["text"] == "Bonjour le monde"
        assert data["usage"]["prompt_tokens"] == 5
        assert data["usage"]["completion_tokens"] == 4
        assert data["performance"]["tokens_per_second"] == 20.0

    def test_model_not_found(self, client, mock_orchestrator):
        from inferall.orchestrator import ModelNotFoundError
        mock_orchestrator.seq2seq_generate.side_effect = ModelNotFoundError("not found")

        resp = client.post("/v1/text/generate", json={
            "model": "missing/model",
            "input": "test",
        })
        assert resp.status_code == 404

    def test_with_language_params(self, client, mock_orchestrator):
        mock_orchestrator.seq2seq_generate.return_value = Seq2SeqResult(
            text="Hola mundo", prompt_tokens=3, completion_tokens=3,
        )

        resp = client.post("/v1/text/generate", json={
            "model": "test/nllb",
            "input": "Hello world",
            "source_lang": "eng_Latn",
            "target_lang": "spa_Latn",
        })
        assert resp.status_code == 200

    def test_health_includes_seq2seq(self, client):
        resp = client.get("/health")
        assert resp.json()["capabilities"]["seq2seq"] is True
