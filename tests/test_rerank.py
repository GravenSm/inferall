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
# Input Sanitization Tests (Bug 1)
# =============================================================================

class TestRerankSanitization:
    """Empty/None/whitespace docs must not reach the tokenizer as seq_len=0."""

    def _patched_backend(self):
        """CrossEncoder-path backend that records what its dispatch receives."""
        from inferall.backends.rerank_backend import CrossEncoderRerankerBackend
        backend = CrossEncoderRerankerBackend()
        captured = {}

        def fake_cross_encoder(loaded, query, documents, params, original_documents=None):
            captured["sanitized"] = list(documents)
            captured["original"] = list(original_documents) if original_documents is not None else None
            return RerankResult(
                results=[{"index": i, "relevance_score": 0.0} for i in range(len(documents))],
                model=loaded.model_id,
                usage={"prompt_tokens": 0},
            )

        backend._rerank_cross_encoder = fake_cross_encoder
        loaded = LoadedModel(
            model_id="test/model",
            backend_name="rerank",
            model=MagicMock(),
            tokenizer=None,
        )
        return backend, loaded, captured

    def test_none_is_replaced_with_space(self):
        backend, loaded, cap = self._patched_backend()
        backend.rerank(loaded, "q", ["valid", None, "also valid"], RerankParams())
        assert cap["sanitized"] == ["valid", " ", "also valid"]

    def test_empty_string_is_replaced_with_space(self):
        backend, loaded, cap = self._patched_backend()
        backend.rerank(loaded, "q", ["", "valid"], RerankParams())
        assert cap["sanitized"] == [" ", "valid"]

    def test_whitespace_only_is_replaced_with_space(self):
        backend, loaded, cap = self._patched_backend()
        backend.rerank(loaded, "q", ["\n\t  ", "valid"], RerankParams())
        assert cap["sanitized"] == [" ", "valid"]

    def test_original_documents_preserved(self):
        """return_documents must echo original text, not the " " placeholder."""
        backend, loaded, cap = self._patched_backend()
        docs = ["", None, "real"]
        backend.rerank(loaded, "q", docs, RerankParams(return_documents=True))
        assert cap["original"] == docs
        assert cap["sanitized"] == [" ", " ", "real"]

    def test_valid_documents_unchanged(self):
        backend, loaded, cap = self._patched_backend()
        docs = ["a", "b c", "d"]
        backend.rerank(loaded, "q", docs, RerankParams())
        assert cap["sanitized"] == docs


# =============================================================================
# Generative Reranker Detection (Bug 2)
# =============================================================================

class TestGenerativeRerankerDetection:
    def test_id_helper_matches_qwen_reranker(self):
        from inferall.backends.rerank_backend import _is_generative_reranker_id
        assert _is_generative_reranker_id("Qwen/Qwen3-Reranker-8B") is True
        assert _is_generative_reranker_id("qwen/qwen3-reranker-0.6b") is True

    def test_id_helper_matches_gemma_reranker(self):
        from inferall.backends.rerank_backend import _is_generative_reranker_id
        assert _is_generative_reranker_id("google/gemma-reranker-v1") is True

    def test_id_helper_rejects_standard_rerankers(self):
        from inferall.backends.rerank_backend import _is_generative_reranker_id
        assert _is_generative_reranker_id("cross-encoder/ms-marco-MiniLM-L-6-v2") is False
        assert _is_generative_reranker_id("BAAI/bge-reranker-v2-m3") is False

    def test_id_helper_rejects_qwen_without_reranker(self):
        from inferall.backends.rerank_backend import _is_generative_reranker_id
        assert _is_generative_reranker_id("Qwen/Qwen3-8B") is False

    def test_instance_detects_qwen_reranker(self):
        from inferall.backends.rerank_backend import CrossEncoderRerankerBackend
        backend = CrossEncoderRerankerBackend()
        loaded = LoadedModel(
            model_id="Qwen/Qwen3-Reranker-8B",
            backend_name="rerank",
            model=MagicMock(),
            tokenizer=MagicMock(),
        )
        assert backend._is_generative_reranker(loaded) is True

    def test_instance_detects_via_chat_template_and_vocab(self):
        """Fallback heuristic: large vocab + chat_template flags a generative LM."""
        from inferall.backends.rerank_backend import CrossEncoderRerankerBackend
        backend = CrossEncoderRerankerBackend()
        tok = MagicMock()
        tok.chat_template = "{{ messages }}"
        model = MagicMock()
        model.config.vocab_size = 151936  # Qwen3 vocab
        loaded = LoadedModel(
            model_id="some/unknown-reranker",  # not id-matched
            backend_name="rerank",
            model=model,
            tokenizer=tok,
        )
        assert backend._is_generative_reranker(loaded) is True

    def test_instance_rejects_small_vocab_reranker(self):
        from inferall.backends.rerank_backend import CrossEncoderRerankerBackend
        backend = CrossEncoderRerankerBackend()
        tok = MagicMock()
        tok.chat_template = None
        model = MagicMock()
        model.config.architectures = ["BertForSequenceClassification"]
        model.config.vocab_size = 30522  # BERT-ish
        loaded = LoadedModel(
            model_id="cross-encoder/ms-marco-MiniLM-L-6-v2",
            backend_name="rerank",
            model=model,
            tokenizer=tok,
        )
        assert backend._is_generative_reranker(loaded) is False


# =============================================================================
# config.json architectures-based Detection (load-time robustness)
# =============================================================================

class TestArchitectureBasedDetection:
    def test_architectures_helper_matches_for_causal_lm(self):
        from inferall.backends.rerank_backend import _architectures_suggest_causal_lm
        assert _architectures_suggest_causal_lm(["Qwen3ForCausalLM"]) is True
        assert _architectures_suggest_causal_lm(["LlamaForCausalLM"]) is True
        assert _architectures_suggest_causal_lm(["GemmaForCausalLM"]) is True

    def test_architectures_helper_rejects_seq_classification(self):
        from inferall.backends.rerank_backend import _architectures_suggest_causal_lm
        assert _architectures_suggest_causal_lm(["BertForSequenceClassification"]) is False
        assert _architectures_suggest_causal_lm(["XLMRobertaForSequenceClassification"]) is False

    def test_architectures_helper_rejects_seq2seq(self):
        """T5/BART-style models have the wrong logit shape for _rerank_generative."""
        from inferall.backends.rerank_backend import _architectures_suggest_causal_lm
        assert _architectures_suggest_causal_lm(["T5ForConditionalGeneration"]) is False
        assert _architectures_suggest_causal_lm(["BartForConditionalGeneration"]) is False

    def test_architectures_helper_empty_list(self):
        from inferall.backends.rerank_backend import _architectures_suggest_causal_lm
        assert _architectures_suggest_causal_lm([]) is False

    def test_read_architectures_missing_dir(self, tmp_path):
        from inferall.backends.rerank_backend import _read_architectures
        assert _read_architectures(tmp_path / "does-not-exist") == []

    def test_read_architectures_missing_config_file(self, tmp_path):
        from inferall.backends.rerank_backend import _read_architectures
        assert _read_architectures(tmp_path) == []

    def test_read_architectures_malformed_json(self, tmp_path):
        from inferall.backends.rerank_backend import _read_architectures
        (tmp_path / "config.json").write_text("{not json")
        assert _read_architectures(tmp_path) == []

    def test_read_architectures_no_architectures_field(self, tmp_path):
        from inferall.backends.rerank_backend import _read_architectures
        (tmp_path / "config.json").write_text('{"vocab_size": 32000}')
        assert _read_architectures(tmp_path) == []

    def test_read_architectures_valid_config(self, tmp_path):
        from inferall.backends.rerank_backend import _read_architectures
        (tmp_path / "config.json").write_text(
            '{"architectures": ["Qwen3ForCausalLM"], "vocab_size": 151936}'
        )
        assert _read_architectures(tmp_path) == ["Qwen3ForCausalLM"]

    def test_read_architectures_filters_non_strings(self, tmp_path):
        """Malformed architectures entries (not strings) shouldn't crash or propagate."""
        from inferall.backends.rerank_backend import _read_architectures
        (tmp_path / "config.json").write_text(
            '{"architectures": ["Qwen3ForCausalLM", null, 42]}'
        )
        assert _read_architectures(tmp_path) == ["Qwen3ForCausalLM"]

    def test_load_detection_combines_id_and_config(self, tmp_path):
        """id-based fast path + config.json fallback together."""
        from inferall.backends.rerank_backend import _detect_generative_reranker_at_load
        from inferall.registry.metadata import ModelFormat, ModelRecord, ModelTask

        # id says generative → True regardless of config
        rec1 = self._mk_record("Qwen/Qwen3-Reranker-8B", tmp_path)
        assert _detect_generative_reranker_at_load(rec1) is True

        # id doesn't match but config.json says ForCausalLM → True
        (tmp_path / "config.json").write_text(
            '{"architectures": ["LlamaForCausalLM"]}'
        )
        rec2 = self._mk_record("some-org/custom-reranker", tmp_path)
        assert _detect_generative_reranker_at_load(rec2) is True

    def test_load_detection_rejects_seq_classification_config(self, tmp_path):
        from inferall.backends.rerank_backend import _detect_generative_reranker_at_load
        (tmp_path / "config.json").write_text(
            '{"architectures": ["XLMRobertaForSequenceClassification"]}'
        )
        rec = self._mk_record("BAAI/bge-reranker-v2-m3", tmp_path)
        assert _detect_generative_reranker_at_load(rec) is False

    def test_load_detection_missing_config_falls_back_safely(self, tmp_path):
        """No config.json and no id match → treat as non-generative (safe default)."""
        from inferall.backends.rerank_backend import _detect_generative_reranker_at_load
        rec = self._mk_record("cross-encoder/ms-marco-MiniLM-L-6-v2", tmp_path)
        assert _detect_generative_reranker_at_load(rec) is False

    def test_instance_detects_via_config_architectures(self):
        """Runtime detection: model whose id doesn't match but whose config does."""
        from inferall.backends.rerank_backend import CrossEncoderRerankerBackend
        backend = CrossEncoderRerankerBackend()
        tok = MagicMock()
        tok.chat_template = None  # isolate the architectures path
        model = MagicMock()
        model.config.architectures = ["Qwen3ForCausalLM"]
        model.config.vocab_size = 151936
        loaded = LoadedModel(
            model_id="some-org/custom-reranker",  # id doesn't match
            backend_name="rerank",
            model=model,
            tokenizer=tok,
        )
        assert backend._is_generative_reranker(loaded) is True

    @staticmethod
    def _mk_record(model_id, local_path):
        from inferall.registry.metadata import ModelFormat, ModelRecord, ModelTask
        return ModelRecord(
            model_id=model_id,
            revision="abc",
            format=ModelFormat.RERANK,
            local_path=local_path,
            file_size_bytes=0,
            param_count=None,
            gguf_variant=None,
            trust_remote_code=False,
            pipeline_tag="text-ranking",
            pulled_at=datetime.now(),
            task=ModelTask.RERANK,
        )


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
