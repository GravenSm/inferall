"""Tests for inferall.api.server — API endpoints, auth, error handling."""

import json
from unittest.mock import MagicMock, patch

import pytest

from inferall.api.server import create_app, _error_response
from inferall.backends.base import GenerationParams, GenerationResult
from inferall.orchestrator import ModelNotFoundError, Orchestrator
from inferall.registry.metadata import ModelFormat, ModelRecord, ModelTask
from inferall.registry.registry import ModelRegistry


@pytest.fixture
def mock_orchestrator():
    orch = MagicMock(spec=Orchestrator)
    orch.list_loaded.return_value = []
    return orch


@pytest.fixture
def mock_registry():
    reg = MagicMock(spec=ModelRegistry)
    reg.list_all.return_value = []
    return reg


@pytest.fixture
def client(mock_orchestrator, mock_registry):
    from starlette.testclient import TestClient
    app = create_app(
        orchestrator=mock_orchestrator,
        registry=mock_registry,
        api_key=None,
        compat_mode="strict",
    )
    return TestClient(app)


@pytest.fixture
def auth_client(mock_orchestrator, mock_registry):
    from starlette.testclient import TestClient
    app = create_app(
        orchestrator=mock_orchestrator,
        registry=mock_registry,
        api_key="test-secret-key",
        compat_mode="strict",
    )
    return TestClient(app)


class TestHealthEndpoint:
    def test_health_ok(self, client):
        resp = client.get("/health")
        assert resp.status_code == 200
        data = resp.json()
        assert data["status"] == "ok"
        assert data["capabilities"]["chat_completions"] is True
        assert data["capabilities"]["tools"] is True

    def test_health_no_auth_required(self, auth_client):
        resp = auth_client.get("/health")
        assert resp.status_code == 200


class TestAPIKeyAuth:
    def test_no_auth_required_when_no_key(self, client):
        resp = client.get("/v1/models")
        assert resp.status_code == 200

    def test_missing_auth_returns_401(self, auth_client):
        resp = auth_client.get("/v1/models")
        assert resp.status_code == 401

    def test_wrong_key_returns_401(self, auth_client):
        resp = auth_client.get(
            "/v1/models",
            headers={"Authorization": "Bearer wrong-key"},
        )
        assert resp.status_code == 401

    def test_correct_key_passes(self, auth_client):
        resp = auth_client.get(
            "/v1/models",
            headers={"Authorization": "Bearer test-secret-key"},
        )
        assert resp.status_code == 200

    def test_malformed_auth_header(self, auth_client):
        resp = auth_client.get(
            "/v1/models",
            headers={"Authorization": "Token test-secret-key"},
        )
        assert resp.status_code == 401


class TestListModels:
    def test_empty_list(self, client):
        resp = client.get("/v1/models")
        assert resp.status_code == 200
        data = resp.json()
        assert data["object"] == "list"
        assert data["data"] == []

    def test_lists_registered_models(self, client, mock_registry):
        from datetime import datetime
        record = MagicMock()
        record.model_id = "test/model"
        record.pulled_at = datetime.now()
        record.task = ModelTask.CHAT
        record.format = ModelFormat.TRANSFORMERS
        record.preferred_engine = None
        mock_registry.list_all.return_value = [record]

        resp = client.get("/v1/models")
        data = resp.json()
        assert len(data["data"]) == 1
        entry = data["data"][0]
        assert entry["id"] == "test/model"
        # New fields the TUI Models tab depends on
        assert entry["format"] == "transformers"
        assert entry["preferred_engine"] is None

    def test_lists_models_with_vllm_preference(self, client, mock_registry):
        """Models opted into the vllm backend must surface preferred_engine='vllm'."""
        from datetime import datetime
        vllm_model = MagicMock()
        vllm_model.model_id = "datalab-to/chandra-ocr-2"
        vllm_model.pulled_at = datetime.now()
        vllm_model.task = ModelTask.VISION_LANGUAGE
        vllm_model.format = ModelFormat.VISION_LANGUAGE
        vllm_model.preferred_engine = "vllm"

        default_model = MagicMock()
        default_model.model_id = "Qwen/Qwen2.5-1.5B-Instruct"
        default_model.pulled_at = datetime.now()
        default_model.task = ModelTask.CHAT
        default_model.format = ModelFormat.TRANSFORMERS
        default_model.preferred_engine = None

        mock_registry.list_all.return_value = [vllm_model, default_model]

        resp = client.get("/v1/models")
        data = resp.json()
        assert len(data["data"]) == 2

        by_id = {m["id"]: m for m in data["data"]}
        assert by_id["datalab-to/chandra-ocr-2"]["preferred_engine"] == "vllm"
        assert by_id["datalab-to/chandra-ocr-2"]["format"] == "vision_language"
        assert by_id["Qwen/Qwen2.5-1.5B-Instruct"]["preferred_engine"] is None
        assert by_id["Qwen/Qwen2.5-1.5B-Instruct"]["format"] == "transformers"

    def test_get_model_includes_preferred_engine(self, client, mock_registry):
        """The single-model endpoint must also expose preferred_engine."""
        from datetime import datetime
        record = MagicMock()
        record.model_id = "test/vllm-model"
        record.pulled_at = datetime.now()
        record.task = ModelTask.CHAT
        record.format = ModelFormat.TRANSFORMERS
        record.file_size_bytes = 1024
        record.param_count = 1_000_000
        record.preferred_engine = "vllm"
        mock_registry.get.return_value = record

        resp = client.get("/v1/models/test/vllm-model")
        assert resp.status_code == 200
        data = resp.json()
        assert data["preferred_engine"] == "vllm"
        assert data["format"] == "transformers"


class TestChatCompletions:
    def test_non_streaming_success(self, client, mock_orchestrator):
        mock_orchestrator.generate.return_value = GenerationResult(
            text="Hello!", prompt_tokens=5, completion_tokens=2, finish_reason="stop"
        )

        resp = client.post("/v1/chat/completions", json={
            "model": "test/model",
            "messages": [{"role": "user", "content": "Hi"}],
        })

        assert resp.status_code == 200
        data = resp.json()
        assert data["choices"][0]["message"]["content"] == "Hello!"
        assert data["choices"][0]["finish_reason"] == "stop"
        assert data["usage"]["prompt_tokens"] == 5
        assert data["usage"]["completion_tokens"] == 2

    def test_model_not_found_returns_404(self, client, mock_orchestrator):
        mock_orchestrator.generate.side_effect = ModelNotFoundError("not found")

        resp = client.post("/v1/chat/completions", json={
            "model": "missing/model",
            "messages": [{"role": "user", "content": "Hi"}],
        })

        assert resp.status_code == 404

    def test_n_greater_than_1_rejected(self, client):
        resp = client.post("/v1/chat/completions", json={
            "model": "test/model",
            "messages": [{"role": "user", "content": "Hi"}],
            "n": 2,
        })
        assert resp.status_code == 400

    def test_tools_passed_through(self, client, mock_orchestrator):
        mock_orchestrator.generate.return_value = GenerationResult(
            text="", prompt_tokens=5, completion_tokens=2, finish_reason="tool_calls"
        )

        resp = client.post("/v1/chat/completions", json={
            "model": "test/model",
            "messages": [{"role": "user", "content": "What's the weather?"}],
            "tools": [{"type": "function", "function": {"name": "get_weather", "parameters": {}}}],
        })
        assert resp.status_code == 200
        # Verify tools were passed to the orchestrator
        call_args = mock_orchestrator.generate.call_args
        params = call_args[0][2]  # 3rd positional arg is params
        assert params.tools is not None

    def test_legacy_functions_converted(self, client, mock_orchestrator):
        mock_orchestrator.generate.return_value = GenerationResult(
            text="OK", prompt_tokens=5, completion_tokens=1, finish_reason="stop"
        )

        resp = client.post("/v1/chat/completions", json={
            "model": "test/model",
            "messages": [{"role": "user", "content": "Hi"}],
            "functions": [{"name": "test", "parameters": {}}],
        })
        assert resp.status_code == 200


class TestChatCompletionsLenient:
    @pytest.fixture
    def lenient_client(self, mock_orchestrator, mock_registry):
        from starlette.testclient import TestClient
        app = create_app(
            orchestrator=mock_orchestrator,
            registry=mock_registry,
            api_key=None,
            compat_mode="lenient",
        )
        return TestClient(app)

    def test_tools_stripped_in_lenient_mode(self, lenient_client, mock_orchestrator):
        mock_orchestrator.generate.return_value = GenerationResult(
            text="OK", prompt_tokens=5, completion_tokens=1, finish_reason="stop"
        )

        resp = lenient_client.post("/v1/chat/completions", json={
            "model": "test/model",
            "messages": [{"role": "user", "content": "Hi"}],
            "tools": [{"type": "function", "function": {"name": "test"}}],
        })
        assert resp.status_code == 200


class TestErrorResponse:
    def test_format(self):
        resp = _error_response(400, "Bad request", code="test_error")
        assert resp.status_code == 400
        body = json.loads(resp.body)
        assert body["error"]["message"] == "Bad request"
        assert body["error"]["code"] == "test_error"
