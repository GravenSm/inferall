"""Tests for OpenAI-compatible endpoints — completions, moderations, translations, models CRUD, variations."""

from unittest.mock import MagicMock

import pytest

from inferall.api.server import create_app
from inferall.backends.base import GenerationResult
from inferall.orchestrator import ModelNotFoundError, Orchestrator
from inferall.registry.metadata import ModelFormat, ModelTask
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
    reg.get.return_value = None
    return reg


@pytest.fixture
def client(mock_orchestrator, mock_registry):
    from starlette.testclient import TestClient
    app = create_app(
        orchestrator=mock_orchestrator,
        registry=mock_registry,
        api_key=None,
    )
    return TestClient(app)


# =============================================================================
# POST /v1/completions (legacy)
# =============================================================================

class TestCompletionsEndpoint:
    def test_text_completion(self, client, mock_orchestrator):
        mock_orchestrator.generate.return_value = GenerationResult(
            text="Hello world!", prompt_tokens=3, completion_tokens=4,
            finish_reason="stop", total_time_ms=100.0, tokens_per_second=40.0,
        )

        resp = client.post("/v1/completions", json={
            "model": "test/model",
            "prompt": "Say hello",
        })

        assert resp.status_code == 200
        data = resp.json()
        assert data["object"] == "text_completion"
        assert data["choices"][0]["text"] == "Hello world!"
        assert data["choices"][0]["finish_reason"] == "stop"
        assert data["usage"]["completion_tokens"] == 4

    def test_completion_with_echo(self, client, mock_orchestrator):
        mock_orchestrator.generate.return_value = GenerationResult(
            text=" world", prompt_tokens=2, completion_tokens=1,
            finish_reason="stop",
        )

        resp = client.post("/v1/completions", json={
            "model": "test/model",
            "prompt": "Hello",
            "echo": True,
        })

        assert resp.status_code == 200
        assert resp.json()["choices"][0]["text"] == "Hello world"

    def test_completion_model_not_found(self, client, mock_orchestrator):
        mock_orchestrator.generate.side_effect = ModelNotFoundError("not found")

        resp = client.post("/v1/completions", json={
            "model": "missing/model",
            "prompt": "test",
        })
        assert resp.status_code == 404

    def test_completion_list_prompt(self, client, mock_orchestrator):
        mock_orchestrator.generate.return_value = GenerationResult(
            text="response", prompt_tokens=2, completion_tokens=1,
            finish_reason="stop",
        )

        resp = client.post("/v1/completions", json={
            "model": "test/model",
            "prompt": ["first prompt", "ignored"],
        })
        assert resp.status_code == 200


# =============================================================================
# POST /v1/moderations
# =============================================================================

class TestModerationsEndpoint:
    def test_moderation_no_model(self, client):
        """Without a moderation model, returns unflagged passthrough."""
        resp = client.post("/v1/moderations", json={
            "input": "Hello world",
        })

        assert resp.status_code == 200
        data = resp.json()
        assert len(data["results"]) == 1
        assert data["results"][0]["flagged"] is False

    def test_moderation_list_input(self, client):
        resp = client.post("/v1/moderations", json={
            "input": ["text one", "text two"],
        })

        assert resp.status_code == 200
        assert len(resp.json()["results"]) == 2


# =============================================================================
# POST /v1/audio/translations
# =============================================================================

class TestAudioTranslationsEndpoint:
    def test_translation_success(self, client, mock_orchestrator):
        from inferall.backends.base import TranscriptionResult
        mock_orchestrator.transcribe.return_value = TranscriptionResult(
            text="Hello world", language="en", duration=2.5, total_time_ms=200.0,
        )

        resp = client.post(
            "/v1/audio/translations",
            data={"model": "test/whisper"},
            files={"file": ("audio.wav", b"fake_audio", "audio/wav")},
        )

        assert resp.status_code == 200
        data = resp.json()
        assert data["text"] == "Hello world"
        assert data["language"] == "en"

    def test_translation_model_not_found(self, client, mock_orchestrator):
        mock_orchestrator.transcribe.side_effect = ModelNotFoundError("not found")

        resp = client.post(
            "/v1/audio/translations",
            data={"model": "missing"},
            files={"file": ("audio.wav", b"bytes", "audio/wav")},
        )
        assert resp.status_code == 404


# =============================================================================
# GET /v1/models/{model}
# =============================================================================

class TestGetModel:
    def test_model_found(self, client, mock_registry):
        from datetime import datetime
        from pathlib import Path
        from inferall.registry.metadata import ModelRecord

        record = ModelRecord(
            model_id="test/model", revision="abc123",
            format=ModelFormat.TRANSFORMERS, local_path=Path("/tmp"),
            file_size_bytes=1_000_000_000, param_count=7_000_000_000,
            gguf_variant=None, trust_remote_code=False,
            pipeline_tag="text-generation", pulled_at=datetime.now(),
            task=ModelTask.CHAT,
        )
        mock_registry.get.return_value = record

        resp = client.get("/v1/models/test/model")
        assert resp.status_code == 200
        data = resp.json()
        assert data["id"] == "test/model"
        assert data["format"] == "transformers"
        assert data["param_count"] == 7_000_000_000

    def test_model_not_found(self, client, mock_registry):
        mock_registry.get.return_value = None

        resp = client.get("/v1/models/nonexistent/model")
        assert resp.status_code == 404


# =============================================================================
# DELETE /v1/models/{model}
# =============================================================================

class TestDeleteModel:
    def test_delete_success(self, client, mock_registry):
        from datetime import datetime
        from pathlib import Path
        from inferall.registry.metadata import ModelRecord

        record = ModelRecord(
            model_id="test/model", revision="abc123",
            format=ModelFormat.TRANSFORMERS, local_path=Path("/tmp"),
            file_size_bytes=1_000_000_000, param_count=None,
            gguf_variant=None, trust_remote_code=False,
            pipeline_tag="text-generation", pulled_at=datetime.now(),
            task=ModelTask.CHAT,
        )
        mock_registry.get.return_value = record

        resp = client.request("DELETE", "/v1/models/test/model")
        assert resp.status_code == 200
        assert resp.json()["deleted"] is True
        mock_registry.remove.assert_called_once()

    def test_delete_not_found(self, client, mock_registry):
        mock_registry.get.return_value = None

        resp = client.request("DELETE", "/v1/models/missing")
        assert resp.status_code == 404


# =============================================================================
# Health capabilities
# =============================================================================

class TestHealthCapabilities:
    def test_new_capabilities(self, client):
        resp = client.get("/health")
        caps = resp.json()["capabilities"]
        assert caps["completions"] is True
        assert caps["moderations"] is True
        assert caps["audio_translations"] is True
        assert caps["image_variations"] is True
