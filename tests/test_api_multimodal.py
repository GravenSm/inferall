"""Tests for multi-modal API endpoints — embeddings, ASR, images, TTS."""

from unittest.mock import MagicMock

import pytest

from inferall.api.server import create_app
from inferall.backends.base import (
    EmbeddingResult,
    ImageGenerationResult,
    TranscriptionResult,
    TTSResult,
)
from inferall.orchestrator import ModelNotFoundError, Orchestrator
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
    )
    return TestClient(app)


class TestEmbeddingsEndpoint:
    def test_single_text(self, client, mock_orchestrator):
        mock_orchestrator.embed.return_value = EmbeddingResult(
            embeddings=[[0.1, 0.2, 0.3]],
            prompt_tokens=3,
            model="test/embedding-model",
        )

        resp = client.post("/v1/embeddings", json={
            "model": "test/embedding-model",
            "input": "Hello world",
        })

        assert resp.status_code == 200
        data = resp.json()
        assert data["object"] == "list"
        assert len(data["data"]) == 1
        assert data["data"][0]["embedding"] == [0.1, 0.2, 0.3]
        assert data["usage"]["prompt_tokens"] == 3

    def test_multiple_texts(self, client, mock_orchestrator):
        mock_orchestrator.embed.return_value = EmbeddingResult(
            embeddings=[[0.1], [0.2]],
            prompt_tokens=5,
            model="test/model",
        )

        resp = client.post("/v1/embeddings", json={
            "model": "test/model",
            "input": ["Hello", "World"],
        })

        assert resp.status_code == 200
        assert len(resp.json()["data"]) == 2

    def test_model_not_found(self, client, mock_orchestrator):
        mock_orchestrator.embed.side_effect = ModelNotFoundError("not found")

        resp = client.post("/v1/embeddings", json={
            "model": "missing/model",
            "input": "Hello",
        })
        assert resp.status_code == 404


class TestImageGenerationEndpoint:
    def test_success(self, client, mock_orchestrator):
        mock_orchestrator.generate_image.return_value = ImageGenerationResult(
            images=["base64encodedimage"],
        )

        resp = client.post("/v1/images/generations", json={
            "model": "test/diffusion",
            "prompt": "a cat sitting on a chair",
        })

        assert resp.status_code == 200
        data = resp.json()
        assert len(data["data"]) == 1
        assert data["data"][0]["b64_json"] == "base64encodedimage"

    def test_model_not_found(self, client, mock_orchestrator):
        mock_orchestrator.generate_image.side_effect = ModelNotFoundError("not found")

        resp = client.post("/v1/images/generations", json={
            "model": "missing/model",
            "prompt": "a cat",
        })
        assert resp.status_code == 404


class TestTTSEndpoint:
    def test_success(self, client, mock_orchestrator):
        mock_orchestrator.synthesize.return_value = TTSResult(
            audio_bytes=b"fake_wav_data",
            content_type="audio/wav",
            sample_rate=24000,
        )

        resp = client.post("/v1/audio/speech", json={
            "model": "test/tts",
            "input": "Hello world",
        })

        assert resp.status_code == 200
        assert resp.headers["content-type"] == "audio/wav"
        assert resp.content == b"fake_wav_data"

    def test_model_not_found(self, client, mock_orchestrator):
        mock_orchestrator.synthesize.side_effect = ModelNotFoundError("not found")

        resp = client.post("/v1/audio/speech", json={
            "model": "missing/model",
            "input": "Hello",
        })
        assert resp.status_code == 404


class TestTranscriptionEndpoint:
    def test_success(self, client, mock_orchestrator):
        mock_orchestrator.transcribe.return_value = TranscriptionResult(
            text="Hello world",
            language="en",
            duration=2.5,
        )

        resp = client.post(
            "/v1/audio/transcriptions",
            data={"model": "test/whisper"},
            files={"file": ("audio.wav", b"fake_audio_bytes", "audio/wav")},
        )

        assert resp.status_code == 200
        data = resp.json()
        assert data["text"] == "Hello world"
        assert data["language"] == "en"

    def test_model_not_found(self, client, mock_orchestrator):
        mock_orchestrator.transcribe.side_effect = ModelNotFoundError("not found")

        resp = client.post(
            "/v1/audio/transcriptions",
            data={"model": "missing/model"},
            files={"file": ("audio.wav", b"bytes", "audio/wav")},
        )
        assert resp.status_code == 404
