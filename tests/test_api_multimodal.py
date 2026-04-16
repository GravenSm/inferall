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

    def test_verbose_json_includes_segments_and_task(self, client, mock_orchestrator):
        """response_format=verbose_json must surface backend-populated segments."""
        mock_orchestrator.transcribe.return_value = TranscriptionResult(
            text="Hello world",
            language="en",
            duration=2.5,
            segments=[
                {"id": 0, "start": 0.0, "end": 1.2, "text": "Hello "},
                {"id": 1, "start": 1.2, "end": 2.5, "text": "world"},
            ],
        )

        resp = client.post(
            "/v1/audio/transcriptions",
            data={"model": "test/whisper", "response_format": "verbose_json"},
            files={"file": ("audio.wav", b"fake", "audio/wav")},
        )

        assert resp.status_code == 200
        data = resp.json()
        assert data["text"] == "Hello world"
        assert data["language"] == "en"
        assert data["duration"] == 2.5
        assert data["task"] == "transcribe"
        assert len(data["segments"]) == 2
        assert data["segments"][0]["text"] == "Hello "
        assert data["segments"][0]["start"] == 0.0
        assert data["segments"][0]["end"] == 1.2

    def test_default_json_omits_segments_when_backend_returns_none(self, client, mock_orchestrator):
        """With response_format=json (default) and no segments, body should not have a segments key."""
        mock_orchestrator.transcribe.return_value = TranscriptionResult(
            text="hi",
            language="en",
        )

        resp = client.post(
            "/v1/audio/transcriptions",
            data={"model": "test/whisper"},
            files={"file": ("a.wav", b"x", "audio/wav")},
        )

        assert resp.status_code == 200
        data = resp.json()
        assert "segments" not in data
        assert "task" not in data  # task only emitted for verbose_json

    def test_default_json_surfaces_segments_if_backend_provided(self, client, mock_orchestrator):
        """Defence-in-depth: if a backend populated segments anyway, don't drop them."""
        mock_orchestrator.transcribe.return_value = TranscriptionResult(
            text="hi",
            language="en",
            segments=[{"id": 0, "start": 0.0, "end": 0.8, "text": "hi"}],
        )

        resp = client.post(
            "/v1/audio/transcriptions",
            data={"model": "test/whisper"},
            files={"file": ("a.wav", b"x", "audio/wav")},
        )

        assert resp.status_code == 200
        data = resp.json()
        assert data["segments"] == [{"id": 0, "start": 0.0, "end": 0.8, "text": "hi"}]

    def test_response_format_forwarded_to_transcription_params(self, client, mock_orchestrator):
        mock_orchestrator.transcribe.return_value = TranscriptionResult(text="x")

        client.post(
            "/v1/audio/transcriptions",
            data={"model": "test/w", "response_format": "verbose_json"},
            files={"file": ("a.wav", b"x", "audio/wav")},
        )

        _, _, params = mock_orchestrator.transcribe.call_args.args
        assert params.response_format == "verbose_json"

    def test_invalid_response_format_rejected(self, client, mock_orchestrator):
        resp = client.post(
            "/v1/audio/transcriptions",
            data={"model": "test/w", "response_format": "bogus"},
            files={"file": ("a.wav", b"x", "audio/wav")},
        )
        assert resp.status_code == 400
        assert "response_format" in resp.text.lower()

    def test_text_response_format_returns_plain_text(self, client, mock_orchestrator):
        mock_orchestrator.transcribe.return_value = TranscriptionResult(text="Hello there")

        resp = client.post(
            "/v1/audio/transcriptions",
            data={"model": "test/w", "response_format": "text"},
            files={"file": ("a.wav", b"x", "audio/wav")},
        )

        assert resp.status_code == 200
        assert resp.headers["content-type"].startswith("text/plain")
        assert resp.text == "Hello there"

    def test_uppercase_response_format_normalised(self, client, mock_orchestrator):
        """Match OpenAI's lenient parsing — VERBOSE_JSON should work."""
        mock_orchestrator.transcribe.return_value = TranscriptionResult(
            text="x",
            segments=[{"id": 0, "start": 0.0, "end": 1.0, "text": "x"}],
        )

        resp = client.post(
            "/v1/audio/transcriptions",
            data={"model": "test/w", "response_format": "VERBOSE_JSON"},
            files={"file": ("a.wav", b"x", "audio/wav")},
        )

        assert resp.status_code == 200
        assert "segments" in resp.json()


class TestTranslationEndpoint:
    def test_verbose_json_includes_segments_and_translate_task(self, client, mock_orchestrator):
        mock_orchestrator.transcribe.return_value = TranscriptionResult(
            text="Hello world",
            language="de",  # source language from backend
            duration=2.5,
            segments=[{"id": 0, "start": 0.0, "end": 2.5, "text": "Hello world"}],
        )

        resp = client.post(
            "/v1/audio/translations",
            data={"model": "test/whisper", "response_format": "verbose_json"},
            files={"file": ("a.wav", b"x", "audio/wav")},
        )

        assert resp.status_code == 200
        data = resp.json()
        assert data["text"] == "Hello world"
        # Translations always report English as the output language
        assert data["language"] == "en"
        assert data["task"] == "translate"
        assert data["segments"][0]["text"] == "Hello world"

    def test_default_json_omits_segments(self, client, mock_orchestrator):
        mock_orchestrator.transcribe.return_value = TranscriptionResult(text="hi", language="fr")

        resp = client.post(
            "/v1/audio/translations",
            data={"model": "test/whisper"},
            files={"file": ("a.wav", b"x", "audio/wav")},
        )

        assert resp.status_code == 200
        data = resp.json()
        assert "segments" not in data
        assert data["language"] == "en"  # always English on /translations

    def test_translation_response_format_forwarded(self, client, mock_orchestrator):
        mock_orchestrator.transcribe.return_value = TranscriptionResult(text="x")

        client.post(
            "/v1/audio/translations",
            data={"model": "test/w", "response_format": "verbose_json"},
            files={"file": ("a.wav", b"x", "audio/wav")},
        )

        _, _, params = mock_orchestrator.transcribe.call_args.args
        assert params.response_format == "verbose_json"
        assert params.task == "translate"
