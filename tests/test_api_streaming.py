"""Tests for streaming chat completion endpoint."""

import json
from unittest.mock import MagicMock, patch

import pytest

from inferall.api.server import create_app
from inferall.backends.base import GenerationResult
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


class TestStreamingChatCompletions:
    def test_streaming_request_accepted(self, client, mock_orchestrator):
        """Test that a streaming request is accepted and returns SSE content type."""
        # Mock the stream method to yield tokens
        mock_orchestrator.stream.return_value = iter(["Hello", " world", "!"])

        resp = client.post("/v1/chat/completions", json={
            "model": "test/model",
            "messages": [{"role": "user", "content": "Hi"}],
            "stream": True,
        })

        assert resp.status_code == 200
        assert "text/event-stream" in resp.headers.get("content-type", "")

    def test_streaming_contains_done_sentinel(self, client, mock_orchestrator):
        """Test that streaming response ends with [DONE]."""
        mock_orchestrator.stream.return_value = iter(["Hello"])

        resp = client.post("/v1/chat/completions", json={
            "model": "test/model",
            "messages": [{"role": "user", "content": "Hi"}],
            "stream": True,
        })

        assert resp.status_code == 200
        body = resp.text
        assert "[DONE]" in body

    def test_streaming_chunks_have_correct_structure(self, client, mock_orchestrator):
        """Test that streamed chunks follow OpenAI SSE format."""
        mock_orchestrator.stream.return_value = iter(["Hi"])

        resp = client.post("/v1/chat/completions", json={
            "model": "test/model",
            "messages": [{"role": "user", "content": "Hi"}],
            "stream": True,
        })

        body = resp.text
        # Should contain data: lines with JSON
        lines = [l for l in body.split("\n") if l.startswith("data: ") and l != "data: [DONE]"]
        assert len(lines) >= 1

        # Parse first non-DONE data line
        for line in lines:
            data = json.loads(line[6:])  # strip "data: "
            assert "choices" in data
            assert data["object"] == "chat.completion.chunk"
            break


class TestNonStreamingFallback:
    def test_stream_false_uses_generate(self, client, mock_orchestrator):
        """Ensure stream=false calls generate(), not stream()."""
        mock_orchestrator.generate.return_value = GenerationResult(
            text="Response", prompt_tokens=5, completion_tokens=3, finish_reason="stop"
        )

        resp = client.post("/v1/chat/completions", json={
            "model": "test/model",
            "messages": [{"role": "user", "content": "Hi"}],
            "stream": False,
        })

        assert resp.status_code == 200
        mock_orchestrator.generate.assert_called_once()
        mock_orchestrator.stream.assert_not_called()
