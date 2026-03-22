"""Tests for orchestrator multi-modal dispatch methods."""

from datetime import datetime
from unittest.mock import MagicMock, patch

import pytest

from inferall.backends.base import (
    EmbeddingParams,
    EmbeddingResult,
    GenerationParams,
    ImageGenerationParams,
    ImageGenerationResult,
    LoadedModel,
    TranscriptionParams,
    TranscriptionResult,
    TTSParams,
    TTSResult,
)
from inferall.config import EngineConfig
from inferall.gpu.allocator import AllocationPlan, GPUAllocator
from inferall.gpu.manager import GPUManager
from inferall.orchestrator import Orchestrator
from inferall.registry.metadata import ModelFormat, ModelTask
from inferall.registry.registry import ModelRegistry


def _make_orchestrator_with_loaded_model(model_id="test/model", backend_name="embedding"):
    """Create an orchestrator with a pre-loaded model."""
    config = EngineConfig(idle_timeout=0)
    registry = MagicMock(spec=ModelRegistry)
    gpu_manager = MagicMock(spec=GPUManager)
    gpu_manager.gpu_assignments = {}
    allocator = MagicMock(spec=GPUAllocator)

    orch = Orchestrator(config, registry, gpu_manager, allocator)

    loaded = LoadedModel(
        model_id=model_id,
        backend_name=backend_name,
        model=MagicMock(),
        tokenizer=MagicMock(),
    )
    orch.loaded_models[model_id] = loaded
    orch._ref_counts[model_id] = 0

    return orch


class TestEmbedDispatch:
    def test_embed_calls_backend(self):
        orch = _make_orchestrator_with_loaded_model(backend_name="embedding")
        params = EmbeddingParams()
        expected = EmbeddingResult(embeddings=[[0.1, 0.2]], prompt_tokens=5, model="test/model")

        with patch.object(orch, '_get_backend') as mock_get:
            mock_get.return_value.embed.return_value = expected
            result = orch.embed("test/model", ["hello"], params)

        assert result.embeddings == [[0.1, 0.2]]
        # Ref count should be back to 0 after release
        assert orch._ref_counts["test/model"] == 0


class TestTranscribeDispatch:
    def test_transcribe_calls_backend(self):
        orch = _make_orchestrator_with_loaded_model(backend_name="asr")
        params = TranscriptionParams()
        expected = TranscriptionResult(text="Hello world", language="en")

        with patch.object(orch, '_get_backend') as mock_get:
            mock_get.return_value.transcribe.return_value = expected
            result = orch.transcribe("test/model", b"audio_bytes", params)

        assert result.text == "Hello world"
        assert orch._ref_counts["test/model"] == 0


class TestGenerateImageDispatch:
    def test_generate_image_calls_backend(self):
        orch = _make_orchestrator_with_loaded_model(backend_name="diffusion")
        params = ImageGenerationParams()
        expected = ImageGenerationResult(images=["base64data"])

        with patch.object(orch, '_get_backend') as mock_get:
            mock_get.return_value.generate_image.return_value = expected
            result = orch.generate_image("test/model", "a cat", params)

        assert result.images == ["base64data"]
        assert orch._ref_counts["test/model"] == 0


class TestSynthesizeDispatch:
    def test_synthesize_calls_backend(self):
        orch = _make_orchestrator_with_loaded_model(backend_name="tts")
        params = TTSParams()
        expected = TTSResult(audio_bytes=b"wav_data", content_type="audio/wav", sample_rate=24000)

        with patch.object(orch, '_get_backend') as mock_get:
            mock_get.return_value.synthesize.return_value = expected
            result = orch.synthesize("test/model", "Hello", params)

        assert result.audio_bytes == b"wav_data"
        assert orch._ref_counts["test/model"] == 0


class TestGenerateDispatch:
    def test_generate_calls_backend_and_releases(self):
        orch = _make_orchestrator_with_loaded_model(backend_name="transformers")
        params = GenerationParams()

        from inferall.backends.base import GenerationResult
        expected = GenerationResult(text="Hi!", prompt_tokens=5, completion_tokens=2, finish_reason="stop")

        with patch.object(orch, '_get_backend') as mock_get:
            mock_get.return_value.generate.return_value = expected
            result = orch.generate("test/model", [{"role": "user", "content": "Hi"}], params)

        assert result.text == "Hi!"
        assert orch._ref_counts["test/model"] == 0
