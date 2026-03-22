"""Tests for inferall.backends — base classes, data structures, and backend properties."""

import threading
from datetime import datetime
from unittest.mock import MagicMock, patch

import pytest

from inferall.backends.base import (
    BaseBackend,
    EmbeddingParams,
    EmbeddingResult,
    GenerationParams,
    GenerationResult,
    ImageGenerationParams,
    ImageGenerationResult,
    LoadedModel,
    TranscriptionParams,
    TranscriptionResult,
    TTSParams,
    TTSResult,
)


class TestGenerationParams:
    def test_defaults(self):
        p = GenerationParams()
        assert p.max_tokens == 2048
        assert p.temperature == 0.7
        assert p.top_p == 0.9
        assert p.top_k == 50
        assert p.repetition_penalty == 1.1
        assert p.stop is None

    def test_custom_values(self):
        p = GenerationParams(max_tokens=512, temperature=0.0, stop=["</s>"])
        assert p.max_tokens == 512
        assert p.temperature == 0.0
        assert p.stop == ["</s>"]


class TestGenerationResult:
    def test_fields(self):
        r = GenerationResult(text="Hello", prompt_tokens=10, completion_tokens=5, finish_reason="stop")
        assert r.text == "Hello"
        assert r.finish_reason == "stop"


class TestEmbeddingParams:
    def test_defaults(self):
        p = EmbeddingParams()
        assert p.normalize is True
        assert p.truncate is True


class TestImageGenerationParams:
    def test_defaults(self):
        p = ImageGenerationParams()
        assert p.n == 1
        assert p.size == "1024x1024"
        assert p.num_inference_steps == 30
        assert p.guidance_scale == 7.5
        assert p.seed is None


class TestTTSParams:
    def test_defaults(self):
        p = TTSParams()
        assert p.voice == "default"
        assert p.speed == 1.0
        assert p.response_format == "wav"


class TestTranscriptionParams:
    def test_defaults(self):
        p = TranscriptionParams()
        assert p.language is None
        assert p.response_format == "json"


class TestLoadedModel:
    def test_creation(self):
        model = LoadedModel(
            model_id="test/model",
            backend_name="transformers",
            model=MagicMock(),
            tokenizer=MagicMock(),
        )
        assert model.model_id == "test/model"
        assert model.backend_name == "transformers"
        assert model.vram_used_bytes == 0

    def test_touch_updates_last_used(self):
        model = LoadedModel(
            model_id="test/model",
            backend_name="transformers",
            model=MagicMock(),
            tokenizer=MagicMock(),
        )
        original_time = model.last_used_at
        # Small sleep to ensure time difference
        import time
        time.sleep(0.01)
        model.touch()
        assert model.last_used_at >= original_time


class TestTransformersBackendProperties:
    def test_name(self):
        from inferall.backends.transformers_backend import TransformersBackend
        backend = TransformersBackend()
        assert backend.name == "transformers"

    def test_strip_thinking_removes_blocks(self):
        from inferall.backends.transformers_backend import TransformersBackend
        text = "<think>some reasoning</think>The actual answer."
        result = TransformersBackend._strip_thinking(text)
        assert result == "The actual answer."

    def test_strip_thinking_removes_unclosed(self):
        from inferall.backends.transformers_backend import TransformersBackend
        text = "<think>still thinking..."
        result = TransformersBackend._strip_thinking(text)
        assert result == ""

    def test_strip_thinking_stops_at_turn_boundary(self):
        from inferall.backends.transformers_backend import TransformersBackend
        text = "Hello there!\nuser: What about..."
        result = TransformersBackend._strip_thinking(text)
        assert result == "Hello there!"

    def test_strip_thinking_removes_role_prefix(self):
        from inferall.backends.transformers_backend import TransformersBackend
        text = "A: The answer is 42."
        result = TransformersBackend._strip_thinking(text)
        assert result == "The answer is 42."


class TestLlamaCppBackendProperties:
    def test_name(self):
        from inferall.backends.llamacpp_backend import LlamaCppBackend
        backend = LlamaCppBackend()
        assert backend.name == "llamacpp"

    def test_find_gguf_file_single(self, tmp_path):
        from inferall.backends.llamacpp_backend import LlamaCppBackend
        backend = LlamaCppBackend()
        gguf_file = tmp_path / "model.gguf"
        gguf_file.touch()
        assert backend._find_gguf_file(tmp_path) == gguf_file

    def test_find_gguf_file_prefers_q4km(self, tmp_path):
        from inferall.backends.llamacpp_backend import LlamaCppBackend
        backend = LlamaCppBackend()
        (tmp_path / "model-Q2_K.gguf").touch()
        (tmp_path / "model-Q4_K_M.gguf").touch()
        (tmp_path / "model-Q8_0.gguf").touch()
        result = backend._find_gguf_file(tmp_path)
        assert "Q4_K_M" in result.name

    def test_find_gguf_file_none_raises(self, tmp_path):
        from inferall.backends.llamacpp_backend import LlamaCppBackend
        backend = LlamaCppBackend()
        with pytest.raises(FileNotFoundError):
            backend._find_gguf_file(tmp_path)

    def test_find_gguf_direct_file(self, tmp_path):
        from inferall.backends.llamacpp_backend import LlamaCppBackend
        backend = LlamaCppBackend()
        gguf_file = tmp_path / "model.gguf"
        gguf_file.touch()
        assert backend._find_gguf_file(gguf_file) == gguf_file
