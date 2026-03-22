"""Tests for multi-modal backend properties and data structures."""

from unittest.mock import MagicMock

import pytest

from inferall.backends.base import (
    ASRBackend,
    DiffusionBackend,
    EmbeddingBackend,
    TTSBackend,
    VisionLanguageBackend,
)


class TestEmbeddingBackendProperties:
    def test_name(self):
        from inferall.backends.embedding_backend import SentenceTransformersBackend
        backend = SentenceTransformersBackend()
        assert backend.name == "embedding"

    def test_resolve_device_with_gpu(self):
        from inferall.backends.embedding_backend import SentenceTransformersBackend
        from inferall.gpu.allocator import AllocationPlan
        backend = SentenceTransformersBackend()
        plan = AllocationPlan(gpu_ids=[0])
        assert backend._resolve_device(plan) == "cuda:0"

    def test_resolve_device_cpu_fallback(self):
        from inferall.backends.embedding_backend import SentenceTransformersBackend
        from inferall.gpu.allocator import AllocationPlan
        backend = SentenceTransformersBackend()
        plan = AllocationPlan(gpu_ids=[])
        assert backend._resolve_device(plan) == "cpu"


class TestVLMBackendProperties:
    def test_name(self):
        from inferall.backends.vlm_backend import VisionLanguageTransformersBackend
        backend = VisionLanguageTransformersBackend()
        assert backend.name == "vlm"


class TestASRBackendProperties:
    def test_name(self):
        from inferall.backends.asr_backend import WhisperBackend
        backend = WhisperBackend()
        assert backend.name == "asr"


class TestDiffusionBackendProperties:
    def test_name(self):
        from inferall.backends.diffusion_backend import DiffusersBackend
        backend = DiffusersBackend()
        assert backend.name == "diffusion"

    def test_parse_size_valid(self):
        from inferall.backends.diffusion_backend import DiffusersBackend
        backend = DiffusersBackend()
        assert backend._parse_size("512x512") == (512, 512)
        assert backend._parse_size("1024x768") == (1024, 768)

    def test_parse_size_invalid(self):
        from inferall.backends.diffusion_backend import DiffusersBackend
        backend = DiffusersBackend()
        assert backend._parse_size("invalid") == (1024, 1024)
        assert backend._parse_size("axb") == (1024, 1024)


class TestTTSBackendProperties:
    def test_name(self):
        from inferall.backends.tts_backend import TTSTransformersBackend
        backend = TTSTransformersBackend()
        assert backend.name == "tts"
