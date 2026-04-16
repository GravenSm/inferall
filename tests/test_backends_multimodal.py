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


class TestWhisperBackendSegmentExtraction:
    """Unit tests for the transformers WhisperBackend segment-extraction helper.

    Exercises _extract_segments directly with fake processors instead of
    running a real model, since the full transcription path needs audio
    tensors + a loaded Whisper that we can't spin up in CI.
    """

    def test_extracts_segments_from_offsets(self):
        from inferall.backends.asr_backend import WhisperBackend
        backend = WhisperBackend()

        fake_processor = MagicMock()
        fake_processor.batch_decode.return_value = [{
            "text": "Hello world.",
            "offsets": [
                {"text": "Hello ", "timestamp": (0.0, 1.2)},
                {"text": "world.", "timestamp": (1.2, 2.4)},
            ],
        }]

        segments = backend._extract_segments(fake_processor, predicted_ids=MagicMock())

        assert segments is not None
        assert len(segments) == 2
        assert segments[0] == {"id": 0, "start": 0.0, "end": 1.2, "text": "Hello "}
        assert segments[1] == {"id": 1, "start": 1.2, "end": 2.4, "text": "world."}

    def test_returns_none_when_decode_raises(self):
        """Different transformers versions expose offsets differently.
        Any failure must degrade gracefully to text-only, not crash the request."""
        from inferall.backends.asr_backend import WhisperBackend
        backend = WhisperBackend()

        fake_processor = MagicMock()
        fake_processor.batch_decode.side_effect = TypeError(
            "output_offsets not supported by this tokenizer"
        )
        assert backend._extract_segments(fake_processor, MagicMock()) is None

    def test_returns_none_when_decode_shape_unexpected(self):
        from inferall.backends.asr_backend import WhisperBackend
        backend = WhisperBackend()

        fake_processor = MagicMock()
        fake_processor.batch_decode.return_value = "unexpected shape"
        assert backend._extract_segments(fake_processor, MagicMock()) is None

    def test_returns_none_on_empty_offsets(self):
        from inferall.backends.asr_backend import WhisperBackend
        backend = WhisperBackend()

        fake_processor = MagicMock()
        fake_processor.batch_decode.return_value = [{"text": "silence", "offsets": []}]
        assert backend._extract_segments(fake_processor, MagicMock()) is None

    def test_handles_missing_timestamp_fields(self):
        """Corrupted offset entries shouldn't crash — unknown times default to 0.0."""
        from inferall.backends.asr_backend import WhisperBackend
        backend = WhisperBackend()

        fake_processor = MagicMock()
        fake_processor.batch_decode.return_value = [{
            "text": "foo bar",
            "offsets": [
                {"text": "foo"},  # no timestamp key at all
                {"text": "bar", "timestamp": (None, None)},
            ],
        }]

        segments = backend._extract_segments(fake_processor, MagicMock())
        assert segments is not None
        assert len(segments) == 2
        assert segments[0] == {"id": 0, "start": 0.0, "end": 0.0, "text": "foo"}
        assert segments[1]["text"] == "bar"


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
