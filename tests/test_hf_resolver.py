"""Tests for inferall.registry.hf_resolver — format detection, pipeline validation, GGUF selection."""

from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from inferall.registry.hf_resolver import (
    HFResolver,
    UnsupportedModelError,
    _BLOCKED_TAGS,
    _DEFAULT_GGUF_VARIANT,
)
from inferall.registry.metadata import ModelFormat, ModelTask


def _make_resolver(tmp_path=None):
    models_dir = tmp_path or Path("/tmp/test_models")
    return HFResolver(models_dir=models_dir)


def _make_model_info(
    pipeline_tag=None,
    tags=None,
    filenames=None,
    sha="abc123def456",
):
    """Create a mock HF model_info response."""
    info = MagicMock()
    info.pipeline_tag = pipeline_tag
    info.tags = tags or []
    info.sha = sha
    info.siblings = []
    if filenames:
        for f in filenames:
            sib = MagicMock()
            sib.rfilename = f
            info.siblings.append(sib)
    return info


class TestPipelineTagValidation:
    def test_known_tags_allowed(self):
        resolver = _make_resolver()
        # Should not raise
        resolver._validate_pipeline_tag("text-generation", "test/model")
        resolver._validate_pipeline_tag("feature-extraction", "test/model")
        resolver._validate_pipeline_tag("automatic-speech-recognition", "test/model")

    def test_none_tag_allowed(self):
        resolver = _make_resolver()
        resolver._validate_pipeline_tag(None, "test/model")

    def test_blocked_tags_raise(self):
        resolver = _make_resolver()
        for tag in _BLOCKED_TAGS:
            with pytest.raises(UnsupportedModelError):
                resolver._validate_pipeline_tag(tag, "test/model")

    def test_warn_tags_allowed(self):
        resolver = _make_resolver()
        resolver._validate_pipeline_tag("text2text-generation", "test/model")

    def test_unknown_tags_allowed(self):
        resolver = _make_resolver()
        resolver._validate_pipeline_tag("custom-fine-tuned-tag", "test/model")


class TestFormatDetection:
    def test_gguf_files_detected(self):
        resolver = _make_resolver()
        info = _make_model_info(filenames=["model-Q4_K_M.gguf"])
        fmt, gguf_file = resolver._detect_format("test/model", info, variant=None)
        assert fmt == ModelFormat.GGUF
        assert gguf_file == "model-Q4_K_M.gguf"

    def test_gptq_tag_detected(self):
        resolver = _make_resolver()
        info = _make_model_info(tags=["gptq"], filenames=["model.safetensors"])
        fmt, _ = resolver._detect_format("test/model", info, variant=None)
        assert fmt == ModelFormat.GPTQ

    def test_awq_tag_detected(self):
        resolver = _make_resolver()
        info = _make_model_info(tags=["awq"], filenames=["model.safetensors"])
        fmt, _ = resolver._detect_format("test/model", info, variant=None)
        assert fmt == ModelFormat.AWQ

    def test_default_is_transformers(self):
        resolver = _make_resolver()
        info = _make_model_info(filenames=["model.safetensors", "config.json"])
        fmt, _ = resolver._detect_format("test/model", info, variant=None)
        assert fmt == ModelFormat.TRANSFORMERS

    def test_embedding_pipeline_tag(self):
        resolver = _make_resolver()
        info = _make_model_info(pipeline_tag="feature-extraction", filenames=["model.safetensors"])
        fmt, _ = resolver._detect_format("test/model", info, variant=None)
        assert fmt == ModelFormat.EMBEDDING

    def test_asr_pipeline_tag(self):
        resolver = _make_resolver()
        info = _make_model_info(pipeline_tag="automatic-speech-recognition")
        fmt, _ = resolver._detect_format("test/model", info, variant=None)
        assert fmt == ModelFormat.ASR

    def test_diffusion_pipeline_tag(self):
        resolver = _make_resolver()
        info = _make_model_info(pipeline_tag="text-to-image")
        fmt, _ = resolver._detect_format("test/model", info, variant=None)
        assert fmt == ModelFormat.DIFFUSION

    def test_tts_pipeline_tag(self):
        resolver = _make_resolver()
        info = _make_model_info(pipeline_tag="text-to-speech")
        fmt, _ = resolver._detect_format("test/model", info, variant=None)
        assert fmt == ModelFormat.TTS

    def test_vlm_pipeline_tag(self):
        resolver = _make_resolver()
        info = _make_model_info(pipeline_tag="image-text-to-text")
        fmt, _ = resolver._detect_format("test/model", info, variant=None)
        assert fmt == ModelFormat.VISION_LANGUAGE

    def test_explicit_variant_forces_gguf(self):
        resolver = _make_resolver()
        info = _make_model_info(filenames=["model-Q4_K_M.gguf", "model-Q8_0.gguf"])
        fmt, gguf_file = resolver._detect_format("test/model", info, variant="Q8_0")
        assert fmt == ModelFormat.GGUF
        assert "Q8_0" in gguf_file


class TestGGUFVariantSelection:
    def test_single_file(self):
        resolver = _make_resolver()
        result = resolver._select_gguf_file(["model.gguf"], variant=None)
        assert result == "model.gguf"

    def test_prefers_q4_k_m(self):
        resolver = _make_resolver()
        files = ["model-Q2_K.gguf", "model-Q4_K_M.gguf", "model-Q8_0.gguf"]
        result = resolver._select_gguf_file(files, variant=None)
        assert "Q4_K_M" in result

    def test_explicit_variant_match(self):
        resolver = _make_resolver()
        files = ["model-Q2_K.gguf", "model-Q4_K_M.gguf", "model-Q8_0.gguf"]
        result = resolver._select_gguf_file(files, variant="Q8_0")
        assert "Q8_0" in result

    def test_variant_not_found_returns_none(self):
        resolver = _make_resolver()
        files = ["model-Q4_K_M.gguf"]
        # Single file: returned even if variant doesn't match
        result = resolver._select_gguf_file(files, variant="Q3_K_S")
        assert result == "model-Q4_K_M.gguf"

    def test_no_gguf_files_returns_none(self):
        resolver = _make_resolver()
        result = resolver._select_gguf_file([], variant=None)
        assert result is None

    def test_fallback_to_first_when_no_default(self):
        resolver = _make_resolver()
        files = ["model-Q2_K.gguf", "model-Q3_K_L.gguf"]  # no Q4_K_M
        result = resolver._select_gguf_file(files, variant=None)
        assert result == files[0]


class TestHelpers:
    def test_get_dir_size(self, tmp_path):
        resolver = _make_resolver(tmp_path)
        (tmp_path / "file1.txt").write_text("hello")
        (tmp_path / "file2.txt").write_text("world!!")
        size = resolver._get_dir_size(tmp_path)
        assert size == 5 + 7  # "hello" + "world!!"

    def test_get_dir_size_single_file(self, tmp_path):
        resolver = _make_resolver(tmp_path)
        f = tmp_path / "model.bin"
        f.write_bytes(b"x" * 100)
        assert resolver._get_dir_size(f) == 100

    def test_extract_variant_names(self):
        resolver = _make_resolver()
        files = ["model.Q4_K_M.gguf", "model.Q8_0.gguf"]
        names = resolver._extract_variant_names(files)
        assert "Q4_K_M" in names
        assert "Q8_0" in names
