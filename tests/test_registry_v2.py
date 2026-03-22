"""Tests for registry v2 migration — task column and multi-modal support."""

from datetime import datetime
from pathlib import Path

import pytest

from inferall.registry.metadata import (
    FORMAT_TO_TASK,
    PIPELINE_TAG_TO_TASK,
    ModelFormat,
    ModelRecord,
    ModelTask,
)
from inferall.registry.registry import ModelRegistry


def _make_record(model_id="test/model", **kwargs):
    defaults = dict(
        model_id=model_id,
        revision="abc123",
        format=ModelFormat.TRANSFORMERS,
        local_path=Path("/tmp/models/test"),
        file_size_bytes=500_000_000,
        param_count=1_000_000_000,
        gguf_variant=None,
        trust_remote_code=False,
        pipeline_tag="text-generation",
        pulled_at=datetime.now(),
        task=ModelTask.CHAT,
    )
    defaults.update(kwargs)
    return ModelRecord(**defaults)


class TestTaskColumn:
    """Test that the task column works correctly after v2 migration."""

    @pytest.fixture
    def registry(self, tmp_path):
        reg = ModelRegistry(tmp_path / "test.db")
        yield reg
        reg.close()

    def test_store_and_retrieve_chat_task(self, registry):
        registry.register(_make_record(task=ModelTask.CHAT))
        record = registry.get("test/model")
        assert record.task == ModelTask.CHAT

    def test_store_and_retrieve_embedding_task(self, registry):
        registry.register(_make_record(
            model_id="sentence-transformers/all-MiniLM-L6-v2",
            format=ModelFormat.EMBEDDING,
            task=ModelTask.EMBEDDING,
        ))
        record = registry.get("sentence-transformers/all-MiniLM-L6-v2")
        assert record.task == ModelTask.EMBEDDING

    def test_store_and_retrieve_asr_task(self, registry):
        registry.register(_make_record(
            model_id="openai/whisper-large-v3",
            format=ModelFormat.ASR,
            task=ModelTask.ASR,
        ))
        record = registry.get("openai/whisper-large-v3")
        assert record.task == ModelTask.ASR

    def test_store_and_retrieve_diffusion_task(self, registry):
        registry.register(_make_record(
            model_id="stabilityai/stable-diffusion-xl-base-1.0",
            format=ModelFormat.DIFFUSION,
            task=ModelTask.DIFFUSION,
        ))
        record = registry.get("stabilityai/stable-diffusion-xl-base-1.0")
        assert record.task == ModelTask.DIFFUSION

    def test_store_and_retrieve_tts_task(self, registry):
        registry.register(_make_record(
            model_id="suno/bark",
            format=ModelFormat.TTS,
            task=ModelTask.TTS,
        ))
        record = registry.get("suno/bark")
        assert record.task == ModelTask.TTS

    def test_store_and_retrieve_vlm_task(self, registry):
        registry.register(_make_record(
            model_id="llava-hf/llava-1.5-7b-hf",
            format=ModelFormat.VISION_LANGUAGE,
            task=ModelTask.VISION_LANGUAGE,
        ))
        record = registry.get("llava-hf/llava-1.5-7b-hf")
        assert record.task == ModelTask.VISION_LANGUAGE


class TestPipelineTagMapping:
    """Test PIPELINE_TAG_TO_TASK mappings."""

    def test_text_generation(self):
        assert PIPELINE_TAG_TO_TASK["text-generation"] == ModelTask.CHAT

    def test_conversational(self):
        assert PIPELINE_TAG_TO_TASK["conversational"] == ModelTask.CHAT

    def test_feature_extraction(self):
        assert PIPELINE_TAG_TO_TASK["feature-extraction"] == ModelTask.EMBEDDING

    def test_asr(self):
        assert PIPELINE_TAG_TO_TASK["automatic-speech-recognition"] == ModelTask.ASR

    def test_text_to_image(self):
        assert PIPELINE_TAG_TO_TASK["text-to-image"] == ModelTask.DIFFUSION

    def test_text_to_speech(self):
        assert PIPELINE_TAG_TO_TASK["text-to-speech"] == ModelTask.TTS

    def test_image_text_to_text(self):
        assert PIPELINE_TAG_TO_TASK["image-text-to-text"] == ModelTask.VISION_LANGUAGE


class TestFormatToTaskMapping:
    """Test FORMAT_TO_TASK mappings."""

    def test_all_formats_mapped(self):
        for fmt in ModelFormat:
            assert fmt in FORMAT_TO_TASK, f"ModelFormat.{fmt.name} missing from FORMAT_TO_TASK"

    def test_chat_formats(self):
        chat_formats = [
            ModelFormat.TRANSFORMERS,
            ModelFormat.TRANSFORMERS_BNB_4BIT,
            ModelFormat.TRANSFORMERS_BNB_8BIT,
            ModelFormat.GPTQ,
            ModelFormat.AWQ,
            ModelFormat.GGUF,
        ]
        for fmt in chat_formats:
            assert FORMAT_TO_TASK[fmt] == ModelTask.CHAT
