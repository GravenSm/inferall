"""Tests for remaining pipeline tasks — detection, segmentation, depth, doc QA, audio processing."""

from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from inferall.backends.base import (
    AudioProcessingParams, AudioProcessingResult,
    DepthEstimationParams, DepthEstimationResult,
    DocumentQAParams, DocumentQAResult,
    ImageSegmentationParams, ImageSegmentationResult,
    LoadedModel,
    ObjectDetectionParams, ObjectDetectionResult,
)
from inferall.registry.metadata import (
    PIPELINE_TAG_TO_TASK, ModelFormat, ModelTask,
)


# =============================================================================
# Metadata Tests
# =============================================================================

class TestNewTaskEnums:
    def test_object_detection(self):
        assert ModelTask.OBJECT_DETECTION.value == "object_detection"
        assert PIPELINE_TAG_TO_TASK["object-detection"] == ModelTask.OBJECT_DETECTION
        assert PIPELINE_TAG_TO_TASK["zero-shot-object-detection"] == ModelTask.OBJECT_DETECTION

    def test_image_segmentation(self):
        assert ModelTask.IMAGE_SEGMENTATION.value == "image_segmentation"
        assert PIPELINE_TAG_TO_TASK["image-segmentation"] == ModelTask.IMAGE_SEGMENTATION
        assert PIPELINE_TAG_TO_TASK["mask-generation"] == ModelTask.IMAGE_SEGMENTATION

    def test_depth_estimation(self):
        assert ModelTask.DEPTH_ESTIMATION.value == "depth_estimation"
        assert PIPELINE_TAG_TO_TASK["depth-estimation"] == ModelTask.DEPTH_ESTIMATION

    def test_document_qa(self):
        assert ModelTask.DOCUMENT_QA.value == "document_qa"
        assert PIPELINE_TAG_TO_TASK["document-question-answering"] == ModelTask.DOCUMENT_QA

    def test_audio_to_audio(self):
        assert ModelTask.AUDIO_TO_AUDIO.value == "audio_to_audio"
        assert PIPELINE_TAG_TO_TASK["audio-to-audio"] == ModelTask.AUDIO_TO_AUDIO

    def test_none_in_blocked_tags(self):
        from inferall.registry.hf_resolver import _BLOCKED_TAGS
        for tag in ("object-detection", "zero-shot-object-detection",
                     "image-segmentation", "mask-generation", "depth-estimation",
                     "document-question-answering", "audio-to-audio"):
            assert tag not in _BLOCKED_TAGS


# =============================================================================
# HF Resolver Detection Tests
# =============================================================================

class TestHFResolverNewTags:
    def _detect(self, tag):
        from inferall.registry.hf_resolver import HFResolver
        resolver = HFResolver(models_dir=Path("/tmp/test"))
        info = MagicMock()
        info.pipeline_tag = tag
        info.tags = []
        info.siblings = []
        return resolver._detect_format("test/model", info, variant=None)[0]

    def test_object_detection(self):
        assert self._detect("object-detection") == ModelFormat.CLASSIFICATION

    def test_zero_shot_detection(self):
        assert self._detect("zero-shot-object-detection") == ModelFormat.CLASSIFICATION

    def test_segmentation(self):
        assert self._detect("image-segmentation") == ModelFormat.CLASSIFICATION

    def test_mask_generation(self):
        assert self._detect("mask-generation") == ModelFormat.CLASSIFICATION

    def test_depth(self):
        assert self._detect("depth-estimation") == ModelFormat.CLASSIFICATION

    def test_document_qa(self):
        assert self._detect("document-question-answering") == ModelFormat.CLASSIFICATION

    def test_audio_to_audio(self):
        assert self._detect("audio-to-audio") == ModelFormat.CLASSIFICATION


# =============================================================================
# Data Structure Tests
# =============================================================================

class TestObjectDetectionParams:
    def test_defaults(self):
        p = ObjectDetectionParams()
        assert p.threshold == 0.5

class TestImageSegmentationParams:
    def test_defaults(self):
        p = ImageSegmentationParams()
        assert p.threshold == 0.5

class TestDepthEstimationParams:
    def test_defaults(self):
        p = DepthEstimationParams()
        assert p.image_b64 == ""

class TestDocumentQAParams:
    def test_defaults(self):
        p = DocumentQAParams()
        assert p.question == ""

class TestAudioProcessingParams:
    def test_defaults(self):
        p = AudioProcessingParams()
        assert p.audio_b64 == ""


# =============================================================================
# Orchestrator Dispatch Tests
# =============================================================================

def _make_orch():
    from inferall.config import EngineConfig
    from inferall.gpu.allocator import GPUAllocator
    from inferall.gpu.manager import GPUManager
    from inferall.orchestrator import Orchestrator
    from inferall.registry.registry import ModelRegistry
    config = EngineConfig(idle_timeout=0)
    return Orchestrator(
        config,
        MagicMock(spec=ModelRegistry),
        MagicMock(spec=GPUManager, gpu_assignments={}),
        MagicMock(spec=GPUAllocator),
    )

def _load_model(orch, model_id="test/model"):
    loaded = LoadedModel(
        model_id=model_id, backend_name="classification",
        model=MagicMock(), tokenizer="image-classification",
    )
    orch.loaded_models[model_id] = loaded
    orch._ref_counts[model_id] = 0
    return loaded


class TestOrchestratorDetect:
    def test_dispatch(self):
        orch = _make_orch()
        _load_model(orch)
        expected = ObjectDetectionResult(detections=[{"label": "cat", "score": 0.9}], model="test/model")
        with patch.object(orch, '_get_backend') as m:
            m.return_value.detect_objects.return_value = expected
            result = orch.detect_objects("test/model", ObjectDetectionParams())
        assert len(result.detections) == 1

class TestOrchestratorSegment:
    def test_dispatch(self):
        orch = _make_orch()
        _load_model(orch)
        expected = ImageSegmentationResult(segments=[{"label": "sky"}], model="test/model")
        with patch.object(orch, '_get_backend') as m:
            m.return_value.segment_image.return_value = expected
            result = orch.segment_image("test/model", ImageSegmentationParams())
        assert len(result.segments) == 1

class TestOrchestratorDepth:
    def test_dispatch(self):
        orch = _make_orch()
        _load_model(orch)
        expected = DepthEstimationResult(depth_map_b64="b64", width=512, height=512, model="test/model")
        with patch.object(orch, '_get_backend') as m:
            m.return_value.estimate_depth.return_value = expected
            result = orch.estimate_depth("test/model", DepthEstimationParams())
        assert result.width == 512

class TestOrchestratorDocQA:
    def test_dispatch(self):
        orch = _make_orch()
        _load_model(orch)
        expected = DocumentQAResult(answer="42", score=0.95, model="test/model")
        with patch.object(orch, '_get_backend') as m:
            m.return_value.answer_document.return_value = expected
            result = orch.answer_document("test/model", DocumentQAParams(question="What?"))
        assert result.answer == "42"

class TestOrchestratorAudioProcess:
    def test_dispatch(self):
        orch = _make_orch()
        _load_model(orch)
        expected = AudioProcessingResult(audio_bytes=b"wav", model="test/model")
        with patch.object(orch, '_get_backend') as m:
            m.return_value.process_audio.return_value = expected
            result = orch.process_audio("test/model", AudioProcessingParams())
        assert result.audio_bytes == b"wav"


# =============================================================================
# API Endpoint Tests
# =============================================================================

@pytest.fixture
def client():
    from inferall.api.server import create_app
    from inferall.orchestrator import Orchestrator
    from inferall.registry.registry import ModelRegistry
    from starlette.testclient import TestClient

    orch = MagicMock(spec=Orchestrator)
    orch.list_loaded.return_value = []
    registry = MagicMock(spec=ModelRegistry)
    registry.list_all.return_value = []
    app = create_app(orchestrator=orch, registry=registry, api_key=None)
    client = TestClient(app)
    client._orch = orch  # stash for assertions
    return client


class TestDetectEndpoint:
    def test_success(self, client):
        client._orch.detect_objects.return_value = ObjectDetectionResult(
            detections=[{"label": "cat", "score": 0.9, "box": {"xmin": 0, "ymin": 0, "xmax": 100, "ymax": 100}}],
            model="test/detr", total_time_ms=50.0,
        )
        resp = client.post("/v1/detect", json={"model": "test/detr", "image": "b64data"})
        assert resp.status_code == 200
        assert len(resp.json()["detections"]) == 1

class TestSegmentEndpoint:
    def test_success(self, client):
        client._orch.segment_image.return_value = ImageSegmentationResult(
            segments=[{"label": "sky", "score": 0.8}], model="test/sam", total_time_ms=100.0,
        )
        resp = client.post("/v1/segment", json={"model": "test/sam", "image": "b64data"})
        assert resp.status_code == 200
        assert len(resp.json()["segments"]) == 1

class TestDepthEndpoint:
    def test_success(self, client):
        client._orch.estimate_depth.return_value = DepthEstimationResult(
            depth_map_b64="depthdata", width=512, height=512, model="test/dpt", total_time_ms=80.0,
        )
        resp = client.post("/v1/depth", json={"model": "test/dpt", "image": "b64data"})
        assert resp.status_code == 200
        assert resp.json()["width"] == 512

class TestDocumentQAEndpoint:
    def test_success(self, client):
        client._orch.answer_document.return_value = DocumentQAResult(
            answer="Invoice #123", score=0.92, model="test/layoutlm", total_time_ms=200.0,
        )
        resp = client.post("/v1/document-qa", json={"model": "test/layoutlm", "image": "b64data", "question": "What is the invoice number?"})
        assert resp.status_code == 200
        assert resp.json()["answer"] == "Invoice #123"

class TestAudioProcessEndpoint:
    def test_success(self, client):
        client._orch.process_audio.return_value = AudioProcessingResult(
            audio_bytes=b"processed_wav", content_type="audio/wav", sample_rate=16000, model="test/enhance", total_time_ms=300.0,
        )
        resp = client.post("/v1/audio/process", json={"model": "test/enhance", "audio": "b64data"})
        assert resp.status_code == 200
        assert resp.content == b"processed_wav"

class TestHealthNewCapabilities:
    def test_all_new_capabilities(self, client):
        resp = client.get("/health")
        caps = resp.json()["capabilities"]
        assert caps["object_detection"] is True
        assert caps["image_segmentation"] is True
        assert caps["depth_estimation"] is True
        assert caps["document_qa"] is True
        assert caps["audio_processing"] is True
