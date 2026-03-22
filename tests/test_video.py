"""Tests for text-to-video support — backend, orchestrator dispatch, API endpoint."""

from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from inferall.backends.base import (
    LoadedModel,
    VideoGenerationBackend,
    VideoGenerationParams,
    VideoGenerationResult,
)
from inferall.registry.metadata import (
    FORMAT_TO_TASK,
    PIPELINE_TAG_TO_TASK,
    ModelFormat,
    ModelTask,
)


# =============================================================================
# Metadata / Enum Tests
# =============================================================================

class TestVideoEnums:
    def test_model_task_exists(self):
        assert ModelTask.TEXT_TO_VIDEO.value == "text_to_video"

    def test_model_format_exists(self):
        assert ModelFormat.TEXT_TO_VIDEO.value == "text_to_video"

    def test_pipeline_tag_mapping(self):
        assert PIPELINE_TAG_TO_TASK["text-to-video"] == ModelTask.TEXT_TO_VIDEO

    def test_format_to_task_mapping(self):
        assert FORMAT_TO_TASK[ModelFormat.TEXT_TO_VIDEO] == ModelTask.TEXT_TO_VIDEO

    def test_not_in_blocked_tags(self):
        from inferall.registry.hf_resolver import _BLOCKED_TAGS
        assert "text-to-video" not in _BLOCKED_TAGS


# =============================================================================
# Backend Property Tests
# =============================================================================

class TestVideoBackendProperties:
    def test_name(self):
        from inferall.backends.video_backend import VideoDiffusersBackend
        backend = VideoDiffusersBackend()
        assert backend.name == "video"

    def test_parse_size_valid(self):
        from inferall.backends.video_backend import VideoDiffusersBackend
        backend = VideoDiffusersBackend()
        assert backend._parse_size("512x512") == (512, 512)
        assert backend._parse_size("1024x576") == (1024, 576)

    def test_parse_size_invalid(self):
        from inferall.backends.video_backend import VideoDiffusersBackend
        backend = VideoDiffusersBackend()
        assert backend._parse_size("bad") == (512, 512)


class TestExtractFrames:
    def test_frames_list_of_lists(self):
        from inferall.backends.video_backend import VideoDiffusersBackend
        backend = VideoDiffusersBackend()
        mock_result = MagicMock()
        mock_result.frames = [["frame1", "frame2"]]
        assert backend._extract_frames(mock_result) == ["frame1", "frame2"]

    def test_frames_flat_list(self):
        from inferall.backends.video_backend import VideoDiffusersBackend
        backend = VideoDiffusersBackend()
        mock_result = MagicMock()
        mock_result.frames = ["frame1", "frame2"]
        # First element is not a list, so returns frames directly
        assert backend._extract_frames(mock_result) == ["frame1", "frame2"]

    def test_fallback_to_images(self):
        from inferall.backends.video_backend import VideoDiffusersBackend
        backend = VideoDiffusersBackend()
        mock_result = MagicMock(spec=[])
        mock_result.frames = None
        mock_result.images = ["img1", "img2"]
        # Need to handle the hasattr check
        del mock_result.frames
        mock_result.images = ["img1", "img2"]
        assert backend._extract_frames(mock_result) == ["img1", "img2"]


# =============================================================================
# Data Structure Tests
# =============================================================================

class TestVideoGenerationParams:
    def test_defaults(self):
        p = VideoGenerationParams()
        assert p.num_frames == 16
        assert p.fps == 8
        assert p.size == "512x512"
        assert p.num_inference_steps == 50
        assert p.guidance_scale == 7.5
        assert p.output_format == "frames+mp4"

    def test_custom_values(self):
        p = VideoGenerationParams(num_frames=24, fps=12, size="1024x576", seed=42)
        assert p.num_frames == 24
        assert p.fps == 12
        assert p.seed == 42


class TestVideoGenerationResult:
    def test_fields(self):
        r = VideoGenerationResult(
            frames=["f1", "f2"],
            video_b64="mp4data",
            num_frames=2,
            fps=8,
            total_time_ms=5000.0,
        )
        assert len(r.frames) == 2
        assert r.video_b64 == "mp4data"
        assert r.total_time_ms == 5000.0


# =============================================================================
# Orchestrator Tests
# =============================================================================

class TestOrchestratorVideoDispatch:
    def test_generate_video_dispatch(self):
        from inferall.config import EngineConfig
        from inferall.gpu.allocator import GPUAllocator
        from inferall.gpu.manager import GPUManager
        from inferall.orchestrator import Orchestrator
        from inferall.registry.registry import ModelRegistry

        config = EngineConfig(idle_timeout=0)
        orch = Orchestrator(
            config,
            MagicMock(spec=ModelRegistry),
            MagicMock(spec=GPUManager, gpu_assignments={}),
            MagicMock(spec=GPUAllocator),
        )

        loaded = LoadedModel(
            model_id="test/video",
            backend_name="video",
            model=MagicMock(),
            tokenizer=None,
        )
        orch.loaded_models["test/video"] = loaded
        orch._ref_counts["test/video"] = 0

        expected = VideoGenerationResult(frames=["f1"], num_frames=1, fps=8)

        with patch.object(orch, '_get_backend') as mock_get:
            mock_get.return_value.generate_video.return_value = expected
            result = orch.generate_video("test/video", "a cat running", VideoGenerationParams())

        assert result.frames == ["f1"]
        assert orch._ref_counts["test/video"] == 0

    def test_backend_selection(self):
        from inferall.config import EngineConfig
        from inferall.gpu.allocator import GPUAllocator
        from inferall.gpu.manager import GPUManager
        from inferall.orchestrator import Orchestrator
        from inferall.registry.registry import ModelRegistry

        config = EngineConfig(idle_timeout=0)
        orch = Orchestrator(
            config,
            MagicMock(spec=ModelRegistry),
            MagicMock(spec=GPUManager, gpu_assignments={}),
            MagicMock(spec=GPUAllocator),
        )
        backend = orch._get_backend(ModelFormat.TEXT_TO_VIDEO)
        assert backend.name == "video"

    def test_format_from_backend_name(self):
        from inferall.config import EngineConfig
        from inferall.gpu.allocator import GPUAllocator
        from inferall.gpu.manager import GPUManager
        from inferall.orchestrator import Orchestrator
        from inferall.registry.registry import ModelRegistry

        config = EngineConfig(idle_timeout=0)
        orch = Orchestrator(
            config,
            MagicMock(spec=ModelRegistry),
            MagicMock(spec=GPUManager, gpu_assignments={}),
            MagicMock(spec=GPUAllocator),
        )
        assert orch._format_from_backend_name("video") == ModelFormat.TEXT_TO_VIDEO


# =============================================================================
# HF Resolver Tests
# =============================================================================

class TestHFResolverVideoDetection:
    def test_text_to_video_pipeline_tag(self):
        from inferall.registry.hf_resolver import HFResolver

        resolver = HFResolver(models_dir=Path("/tmp/test"))
        info = MagicMock()
        info.pipeline_tag = "text-to-video"
        info.tags = []
        info.siblings = []

        fmt, gguf_file = resolver._detect_format("test/model", info, variant=None)
        assert fmt == ModelFormat.TEXT_TO_VIDEO


# =============================================================================
# GPU Allocator Tests
# =============================================================================

class TestAllocatorVideoFormat:
    def test_bytes_per_param_has_video(self):
        from inferall.gpu.allocator import _BYTES_PER_PARAM
        assert ModelFormat.TEXT_TO_VIDEO in _BYTES_PER_PARAM
        assert _BYTES_PER_PARAM[ModelFormat.TEXT_TO_VIDEO] == 3.0


# =============================================================================
# API Endpoint Tests
# =============================================================================

class TestVideoEndpoint:
    @pytest.fixture
    def mock_orchestrator(self):
        from inferall.orchestrator import Orchestrator
        orch = MagicMock(spec=Orchestrator)
        orch.list_loaded.return_value = []
        return orch

    @pytest.fixture
    def client(self, mock_orchestrator):
        from inferall.api.server import create_app
        from inferall.registry.registry import ModelRegistry
        from starlette.testclient import TestClient

        registry = MagicMock(spec=ModelRegistry)
        registry.list_all.return_value = []
        app = create_app(
            orchestrator=mock_orchestrator,
            registry=registry,
            api_key=None,
        )
        return TestClient(app)

    def test_video_success(self, client, mock_orchestrator):
        mock_orchestrator.generate_video.return_value = VideoGenerationResult(
            frames=["frame1_b64", "frame2_b64"],
            video_b64="mp4_b64_data",
            num_frames=2,
            fps=8,
            total_time_ms=10000.0,
        )

        resp = client.post("/v1/videos/generations", json={
            "model": "test/video",
            "prompt": "a cat running in a field",
        })

        assert resp.status_code == 200
        data = resp.json()
        assert len(data["data"]["frames"]) == 2
        assert data["data"]["video_b64"] == "mp4_b64_data"
        assert data["data"]["num_frames"] == 2
        assert data["data"]["fps"] == 8
        assert data["performance"]["total_time_ms"] == 10000.0

    def test_video_model_not_found(self, client, mock_orchestrator):
        from inferall.orchestrator import ModelNotFoundError
        mock_orchestrator.generate_video.side_effect = ModelNotFoundError("not found")

        resp = client.post("/v1/videos/generations", json={
            "model": "missing/model",
            "prompt": "test",
        })
        assert resp.status_code == 404

    def test_video_custom_params(self, client, mock_orchestrator):
        mock_orchestrator.generate_video.return_value = VideoGenerationResult(
            frames=["f1"], num_frames=1, fps=12,
        )

        resp = client.post("/v1/videos/generations", json={
            "model": "test/video",
            "prompt": "a sunset",
            "num_frames": 24,
            "fps": 12,
            "size": "1024x576",
            "seed": 42,
        })
        assert resp.status_code == 200

    def test_health_includes_video(self, client):
        resp = client.get("/health")
        assert resp.json()["capabilities"]["video_generations"] is True
