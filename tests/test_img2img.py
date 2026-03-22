"""Tests for image-to-image support — backend, orchestrator dispatch, API endpoint."""

import base64
from datetime import datetime
from io import BytesIO
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from inferall.backends.base import (
    Img2ImgBackend,
    Img2ImgParams,
    Img2ImgResult,
    LoadedModel,
)
from inferall.registry.metadata import (
    FORMAT_TO_TASK,
    PIPELINE_TAG_TO_TASK,
    ModelFormat,
    ModelTask,
)


def _make_test_image_b64():
    """Create a small test image as base64."""
    from PIL import Image
    img = Image.new("RGB", (64, 64), "red")
    buf = BytesIO()
    img.save(buf, format="PNG")
    return base64.b64encode(buf.getvalue()).decode()


# =============================================================================
# Metadata / Enum Tests
# =============================================================================

class TestImg2ImgEnums:
    def test_model_task_exists(self):
        assert ModelTask.IMAGE_TO_IMAGE.value == "image_to_image"

    def test_model_format_exists(self):
        assert ModelFormat.IMAGE_TO_IMAGE.value == "image_to_image"

    def test_pipeline_tag_mapping(self):
        assert PIPELINE_TAG_TO_TASK["image-to-image"] == ModelTask.IMAGE_TO_IMAGE

    def test_format_to_task_mapping(self):
        assert FORMAT_TO_TASK[ModelFormat.IMAGE_TO_IMAGE] == ModelTask.IMAGE_TO_IMAGE

    def test_not_in_blocked_tags(self):
        from inferall.registry.hf_resolver import _BLOCKED_TAGS
        assert "image-to-image" not in _BLOCKED_TAGS


# =============================================================================
# Backend Property Tests
# =============================================================================

class TestImg2ImgBackendProperties:
    def test_name(self):
        from inferall.backends.img2img_backend import Img2ImgDiffusersBackend
        backend = Img2ImgDiffusersBackend()
        assert backend.name == "img2img"

    def test_parse_size_valid(self):
        from inferall.backends.img2img_backend import Img2ImgDiffusersBackend
        backend = Img2ImgDiffusersBackend()
        assert backend._parse_size("512x512") == (512, 512)
        assert backend._parse_size("1024x768") == (1024, 768)

    def test_parse_size_invalid(self):
        from inferall.backends.img2img_backend import Img2ImgDiffusersBackend
        backend = Img2ImgDiffusersBackend()
        assert backend._parse_size("invalid") == (512, 512)


# =============================================================================
# Data Structure Tests
# =============================================================================

class TestImg2ImgParams:
    def test_defaults(self):
        p = Img2ImgParams()
        assert p.strength == 0.75
        assert p.n == 1
        assert p.size is None
        assert p.num_inference_steps == 30
        assert p.guidance_scale == 7.5
        assert p.seed is None

    def test_custom_values(self):
        p = Img2ImgParams(strength=0.5, seed=42, size="256x256")
        assert p.strength == 0.5
        assert p.seed == 42
        assert p.size == "256x256"


class TestImg2ImgResult:
    def test_fields(self):
        r = Img2ImgResult(images=["base64data"], total_time_ms=100.0)
        assert r.images == ["base64data"]
        assert r.total_time_ms == 100.0


# =============================================================================
# Orchestrator Tests
# =============================================================================

class TestOrchestratorImg2ImgDispatch:
    def test_edit_image_dispatch(self):
        from inferall.config import EngineConfig
        from inferall.gpu.allocator import GPUAllocator
        from inferall.gpu.manager import GPUManager
        from inferall.orchestrator import Orchestrator
        from inferall.registry.registry import ModelRegistry

        config = EngineConfig(idle_timeout=0)
        registry = MagicMock(spec=ModelRegistry)
        gpu_manager = MagicMock(spec=GPUManager)
        gpu_manager.gpu_assignments = {}
        allocator = MagicMock(spec=GPUAllocator)

        orch = Orchestrator(config, registry, gpu_manager, allocator)

        loaded = LoadedModel(
            model_id="test/img2img",
            backend_name="img2img",
            model=MagicMock(),
            tokenizer=None,
        )
        orch.loaded_models["test/img2img"] = loaded
        orch._ref_counts["test/img2img"] = 0

        expected = Img2ImgResult(images=["b64data"])

        with patch.object(orch, '_get_backend') as mock_get:
            mock_get.return_value.edit_image.return_value = expected
            result = orch.edit_image("test/img2img", "make it blue", Img2ImgParams())

        assert result.images == ["b64data"]
        assert orch._ref_counts["test/img2img"] == 0

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
        backend = orch._get_backend(ModelFormat.IMAGE_TO_IMAGE)
        assert backend.name == "img2img"

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
        assert orch._format_from_backend_name("img2img") == ModelFormat.IMAGE_TO_IMAGE


# =============================================================================
# HF Resolver Tests
# =============================================================================

class TestHFResolverImg2ImgDetection:
    def test_image_to_image_pipeline_tag(self):
        from inferall.registry.hf_resolver import HFResolver

        resolver = HFResolver(models_dir=Path("/tmp/test"))
        info = MagicMock()
        info.pipeline_tag = "image-to-image"
        info.tags = []
        info.siblings = []

        fmt, gguf_file = resolver._detect_format("test/model", info, variant=None)
        assert fmt == ModelFormat.IMAGE_TO_IMAGE
        assert gguf_file is None


# =============================================================================
# GPU Allocator Tests
# =============================================================================

class TestAllocatorImg2ImgFormat:
    def test_bytes_per_param_has_img2img(self):
        from inferall.gpu.allocator import _BYTES_PER_PARAM
        assert ModelFormat.IMAGE_TO_IMAGE in _BYTES_PER_PARAM


# =============================================================================
# API Endpoint Tests
# =============================================================================

class TestImageEditEndpoint:
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

    def test_edit_image_success(self, client, mock_orchestrator):
        mock_orchestrator.edit_image.return_value = Img2ImgResult(
            images=["edited_image_b64"],
            total_time_ms=500.0,
        )

        resp = client.post("/v1/images/edits", json={
            "model": "test/img2img",
            "prompt": "make it blue",
            "image": "fake_base64_image_data",
            "strength": 0.5,
        })

        assert resp.status_code == 200
        data = resp.json()
        assert len(data["data"]) == 1
        assert data["data"][0]["b64_json"] == "edited_image_b64"
        assert data["performance"]["total_time_ms"] == 500.0

    def test_edit_image_model_not_found(self, client, mock_orchestrator):
        from inferall.orchestrator import ModelNotFoundError
        mock_orchestrator.edit_image.side_effect = ModelNotFoundError("not found")

        resp = client.post("/v1/images/edits", json={
            "model": "missing/model",
            "prompt": "edit",
            "image": "data",
        })
        assert resp.status_code == 404

    def test_invalid_strength_rejected(self, client):
        resp = client.post("/v1/images/edits", json={
            "model": "test/model",
            "prompt": "edit",
            "image": "data",
            "strength": 1.5,
        })
        assert resp.status_code == 400

    def test_health_includes_image_edits(self, client):
        resp = client.get("/health")
        assert resp.json()["capabilities"]["image_edits"] is True
