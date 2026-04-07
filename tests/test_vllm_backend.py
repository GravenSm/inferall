"""
Tests for the vLLM backend (subprocess + HTTP).

These tests run without vllm installed — the actual subprocess and HTTP
calls are mocked. Integration with a real vllm runtime is exercised by
the end-to-end benchmark, not unit tests.
"""

import json
import sys
import threading
from datetime import datetime
from io import BytesIO
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from inferall.backends.base import GenerationParams, LoadedModel
from inferall.backends.vllm_backend import (
    VLLMTextBackend,
    VLLMVisionBackend,
    _parse_size,
    _pick_free_port,
    _VLLMProcess,
)
from inferall.gpu.allocator import AllocationPlan
from inferall.registry.metadata import ModelFormat, ModelRecord


# =============================================================================
# Helpers
# =============================================================================

def _make_record(tmp_path, **kwargs) -> ModelRecord:
    defaults = dict(
        model_id="test/vllm-model",
        revision="abc123",
        format=ModelFormat.TRANSFORMERS,
        local_path=tmp_path,
        file_size_bytes=1024,
        param_count=7_000_000_000,
        gguf_variant=None,
        trust_remote_code=False,
        pipeline_tag="text-generation",
        pulled_at=datetime(2026, 1, 1),
        preferred_engine="vllm",
    )
    defaults.update(kwargs)
    return ModelRecord(**defaults)


def _make_loaded_with_mock_proc(model_id: str = "test/vllm-model") -> LoadedModel:
    """Build a LoadedModel whose .model is a fake _VLLMProcess."""
    fake_proc = MagicMock()
    fake_proc.poll.return_value = None  # Still running
    fake_proc.pid = 12345
    state = _VLLMProcess(
        model_id=model_id,
        proc=fake_proc,
        port=8765,
        base_url="http://127.0.0.1:8765",
        log_file=Path("/tmp/test.log"),
    )
    return LoadedModel(
        model_id=model_id,
        backend_name="vllm",
        model=state,
        tokenizer=None,
    )


# =============================================================================
# Helper unit tests
# =============================================================================

class TestParseSize:
    def test_int_passes_through(self):
        assert _parse_size(1024) == 1024

    def test_gib(self):
        assert _parse_size("10GiB") == 10 * 1024**3

    def test_gb(self):
        assert _parse_size("4GB") == 4 * 1000**3

    def test_mb(self):
        assert _parse_size("512MB") == 512 * 1000**2

    def test_bare_number_string(self):
        assert _parse_size("2048") == 2048


class TestPickFreePort:
    def test_returns_int_in_valid_range(self):
        port = _pick_free_port()
        assert isinstance(port, int)
        assert 1024 <= port <= 65535


# =============================================================================
# Backend properties
# =============================================================================

class TestBackendProperties:
    def test_text_backend_name(self):
        assert VLLMTextBackend().name == "vllm"

    def test_vision_backend_name(self):
        assert VLLMVisionBackend().name == "vllm"

    def test_text_backend_is_basebackend(self):
        from inferall.backends.base import BaseBackend
        assert isinstance(VLLMTextBackend(), BaseBackend)

    def test_vision_backend_is_vlmbackend(self):
        from inferall.backends.base import VisionLanguageBackend
        assert isinstance(VLLMVisionBackend(), VisionLanguageBackend)


# =============================================================================
# Allocation translation
# =============================================================================

class TestGpuMemoryUtilizationMapping:
    def test_env_override(self, monkeypatch):
        backend = VLLMTextBackend()
        monkeypatch.setenv("INFERALL_VLLM_GPU_MEMORY_UTILIZATION", "0.55")
        plan = AllocationPlan()
        assert backend._compute_gpu_memory_utilization(plan) == 0.55

    def test_env_override_clamped(self, monkeypatch):
        backend = VLLMTextBackend()
        monkeypatch.setenv("INFERALL_VLLM_GPU_MEMORY_UTILIZATION", "1.5")
        plan = AllocationPlan()
        assert backend._compute_gpu_memory_utilization(plan) == 0.95

    def test_dynamic_from_free_vram(self):
        """When free vram is plentiful, fraction should reflect free memory minus a buffer."""
        backend = VLLMTextBackend()
        plan = AllocationPlan(gpu_ids=[0])
        # 18 GiB free out of 24 GiB total — buffer is 1.5 GiB → (18-1.5)/24 ≈ 0.69
        with patch("torch.cuda.is_available", return_value=True), \
             patch("torch.cuda.mem_get_info", return_value=(18 * 1024**3, 24 * 1024**3)):
            frac = backend._compute_gpu_memory_utilization(plan)
        assert 0.65 < frac < 0.72

    def test_dynamic_clamped_below_max(self):
        """An idle GPU should never push the fraction above 0.85."""
        backend = VLLMTextBackend()
        plan = AllocationPlan(gpu_ids=[0])
        with patch("torch.cuda.is_available", return_value=True), \
             patch("torch.cuda.mem_get_info", return_value=(24 * 1024**3, 24 * 1024**3)):
            frac = backend._compute_gpu_memory_utilization(plan)
        assert frac == 0.85

    def test_dynamic_clamped_above_min(self):
        """A nearly-full GPU should still return at least 0.30 (smaller can't fit kv cache)."""
        backend = VLLMTextBackend()
        plan = AllocationPlan(gpu_ids=[0])
        with patch("torch.cuda.is_available", return_value=True), \
             patch("torch.cuda.mem_get_info", return_value=(100 * 1024**2, 24 * 1024**3)):
            frac = backend._compute_gpu_memory_utilization(plan)
        assert frac == 0.30


class TestMaxModelLen:
    def test_returns_default_cap_when_no_config(self, tmp_path):
        backend = VLLMTextBackend()
        record = _make_record(tmp_path)
        assert backend._compute_max_model_len(record) == backend._MAX_MODEL_LEN_CAP

    def test_reads_max_position_embeddings(self, tmp_path):
        backend = VLLMTextBackend()
        (tmp_path / "config.json").write_text(json.dumps({
            "max_position_embeddings": 4096,
        }))
        record = _make_record(tmp_path)
        assert backend._compute_max_model_len(record) == 4096

    def test_caps_huge_context(self, tmp_path):
        backend = VLLMTextBackend()
        (tmp_path / "config.json").write_text(json.dumps({
            "max_position_embeddings": 1_000_000,
        }))
        record = _make_record(tmp_path)
        assert backend._compute_max_model_len(record) == backend._MAX_MODEL_LEN_CAP

    def test_env_override(self, tmp_path, monkeypatch):
        backend = VLLMTextBackend()
        monkeypatch.setenv("INFERALL_VLLM_MAX_MODEL_LEN", "16384")
        record = _make_record(tmp_path)
        assert backend._compute_max_model_len(record) == 16384


# =============================================================================
# Generate (HTTP mocked)
# =============================================================================

class _FakeResponse:
    """Minimal urlopen() context-manager response."""

    def __init__(self, body: dict, status: int = 200):
        self._body = json.dumps(body).encode("utf-8")
        self.status = status

    def read(self):
        return self._body

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        return False


class TestGenerate:
    def test_generate_posts_to_chat_completions(self):
        backend = VLLMTextBackend()
        loaded = _make_loaded_with_mock_proc()
        params = GenerationParams(max_tokens=100, temperature=0.0)

        fake_resp_body = {
            "choices": [{
                "message": {"role": "assistant", "content": "hello world"},
                "finish_reason": "stop",
            }],
            "usage": {"prompt_tokens": 5, "completion_tokens": 2},
        }

        with patch("urllib.request.urlopen") as mock_open:
            mock_open.return_value = _FakeResponse(fake_resp_body)
            result = backend.generate(loaded, [{"role": "user", "content": "hi"}], params)

        assert result.text == "hello world"
        assert result.prompt_tokens == 5
        assert result.completion_tokens == 2
        assert result.finish_reason == "stop"

        # Verify the request body contains expected fields
        call_req = mock_open.call_args[0][0]
        body = json.loads(call_req.data.decode())
        assert body["model"] == loaded.model_id
        assert body["max_tokens"] == 100
        assert body["temperature"] == 0.0
        assert body["stream"] is False

    def test_generate_passes_optional_fields(self):
        backend = VLLMTextBackend()
        loaded = _make_loaded_with_mock_proc()
        params = GenerationParams(
            max_tokens=50,
            temperature=0.5,
            stop=["</s>"],
            tools=[{"type": "function", "function": {"name": "f"}}],
            tool_choice="auto",
            response_format={"type": "json_object"},
        )

        fake_resp_body = {
            "choices": [{"message": {"content": ""}, "finish_reason": "stop"}],
            "usage": {"prompt_tokens": 1, "completion_tokens": 0},
        }

        with patch("urllib.request.urlopen") as mock_open:
            mock_open.return_value = _FakeResponse(fake_resp_body)
            backend.generate(loaded, [{"role": "user", "content": "x"}], params)

        body = json.loads(mock_open.call_args[0][0].data.decode())
        assert body["stop"] == ["</s>"]
        assert body["tools"] == [{"type": "function", "function": {"name": "f"}}]
        assert body["tool_choice"] == "auto"
        assert body["response_format"] == {"type": "json_object"}

    def test_generate_raises_when_subprocess_dead(self):
        backend = VLLMTextBackend()
        loaded = _make_loaded_with_mock_proc()
        loaded.model.proc.poll.return_value = 1  # Exited
        params = GenerationParams()
        with pytest.raises(RuntimeError, match="has died"):
            backend.generate(loaded, [{"role": "user", "content": "x"}], params)


# =============================================================================
# Stream (SSE mocked)
# =============================================================================

class _FakeSSEResponse:
    """Iterable urlopen() result that yields SSE lines."""

    def __init__(self, lines):
        self._lines = [l.encode() for l in lines]

    def __iter__(self):
        return iter(self._lines)

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        return False


class TestStream:
    def test_stream_yields_token_deltas(self):
        backend = VLLMTextBackend()
        loaded = _make_loaded_with_mock_proc()
        params = GenerationParams(max_tokens=10, temperature=0.0)

        sse_lines = [
            'data: {"choices":[{"delta":{"content":"Hello"}}]}\n',
            'data: {"choices":[{"delta":{"content":" world"}}]}\n',
            'data: [DONE]\n',
        ]

        with patch("urllib.request.urlopen") as mock_open:
            mock_open.return_value = _FakeSSEResponse(sse_lines)
            tokens = list(backend.stream(loaded, [{"role": "user", "content": "hi"}], params))

        assert tokens == ["Hello", " world"]

        body = json.loads(mock_open.call_args[0][0].data.decode())
        assert body["stream"] is True

    def test_stream_respects_cancel_event(self):
        backend = VLLMTextBackend()
        loaded = _make_loaded_with_mock_proc()
        params = GenerationParams()

        sse_lines = [
            'data: {"choices":[{"delta":{"content":"a"}}]}\n',
            'data: {"choices":[{"delta":{"content":"b"}}]}\n',
            'data: {"choices":[{"delta":{"content":"c"}}]}\n',
            'data: [DONE]\n',
        ]

        cancel = threading.Event()
        cancel.set()  # Pre-cancel

        with patch("urllib.request.urlopen") as mock_open:
            mock_open.return_value = _FakeSSEResponse(sse_lines)
            tokens = list(backend.stream(loaded, [{"role": "user", "content": "hi"}], params, cancel))

        assert tokens == []  # Cancelled before any yield


# =============================================================================
# Unload
# =============================================================================

class TestUnload:
    def test_unload_terminates_process(self):
        backend = VLLMTextBackend()
        loaded = _make_loaded_with_mock_proc()
        proc = loaded.model.proc

        # Make the wait() return cleanly without going to SIGKILL
        proc.wait.return_value = 0

        with patch("os.killpg") as mock_killpg, \
             patch("os.getpgid", return_value=12345):
            backend.unload(loaded)

        # We sent SIGTERM via killpg
        assert mock_killpg.called
        assert loaded.model is None

    def test_unload_handles_already_dead_process(self):
        backend = VLLMTextBackend()
        loaded = _make_loaded_with_mock_proc()
        loaded.model.proc.poll.return_value = 0  # Already exited
        # Should not raise
        backend.unload(loaded)
        assert loaded.model is None


# =============================================================================
# Orchestrator dispatch wiring
# =============================================================================

class TestOrchestratorDispatch:
    """Verify that preferred_engine='vllm' routes to the vllm backend."""

    def test_get_backend_for_record_with_vllm_preference(self, tmp_path):
        from inferall.config import EngineConfig
        from inferall.gpu.allocator import GPUAllocator
        from inferall.gpu.manager import GPUManager
        from inferall.orchestrator import Orchestrator
        from inferall.registry.registry import ModelRegistry

        config = EngineConfig(
            base_dir=tmp_path,
            registry_path=tmp_path / "reg.db",
        )
        registry = ModelRegistry(config.registry_path)
        gpu_mgr = MagicMock(spec=GPUManager)
        gpu_mgr.n_gpus = 0
        allocator = MagicMock(spec=GPUAllocator)
        orch = Orchestrator(config, registry, gpu_mgr, allocator)

        # Text-format record with vllm preference -> text vllm backend
        text_record = _make_record(tmp_path, format=ModelFormat.TRANSFORMERS)
        backend = orch._get_backend_for_record(text_record)
        assert isinstance(backend, VLLMTextBackend)

        # VLM-format record with vllm preference -> vision vllm backend
        vlm_record = _make_record(tmp_path, format=ModelFormat.VISION_LANGUAGE)
        backend = orch._get_backend_for_record(vlm_record)
        assert isinstance(backend, VLLMVisionBackend)

    def test_get_backend_for_record_without_preference(self, tmp_path):
        """When preferred_engine is None, fall through to format-based dispatch."""
        from inferall.config import EngineConfig
        from inferall.gpu.allocator import GPUAllocator
        from inferall.gpu.manager import GPUManager
        from inferall.orchestrator import Orchestrator
        from inferall.registry.registry import ModelRegistry

        config = EngineConfig(
            base_dir=tmp_path,
            registry_path=tmp_path / "reg.db",
        )
        registry = ModelRegistry(config.registry_path)
        gpu_mgr = MagicMock(spec=GPUManager)
        gpu_mgr.n_gpus = 0
        allocator = MagicMock(spec=GPUAllocator)
        orch = Orchestrator(config, registry, gpu_mgr, allocator)

        record = _make_record(tmp_path, preferred_engine=None)
        backend = orch._get_backend_for_record(record)
        # Should be the standard transformers backend, not vllm
        assert backend.name == "transformers"
