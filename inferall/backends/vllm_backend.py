"""
vLLM Backend
------------
High-throughput inference backend powered by vLLM running as an isolated
subprocess. Talks to vllm's OpenAI-compatible HTTP server.

Why subprocess instead of embedded?

  vLLM pins ``transformers<5`` (even on main as of vLLM 0.19.x) while
  inferall uses transformers 5.x. Embedding vllm in our process would force
  a transformers downgrade and rewrite of every other backend. Running
  vllm in its own venv (~/.cache/inferall/vllm-venv) sidesteps the conflict
  entirely. This is also how chandra and most production deployments run
  it. See ``vllm_runtime.py`` for venv discovery + bootstrap.

Lifecycle:
  load()    -> spawn ``python -m vllm.entrypoints.openai.api_server`` and
              poll /health until ready (or timeout).
  generate()/stream()  -> POST /v1/chat/completions on the spawned server.
  unload()  -> SIGTERM the process; SIGKILL on timeout. CUDA memory is
              freed when the process exits.

Implements both BaseBackend (text models) and VisionLanguageBackend (VLMs).
vLLM handles both modalities through the same /v1/chat/completions endpoint,
so the implementations are nearly identical — they only differ in how the
``messages`` payload is constructed.
"""

import json
import logging
import os
import signal
import socket
import subprocess
import threading
import time
import urllib.error
import urllib.request
from dataclasses import dataclass, field
from pathlib import Path
from typing import Iterator, List, Optional

from inferall.backends.base import (
    BaseBackend,
    GenerationParams,
    GenerationResult,
    LoadedModel,
    VisionLanguageBackend,
)
from inferall.backends.vllm_runtime import find_vllm_python
from inferall.gpu.allocator import AllocationPlan
from inferall.registry.metadata import ModelRecord

logger = logging.getLogger(__name__)


# How long to wait for ``/health`` after starting the subprocess
_HEALTH_TIMEOUT_SECONDS = 600  # 10 minutes — first load can include weight download
# Polling interval for /health checks
_HEALTH_POLL_INTERVAL = 1.0
# Per-request HTTP timeouts
_REQUEST_CONNECT_TIMEOUT = 10
_REQUEST_READ_TIMEOUT = 600
# Time to wait for graceful shutdown before SIGKILL
_SHUTDOWN_TIMEOUT_SECONDS = 30


@dataclass
class _VLLMProcess:
    """
    Per-model state for a running vllm subprocess.

    Stored as ``LoadedModel.model`` so the orchestrator can dispatch
    generate/stream/unload through the standard backend interface.
    """

    model_id: str           # The vllm --served-model-name
    proc: subprocess.Popen
    port: int
    base_url: str
    log_file: Path

    def is_alive(self) -> bool:
        return self.proc.poll() is None


# =============================================================================
# Shared implementation — text and VLM share most of their code paths
# =============================================================================

class _VLLMBackendBase:
    """
    Implementation shared by VLLMTextBackend and VLLMVisionBackend.

    Both wrap the same lifecycle around vllm's OpenAI-compatible server. The
    only difference is which abstract base class they inherit from, so the
    orchestrator's existing dispatch can keep working without conditional
    typing.
    """

    @property
    def name(self) -> str:
        return "vllm"

    # -------------------------------------------------------------------------
    # Load — spawn vllm subprocess
    # -------------------------------------------------------------------------

    def load(self, record: ModelRecord, allocation: AllocationPlan) -> LoadedModel:
        """Start a vllm OpenAI-compatible server for this model."""
        vllm_py = find_vllm_python()
        port = _pick_free_port()
        log_path = _log_path_for(record.model_id)

        # Translate inferall's allocation into vllm flags
        gpu_mem_util = self._compute_gpu_memory_utilization(allocation)
        max_model_len = self._compute_max_model_len(record)

        max_num_seqs = self._compute_max_num_seqs()

        cmd = [
            str(vllm_py), "-m", "vllm.entrypoints.openai.api_server",
            "--model", str(record.local_path),
            "--served-model-name", record.model_id,
            "--host", "127.0.0.1",
            "--port", str(port),
            "--gpu-memory-utilization", f"{gpu_mem_util:.2f}",
            "--dtype", "bfloat16",
            "--max-num-seqs", str(max_num_seqs),
            # vllm 0.19+ uses --no-enable-log-requests; 0.10/0.11 had --disable-log-requests
            "--no-enable-log-requests",
        ]
        if record.trust_remote_code:
            cmd.append("--trust-remote-code")
        if max_model_len is not None:
            cmd.extend(["--max-model-len", str(max_model_len)])

        # Pin to a specific GPU if the allocator chose one (single-GPU case).
        # Multi-GPU tensor-parallel would need --tensor-parallel-size; we keep
        # the simple case for now and let the allocator deal with placement.
        env = os.environ.copy()
        if allocation.gpu_ids and len(allocation.gpu_ids) == 1:
            env["CUDA_VISIBLE_DEVICES"] = str(allocation.gpu_ids[0])
        elif allocation.gpu_ids and len(allocation.gpu_ids) > 1:
            env["CUDA_VISIBLE_DEVICES"] = ",".join(str(g) for g in allocation.gpu_ids)
            cmd.extend(["--tensor-parallel-size", str(len(allocation.gpu_ids))])

        logger.info(
            "Starting vllm subprocess for %s on port %d (gpu_mem_util=%.2f)",
            record.model_id, port, gpu_mem_util,
        )
        logger.debug("vllm cmd: %s", " ".join(cmd))

        log_path.parent.mkdir(parents=True, exist_ok=True)
        log_fh = open(log_path, "w")
        proc = subprocess.Popen(
            cmd,
            stdout=log_fh,
            stderr=subprocess.STDOUT,
            env=env,
            # New process group so we can SIGTERM the whole tree on unload
            preexec_fn=os.setsid if hasattr(os, "setsid") else None,
        )

        base_url = f"http://127.0.0.1:{port}"

        # Wait for /health
        try:
            self._wait_for_health(proc, base_url, log_path)
        except Exception:
            # If startup failed, terminate the process before re-raising so we
            # don't leak a half-initialized subprocess.
            _terminate_process(proc)
            log_fh.close()
            raise

        vllm_state = _VLLMProcess(
            model_id=record.model_id,
            proc=proc,
            port=port,
            base_url=base_url,
            log_file=log_path,
        )

        logger.info(
            "vllm server ready for %s at %s (pid=%d)",
            record.model_id, base_url, proc.pid,
        )

        return LoadedModel(
            model_id=record.model_id,
            backend_name=self.name,
            model=vllm_state,
            tokenizer=None,  # vllm owns tokenization in its own process
            vram_used_bytes=allocation.estimated_vram_bytes,
        )

    def _compute_gpu_memory_utilization(self, allocation: AllocationPlan) -> float:
        """
        Compute vLLM's ``gpu_memory_utilization`` flag (a fraction of *total*
        GPU memory that vllm is allowed to claim).

        Two sources, in order of preference:

        1. ``INFERALL_VLLM_GPU_MEMORY_UTILIZATION`` env var — explicit override.
        2. Live free memory query: aim to fit within the *currently free* VRAM
           on the target GPU, leaving a safety buffer for activation peaks
           and other processes that may grow during inference.

        vLLM's own default is 0.9 which OOMs aggressively in a multi-tenant
        environment like inferall. We deliberately stay conservative: it's
        much better to under-claim memory and leave headroom than to crash
        the subprocess on first request.
        """
        # Explicit override
        env_override = os.environ.get("INFERALL_VLLM_GPU_MEMORY_UTILIZATION")
        if env_override:
            try:
                return max(0.1, min(0.95, float(env_override)))
            except ValueError:
                logger.warning(
                    "Invalid INFERALL_VLLM_GPU_MEMORY_UTILIZATION=%r, ignoring",
                    env_override,
                )

        # Dynamic: compute from currently free VRAM
        try:
            import torch
            if torch.cuda.is_available():
                gpu_id = allocation.gpu_ids[0] if allocation.gpu_ids else 0
                free_bytes, total_bytes = torch.cuda.mem_get_info(gpu_id)
                # Leave 1.5 GiB buffer for activation peaks during prefill
                # plus headroom for other processes that may grow.
                safety_buffer = 1536 * 1024**2
                usable = max(0, free_bytes - safety_buffer)
                fraction = usable / total_bytes
                # Clamp to a sane range — never go above 0.85 even with a
                # totally idle GPU, never below 0.30 (smaller fractions
                # often can't fit even tiny KV caches).
                fraction = max(0.30, min(0.85, fraction))
                logger.info(
                    "vllm gpu_memory_utilization=%.2f (free=%.1f GiB, "
                    "total=%.1f GiB on gpu %d)",
                    fraction, free_bytes / 1024**3, total_bytes / 1024**3, gpu_id,
                )
                return fraction
        except Exception as e:
            logger.debug("Could not query free GPU memory: %s", e)

        return 0.70

    # Cap on context length, in tokens. Larger means bigger KV cache, which
    # can OOM the GPU even before any inference happens. 4k is enough for
    # almost every interactive chat use case, including single-page OCR.
    # Override per model with INFERALL_VLLM_MAX_MODEL_LEN.
    _MAX_MODEL_LEN_CAP = 4096

    # Cap on concurrent in-flight sequences. vLLM's default is 256 which
    # is sized for shared serving infrastructure; for an interactive engine
    # like inferall a much smaller pool keeps the KV cache footprint sane.
    # Override with INFERALL_VLLM_MAX_NUM_SEQS.
    _MAX_NUM_SEQS_CAP = 8

    def _compute_max_num_seqs(self) -> int:
        """Concurrent in-flight sequences. Override with INFERALL_VLLM_MAX_NUM_SEQS."""
        env_override = os.environ.get("INFERALL_VLLM_MAX_NUM_SEQS")
        if env_override:
            try:
                return max(1, int(env_override))
            except ValueError:
                logger.warning("Invalid INFERALL_VLLM_MAX_NUM_SEQS=%r", env_override)
        return self._MAX_NUM_SEQS_CAP

    def _compute_max_model_len(self, record: ModelRecord) -> Optional[int]:
        """
        Compute vllm's --max-model-len.

        vLLM defaults to the model's full context length, which can OOM the
        KV cache on large-context models long before any request comes in.
        We cap aggressively (8k) by default, which is enough for chat and
        single-document OCR. Override with ``INFERALL_VLLM_MAX_MODEL_LEN``.
        """
        env_override = os.environ.get("INFERALL_VLLM_MAX_MODEL_LEN")
        if env_override:
            try:
                return int(env_override)
            except ValueError:
                logger.warning("Invalid INFERALL_VLLM_MAX_MODEL_LEN=%r", env_override)

        try:
            config_path = Path(record.local_path) / "config.json"
            if not config_path.exists():
                return self._MAX_MODEL_LEN_CAP
            with open(config_path) as f:
                cfg = json.load(f)
            for key in ("max_position_embeddings", "model_max_length", "max_seq_len"):
                if key in cfg and isinstance(cfg[key], int):
                    return min(cfg[key], self._MAX_MODEL_LEN_CAP)
        except Exception as e:
            logger.debug("Could not read max_model_len from config: %s", e)
        return self._MAX_MODEL_LEN_CAP

    def _wait_for_health(
        self,
        proc: subprocess.Popen,
        base_url: str,
        log_path: Path,
    ) -> None:
        """Poll /health until the server responds, the process exits, or we time out."""
        deadline = time.monotonic() + _HEALTH_TIMEOUT_SECONDS
        health_url = f"{base_url}/health"

        while time.monotonic() < deadline:
            # Did the subprocess die?
            if proc.poll() is not None:
                tail = _tail_log(log_path, lines=40)
                raise RuntimeError(
                    f"vllm subprocess exited with code {proc.returncode} during "
                    f"startup. Last lines of {log_path}:\n{tail}"
                )

            try:
                with urllib.request.urlopen(health_url, timeout=2) as resp:
                    if resp.status == 200:
                        return
            except (urllib.error.URLError, ConnectionError, socket.timeout):
                pass  # Server not up yet

            time.sleep(_HEALTH_POLL_INTERVAL)

        raise TimeoutError(
            f"vllm did not become healthy within {_HEALTH_TIMEOUT_SECONDS}s. "
            f"See {log_path} for details."
        )

    # -------------------------------------------------------------------------
    # Generate — non-streaming
    # -------------------------------------------------------------------------

    def generate(
        self,
        loaded: LoadedModel,
        messages: List[dict],
        params: GenerationParams,
    ) -> GenerationResult:
        """Forward to /v1/chat/completions."""
        loaded.touch()
        state: _VLLMProcess = loaded.model

        if not state.is_alive():
            raise RuntimeError(
                f"vllm subprocess for {state.model_id} has died. "
                f"See {state.log_file} for details."
            )

        body = {
            "model": state.model_id,
            "messages": messages,
            "max_tokens": params.max_tokens,
            "temperature": params.temperature,
            "top_p": params.top_p,
            "stream": False,
        }
        if params.stop:
            body["stop"] = params.stop
        if params.tools:
            body["tools"] = params.tools
        if params.tool_choice is not None:
            body["tool_choice"] = params.tool_choice
        if params.response_format:
            body["response_format"] = params.response_format

        resp = _http_post_json(f"{state.base_url}/v1/chat/completions", body)

        choice = resp["choices"][0]
        msg = choice["message"]
        usage = resp.get("usage", {})

        return GenerationResult(
            text=msg.get("content") or "",
            prompt_tokens=int(usage.get("prompt_tokens", 0)),
            completion_tokens=int(usage.get("completion_tokens", 0)),
            finish_reason=choice.get("finish_reason", "stop"),
        )

    # -------------------------------------------------------------------------
    # Stream — Server-Sent Events
    # -------------------------------------------------------------------------

    def stream(
        self,
        loaded: LoadedModel,
        messages: List[dict],
        params: GenerationParams,
        cancel: Optional[threading.Event] = None,
    ) -> Iterator[str]:
        """Stream tokens via vllm's SSE response from /v1/chat/completions."""
        loaded.touch()
        state: _VLLMProcess = loaded.model

        if not state.is_alive():
            raise RuntimeError(
                f"vllm subprocess for {state.model_id} has died. "
                f"See {state.log_file} for details."
            )

        body = {
            "model": state.model_id,
            "messages": messages,
            "max_tokens": params.max_tokens,
            "temperature": params.temperature,
            "top_p": params.top_p,
            "stream": True,
        }
        if params.stop:
            body["stop"] = params.stop

        req = urllib.request.Request(
            f"{state.base_url}/v1/chat/completions",
            data=json.dumps(body).encode("utf-8"),
            headers={
                "Content-Type": "application/json",
                "Accept": "text/event-stream",
            },
            method="POST",
        )

        # urllib doesn't natively stream chunks; we read the response line by
        # line which works because vllm sends one ``data: ...`` per line.
        try:
            with urllib.request.urlopen(req, timeout=_REQUEST_READ_TIMEOUT) as resp:
                for raw_line in resp:
                    if cancel is not None and cancel.is_set():
                        break
                    line = raw_line.decode("utf-8", errors="replace").strip()
                    if not line or not line.startswith("data:"):
                        continue
                    payload = line[5:].strip()
                    if payload == "[DONE]":
                        break
                    try:
                        chunk = json.loads(payload)
                    except json.JSONDecodeError:
                        continue
                    choices = chunk.get("choices") or []
                    if not choices:
                        continue
                    delta = choices[0].get("delta") or {}
                    token = delta.get("content")
                    if token:
                        yield token
        except urllib.error.HTTPError as e:
            err_body = e.read().decode("utf-8", errors="replace")
            raise RuntimeError(f"vllm stream HTTP {e.code}: {err_body}") from e

    # -------------------------------------------------------------------------
    # Unload
    # -------------------------------------------------------------------------

    def unload(self, loaded: LoadedModel) -> None:
        """Terminate the vllm subprocess and free its GPU memory."""
        state: _VLLMProcess = loaded.model
        if state is None:
            return
        logger.info("Stopping vllm subprocess for %s (pid=%d)",
                    state.model_id, state.proc.pid)
        _terminate_process(state.proc)
        loaded.model = None


# =============================================================================
# Concrete backends
# =============================================================================

class VLLMTextBackend(_VLLMBackendBase, BaseBackend):
    """vLLM backend for text-only chat models. Implements BaseBackend."""


class VLLMVisionBackend(_VLLMBackendBase, VisionLanguageBackend):
    """vLLM backend for vision-language models. Implements VisionLanguageBackend.

    The implementation is identical to the text backend — vLLM's
    /v1/chat/completions endpoint accepts OpenAI-style multimodal messages
    natively (``{"type": "image_url", "image_url": {"url": "data:..."}}``)
    so we can pass them straight through.
    """


# =============================================================================
# Helpers
# =============================================================================

def _pick_free_port() -> int:
    """Bind to port 0 to let the OS pick a free port, then release it."""
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.bind(("127.0.0.1", 0))
        return s.getsockname()[1]


def _log_path_for(model_id: str) -> Path:
    """Return a per-model log file under the inferall cache directory."""
    safe = model_id.replace("/", "_").replace(":", "_")
    return Path.home() / ".cache" / "inferall" / "vllm-logs" / f"{safe}.log"


def _terminate_process(proc: subprocess.Popen) -> None:
    """Stop a vllm subprocess gracefully, then forcefully if needed."""
    if proc.poll() is not None:
        return  # Already exited

    # Send SIGTERM to the whole process group (vllm spawns workers)
    try:
        if hasattr(os, "killpg") and hasattr(os, "getpgid"):
            os.killpg(os.getpgid(proc.pid), signal.SIGTERM)
        else:
            proc.terminate()
    except ProcessLookupError:
        return

    try:
        proc.wait(timeout=_SHUTDOWN_TIMEOUT_SECONDS)
        return
    except subprocess.TimeoutExpired:
        logger.warning("vllm subprocess did not exit in %ds, sending SIGKILL",
                       _SHUTDOWN_TIMEOUT_SECONDS)

    try:
        if hasattr(os, "killpg") and hasattr(os, "getpgid"):
            os.killpg(os.getpgid(proc.pid), signal.SIGKILL)
        else:
            proc.kill()
    except ProcessLookupError:
        return
    proc.wait(timeout=5)


def _http_post_json(url: str, body: dict) -> dict:
    """POST JSON, raise on HTTP error, return parsed JSON response."""
    req = urllib.request.Request(
        url,
        data=json.dumps(body).encode("utf-8"),
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    try:
        with urllib.request.urlopen(req, timeout=_REQUEST_READ_TIMEOUT) as resp:
            return json.loads(resp.read().decode("utf-8"))
    except urllib.error.HTTPError as e:
        err_body = e.read().decode("utf-8", errors="replace")
        raise RuntimeError(f"vllm HTTP {e.code}: {err_body}") from e


def _tail_log(path: Path, lines: int = 20) -> str:
    """Return the last N lines of a log file as a string. Best-effort."""
    try:
        with open(path, "rb") as f:
            f.seek(0, os.SEEK_END)
            size = f.tell()
            chunk = min(size, 8192)
            f.seek(size - chunk)
            data = f.read().decode("utf-8", errors="replace")
        return "\n".join(data.splitlines()[-lines:])
    except Exception:
        return "<log unavailable>"


def _parse_size(s) -> int:
    """Parse '10GiB' / '4096MB' / int → bytes."""
    if isinstance(s, (int, float)):
        return int(s)
    s = str(s).strip()
    suffixes = {
        "GIB": 1024**3, "GB": 1000**3, "G": 1024**3,
        "MIB": 1024**2, "MB": 1000**2, "M": 1024**2,
        "KIB": 1024,    "KB": 1000,    "K": 1024,
        "B": 1,
    }
    upper = s.upper()
    for suffix, mul in sorted(suffixes.items(), key=lambda x: -len(x[0])):
        if upper.endswith(suffix):
            try:
                return int(float(upper[: -len(suffix)]) * mul)
            except ValueError:
                break
    return int(s)
