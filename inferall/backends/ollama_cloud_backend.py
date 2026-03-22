"""
Ollama Cloud Backend
---------------------
Proxies inference requests to Ollama's cloud API for remote models
(e.g., nemotron-3-super:cloud).

No local model loading — all computation happens on Ollama's servers.
Requires OLLAMA_API_KEY environment variable or config.
"""

import json
import logging
import os
import threading
import time
from pathlib import Path
from typing import Iterator, List, Optional
from urllib.request import Request, urlopen
from urllib.error import HTTPError

from inferall.backends.base import (
    BaseBackend,
    GenerationParams,
    GenerationResult,
    LoadedModel,
)
from inferall.gpu.allocator import AllocationPlan
from inferall.registry.metadata import ModelRecord

logger = logging.getLogger(__name__)


class OllamaCloudBackend(BaseBackend):
    """Backend that proxies to Ollama's cloud inference API."""

    @property
    def name(self) -> str:
        return "ollama_cloud"

    # -------------------------------------------------------------------------
    # Load (no-op — cloud model, nothing to load locally)
    # -------------------------------------------------------------------------

    def load(self, record: ModelRecord, allocation: AllocationPlan) -> LoadedModel:
        """Register a cloud model. No local loading needed."""
        # Read cloud metadata
        meta_path = record.local_path / "ollama_metadata.json"
        metadata = {}
        if meta_path.exists():
            with open(meta_path) as f:
                metadata = json.load(f)

        remote_host = metadata.get("remote_host", "https://ollama.com:443")
        remote_model = metadata.get("remote_model", record.model_id)

        logger.info(
            "Cloud model registered: %s → %s/%s",
            record.model_id, remote_host, remote_model,
        )

        # Store metadata as the "model" object — it's all we need
        cloud_config = {
            "remote_host": remote_host,
            "remote_model": remote_model,
            "capabilities": metadata.get("capabilities", []),
            "context_length": metadata.get("context_length", 4096),
            "api_key": self._resolve_api_key(),
        }

        return LoadedModel(
            model_id=record.model_id,
            backend_name=self.name,
            model=cloud_config,
            tokenizer=None,
            vram_used_bytes=0,  # No local VRAM usage
        )

    # -------------------------------------------------------------------------
    # Generate (proxy to Ollama API)
    # -------------------------------------------------------------------------

    def generate(
        self,
        loaded: LoadedModel,
        messages: List[dict],
        params: GenerationParams,
    ) -> GenerationResult:
        """Generate via Ollama cloud API."""
        loaded.touch()

        config = loaded.model
        url = f"{config['remote_host']}/api/chat"

        payload = {
            "model": config["remote_model"],
            "messages": messages,
            "stream": False,
            "options": {
                "temperature": params.temperature,
                "top_p": params.top_p,
                "top_k": params.top_k,
                "num_predict": params.max_tokens,
                "repeat_penalty": params.repetition_penalty,
            },
        }

        if params.stop:
            payload["options"]["stop"] = params.stop
        if params.tools:
            payload["tools"] = params.tools
        if params.response_format and params.response_format.get("type") == "json_object":
            payload["format"] = "json"
        elif params.response_format and params.response_format.get("type") == "json_schema":
            schema = params.response_format.get("json_schema", {}).get("schema")
            if schema:
                payload["format"] = schema

        t0 = time.perf_counter()
        api_key = config.get("api_key") or self._resolve_api_key()
        response = self._api_request(url, payload, api_key)
        elapsed_ms = (time.perf_counter() - t0) * 1000.0

        # Parse Ollama response
        message = response.get("message", {})
        text = message.get("content", "")
        prompt_tokens = response.get("prompt_eval_count", 0)
        completion_tokens = response.get("eval_count", 0)

        # Parse tool calls if present
        tool_calls = None
        if message.get("tool_calls"):
            from inferall.backends.base import ToolCall
            import uuid as uuid_mod
            tool_calls = [
                ToolCall(
                    id=f"call_{uuid_mod.uuid4().hex[:24]}",
                    function_name=tc.get("function", {}).get("name", ""),
                    function_arguments=json.dumps(tc.get("function", {}).get("arguments", {})),
                )
                for tc in message["tool_calls"]
            ]

        # Determine finish reason
        done_reason = response.get("done_reason", "stop")
        if tool_calls:
            finish_reason = "tool_calls"
        elif done_reason == "length":
            finish_reason = "length"
        else:
            finish_reason = "stop"

        return GenerationResult(
            text=text,
            prompt_tokens=prompt_tokens,
            completion_tokens=completion_tokens,
            finish_reason=finish_reason,
            tool_calls=tool_calls,
        )

    # -------------------------------------------------------------------------
    # Stream (proxy streaming from Ollama API)
    # -------------------------------------------------------------------------

    def stream(
        self,
        loaded: LoadedModel,
        messages: List[dict],
        params: GenerationParams,
        cancel: Optional[threading.Event] = None,
    ) -> Iterator[str]:
        """Stream tokens via Ollama cloud API."""
        loaded.touch()

        config = loaded.model
        url = f"{config['remote_host']}/api/chat"

        payload = {
            "model": config["remote_model"],
            "messages": messages,
            "stream": True,
            "options": {
                "temperature": params.temperature,
                "top_p": params.top_p,
                "top_k": params.top_k,
                "num_predict": params.max_tokens,
                "repeat_penalty": params.repetition_penalty,
            },
        }

        if params.stop:
            payload["options"]["stop"] = params.stop

        data = json.dumps(payload).encode()
        headers = {"Content-Type": "application/json"}
        api_key = config.get("api_key") or self._resolve_api_key()
        if api_key:
            headers["Authorization"] = f"Bearer {api_key}"

        req = Request(url, data=data, headers=headers, method="POST")

        try:
            with urlopen(req, timeout=120) as resp:
                for line in resp:
                    if cancel and cancel.is_set():
                        break

                    line = line.strip()
                    if not line:
                        continue

                    try:
                        chunk = json.loads(line)
                    except json.JSONDecodeError:
                        continue

                    content = chunk.get("message", {}).get("content", "")
                    if content:
                        yield content

                    if chunk.get("done", False):
                        break
        except HTTPError as e:
            raise RuntimeError(
                f"Ollama cloud API error: HTTP {e.code} {e.reason}"
            )

    # -------------------------------------------------------------------------
    # Unload (no-op for cloud models)
    # -------------------------------------------------------------------------

    def unload(self, loaded: LoadedModel) -> None:
        """Nothing to unload for cloud models."""
        logger.info("Unregistered cloud model: %s", loaded.model_id)

    # -------------------------------------------------------------------------
    # Helpers
    # -------------------------------------------------------------------------

    def _resolve_api_key(self) -> str:
        """Resolve the Ollama API key from env var or config file."""
        # Env var takes priority
        key = os.environ.get("OLLAMA_API_KEY", "")
        if key:
            return key

        # Try config file
        try:
            from inferall.config import EngineConfig
            config = EngineConfig.load()
            if config.ollama_api_key:
                return config.ollama_api_key
        except Exception:
            pass

        return ""

    def _api_request(self, url: str, payload: dict, api_key: str) -> dict:
        """Make a request to the Ollama cloud API."""
        data = json.dumps(payload).encode()
        headers = {"Content-Type": "application/json"}

        if api_key:
            headers["Authorization"] = f"Bearer {api_key}"

        req = Request(url, data=data, headers=headers, method="POST")

        try:
            with urlopen(req, timeout=120) as resp:
                return json.loads(resp.read())
        except HTTPError as e:
            body = e.read().decode("utf-8", errors="replace")[:500]
            if e.code == 401:
                raise RuntimeError(
                    "Ollama cloud API: authentication failed. "
                    "Set OLLAMA_API_KEY environment variable."
                )
            raise RuntimeError(
                f"Ollama cloud API error: HTTP {e.code} {e.reason}\n{body}"
            )
