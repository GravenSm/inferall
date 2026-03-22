"""
llama-cpp-python Backend
------------------------
Handles GGUF model files via llama-cpp-python.

Features:
- Multi-GPU via tensor_split
- Built-in chat template support
- Streaming via create_chat_completion(stream=True)
"""

import logging
import threading
from datetime import datetime
from pathlib import Path
from typing import Iterator, List, Optional

from inferall.backends.base import (
    BaseBackend,
    GenerationParams,
    GenerationResult,
    LoadedModel,
)
from inferall.gpu.allocator import AllocationPlan
from inferall.registry.metadata import ModelRecord

logger = logging.getLogger(__name__)


class LlamaCppBackend(BaseBackend):
    """Backend for GGUF models via llama-cpp-python."""

    @property
    def name(self) -> str:
        return "llamacpp"

    # -------------------------------------------------------------------------
    # Load
    # -------------------------------------------------------------------------

    def load(self, record: ModelRecord, allocation: AllocationPlan) -> LoadedModel:
        """Load a GGUF model."""
        try:
            from llama_cpp import Llama
        except ImportError:
            raise RuntimeError(
                "llama-cpp-python is required for GGUF models. "
                "Install with: pip install llama-cpp-python"
            )

        # Find the GGUF file
        model_path = self._find_gguf_file(record.local_path)
        logger.info("Loading GGUF model: %s", model_path)

        # Build load kwargs
        load_kwargs = {
            "model_path": str(model_path),
            "n_gpu_layers": allocation.n_gpu_layers,
            "verbose": False,
        }

        # Multi-GPU tensor split
        if allocation.tensor_split:
            load_kwargs["tensor_split"] = allocation.tensor_split
            logger.info("Using tensor_split: %s", allocation.tensor_split)

        # Context length — default to 4096
        load_kwargs["n_ctx"] = 4096

        llm = Llama(**load_kwargs)

        loaded = LoadedModel(
            model_id=record.model_id,
            backend_name=self.name,
            model=llm,
            tokenizer=None,  # llama.cpp handles tokenization internally
            vram_used_bytes=allocation.estimated_vram_bytes,
        )

        logger.info("Loaded GGUF model: %s", record.model_id)
        return loaded

    def _find_gguf_file(self, local_path: Path) -> Path:
        """Find the GGUF file in the model directory."""
        if local_path.is_file() and local_path.suffix == ".gguf":
            return local_path

        gguf_files = list(local_path.rglob("*.gguf"))
        if not gguf_files:
            raise FileNotFoundError(
                f"No .gguf files found in {local_path}. "
                f"The model may not have downloaded correctly."
            )

        if len(gguf_files) == 1:
            return gguf_files[0]

        # Multiple GGUF files — prefer Q4_K_M
        for f in gguf_files:
            if "Q4_K_M" in f.name.upper():
                return f

        # Fall back to first file
        return gguf_files[0]

    # -------------------------------------------------------------------------
    # Generate (non-streaming)
    # -------------------------------------------------------------------------

    def generate(
        self,
        loaded: LoadedModel,
        messages: List[dict],
        params: GenerationParams,
    ) -> GenerationResult:
        """Generate a complete response using llama.cpp."""
        loaded.touch()

        kwargs = {
            "messages": messages,
            "max_tokens": params.max_tokens,
            "temperature": params.temperature,
            "top_p": params.top_p,
            "top_k": params.top_k,
            "repeat_penalty": params.repetition_penalty,
            "stop": params.stop,
        }

        if params.tools:
            kwargs["tools"] = params.tools
        if params.tool_choice is not None:
            kwargs["tool_choice"] = params.tool_choice
        if params.response_format:
            kwargs["response_format"] = params.response_format

        response = loaded.model.create_chat_completion(**kwargs)

        choice = response["choices"][0]
        usage = response.get("usage", {})
        message = choice.get("message", {})

        # Parse tool calls if present
        tool_calls = None
        if message.get("tool_calls"):
            from inferall.backends.base import ToolCall
            tool_calls = [
                ToolCall(
                    id=tc.get("id", f"call_{i}"),
                    type=tc.get("type", "function"),
                    function_name=tc.get("function", {}).get("name", ""),
                    function_arguments=tc.get("function", {}).get("arguments", "{}"),
                )
                for i, tc in enumerate(message["tool_calls"])
            ]

        return GenerationResult(
            text=message.get("content") or "",
            prompt_tokens=usage.get("prompt_tokens", 0),
            completion_tokens=usage.get("completion_tokens", 0),
            finish_reason=choice.get("finish_reason", "stop"),
            tool_calls=tool_calls,
        )

    # -------------------------------------------------------------------------
    # Stream
    # -------------------------------------------------------------------------

    def stream(
        self,
        loaded: LoadedModel,
        messages: List[dict],
        params: GenerationParams,
        cancel: Optional[threading.Event] = None,
    ) -> Iterator[str]:
        """Stream tokens from llama.cpp."""
        loaded.touch()

        stream = loaded.model.create_chat_completion(
            messages=messages,
            max_tokens=params.max_tokens,
            temperature=params.temperature,
            top_p=params.top_p,
            top_k=params.top_k,
            repeat_penalty=params.repetition_penalty,
            stop=params.stop,
            stream=True,
        )

        for chunk in stream:
            if cancel is not None and cancel.is_set():
                break

            choices = chunk.get("choices", [])
            if choices:
                delta = choices[0].get("delta", {})
                content = delta.get("content")
                if content:
                    yield content

    # -------------------------------------------------------------------------
    # Unload
    # -------------------------------------------------------------------------

    def unload(self, loaded: LoadedModel) -> None:
        """Unload the llama.cpp model."""
        logger.info("Unloading GGUF model: %s", loaded.model_id)

        if loaded.model is not None:
            # llama-cpp-python handles cleanup via __del__
            del loaded.model
            loaded.model = None

        logger.debug("Unloaded %s", loaded.model_id)
