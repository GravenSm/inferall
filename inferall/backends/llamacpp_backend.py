"""
llama-cpp-python Backend
------------------------
Handles GGUF model files via llama-cpp-python.

Features:
- Multi-GPU via tensor_split
- Built-in chat template support (the Jinja template embedded in the GGUF)
- Streaming via create_chat_completion(stream=True)
- Tool calling for models that emit text-format tool calls (Qwen 2.5/3,
  Hermes, Llama 3.1, Mistral): llama-cpp-python only returns structured
  ``tool_calls`` when ``tool_choice`` names one specific function, so for
  the usual ``tool_choice="auto"`` case the reply text is parsed here.
"""

import logging
import threading
from pathlib import Path
from typing import Iterator, List, Optional

from inferall.backends.base import (
    BaseBackend,
    GenerationParams,
    GenerationResult,
    LoadedModel,
    ToolCall,
)
from inferall.backends.tool_parsing import (
    ThinkingStreamFilter,
    normalize_tool_call_messages,
    parse_tool_calls,
    strip_thinking,
)
from inferall.gpu.allocator import AllocationPlan
from inferall.registry.metadata import ModelRecord

logger = logging.getLogger(__name__)

# Context window used when none is configured. Tool schemas plus a few turns
# of tool results overflow 4096 quickly; 0 means "use the model's trained
# context length" (llama-cpp-python semantics).
DEFAULT_N_CTX = 8192


class LlamaCppBackend(BaseBackend):
    """Backend for GGUF models via llama-cpp-python."""

    def __init__(self, n_ctx: int = DEFAULT_N_CTX):
        self.n_ctx = n_ctx

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
            "n_ctx": self.n_ctx,
            "verbose": False,
        }

        # Multi-GPU tensor split
        if allocation.tensor_split:
            load_kwargs["tensor_split"] = allocation.tensor_split
            logger.info("Using tensor_split: %s", allocation.tensor_split)

        llm = Llama(**load_kwargs)

        loaded = LoadedModel(
            model_id=record.model_id,
            backend_name=self.name,
            model=llm,
            tokenizer=None,  # llama.cpp handles tokenization internally
            vram_used_bytes=allocation.estimated_vram_bytes,
        )

        logger.info(
            "Loaded GGUF model: %s (n_ctx=%s)",
            record.model_id, self.n_ctx or "model default",
        )
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
    # Request building
    # -------------------------------------------------------------------------

    @staticmethod
    def _build_kwargs(messages: List[dict], params: GenerationParams) -> dict:
        """Build create_chat_completion kwargs shared by generate() and stream()."""
        kwargs = {
            "messages": normalize_tool_call_messages(messages),
            "max_tokens": params.max_tokens,
            "temperature": params.temperature,
            "top_p": params.top_p,
            "top_k": params.top_k,
            "repeat_penalty": params.repetition_penalty,
            "stop": params.stop,
        }

        # tool_choice="none": the model must not call tools, so do not even
        # render them into the prompt.
        if params.tools and params.tool_choice != "none":
            kwargs["tools"] = params.tools
            if params.tool_choice is not None:
                kwargs["tool_choice"] = params.tool_choice
        if params.response_format:
            kwargs["response_format"] = params.response_format

        return kwargs

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

        kwargs = self._build_kwargs(messages, params)
        response = loaded.model.create_chat_completion(**kwargs)

        choice = response["choices"][0]
        usage = response.get("usage", {})
        message = choice.get("message", {})
        finish_reason = choice.get("finish_reason", "stop")

        # Reasoning models (Qwen3) think before answering or calling a tool.
        text = strip_thinking(message.get("content") or "")

        # Structured tool_calls: llama-cpp-python produces these only when
        # tool_choice named a specific function (grammar-constrained).
        tool_calls = None
        if message.get("tool_calls"):
            tool_calls = [
                ToolCall(
                    id=tc.get("id") or f"call_{i}",
                    type=tc.get("type", "function"),
                    function_name=tc.get("function", {}).get("name", ""),
                    function_arguments=tc.get("function", {}).get("arguments", "{}"),
                )
                for i, tc in enumerate(message["tool_calls"])
            ]
        # Otherwise the model wrote its tool call as text in its own format.
        elif "tools" in kwargs:
            tool_calls, text = parse_tool_calls(text)

        if tool_calls:
            finish_reason = "tool_calls"

        return GenerationResult(
            text=text,
            prompt_tokens=usage.get("prompt_tokens", 0),
            completion_tokens=usage.get("completion_tokens", 0),
            finish_reason=finish_reason,
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
        """
        Stream tokens from llama.cpp.

        Tools and response_format are rendered into the prompt exactly as in
        generate(). Tool calls still arrive as text; the API layer extracts
        them from the stream.
        """
        loaded.touch()

        kwargs = self._build_kwargs(messages, params)
        stream = loaded.model.create_chat_completion(stream=True, **kwargs)
        think_filter = ThinkingStreamFilter()

        for chunk in stream:
            if cancel is not None and cancel.is_set():
                break

            choices = chunk.get("choices", [])
            if not choices:
                continue
            delta = choices[0].get("delta", {})
            content = delta.get("content")
            if content:
                out = think_filter.feed(content)
                if out:
                    yield out

        tail = think_filter.flush()
        if tail:
            yield tail

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
