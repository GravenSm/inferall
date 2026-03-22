"""
Vision-Language Backend
-----------------------
Handles VLMs that accept images + text and produce text.

Uses AutoProcessor for input handling and supports streaming via
TextIteratorStreamer (same pattern as TransformersBackend).
"""

import base64
import io
import logging
import threading
from typing import Iterator, List, Optional

import torch
from transformers import TextIteratorStreamer

from inferall.backends.base import (
    GenerationParams,
    GenerationResult,
    LoadedModel,
    VisionLanguageBackend,
)
from inferall.gpu.allocator import AllocationPlan
from inferall.registry.metadata import ModelRecord

logger = logging.getLogger(__name__)


class VisionLanguageTransformersBackend(VisionLanguageBackend):
    """VLM backend using AutoProcessor + AutoModel."""

    @property
    def name(self) -> str:
        return "vlm"

    # -------------------------------------------------------------------------
    # Load
    # -------------------------------------------------------------------------

    def load(self, record: ModelRecord, allocation: AllocationPlan) -> LoadedModel:
        """Load a vision-language model."""
        model_path = str(record.local_path)
        trust = record.trust_remote_code

        logger.info("Loading VLM %s", record.model_id)

        from transformers import AutoProcessor

        processor = AutoProcessor.from_pretrained(model_path, trust_remote_code=trust)

        # Try AutoModelForVision2Seq, fall back to AutoModelForCausalLM
        load_kwargs = {
            "pretrained_model_name_or_path": model_path,
            "trust_remote_code": trust,
            "torch_dtype": "auto",
        }
        if allocation.max_memory:
            load_kwargs["device_map"] = "auto"
            load_kwargs["max_memory"] = allocation.max_memory
        else:
            load_kwargs["device_map"] = allocation.device_map

        model = self._load_model(load_kwargs)

        logger.info("Loaded VLM %s", record.model_id)

        return LoadedModel(
            model_id=record.model_id,
            backend_name=self.name,
            model=model,
            tokenizer=processor,  # Store processor as "tokenizer"
            vram_used_bytes=allocation.estimated_vram_bytes,
        )

    def _load_model(self, load_kwargs: dict):
        """Load model with fallback chain."""
        # Try the most specific VLM auto class first, then broaden
        _auto_classes = [
            "AutoModelForImageTextToText",   # transformers >= 5.x
            "AutoModelForVision2Seq",        # transformers < 5.x
            "AutoModelForCausalLM",          # multimodal causal LMs (LLaVA-style)
        ]
        import transformers

        last_err = None
        for cls_name in _auto_classes:
            cls = getattr(transformers, cls_name, None)
            if cls is None:
                continue
            try:
                logger.info("Trying %s", cls_name)
                return cls.from_pretrained(**load_kwargs)
            except Exception as e:
                logger.info("%s failed: %s", cls_name, e)
                last_err = e

        raise RuntimeError(
            f"Could not load VLM with any auto class. Last error: {last_err}"
        )

    # -------------------------------------------------------------------------
    # Generate
    # -------------------------------------------------------------------------

    def generate(
        self,
        loaded: LoadedModel,
        messages: List[dict],
        params: GenerationParams,
    ) -> GenerationResult:
        """Generate a response from multimodal messages."""
        loaded.touch()

        inputs, images = self._process_messages(loaded, messages)
        prompt_tokens = inputs["input_ids"].shape[1]

        gen_kwargs = {
            **inputs,
            "max_new_tokens": params.max_tokens,
            "do_sample": params.temperature > 0,
        }
        if params.temperature > 0:
            gen_kwargs["temperature"] = params.temperature
            gen_kwargs["top_p"] = params.top_p

        with torch.inference_mode():
            output_ids = loaded.model.generate(**gen_kwargs)

        new_tokens = output_ids[0][prompt_tokens:]
        text = loaded.tokenizer.decode(new_tokens, skip_special_tokens=True)
        completion_tokens = len(new_tokens)

        finish_reason = "length" if completion_tokens >= params.max_tokens else "stop"

        return GenerationResult(
            text=text,
            prompt_tokens=prompt_tokens,
            completion_tokens=completion_tokens,
            finish_reason=finish_reason,
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
        """Stream tokens from multimodal messages."""
        loaded.touch()

        inputs, images = self._process_messages(loaded, messages)

        gen_kwargs = {
            **inputs,
            "max_new_tokens": params.max_tokens,
            "do_sample": params.temperature > 0,
        }
        if params.temperature > 0:
            gen_kwargs["temperature"] = params.temperature
            gen_kwargs["top_p"] = params.top_p

        # Set up streamer
        streamer = TextIteratorStreamer(
            loaded.tokenizer,
            skip_prompt=True,
            skip_special_tokens=True,
        )
        gen_kwargs["streamer"] = streamer

        gen_thread = threading.Thread(
            target=self._generate_with_streamer,
            args=(loaded.model, gen_kwargs),
            daemon=True,
        )
        gen_thread.start()

        try:
            for token in streamer:
                if cancel is not None and cancel.is_set():
                    break
                if token:
                    yield token
        finally:
            gen_thread.join(timeout=5.0)

    def _generate_with_streamer(self, model, gen_kwargs):
        """Run model.generate in a thread (for streaming)."""
        try:
            with torch.inference_mode():
                model.generate(**gen_kwargs)
        except Exception:
            logger.error("VLM generation error in streamer thread", exc_info=True)

    # -------------------------------------------------------------------------
    # Unload
    # -------------------------------------------------------------------------

    def unload(self, loaded: LoadedModel) -> None:
        """Unload VLM and free resources."""
        logger.info("Unloading VLM %s", loaded.model_id)

        del loaded.model
        del loaded.tokenizer
        loaded.model = None
        loaded.tokenizer = None

        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    # -------------------------------------------------------------------------
    # Helpers
    # -------------------------------------------------------------------------

    def _process_messages(self, loaded: LoadedModel, messages: List[dict]):
        """
        Parse messages for text and image content.

        Handles OpenAI multimodal format:
        {"type": "image_url", "image_url": {"url": "data:image/png;base64,..."}}

        Inserts the processor's image token (e.g., <image>) into the text
        so the processor can match images to their positions.
        """
        from PIL import Image

        # Detect the processor's image placeholder token
        processor = loaded.tokenizer
        image_token = getattr(processor, "image_token", "<image>")

        images = []
        text_parts = []

        for msg in messages:
            content = msg.get("content", "")
            if isinstance(content, str):
                text_parts.append(f"{msg['role']}: {content}")
                continue

            # Multimodal content (list of parts)
            for part in content:
                if isinstance(part, str):
                    text_parts.append(part)
                elif isinstance(part, dict):
                    if part.get("type") == "text":
                        text_parts.append(part["text"])
                    elif part.get("type") == "image_url":
                        image = self._load_image(part["image_url"]["url"])
                        images.append(image)
                        text_parts.append(image_token)

        text = "\n".join(text_parts)

        # Build processor inputs
        processor = loaded.tokenizer
        if images:
            inputs = processor(
                text=text,
                images=images,
                return_tensors="pt",
            )
        else:
            inputs = processor(
                text=text,
                return_tensors="pt",
            )

        # Move to model device
        device = next(loaded.model.parameters()).device
        inputs = {k: v.to(device) if hasattr(v, "to") else v for k, v in inputs.items()}

        return inputs, images

    def _load_image(self, url: str) -> "Image.Image":
        """Load an image from a URL or base64 data URI."""
        from PIL import Image

        if url.startswith("data:"):
            # Base64 data URI: data:image/png;base64,iVBOR...
            header, data = url.split(",", 1)
            image_bytes = base64.b64decode(data)
            return Image.open(io.BytesIO(image_bytes)).convert("RGB")

        # HTTP URL — use urllib (stdlib) to avoid requests/httpx dep issues
        from urllib.request import urlopen, Request
        req = Request(url, headers={"User-Agent": "model-engine/0.1"})
        with urlopen(req, timeout=30) as resp:
            return Image.open(io.BytesIO(resp.read())).convert("RGB")
