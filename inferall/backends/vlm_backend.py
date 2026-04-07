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

        gen_kwargs = self._build_gen_kwargs(loaded, inputs, params)

        with torch.inference_mode():
            output_ids = loaded.model.generate(**gen_kwargs)

        new_tokens = output_ids[0][prompt_tokens:]
        text = loaded.tokenizer.decode(new_tokens, skip_special_tokens=True)
        text = self._strip_thinking_and_turns(text)
        completion_tokens = len(new_tokens)

        finish_reason = "length" if completion_tokens >= params.max_tokens else "stop"

        return GenerationResult(
            text=text,
            prompt_tokens=prompt_tokens,
            completion_tokens=completion_tokens,
            finish_reason=finish_reason,
        )

    def _build_gen_kwargs(self, loaded: LoadedModel, inputs: dict, params: GenerationParams) -> dict:
        """Build generation kwargs with proper EOS token IDs and stopping criteria."""
        processor = loaded.tokenizer
        # Get the underlying tokenizer (processors wrap one)
        tokenizer = getattr(processor, "tokenizer", processor)

        # Collect EOS-like token IDs so the model stops at turn boundaries
        eos_ids = set()
        if hasattr(tokenizer, "eos_token_id") and tokenizer.eos_token_id is not None:
            eos_ids.add(tokenizer.eos_token_id)

        unk_id = getattr(tokenizer, "unk_token_id", None)
        for token_name in ["<|im_end|>", "<|eot_id|>", "<|end|>",
                           "<|im_start|>", "<|endoftext|>"]:
            try:
                tid = tokenizer.convert_tokens_to_ids(token_name)
                if tid is not None and (unk_id is None or tid != unk_id):
                    eos_ids.add(tid)
            except Exception:
                pass

        gen_kwargs = {
            **inputs,
            "max_new_tokens": params.max_tokens,
            "do_sample": params.temperature > 0,
        }

        if eos_ids:
            gen_kwargs["eos_token_id"] = list(eos_ids) if len(eos_ids) > 1 else next(iter(eos_ids))
            pad_id = getattr(tokenizer, "pad_token_id", None) or getattr(tokenizer, "eos_token_id", None)
            if pad_id is not None:
                gen_kwargs["pad_token_id"] = pad_id

        if params.temperature > 0:
            gen_kwargs["temperature"] = params.temperature
            gen_kwargs["top_p"] = params.top_p

        return gen_kwargs

    @staticmethod
    def _strip_thinking_and_turns(text: str) -> str:
        """Strip <think> blocks, role artifacts, and stop at turn boundaries."""
        import re as _re
        # Remove complete thinking blocks
        text = _re.sub(r"<think>.*?</think>\s*", "", text, flags=_re.DOTALL)
        # Remove unclosed thinking block at start
        text = _re.sub(r"^<think>.*", "", text, flags=_re.DOTALL)
        # Stop at turn markers (model generating fake conversation)
        for marker in ("\nassistant\n", "\nuser\n", "\nsystem\n",
                       "\nassistant:", "\nuser:", "\nsystem:",
                       "\nA:", "\nhuman:", "\nHuman:", "\nAssistant:", "\nUser:"):
            idx = text.find(marker)
            if idx != -1:
                text = text[:idx]
        return text.strip()

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
        gen_kwargs = self._build_gen_kwargs(loaded, inputs, params)

        # Use the underlying tokenizer for streaming
        processor = loaded.tokenizer
        stream_tokenizer = getattr(processor, "tokenizer", processor)

        streamer = TextIteratorStreamer(
            stream_tokenizer,
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

        # Same turn-boundary detection as TransformersBackend
        _TURN_MARKERS = ("\nassistant\n", "\nuser\n", "\nsystem\n",
                         "\nassistant:", "\nuser:", "\nsystem:",
                         "\nA:", "\nhuman:", "\nHuman:",
                         "\nAssistant:", "\nUser:")
        try:
            in_think = False
            buf = ""
            stopped = False
            for token in streamer:
                if stopped:
                    continue
                if cancel is not None and cancel.is_set():
                    break
                if not token:
                    continue

                buf += token

                while buf and not stopped:
                    if in_think:
                        close_idx = buf.find("</think>")
                        if close_idx != -1:
                            buf = buf[close_idx + 8:].lstrip()
                            in_think = False
                            continue
                        else:
                            break
                    else:
                        open_idx = buf.find("<think>")
                        if open_idx != -1:
                            if open_idx > 0:
                                yield buf[:open_idx]
                            buf = buf[open_idx + 7:]
                            in_think = True
                            continue

                        # Turn marker check
                        marker_hit = False
                        for marker in _TURN_MARKERS:
                            m_idx = buf.find(marker)
                            if m_idx != -1:
                                if m_idx > 0:
                                    yield buf[:m_idx]
                                buf = ""
                                stopped = True
                                marker_hit = True
                                break
                        if marker_hit:
                            break

                        # Hold buffer if it ends with a partial marker
                        hold = False
                        if buf.endswith(("<", "<t", "<th", "<thi", "<thin", "<think")):
                            hold = True
                        if not hold:
                            for marker in _TURN_MARKERS:
                                for i in range(1, len(marker)):
                                    if buf.endswith(marker[:i]):
                                        hold = True
                                        break
                                if hold:
                                    break
                        if hold:
                            break

                        yield buf
                        buf = ""

            if buf and not in_think and not stopped:
                yield buf
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

        Strategy:
        1. Convert OpenAI messages → HF chat-template format with PIL images inline
        2. Try processor.apply_chat_template() — modern multimodal processors handle
           role tokens, image placeholders, and generation prompts automatically
        3. Fall back to manual text concatenation for older processors
        """
        from PIL import Image
        processor = loaded.tokenizer

        # Step 1: Convert OpenAI messages → HF format, extracting PIL images
        hf_messages = []
        all_images = []

        for msg in messages:
            role = msg.get("role", "user")
            content = msg.get("content", "")

            if isinstance(content, str):
                hf_messages.append({"role": role, "content": [{"type": "text", "text": content}]})
                continue

            # Multimodal content list
            hf_content = []
            for part in content:
                if isinstance(part, str):
                    hf_content.append({"type": "text", "text": part})
                elif isinstance(part, dict):
                    ptype = part.get("type")
                    if ptype == "text":
                        hf_content.append({"type": "text", "text": part["text"]})
                    elif ptype == "image_url":
                        image = self._load_image(part["image_url"]["url"])
                        all_images.append(image)
                        hf_content.append({"type": "image"})
                    elif ptype == "image":
                        # Already in HF format
                        if "image" in part:
                            all_images.append(part["image"])
                        hf_content.append({"type": "image"})
            hf_messages.append({"role": role, "content": hf_content})

        # Step 2: Try apply_chat_template (modern path)
        inputs = None
        try:
            template_kwargs = {
                "add_generation_prompt": True,
                "tokenize": True,
                "return_dict": True,
                "return_tensors": "pt",
            }
            # Some processors accept images as a kwarg
            if all_images:
                try:
                    inputs = processor.apply_chat_template(
                        hf_messages, **template_kwargs,
                    )
                    # If the processor didn't fold images in, do it via processor()
                    if "pixel_values" not in inputs and "image_grid_thw" not in inputs:
                        # Re-run with explicit images
                        text_only = processor.apply_chat_template(
                            hf_messages, add_generation_prompt=True, tokenize=False,
                        )
                        inputs = processor(
                            text=text_only, images=all_images, return_tensors="pt",
                        )
                except TypeError:
                    # Some processors need images passed differently
                    text_only = processor.apply_chat_template(
                        hf_messages, add_generation_prompt=True, tokenize=False,
                    )
                    inputs = processor(
                        text=text_only, images=all_images, return_tensors="pt",
                    )
            else:
                inputs = processor.apply_chat_template(
                    hf_messages, **template_kwargs,
                )
        except Exception as e:
            logger.info("apply_chat_template failed (%s), using manual fallback", e)
            inputs = None

        # Step 3: Fallback — manual text concatenation
        if inputs is None:
            image_token = getattr(processor, "image_token", "<image>")
            text_parts = []
            for msg in messages:
                content = msg.get("content", "")
                if isinstance(content, str):
                    text_parts.append(f"{msg['role']}: {content}")
                else:
                    for part in content:
                        if isinstance(part, dict):
                            if part.get("type") == "text":
                                text_parts.append(part["text"])
                            elif part.get("type") == "image_url":
                                text_parts.append(image_token)
            text = "\n".join(text_parts)

            if all_images:
                inputs = processor(
                    text=text, images=all_images, return_tensors="pt",
                )
            else:
                inputs = processor(text=text, return_tensors="pt")

        # Move to model device
        device = next(loaded.model.parameters()).device
        inputs = {k: v.to(device) if hasattr(v, "to") else v for k, v in inputs.items()}

        return inputs, all_images

    def _load_image(self, url: str) -> "Image.Image":
        """Load an image from a URL or base64 data URI."""
        from PIL import Image

        if url.startswith("data:"):
            # Base64 data URI: data:image/png;base64,iVBOR...
            header, data = url.split(",", 1)
            image_bytes = base64.b64decode(data)
            return Image.open(io.BytesIO(image_bytes)).convert("RGB")

        # HTTP URL — validate scheme to prevent SSRF
        from urllib.parse import urlparse
        parsed = urlparse(url)
        if parsed.scheme not in ("http", "https"):
            raise ValueError(f"Unsupported URL scheme: {parsed.scheme}")

        # Block private/internal network ranges
        import socket
        try:
            ip = socket.gethostbyname(parsed.hostname)
            import ipaddress
            addr = ipaddress.ip_address(ip)
            if addr.is_private or addr.is_loopback or addr.is_reserved or addr.is_link_local:
                raise ValueError(f"URL resolves to private/internal address: {ip}")
        except (socket.gaierror, ValueError) as e:
            if "private" in str(e) or "internal" in str(e):
                raise
            # If DNS fails, let the request fail naturally below

        from urllib.request import urlopen, Request
        req = Request(url, headers={"User-Agent": "inferall/0.1"})
        with urlopen(req, timeout=30) as resp:
            return Image.open(io.BytesIO(resp.read())).convert("RGB")
