"""
Transformers Backend
--------------------
Handles all HF-native model formats:
- Native transformers (fp16/fp32/bf16)
- bitsandbytes 4-bit and 8-bit
- GPTQ (native transformers, fallback to auto-gptq)
- AWQ (native transformers, fallback to autoawq)

Uses accelerate for multi-GPU distribution via device_map + max_memory.
"""

import json
import logging
import re
import threading
from datetime import datetime
from typing import Iterator, List, Optional

import torch
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    TextIteratorStreamer,
)

from inferall.backends.base import (
    BaseBackend,
    GenerationParams,
    GenerationResult,
    LoadedModel,
)
from inferall.gpu.allocator import AllocationPlan
from inferall.registry.metadata import ModelFormat, ModelRecord

logger = logging.getLogger(__name__)

# Generic ChatML template for models without one
_CHATML_TEMPLATE = (
    "{% for message in messages %}"
    "<|im_start|>{{ message['role'] }}\n{{ message['content'] }}<|im_end|>\n"
    "{% endfor %}"
    "<|im_start|>assistant\n"
)


class TransformersBackend(BaseBackend):
    """Backend for HuggingFace transformers models."""

    @property
    def name(self) -> str:
        return "transformers"

    # -------------------------------------------------------------------------
    # Load
    # -------------------------------------------------------------------------

    def load(self, record: ModelRecord, allocation: AllocationPlan) -> LoadedModel:
        """
        Load a model using transformers + accelerate.

        Handles native, BNB, GPTQ, and AWQ formats with appropriate
        loading strategies and fallback chains.
        """
        model_path = str(record.local_path)
        trust = record.trust_remote_code

        logger.info(
            "Loading %s (format=%s, trust_remote_code=%s)",
            record.model_id, record.format.value, trust,
        )

        # Load tokenizer
        tokenizer = AutoTokenizer.from_pretrained(
            model_path, trust_remote_code=trust,
        )

        # Ensure chat template exists
        if tokenizer.chat_template is None:
            logger.info("No chat template found, using generic ChatML template")
            tokenizer.chat_template = _CHATML_TEMPLATE

        # Build loading kwargs
        load_kwargs = self._build_load_kwargs(record, allocation)

        # Load model with format-specific strategy
        model = self._load_model(record, load_kwargs)

        # Log device map
        if hasattr(model, "hf_device_map"):
            logger.info("Device map: %s", model.hf_device_map)

        loaded = LoadedModel(
            model_id=record.model_id,
            backend_name=self.name,
            model=model,
            tokenizer=tokenizer,
            vram_used_bytes=allocation.estimated_vram_bytes,
        )

        # Ensure the model's generation config has the correct eos_token_id.
        # Some models (e.g., Qwen3) have eos_token_id=None in config.json
        # but the tokenizer knows the correct value.
        if hasattr(model, "generation_config") and tokenizer.eos_token_id is not None:
            if model.generation_config.eos_token_id != tokenizer.eos_token_id:
                model.generation_config.eos_token_id = tokenizer.eos_token_id
                model.generation_config.pad_token_id = tokenizer.eos_token_id
                logger.info(
                    "Fixed generation_config.eos_token_id to %d",
                    tokenizer.eos_token_id,
                )

        logger.info("Loaded %s successfully", record.model_id)
        return loaded

    def _build_load_kwargs(self, record: ModelRecord, allocation: AllocationPlan) -> dict:
        """Build kwargs for AutoModelForCausalLM.from_pretrained()."""
        kwargs = {
            "pretrained_model_name_or_path": str(record.local_path),
            "trust_remote_code": record.trust_remote_code,
        }

        # Device placement
        if allocation.offload_to_cpu:
            kwargs["device_map"] = allocation.device_map
            if allocation.max_memory:
                kwargs["max_memory"] = allocation.max_memory
        elif allocation.max_memory:
            kwargs["device_map"] = "auto"
            kwargs["max_memory"] = allocation.max_memory
        else:
            kwargs["device_map"] = allocation.device_map

        # Dtype — default to auto for most formats
        kwargs["torch_dtype"] = "auto"

        # BitsAndBytes quantization
        if record.format == ModelFormat.TRANSFORMERS_BNB_4BIT:
            kwargs["quantization_config"] = self._bnb_config(bits=4)
        elif record.format == ModelFormat.TRANSFORMERS_BNB_8BIT:
            kwargs["quantization_config"] = self._bnb_config(bits=8)

        return kwargs

    def _bnb_config(self, bits: int):
        """Create a BitsAndBytesConfig."""
        try:
            from transformers import BitsAndBytesConfig
            if bits == 4:
                return BitsAndBytesConfig(
                    load_in_4bit=True,
                    bnb_4bit_compute_dtype=torch.float16,
                    bnb_4bit_quant_type="nf4",
                )
            else:
                return BitsAndBytesConfig(load_in_8bit=True)
        except ImportError:
            raise RuntimeError(
                "bitsandbytes is required for quantized loading. "
                "Install with: pip install bitsandbytes"
            )

    def _load_model(self, record: ModelRecord, load_kwargs: dict):
        """
        Load model with format-specific strategy.

        GPTQ/AWQ: try transformers native first, fallback to dedicated packages.
        """
        if record.format in (ModelFormat.GPTQ, ModelFormat.AWQ):
            return self._load_with_fallback(record, load_kwargs)

        logger.debug("Loading with AutoModelForCausalLM: %s", load_kwargs.keys())
        return AutoModelForCausalLM.from_pretrained(**load_kwargs)

    def _load_with_fallback(self, record: ModelRecord, load_kwargs: dict):
        """
        GPTQ/AWQ fallback chain:
        1. Try transformers native
        2. Try auto-gptq / autoawq
        3. Fail with install instructions
        """
        # Step 1: try transformers native
        native_error = None
        try:
            logger.info("Trying native transformers loading for %s", record.format.value)
            model = AutoModelForCausalLM.from_pretrained(**load_kwargs)
            logger.info("Loaded %s natively via transformers", record.format.value)
            return model
        except Exception as e:
            native_error = e
            logger.warning(
                "Native transformers loading failed for %s: %s",
                record.format.value, e,
            )

        # Step 2: try dedicated package
        model_path = str(record.local_path)
        trust = record.trust_remote_code

        if record.format == ModelFormat.GPTQ:
            try:
                from auto_gptq import AutoGPTQForCausalLM
                logger.info("Falling back to auto-gptq for %s", record.model_id)
                model = AutoGPTQForCausalLM.from_quantized(
                    model_path,
                    trust_remote_code=trust,
                    device_map=load_kwargs.get("device_map", "auto"),
                    max_memory=load_kwargs.get("max_memory"),
                )
                logger.info("Loaded via auto-gptq")
                return model
            except ImportError:
                raise RuntimeError(
                    f"Native GPTQ loading failed and auto-gptq is not installed.\n"
                    f"Install with: pip install auto-gptq\n"
                    f"Original error: {native_error}"
                )
            except Exception as gptq_err:
                raise RuntimeError(
                    f"Both native and auto-gptq loading failed for {record.model_id}.\n"
                    f"Native error: {native_error}\n"
                    f"auto-gptq error: {gptq_err}"
                )

        if record.format == ModelFormat.AWQ:
            try:
                from awq import AutoAWQForCausalLM
                logger.info("Falling back to autoawq for %s", record.model_id)
                model = AutoAWQForCausalLM.from_quantized(
                    model_path,
                    trust_remote_code=trust,
                    device_map=load_kwargs.get("device_map", "auto"),
                    max_memory=load_kwargs.get("max_memory"),
                )
                logger.info("Loaded via autoawq")
                return model
            except ImportError:
                raise RuntimeError(
                    f"Native AWQ loading failed and autoawq is not installed.\n"
                    f"Install with: pip install autoawq\n"
                    f"Original error: {native_error}"
                )
            except Exception as awq_err:
                raise RuntimeError(
                    f"Both native and autoawq loading failed for {record.model_id}.\n"
                    f"Native error: {native_error}\n"
                    f"autoawq error: {awq_err}"
                )

        raise RuntimeError(f"Unsupported format for fallback: {record.format}")

    # -------------------------------------------------------------------------
    # Generate (non-streaming)
    # -------------------------------------------------------------------------

    def generate(
        self,
        loaded: LoadedModel,
        messages: List[dict],
        params: GenerationParams,
    ) -> GenerationResult:
        """Generate a complete response."""
        loaded.touch()

        # For JSON mode, inject system prompt guidance
        if params.response_format:
            messages = self._inject_response_format(messages, params.response_format)

        # Apply chat template (with tools if provided)
        input_ids, attention_mask = self._apply_chat_template(
            loaded, messages, tools=params.tools,
        )
        prompt_tokens = input_ids.shape[1]

        # Build generation kwargs
        gen_kwargs = self._build_gen_kwargs(loaded, input_ids, attention_mask, params)

        # Generate
        with torch.inference_mode():
            output_ids = loaded.model.generate(**gen_kwargs)

        # Decode only the new tokens
        new_tokens = output_ids[0][prompt_tokens:]
        text = loaded.tokenizer.decode(new_tokens, skip_special_tokens=True)
        text = self._strip_thinking(text)
        completion_tokens = len(new_tokens)

        # Parse tool calls from output if tools were requested
        tool_calls = None
        if params.tools:
            tool_calls, text = self._parse_tool_calls(text)

        # Determine finish reason
        if tool_calls:
            finish_reason = "tool_calls"
        elif completion_tokens >= params.max_tokens:
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
        Stream tokens using TextIteratorStreamer.

        Runs model.generate() in a background thread and yields tokens
        as they're produced.
        """
        loaded.touch()

        input_ids, attention_mask = self._apply_chat_template(loaded, messages)
        gen_kwargs = self._build_gen_kwargs(loaded, input_ids, attention_mask, params)

        # Set up streamer
        streamer = TextIteratorStreamer(
            loaded.tokenizer,
            skip_prompt=True,
            skip_special_tokens=True,
        )
        gen_kwargs["streamer"] = streamer

        # Run generation in a thread
        gen_thread = threading.Thread(
            target=self._generate_with_streamer,
            args=(loaded.model, gen_kwargs),
            daemon=True,
        )
        gen_thread.start()

        # Yield tokens, stripping <think>...</think> blocks and stopping
        # at turn boundaries (some models generate "user:" as plain text
        # instead of using special tokens like <|im_start|>).
        _TURN_MARKERS = ("\nuser:", "\nUser:", "\nassistant:", "\nAssistant:",
                         "\nA:", "\nhuman:", "\nHuman:")
        try:
            in_think = False
            first_content = True  # Track if we've yielded any real content yet
            buf = ""
            stopped = False
            for token in streamer:
                if stopped:
                    continue  # Drain streamer without yielding
                if cancel is not None and cancel.is_set():
                    break
                if not token:
                    continue

                buf += token

                # Process buffer for think tags and turn markers
                while buf and not stopped:
                    if in_think:
                        close_idx = buf.find("</think>")
                        if close_idx != -1:
                            buf = buf[close_idx + 8:].lstrip()
                            in_think = False
                            first_content = True  # Next output is real content
                            continue
                        else:
                            break  # Keep buffering
                    else:
                        # Check for think opening tag
                        open_idx = buf.find("<think>")
                        if open_idx != -1:
                            if open_idx > 0:
                                yield buf[:open_idx]
                            buf = buf[open_idx + 7:]
                            in_think = True
                            continue

                        # Check for turn markers (model generating a new turn)
                        for marker in _TURN_MARKERS:
                            m_idx = buf.find(marker)
                            if m_idx != -1:
                                # Yield text before the marker, then stop
                                if m_idx > 0:
                                    yield buf[:m_idx]
                                buf = ""
                                stopped = True
                                break

                        if stopped:
                            break

                        # Check for partial tags at buffer end
                        if buf.endswith(("<", "<t", "<th", "<thi",
                                         "<thin", "<think")):
                            break

                        # Check for partial turn markers at end
                        hold = False
                        for marker in _TURN_MARKERS:
                            for i in range(1, len(marker)):
                                if buf.endswith(marker[:i]):
                                    hold = True
                                    break
                            if hold:
                                break

                        if hold:
                            break

                        # Safe to yield — strip role prefixes from first chunk
                        if first_content:
                            buf = re.sub(r"^[\s]*(?:[A-Z]:\s*)?", "", buf)
                            if not buf:
                                break  # Stripped everything, wait for more
                            first_content = False
                        yield buf
                        buf = ""

            # Flush remaining buffer
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
            logger.error("Generation error in streamer thread", exc_info=True)

    # -------------------------------------------------------------------------
    # Unload
    # -------------------------------------------------------------------------

    def unload(self, loaded: LoadedModel) -> None:
        """Unload model and free GPU memory."""
        logger.info("Unloading %s", loaded.model_id)

        del loaded.model
        del loaded.tokenizer
        loaded.model = None
        loaded.tokenizer = None

        if torch.cuda.is_available():
            torch.cuda.empty_cache()

        logger.debug("Unloaded %s, CUDA cache cleared", loaded.model_id)

    # -------------------------------------------------------------------------
    # Helpers
    # -------------------------------------------------------------------------

    def _apply_chat_template(self, loaded: LoadedModel, messages: List[dict], tools=None):
        """Apply the tokenizer's chat template and return (input_ids, attention_mask) on the model's device."""
        # Detect thinking-capable models and disable thinking by default
        template_kwargs = {}
        template_src = loaded.tokenizer.chat_template or ""
        if "enable_thinking" in template_src:
            template_kwargs["enable_thinking"] = False

        # Pass tools if the template supports them
        if tools and "tools" in template_src:
            template_kwargs["tools"] = tools

        result = loaded.tokenizer.apply_chat_template(
            messages,
            return_tensors="pt",
            add_generation_prompt=True,
            **template_kwargs,
        )

        # apply_chat_template may return a BatchEncoding or a bare tensor
        if hasattr(result, "input_ids"):
            input_ids = result["input_ids"]
            attention_mask = result.get("attention_mask", torch.ones_like(result["input_ids"]))
        else:
            input_ids = result
            attention_mask = torch.ones_like(input_ids)

        # Move to model's device
        if hasattr(loaded.model, "device"):
            device = loaded.model.device
        else:
            # Multi-GPU models — first parameter's device
            device = next(loaded.model.parameters()).device
        return input_ids.to(device), attention_mask.to(device)

    def _build_gen_kwargs(
        self,
        loaded: LoadedModel,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        params: GenerationParams,
    ) -> dict:
        """Build kwargs for model.generate()."""
        # Collect all EOS-like token IDs to ensure model stops at turn boundaries.
        # Many models (e.g., Qwen3) have eos_token_id in tokenizer but not in
        # model config, so we must explicitly set it.
        eos_ids = set()
        if loaded.tokenizer.eos_token_id is not None:
            eos_ids.add(loaded.tokenizer.eos_token_id)
        # Add common turn-end / next-turn tokens
        unk_id = loaded.tokenizer.unk_token_id
        for token_name in ["<|im_end|>", "<|eot_id|>", "<|end|>",
                           "<|im_start|>", "<|endoftext|>"]:
            tid = loaded.tokenizer.convert_tokens_to_ids(token_name)
            if tid is not None and (unk_id is None or tid != unk_id):
                eos_ids.add(tid)

        kwargs = {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "max_new_tokens": params.max_tokens,
            "do_sample": params.temperature > 0,
            "pad_token_id": loaded.tokenizer.eos_token_id,
            "eos_token_id": list(eos_ids) if len(eos_ids) > 1 else loaded.tokenizer.eos_token_id,
        }

        if params.temperature > 0:
            kwargs["temperature"] = params.temperature
            kwargs["top_p"] = params.top_p
            kwargs["top_k"] = params.top_k
            kwargs["repetition_penalty"] = params.repetition_penalty

        if params.stop:
            # Convert stop strings to token IDs for stopping criteria
            stop_ids = []
            for s in params.stop:
                ids = loaded.tokenizer.encode(s, add_special_tokens=False)
                if ids:
                    stop_ids.append(ids[-1])
            if stop_ids:
                kwargs["eos_token_id"] = stop_ids

        return kwargs

    @staticmethod
    def _strip_thinking(text: str) -> str:
        """Strip <think>...</think> reasoning blocks and turn artifacts from model output."""
        # Remove complete thinking blocks (possibly spanning multiple lines)
        stripped = re.sub(r"<think>.*?</think>\s*", "", text, flags=re.DOTALL)
        # Remove unclosed thinking block at the start (model still thinking)
        stripped = re.sub(r"^<think>.*", "", stripped, flags=re.DOTALL)
        # Remove any leading role prefix artifacts (e.g., "A: ")
        stripped = re.sub(r"^[A-Z]:\s*", "", stripped)
        # Stop at turn boundaries (model generating fake conversation)
        for marker in ("\nuser:", "\nUser:", "\nassistant:", "\nAssistant:",
                       "\nA:", "\nhuman:", "\nHuman:"):
            idx = stripped.find(marker)
            if idx != -1:
                stripped = stripped[:idx]
        return stripped.strip()

    @staticmethod
    def _parse_tool_calls(text: str):
        """
        Parse tool calls from model output text.

        Handles formats from Qwen, Llama 3.1+, Mistral, and generic JSON.
        Returns (list_of_ToolCall_or_None, remaining_text).
        """
        import json as json_mod
        import uuid as uuid_mod
        from inferall.backends.base import ToolCall

        tool_calls = []

        # Pattern 1: Qwen format — <tool_call>{"name": ..., "arguments": ...}</tool_call>
        qwen_pattern = re.compile(
            r'<tool_call>\s*(\{.*?\})\s*</tool_call>', re.DOTALL
        )
        for match in qwen_pattern.finditer(text):
            try:
                data = json_mod.loads(match.group(1))
                tool_calls.append(ToolCall(
                    id=f"call_{uuid_mod.uuid4().hex[:24]}",
                    function_name=data.get("name", ""),
                    function_arguments=json_mod.dumps(data.get("arguments", {})),
                ))
            except json_mod.JSONDecodeError:
                continue
        if tool_calls:
            remaining = qwen_pattern.sub("", text).strip()
            return tool_calls, remaining

        # Pattern 2: Mistral format — [TOOL_CALLS] [{"name": ..., "arguments": ...}]
        mistral_pattern = re.compile(
            r'\[TOOL_CALLS\]\s*(\[.*?\])', re.DOTALL
        )
        match = mistral_pattern.search(text)
        if match:
            try:
                calls = json_mod.loads(match.group(1))
                for call in calls:
                    tool_calls.append(ToolCall(
                        id=call.get("id", f"call_{uuid_mod.uuid4().hex[:24]}"),
                        function_name=call.get("name", ""),
                        function_arguments=json_mod.dumps(call.get("arguments", {})),
                    ))
                remaining = text[:match.start()].strip()
                return tool_calls, remaining
            except json_mod.JSONDecodeError:
                pass

        # Pattern 3: Llama 3.1 format — {"name": "func", "parameters": {...}}
        llama_pattern = re.compile(
            r'<function=(\w+)>(.*?)</function>', re.DOTALL
        )
        for match in llama_pattern.finditer(text):
            func_name = match.group(1)
            try:
                args = json_mod.loads(match.group(2))
            except json_mod.JSONDecodeError:
                args = {}
            tool_calls.append(ToolCall(
                id=f"call_{uuid_mod.uuid4().hex[:24]}",
                function_name=func_name,
                function_arguments=json_mod.dumps(args),
            ))
        if tool_calls:
            remaining = llama_pattern.sub("", text).strip()
            return tool_calls, remaining

        # Pattern 4: Generic JSON — try to parse as {"name": ..., "arguments": ...}
        generic_pattern = re.compile(
            r'\{"name"\s*:\s*"[^"]+"\s*,\s*"(?:arguments|parameters)"\s*:\s*\{.*?\}\s*\}',
            re.DOTALL,
        )
        for match in generic_pattern.finditer(text):
            try:
                data = json_mod.loads(match.group(0))
                args = data.get("arguments", data.get("parameters", {}))
                tool_calls.append(ToolCall(
                    id=f"call_{uuid_mod.uuid4().hex[:24]}",
                    function_name=data["name"],
                    function_arguments=json_mod.dumps(args),
                ))
            except (json_mod.JSONDecodeError, KeyError):
                continue
        if tool_calls:
            remaining = generic_pattern.sub("", text).strip()
            return tool_calls, remaining

        return None, text

    @staticmethod
    def _inject_response_format(messages: List[dict], response_format: dict) -> List[dict]:
        """Inject system prompt guidance for JSON mode."""
        fmt_type = response_format.get("type", "text")
        if fmt_type == "text":
            return messages

        messages = list(messages)  # Don't mutate original

        if fmt_type == "json_object":
            guidance = "Respond with valid JSON only. Do not include any text outside the JSON object."
        elif fmt_type == "json_schema":
            schema = response_format.get("json_schema", {})
            schema_str = json.dumps(schema.get("schema", {}), indent=2) if isinstance(schema, dict) else str(schema)
            guidance = f"Respond with valid JSON matching this schema:\n{schema_str}\nDo not include any text outside the JSON."
        else:
            return messages

        # Prepend to existing system message or add one
        if messages and messages[0].get("role") == "system":
            messages[0] = dict(messages[0])
            messages[0]["content"] = guidance + "\n\n" + (messages[0]["content"] or "")
        else:
            messages.insert(0, {"role": "system", "content": guidance})

        return messages
