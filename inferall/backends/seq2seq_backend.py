"""
Seq2Seq Backend
---------------
Handles encoder-decoder models for translation, summarization, and
general text-to-text generation.

Models: T5, mT5, mBART, NLLB, MarianMT, FLAN-T5, etc.
Uses AutoModelForSeq2SeqLM + AutoTokenizer.
"""

import logging
from typing import Optional

import torch
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer

from inferall.backends.base import (
    LoadedModel,
    Seq2SeqBackend,
    Seq2SeqParams,
    Seq2SeqResult,
)
from inferall.gpu.allocator import AllocationPlan
from inferall.registry.metadata import ModelRecord

logger = logging.getLogger(__name__)


class Seq2SeqTransformersBackend(Seq2SeqBackend):
    """Seq2seq backend using AutoModelForSeq2SeqLM."""

    @property
    def name(self) -> str:
        return "seq2seq"

    # -------------------------------------------------------------------------
    # Load
    # -------------------------------------------------------------------------

    def load(self, record: ModelRecord, allocation: AllocationPlan) -> LoadedModel:
        """Load a seq2seq model."""
        model_path = str(record.local_path)
        trust = record.trust_remote_code

        logger.info("Loading seq2seq model %s", record.model_id)

        tokenizer = AutoTokenizer.from_pretrained(
            model_path, trust_remote_code=trust,
        )

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

        model = AutoModelForSeq2SeqLM.from_pretrained(**load_kwargs)

        logger.info("Loaded seq2seq model %s", record.model_id)

        return LoadedModel(
            model_id=record.model_id,
            backend_name=self.name,
            model=model,
            tokenizer=tokenizer,
            vram_used_bytes=allocation.estimated_vram_bytes,
        )

    # -------------------------------------------------------------------------
    # Generate
    # -------------------------------------------------------------------------

    def generate(
        self,
        loaded: LoadedModel,
        text: str,
        params: Seq2SeqParams,
    ) -> Seq2SeqResult:
        """Generate text from input (translate, summarize, etc.)."""
        loaded.touch()

        tokenizer = loaded.tokenizer
        model = loaded.model

        # Handle language codes for translation models
        self._set_language_codes(tokenizer, params)

        # Tokenize input
        inputs = tokenizer(
            text,
            return_tensors="pt",
            truncation=True,
            max_length=1024,
        )

        # Move to model device
        if hasattr(model, "device"):
            device = model.device
        else:
            device = next(model.parameters()).device
        inputs = {k: v.to(device) for k, v in inputs.items()}

        prompt_tokens = inputs["input_ids"].shape[1]

        # Build generation kwargs
        gen_kwargs = {
            **inputs,
            "max_new_tokens": params.max_tokens,
            "num_beams": params.num_beams,
        }

        if params.temperature > 0 and params.temperature != 1.0:
            gen_kwargs["do_sample"] = True
            gen_kwargs["temperature"] = params.temperature
        else:
            gen_kwargs["do_sample"] = False

        # Handle forced_bos_token_id for NLLB/mBART target language
        forced_bos = self._get_forced_bos_token_id(tokenizer, params)
        if forced_bos is not None:
            gen_kwargs["forced_bos_token_id"] = forced_bos

        with torch.inference_mode():
            output_ids = model.generate(**gen_kwargs)

        # Decode
        output_text = tokenizer.decode(output_ids[0], skip_special_tokens=True)
        completion_tokens = len(output_ids[0])

        return Seq2SeqResult(
            text=output_text,
            prompt_tokens=prompt_tokens,
            completion_tokens=completion_tokens,
        )

    # -------------------------------------------------------------------------
    # Unload
    # -------------------------------------------------------------------------

    def unload(self, loaded: LoadedModel) -> None:
        """Unload seq2seq model and free resources."""
        logger.info("Unloading seq2seq model %s", loaded.model_id)

        del loaded.model
        del loaded.tokenizer
        loaded.model = None
        loaded.tokenizer = None

        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    # -------------------------------------------------------------------------
    # Language Code Helpers
    # -------------------------------------------------------------------------

    def _set_language_codes(self, tokenizer, params: Seq2SeqParams) -> None:
        """Set source/target language on the tokenizer (mBART/NLLB style)."""
        if params.source_lang and hasattr(tokenizer, "src_lang"):
            tokenizer.src_lang = params.source_lang
        if params.target_lang and hasattr(tokenizer, "tgt_lang"):
            tokenizer.tgt_lang = params.target_lang

    def _get_forced_bos_token_id(self, tokenizer, params: Seq2SeqParams) -> Optional[int]:
        """Get forced_bos_token_id for target language (NLLB/mBART)."""
        if params.target_lang is None:
            return None

        # NLLB and mBART use lang_code_to_id or convert_tokens_to_ids
        if hasattr(tokenizer, "lang_code_to_id"):
            lang_id = tokenizer.lang_code_to_id.get(params.target_lang)
            if lang_id is not None:
                return lang_id

        # Try converting the target lang as a token
        try:
            token_id = tokenizer.convert_tokens_to_ids(params.target_lang)
            unk_id = tokenizer.unk_token_id
            if token_id is not None and (unk_id is None or token_id != unk_id):
                return token_id
        except Exception:
            pass

        return None
