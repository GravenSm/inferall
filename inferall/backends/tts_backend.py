"""
TTS Backend
-----------
Handles text-to-speech models (Bark, SpeechT5, etc.).

Uses transformers AutoProcessor + AutoModel. Initially targets Bark,
the most straightforward HF TTS model.
"""

import io
import logging

import torch

from inferall.backends.base import (
    LoadedModel,
    TTSBackend,
    TTSParams,
    TTSResult,
)
from inferall.gpu.allocator import AllocationPlan
from inferall.registry.metadata import ModelRecord

logger = logging.getLogger(__name__)


class TTSTransformersBackend(TTSBackend):
    """TTS backend for Bark and similar HF TTS models."""

    @property
    def name(self) -> str:
        return "tts"

    # -------------------------------------------------------------------------
    # Load
    # -------------------------------------------------------------------------

    def load(self, record: ModelRecord, allocation: AllocationPlan) -> LoadedModel:
        """Load a TTS model."""
        model_path = str(record.local_path)
        trust = record.trust_remote_code

        logger.info("Loading TTS model %s", record.model_id)

        from transformers import AutoModel, AutoProcessor

        processor = AutoProcessor.from_pretrained(model_path, trust_remote_code=trust)

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

        model = AutoModel.from_pretrained(**load_kwargs)

        logger.info("Loaded TTS model %s", record.model_id)

        return LoadedModel(
            model_id=record.model_id,
            backend_name=self.name,
            model=model,
            tokenizer=processor,  # Store processor as "tokenizer"
            vram_used_bytes=allocation.estimated_vram_bytes,
        )

    # -------------------------------------------------------------------------
    # Synthesize
    # -------------------------------------------------------------------------

    def synthesize(
        self,
        loaded: LoadedModel,
        text: str,
        params: TTSParams,
    ) -> TTSResult:
        """Synthesize speech from text."""
        loaded.touch()

        import numpy as np

        processor = loaded.tokenizer
        model = loaded.model

        # Build processor inputs
        proc_kwargs = {"text": [text], "return_tensors": "pt"}
        if params.voice and params.voice != "default":
            proc_kwargs["voice_preset"] = params.voice

        inputs = processor(**proc_kwargs)

        # Move to model device
        if hasattr(model, "device"):
            device = model.device
        else:
            device = next(model.parameters()).device

        inputs = {k: v.to(device) if hasattr(v, "to") else v for k, v in inputs.items()}

        with torch.inference_mode():
            speech_output = model.generate(**inputs)

        # Convert to audio numpy array
        audio_array = speech_output.cpu().numpy().squeeze()

        # Get sample rate from model config
        # v5: generation_config may not exist on non-generative models
        gen_config = getattr(model, "generation_config", None)
        sample_rate = getattr(
            gen_config, "sample_rate",
            getattr(model.config, "sample_rate", 24000),
        )

        # Convert to WAV bytes
        audio_bytes = self._to_wav(audio_array, sample_rate)
        content_type = "audio/wav"

        return TTSResult(
            audio_bytes=audio_bytes,
            content_type=content_type,
            sample_rate=sample_rate,
        )

    # -------------------------------------------------------------------------
    # Unload
    # -------------------------------------------------------------------------

    def unload(self, loaded: LoadedModel) -> None:
        """Unload TTS model and free resources."""
        logger.info("Unloading TTS model %s", loaded.model_id)

        del loaded.model
        del loaded.tokenizer
        loaded.model = None
        loaded.tokenizer = None

        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    # -------------------------------------------------------------------------
    # Helpers
    # -------------------------------------------------------------------------

    def _to_wav(self, audio_array, sample_rate: int) -> bytes:
        """Convert numpy audio array to WAV bytes."""
        import numpy as np

        # Normalize to int16 range
        if audio_array.dtype == np.float32 or audio_array.dtype == np.float64:
            audio_array = np.clip(audio_array, -1.0, 1.0)
            audio_array = (audio_array * 32767).astype(np.int16)

        try:
            import scipy.io.wavfile as wavfile
            buffer = io.BytesIO()
            wavfile.write(buffer, sample_rate, audio_array)
            return buffer.getvalue()
        except ImportError:
            raise RuntimeError(
                "scipy is required for WAV encoding. "
                "Install with: pip install scipy"
            )
