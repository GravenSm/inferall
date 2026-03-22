"""
ASR Backend
-----------
Handles automatic speech recognition models (Whisper, etc.).

Uses transformers AutoProcessor + AutoModelForSpeechSeq2Seq.
Audio decoding via soundfile, with librosa fallback for resampling.
"""

import io
import logging
from typing import Optional

import torch

from inferall.backends.base import (
    ASRBackend,
    LoadedModel,
    TranscriptionParams,
    TranscriptionResult,
)
from inferall.gpu.allocator import AllocationPlan
from inferall.registry.metadata import ModelRecord

logger = logging.getLogger(__name__)

# Whisper requires 16kHz audio
_TARGET_SAMPLE_RATE = 16000


class WhisperBackend(ASRBackend):
    """ASR backend for Whisper-family models."""

    @property
    def name(self) -> str:
        return "asr"

    # -------------------------------------------------------------------------
    # Load
    # -------------------------------------------------------------------------

    def load(self, record: ModelRecord, allocation: AllocationPlan) -> LoadedModel:
        """Load an ASR model."""
        model_path = str(record.local_path)

        logger.info("Loading ASR model %s", record.model_id)

        from transformers import AutoModelForSpeechSeq2Seq, AutoProcessor

        processor = AutoProcessor.from_pretrained(model_path)

        load_kwargs = {
            "pretrained_model_name_or_path": model_path,
            "torch_dtype": "auto",
        }
        if allocation.max_memory:
            load_kwargs["device_map"] = "auto"
            load_kwargs["max_memory"] = allocation.max_memory
        else:
            load_kwargs["device_map"] = allocation.device_map

        model = AutoModelForSpeechSeq2Seq.from_pretrained(**load_kwargs)

        logger.info("Loaded ASR model %s", record.model_id)

        return LoadedModel(
            model_id=record.model_id,
            backend_name=self.name,
            model=model,
            tokenizer=processor,  # Store processor as "tokenizer"
            vram_used_bytes=allocation.estimated_vram_bytes,
        )

    # -------------------------------------------------------------------------
    # Transcribe
    # -------------------------------------------------------------------------

    def transcribe(
        self,
        loaded: LoadedModel,
        audio_bytes: bytes,
        params: TranscriptionParams,
    ) -> TranscriptionResult:
        """Transcribe audio bytes to text."""
        loaded.touch()

        audio_array, sample_rate = self._decode_audio(audio_bytes)

        # Resample to target rate if needed
        if sample_rate != _TARGET_SAMPLE_RATE:
            audio_array = self._resample(audio_array, sample_rate, _TARGET_SAMPLE_RATE)

        # Process through processor
        processor = loaded.tokenizer
        inputs = processor(
            audio_array,
            sampling_rate=_TARGET_SAMPLE_RATE,
            return_tensors="pt",
        )

        # Move to model device
        if hasattr(loaded.model, "device"):
            device = loaded.model.device
        else:
            device = next(loaded.model.parameters()).device

        inputs = {k: v.to(device) if hasattr(v, "to") else v for k, v in inputs.items()}

        # Generate kwargs
        gen_kwargs = {}
        gen_kwargs["task"] = params.task  # "transcribe" or "translate"
        if params.language:
            gen_kwargs["language"] = params.language

        with torch.inference_mode():
            predicted_ids = loaded.model.generate(**inputs, **gen_kwargs)

        text = processor.decode(predicted_ids[0], skip_special_tokens=True)

        return TranscriptionResult(
            text=text.strip(),
            language=params.language,
        )

    # -------------------------------------------------------------------------
    # Unload
    # -------------------------------------------------------------------------

    def unload(self, loaded: LoadedModel) -> None:
        """Unload ASR model and free resources."""
        logger.info("Unloading ASR model %s", loaded.model_id)

        del loaded.model
        del loaded.tokenizer
        loaded.model = None
        loaded.tokenizer = None

        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    # -------------------------------------------------------------------------
    # Helpers
    # -------------------------------------------------------------------------

    def _decode_audio(self, audio_bytes: bytes):
        """Decode audio bytes to numpy array + sample rate."""
        import numpy as np

        try:
            import soundfile as sf
            audio_array, sample_rate = sf.read(io.BytesIO(audio_bytes))
        except Exception:
            # Fallback to librosa for broader format support (MP3, etc.)
            try:
                import librosa
                audio_array, sample_rate = librosa.load(
                    io.BytesIO(audio_bytes), sr=None,
                )
            except ImportError:
                raise RuntimeError(
                    "Could not decode audio. Install soundfile or librosa: "
                    "pip install soundfile librosa"
                )

        # Convert stereo to mono if needed
        if len(audio_array.shape) > 1:
            audio_array = np.mean(audio_array, axis=1)

        return audio_array, sample_rate

    def _resample(self, audio, orig_sr: int, target_sr: int):
        """Resample audio to target sample rate."""
        try:
            import librosa
            return librosa.resample(audio, orig_sr=orig_sr, target_sr=target_sr)
        except ImportError:
            pass

        # Fallback: scipy
        try:
            from scipy.signal import resample
            import numpy as np
            num_samples = int(len(audio) * target_sr / orig_sr)
            return resample(audio, num_samples)
        except ImportError:
            raise RuntimeError(
                "Audio resampling requires librosa or scipy. "
                "Install with: pip install librosa"
            )
