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

        # When the client asks for verbose_json we need per-segment timestamps,
        # which Whisper's `generate` produces via return_timestamps. This
        # changes the shape of predicted_ids subtly, which is why we only
        # enable it on-demand rather than unconditionally.
        want_timestamps = (params.response_format == "verbose_json")
        if want_timestamps:
            gen_kwargs["return_timestamps"] = True

        with torch.inference_mode():
            predicted_ids = loaded.model.generate(**inputs, **gen_kwargs)

        text = processor.decode(predicted_ids[0], skip_special_tokens=True)

        segments = None
        if want_timestamps:
            segments = self._extract_segments(processor, predicted_ids)

        return TranscriptionResult(
            text=text.strip(),
            language=params.language,
            segments=segments,
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

    def _extract_segments(self, processor, predicted_ids):
        """Decode Whisper predicted_ids with offset timestamps into segment dicts.

        Shape returned matches the faster-whisper / OpenAI verbose_json
        segment objects: ``{"id": int, "start": float, "end": float, "text": str}``.
        Returns None on any decode failure (different transformers versions
        expose the offsets API slightly differently) so the transcription
        path never regresses to an error — the caller just omits segments.
        """
        try:
            decoded = processor.batch_decode(
                predicted_ids,
                skip_special_tokens=True,
                output_offsets=True,
            )
        except Exception:
            logger.debug(
                "Whisper timestamp decode failed; returning text-only result",
                exc_info=True,
            )
            return None

        if not decoded or not isinstance(decoded, list):
            return None

        first = decoded[0]
        if not isinstance(first, dict):
            return None
        chunks = first.get("offsets") or []

        segments = []
        for i, chunk in enumerate(chunks):
            if not isinstance(chunk, dict):
                continue
            timestamp = chunk.get("timestamp") or (None, None)
            try:
                start, end = timestamp[0], timestamp[1]
            except (TypeError, IndexError):
                start, end = None, None
            segments.append({
                "id": i,
                "start": float(start) if start is not None else 0.0,
                "end": float(end) if end is not None else 0.0,
                "text": chunk.get("text", ""),
            })
        return segments or None

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
