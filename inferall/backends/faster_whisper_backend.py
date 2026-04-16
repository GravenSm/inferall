"""
Faster-Whisper Backend
----------------------
ASR backend for CTranslate2-format Whisper models (Systran/faster-whisper-*,
and any other repo that publishes its weights as a CT2 `model.bin`).

These models ship a CT2-native config.json that has no `model_type` field,
so `transformers.AutoModelForSpeechSeq2Seq.from_pretrained` cannot load them.
We delegate to the `faster_whisper` library instead, which wraps CTranslate2.

Import of `faster_whisper` is deferred until load() so the module is safe to
import without the dependency installed — the orchestrator can still be
constructed and other backends can run.
"""

import io
import logging

from inferall.backends.base import (
    ASRBackend,
    LoadedModel,
    TranscriptionParams,
    TranscriptionResult,
)
from inferall.gpu.allocator import AllocationPlan
from inferall.registry.metadata import ModelRecord

logger = logging.getLogger(__name__)


class FasterWhisperBackend(ASRBackend):
    """ASR backend for CTranslate2 / faster-whisper Whisper models."""

    @property
    def name(self) -> str:
        return "faster_whisper"

    # -------------------------------------------------------------------------
    # Load
    # -------------------------------------------------------------------------

    def load(self, record: ModelRecord, allocation: AllocationPlan) -> LoadedModel:
        """Load a faster-whisper model from its local CT2 directory."""
        try:
            from faster_whisper import WhisperModel
        except ImportError as e:
            raise RuntimeError(
                "faster-whisper is required for CTranslate2 Whisper models. "
                "Install with: pip install 'inferall[asr]' (or `pip install faster-whisper`)."
            ) from e

        model_path = str(record.local_path)
        device, compute_type = self._select_device_and_dtype(allocation)

        logger.info(
            "Loading faster-whisper model %s on %s (compute_type=%s)",
            record.model_id, device, compute_type,
        )

        model = WhisperModel(
            model_path,
            device=device,
            compute_type=compute_type,
        )

        logger.info("Loaded faster-whisper model %s", record.model_id)

        return LoadedModel(
            model_id=record.model_id,
            backend_name=self.name,
            model=model,
            tokenizer=None,  # faster-whisper owns its tokenizer internally
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
        """Transcribe audio bytes to text via faster-whisper."""
        loaded.touch()

        # faster-whisper accepts a file-like object; it handles decoding +
        # resampling internally (via its own FFmpeg/av pipeline), so we don't
        # need the soundfile/librosa dance that the transformers path uses.
        audio_stream = io.BytesIO(audio_bytes)

        segments_iter, info = loaded.model.transcribe(
            audio_stream,
            language=params.language,
            task=params.task,  # "transcribe" or "translate"
        )

        # segments is a generator; materialise so we can sum text + attach
        # them to the result when verbose_json is requested.
        segment_list = []
        for seg in segments_iter:
            segment_list.append({
                "id": getattr(seg, "id", len(segment_list)),
                "start": seg.start,
                "end": seg.end,
                "text": seg.text,
            })

        text = "".join(s["text"] for s in segment_list).strip()
        detected_language = getattr(info, "language", None) or params.language
        duration = getattr(info, "duration", None)

        return TranscriptionResult(
            text=text,
            language=detected_language,
            duration=duration,
            segments=(
                segment_list if params.response_format == "verbose_json" else None
            ),
        )

    # -------------------------------------------------------------------------
    # Unload
    # -------------------------------------------------------------------------

    def unload(self, loaded: LoadedModel) -> None:
        """Unload faster-whisper model and release GPU memory (if any)."""
        logger.info("Unloading faster-whisper model %s", loaded.model_id)

        del loaded.model
        loaded.model = None
        loaded.tokenizer = None

        # ctranslate2 releases GPU memory on WhisperModel deletion; no
        # explicit cuda.empty_cache() call needed (torch may not even be
        # the framework backing ctranslate2 here).

    # -------------------------------------------------------------------------
    # Helpers
    # -------------------------------------------------------------------------

    def _select_device_and_dtype(self, allocation: AllocationPlan):
        """Choose faster-whisper device + compute_type from the allocation.

        Defaults follow upstream faster-whisper recommendations: float16 on
        GPU (smallest memory footprint with good accuracy), int8 on CPU
        (dramatic speedup versus float32 on typical x86 hardware).
        """
        if allocation.gpu_ids:
            return "cuda", "float16"
        return "cpu", "int8"
