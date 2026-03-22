"""
Orchestrator
------------
Central model lifecycle manager. Handles:
- Backend selection (transformers vs llama.cpp) based on ModelFormat
- Model loading with GPU allocation
- LRU eviction when max_loaded_models is reached
- Per-model locking (two-phase) to prevent concurrent double-loading
- Reference counting to prevent eviction during active inference
- Idle timeout eviction (background thread)

Entirely synchronous — called from thread pool workers only, never from
the async event loop. Uses threading.Lock throughout.
"""

import logging
import threading
import time
from dataclasses import dataclass
from datetime import datetime
from typing import Iterator, List, Optional

from inferall.backends.base import (
    BaseBackend,
    EmbeddingParams,
    EmbeddingResult,
    GenerationParams,
    GenerationResult,
    ImageGenerationParams,
    ImageGenerationResult,
    LoadedModel,
    TTSParams,
    TTSResult,
    Img2ImgParams,
    Img2ImgResult,
    RerankParams,
    AudioProcessingParams,
    AudioProcessingResult,
    ClassificationParams,
    ClassificationResult,
    DepthEstimationParams,
    DepthEstimationResult,
    DocumentQAParams,
    DocumentQAResult,
    ImageSegmentationParams,
    ImageSegmentationResult,
    ObjectDetectionParams,
    ObjectDetectionResult,
    Seq2SeqParams,
    Seq2SeqResult,
    VideoGenerationParams,
    VideoGenerationResult,
    RerankResult,
    TranscriptionParams,
    TranscriptionResult,
)
from inferall.backends.llamacpp_backend import LlamaCppBackend
from inferall.backends.transformers_backend import TransformersBackend
from inferall.config import EngineConfig
from inferall.gpu.allocator import GPUAllocator
from inferall.gpu.manager import GPUManager
from inferall.registry.metadata import ModelFormat, ModelRecord
from inferall.registry.registry import ModelRegistry

logger = logging.getLogger(__name__)

# Formats handled by each backend
_LLAMACPP_FORMATS = {ModelFormat.GGUF}
_TRANSFORMERS_FORMATS = {
    ModelFormat.TRANSFORMERS,
    ModelFormat.TRANSFORMERS_BNB_4BIT,
    ModelFormat.TRANSFORMERS_BNB_8BIT,
    ModelFormat.GPTQ,
    ModelFormat.AWQ,
}
_EMBEDDING_FORMATS = {ModelFormat.EMBEDDING}
_VLM_FORMATS = {ModelFormat.VISION_LANGUAGE}
_ASR_FORMATS = {ModelFormat.ASR}
_DIFFUSION_FORMATS = {ModelFormat.DIFFUSION}
_TTS_FORMATS = {ModelFormat.TTS}
_RERANK_FORMATS = {ModelFormat.RERANK}
_IMG2IMG_FORMATS = {ModelFormat.IMAGE_TO_IMAGE}
_VIDEO_FORMATS = {ModelFormat.TEXT_TO_VIDEO}
_SEQ2SEQ_FORMATS = {ModelFormat.SEQ2SEQ}
_CLASSIFICATION_FORMATS = {ModelFormat.CLASSIFICATION}
_OLLAMA_CLOUD_FORMATS = {ModelFormat.OLLAMA_CLOUD}


@dataclass
class LoadedModelInfo:
    """Public info about a loaded model (for status display)."""

    model_id: str
    backend_name: str
    loaded_at: datetime
    last_used_at: datetime
    vram_used_bytes: int
    ref_count: int


class ModelNotFoundError(Exception):
    """Raised when a model is not in the registry."""


class Orchestrator:
    """
    Entirely synchronous model lifecycle manager.

    Uses two-phase locking to prevent deadlocks:
    - _global_lock: protects dict mutations (loaded_models, _ref_counts, _model_locks)
    - Per-model locks: prevent concurrent double-loading of the same model

    Lock ordering: _global_lock is NEVER held while acquiring a model lock.
    """

    def __init__(
        self,
        config: EngineConfig,
        registry: ModelRegistry,
        gpu_manager: GPUManager,
        allocator: GPUAllocator,
    ):
        self.config = config
        self.registry = registry
        self.gpu_manager = gpu_manager
        self.allocator = allocator

        # Model state
        self.loaded_models: dict[str, LoadedModel] = {}
        self._model_locks: dict[str, threading.Lock] = {}
        self._ref_counts: dict[str, int] = {}
        self._global_lock = threading.Lock()

        # Backends (lazily instantiated singletons)
        self._transformers_backend: Optional[TransformersBackend] = None
        self._llamacpp_backend: Optional[LlamaCppBackend] = None
        self._embedding_backend = None
        self._vlm_backend = None
        self._asr_backend = None
        self._diffusion_backend = None
        self._tts_backend = None
        self._rerank_backend = None
        self._img2img_backend = None
        self._video_backend = None
        self._seq2seq_backend = None
        self._classification_backend = None
        self._ollama_cloud_backend = None

        # Idle timeout thread
        self._shutdown_event = threading.Event()
        self._idle_thread: Optional[threading.Thread] = None
        if config.idle_timeout > 0:
            self._start_idle_monitor()

    # -------------------------------------------------------------------------
    # Backend Selection
    # -------------------------------------------------------------------------

    def _get_backend(self, fmt: ModelFormat):
        """Select the appropriate backend for the model format."""
        if fmt in _LLAMACPP_FORMATS:
            if self._llamacpp_backend is None:
                self._llamacpp_backend = LlamaCppBackend()
            return self._llamacpp_backend
        if fmt in _TRANSFORMERS_FORMATS:
            if self._transformers_backend is None:
                self._transformers_backend = TransformersBackend()
            return self._transformers_backend
        if fmt in _EMBEDDING_FORMATS:
            if self._embedding_backend is None:
                from inferall.backends.embedding_backend import SentenceTransformersBackend
                self._embedding_backend = SentenceTransformersBackend()
            return self._embedding_backend
        if fmt in _VLM_FORMATS:
            if self._vlm_backend is None:
                from inferall.backends.vlm_backend import VisionLanguageTransformersBackend
                self._vlm_backend = VisionLanguageTransformersBackend()
            return self._vlm_backend
        if fmt in _ASR_FORMATS:
            if self._asr_backend is None:
                from inferall.backends.asr_backend import WhisperBackend
                self._asr_backend = WhisperBackend()
            return self._asr_backend
        if fmt in _DIFFUSION_FORMATS:
            if self._diffusion_backend is None:
                from inferall.backends.diffusion_backend import DiffusersBackend
                self._diffusion_backend = DiffusersBackend()
            return self._diffusion_backend
        if fmt in _TTS_FORMATS:
            if self._tts_backend is None:
                from inferall.backends.tts_backend import TTSTransformersBackend
                self._tts_backend = TTSTransformersBackend()
            return self._tts_backend
        if fmt in _RERANK_FORMATS:
            if self._rerank_backend is None:
                from inferall.backends.rerank_backend import CrossEncoderRerankerBackend
                self._rerank_backend = CrossEncoderRerankerBackend()
            return self._rerank_backend
        if fmt in _IMG2IMG_FORMATS:
            if self._img2img_backend is None:
                from inferall.backends.img2img_backend import Img2ImgDiffusersBackend
                self._img2img_backend = Img2ImgDiffusersBackend()
            return self._img2img_backend
        if fmt in _VIDEO_FORMATS:
            if self._video_backend is None:
                from inferall.backends.video_backend import VideoDiffusersBackend
                self._video_backend = VideoDiffusersBackend()
            return self._video_backend
        if fmt in _SEQ2SEQ_FORMATS:
            if self._seq2seq_backend is None:
                from inferall.backends.seq2seq_backend import Seq2SeqTransformersBackend
                self._seq2seq_backend = Seq2SeqTransformersBackend()
            return self._seq2seq_backend
        if fmt in _CLASSIFICATION_FORMATS:
            if self._classification_backend is None:
                from inferall.backends.classification_backend import TransformersClassificationBackend
                self._classification_backend = TransformersClassificationBackend()
            return self._classification_backend
        if fmt in _OLLAMA_CLOUD_FORMATS:
            if self._ollama_cloud_backend is None:
                from inferall.backends.ollama_cloud_backend import OllamaCloudBackend
                self._ollama_cloud_backend = OllamaCloudBackend()
            return self._ollama_cloud_backend
        raise ValueError(f"No backend for format: {fmt}")

    # -------------------------------------------------------------------------
    # get_or_load — Two-Phase Locking
    # -------------------------------------------------------------------------

    def get_or_load(self, model_id: str) -> LoadedModel:
        """
        Get a loaded model or load it. Increments ref count.

        Caller MUST call release(model_id) when done (use try/finally).

        Two-phase locking:
        1. Fast path: check under global lock, return if loaded
        2. Slow path: acquire per-model lock, load if needed
        """
        # --- Phase 1: fast path under global lock ---
        with self._global_lock:
            if model_id in self.loaded_models:
                self._ref_counts[model_id] += 1
                loaded = self.loaded_models[model_id]
                loaded.touch()
                return loaded

            # Get or create per-model lock
            if model_id not in self._model_locks:
                self._model_locks[model_id] = threading.Lock()
            model_lock = self._model_locks[model_id]
        # --- global_lock is RELEASED ---

        # --- Phase 2: slow path under model lock only ---
        with model_lock:
            # Re-check: another thread may have loaded it while we waited
            with self._global_lock:
                if model_id in self.loaded_models:
                    self._ref_counts[model_id] += 1
                    loaded = self.loaded_models[model_id]
                    loaded.touch()
                    return loaded

            # Look up in registry
            record = self.registry.get(model_id)
            if record is None:
                raise ModelNotFoundError(
                    f"Model '{model_id}' not found. "
                    f"Run: inferall pull {model_id}"
                )

            # Evict if needed (before computing allocation)
            self._evict_if_needed()

            # Compute allocation
            backend = self._get_backend(record.format)
            allocation = self.allocator.compute_allocation(record)

            # Load the model (slow — no locks held)
            logger.info("Loading model: %s (backend=%s)", model_id, backend.name)
            loaded = backend.load(record, allocation)

            # Record in GPU manager (one allocation per GPU used)
            if allocation.gpu_ids:
                vram_per_gpu = allocation.estimated_vram_bytes // len(allocation.gpu_ids)
                for gpu_id in allocation.gpu_ids:
                    self.gpu_manager.record_allocation(
                        f"{model_id}@gpu{gpu_id}", gpu_id, vram_per_gpu,
                    )

            # Update registry last_used_at
            self.registry.update_last_used(model_id)

            # Store under global lock
            with self._global_lock:
                self.loaded_models[model_id] = loaded
                self._ref_counts[model_id] = 1

            logger.info("Model loaded: %s", model_id)
            return loaded

    # -------------------------------------------------------------------------
    # Release / Unload
    # -------------------------------------------------------------------------

    def release(self, model_id: str) -> None:
        """Decrement ref count for a model. Call in a finally block."""
        with self._global_lock:
            if model_id in self._ref_counts:
                self._ref_counts[model_id] = max(0, self._ref_counts[model_id] - 1)

    def unload(self, model_id: str) -> None:
        """
        Unload a model and free its resources.

        Refuses if the model has active references (ref_count > 0).
        """
        with self._global_lock:
            if model_id not in self.loaded_models:
                logger.warning("Cannot unload %s: not loaded", model_id)
                return
            if self._ref_counts.get(model_id, 0) > 0:
                logger.warning(
                    "Cannot unload %s: %d active references",
                    model_id, self._ref_counts[model_id],
                )
                return
            loaded = self.loaded_models.pop(model_id)
            self._ref_counts.pop(model_id, None)
        # global_lock released — do the actual unload outside

        backend = self._get_backend(
            ModelFormat(loaded.backend_name)
            if loaded.backend_name in [f.value for f in ModelFormat]
            else self._format_from_backend_name(loaded.backend_name)
        )
        backend.unload(loaded)

        # Release GPU allocation
        self._release_gpu_allocations(model_id)
        logger.info("Unloaded model: %s", model_id)

    def _format_from_backend_name(self, backend_name: str) -> ModelFormat:
        """Map backend name to a representative format for backend selection."""
        _name_to_format = {
            "llamacpp": ModelFormat.GGUF,
            "transformers": ModelFormat.TRANSFORMERS,
            "embedding": ModelFormat.EMBEDDING,
            "vlm": ModelFormat.VISION_LANGUAGE,
            "asr": ModelFormat.ASR,
            "diffusion": ModelFormat.DIFFUSION,
            "tts": ModelFormat.TTS,
            "rerank": ModelFormat.RERANK,
            "img2img": ModelFormat.IMAGE_TO_IMAGE,
            "video": ModelFormat.TEXT_TO_VIDEO,
            "seq2seq": ModelFormat.SEQ2SEQ,
            "classification": ModelFormat.CLASSIFICATION,
            "ollama_cloud": ModelFormat.OLLAMA_CLOUD,
        }
        return _name_to_format.get(backend_name, ModelFormat.TRANSFORMERS)

    def _release_gpu_allocations(self, model_id: str) -> None:
        """Release all GPU allocation records for a model."""
        # We record allocations as model_id@gpuN
        keys_to_release = [
            k for k in list(self.gpu_manager.gpu_assignments.keys())
            if k.startswith(f"{model_id}@gpu")
        ]
        for key in keys_to_release:
            self.gpu_manager.release_allocation(key)
        # Also try the bare model_id (for single-GPU or tests)
        if not keys_to_release:
            self.gpu_manager.release_allocation(model_id)

    # -------------------------------------------------------------------------
    # Generate / Stream
    # -------------------------------------------------------------------------

    def generate(
        self,
        model_id: str,
        messages: List[dict],
        params: GenerationParams,
    ) -> GenerationResult:
        """Generate a complete response. Handles get_or_load + release."""
        loaded = self.get_or_load(model_id)
        try:
            backend = self._get_backend(
                self._format_from_backend_name(loaded.backend_name)
            )
            t0 = time.perf_counter()
            result = backend.generate(loaded, messages, params)
            elapsed_ms = (time.perf_counter() - t0) * 1000.0
            result.total_time_ms = elapsed_ms
            if result.completion_tokens > 0 and elapsed_ms > 0:
                result.tokens_per_second = result.completion_tokens / (elapsed_ms / 1000.0)
            logger.info(
                "Generate %s: %d tokens in %.0fms (%.1f tok/s)",
                model_id, result.completion_tokens, elapsed_ms,
                result.tokens_per_second or 0,
            )
            return result
        finally:
            self.release(model_id)

    def stream(
        self,
        model_id: str,
        messages: List[dict],
        params: GenerationParams,
        cancel: Optional[threading.Event] = None,
    ) -> Iterator[str]:
        """
        Stream tokens. Handles get_or_load + release.

        The caller should iterate the returned iterator. Release happens
        when the iterator is exhausted or an exception occurs.
        """
        loaded = self.get_or_load(model_id)
        try:
            backend = self._get_backend(
                self._format_from_backend_name(loaded.backend_name)
            )
            yield from backend.stream(loaded, messages, params, cancel)
        finally:
            self.release(model_id)

    # -------------------------------------------------------------------------
    # Task-Specific Dispatch (Multi-Modal)
    # -------------------------------------------------------------------------

    def embed(
        self,
        model_id: str,
        texts: List[str],
        params: EmbeddingParams,
    ) -> EmbeddingResult:
        """Generate embeddings. Handles get_or_load + release."""
        loaded = self.get_or_load(model_id)
        try:
            backend = self._get_backend(
                self._format_from_backend_name(loaded.backend_name)
            )
            t0 = time.perf_counter()
            result = backend.embed(loaded, texts, params)
            elapsed_ms = (time.perf_counter() - t0) * 1000.0
            result.total_time_ms = elapsed_ms
            logger.info(
                "Embed %s: %d texts, %d tokens in %.0fms",
                model_id, len(texts), result.prompt_tokens, elapsed_ms,
            )
            return result
        finally:
            self.release(model_id)

    def transcribe(
        self,
        model_id: str,
        audio_bytes: bytes,
        params: TranscriptionParams,
    ) -> TranscriptionResult:
        """Transcribe audio. Handles get_or_load + release."""
        loaded = self.get_or_load(model_id)
        try:
            backend = self._get_backend(
                self._format_from_backend_name(loaded.backend_name)
            )
            t0 = time.perf_counter()
            result = backend.transcribe(loaded, audio_bytes, params)
            elapsed_ms = (time.perf_counter() - t0) * 1000.0
            result.total_time_ms = elapsed_ms
            logger.info("Transcribe %s: completed in %.0fms", model_id, elapsed_ms)
            return result
        finally:
            self.release(model_id)

    def generate_image(
        self,
        model_id: str,
        prompt: str,
        params: ImageGenerationParams,
    ) -> ImageGenerationResult:
        """Generate images. Handles get_or_load + release."""
        loaded = self.get_or_load(model_id)
        try:
            backend = self._get_backend(
                self._format_from_backend_name(loaded.backend_name)
            )
            t0 = time.perf_counter()
            result = backend.generate_image(loaded, prompt, params)
            elapsed_ms = (time.perf_counter() - t0) * 1000.0
            result.total_time_ms = elapsed_ms
            logger.info(
                "Generate image %s: %d image(s) in %.0fms",
                model_id, len(result.images), elapsed_ms,
            )
            return result
        finally:
            self.release(model_id)

    def edit_image(
        self,
        model_id: str,
        prompt: str,
        params: Img2ImgParams,
    ) -> Img2ImgResult:
        """Edit an image via img2img. Handles get_or_load + release."""
        loaded = self.get_or_load(model_id)
        try:
            backend = self._get_backend(
                self._format_from_backend_name(loaded.backend_name)
            )
            t0 = time.perf_counter()
            result = backend.edit_image(loaded, prompt, params)
            elapsed_ms = (time.perf_counter() - t0) * 1000.0
            result.total_time_ms = elapsed_ms
            logger.info(
                "Edit image %s: %d image(s) in %.0fms",
                model_id, len(result.images), elapsed_ms,
            )
            return result
        finally:
            self.release(model_id)

    def generate_video(
        self,
        model_id: str,
        prompt: str,
        params: VideoGenerationParams,
    ) -> VideoGenerationResult:
        """Generate a video. Handles get_or_load + release."""
        loaded = self.get_or_load(model_id)
        try:
            backend = self._get_backend(
                self._format_from_backend_name(loaded.backend_name)
            )
            t0 = time.perf_counter()
            result = backend.generate_video(loaded, prompt, params)
            elapsed_ms = (time.perf_counter() - t0) * 1000.0
            result.total_time_ms = elapsed_ms
            logger.info(
                "Generate video %s: %d frames in %.0fms",
                model_id, result.num_frames, elapsed_ms,
            )
            return result
        finally:
            self.release(model_id)

    def synthesize(
        self,
        model_id: str,
        text: str,
        params: TTSParams,
    ) -> TTSResult:
        """Synthesize speech. Handles get_or_load + release."""
        loaded = self.get_or_load(model_id)
        try:
            backend = self._get_backend(
                self._format_from_backend_name(loaded.backend_name)
            )
            t0 = time.perf_counter()
            result = backend.synthesize(loaded, text, params)
            elapsed_ms = (time.perf_counter() - t0) * 1000.0
            result.total_time_ms = elapsed_ms
            logger.info("Synthesize %s: completed in %.0fms", model_id, elapsed_ms)
            return result
        finally:
            self.release(model_id)

    def rerank(
        self,
        model_id: str,
        query: str,
        documents: List[str],
        params: RerankParams,
    ) -> RerankResult:
        """Rerank documents by relevance. Handles get_or_load + release."""
        loaded = self.get_or_load(model_id)
        try:
            backend = self._get_backend(
                self._format_from_backend_name(loaded.backend_name)
            )
            t0 = time.perf_counter()
            result = backend.rerank(loaded, query, documents, params)
            elapsed_ms = (time.perf_counter() - t0) * 1000.0
            result.total_time_ms = elapsed_ms
            logger.info(
                "Rerank %s: %d docs in %.0fms",
                model_id, len(documents), elapsed_ms,
            )
            return result
        finally:
            self.release(model_id)

    def seq2seq_generate(
        self,
        model_id: str,
        text: str,
        params: Seq2SeqParams,
    ) -> Seq2SeqResult:
        """Seq2seq generation (translate, summarize). Handles get_or_load + release."""
        loaded = self.get_or_load(model_id)
        try:
            backend = self._get_backend(
                self._format_from_backend_name(loaded.backend_name)
            )
            t0 = time.perf_counter()
            result = backend.generate(loaded, text, params)
            elapsed_ms = (time.perf_counter() - t0) * 1000.0
            result.total_time_ms = elapsed_ms
            if result.completion_tokens > 0 and elapsed_ms > 0:
                result.tokens_per_second = result.completion_tokens / (elapsed_ms / 1000.0)
            logger.info(
                "Seq2seq %s: %d tokens in %.0fms (%.1f tok/s)",
                model_id, result.completion_tokens, elapsed_ms,
                result.tokens_per_second or 0,
            )
            return result
        finally:
            self.release(model_id)

    def classify(
        self,
        model_id: str,
        text: str,
        params: ClassificationParams,
    ) -> ClassificationResult:
        """Classify input. Handles get_or_load + release."""
        loaded = self.get_or_load(model_id)
        try:
            backend = self._get_backend(
                self._format_from_backend_name(loaded.backend_name)
            )
            t0 = time.perf_counter()
            result = backend.classify(loaded, text, params)
            elapsed_ms = (time.perf_counter() - t0) * 1000.0
            result.total_time_ms = elapsed_ms
            logger.info("Classify %s: completed in %.0fms", model_id, elapsed_ms)
            return result
        finally:
            self.release(model_id)

    def detect_objects(self, model_id: str, params: ObjectDetectionParams) -> ObjectDetectionResult:
        """Detect objects in an image. Handles get_or_load + release."""
        loaded = self.get_or_load(model_id)
        try:
            backend = self._get_backend(self._format_from_backend_name(loaded.backend_name))
            t0 = time.perf_counter()
            result = backend.detect_objects(loaded, params)
            elapsed_ms = (time.perf_counter() - t0) * 1000.0
            result.total_time_ms = elapsed_ms
            logger.info("Detect %s: %d objects in %.0fms", model_id, len(result.detections), elapsed_ms)
            return result
        finally:
            self.release(model_id)

    def segment_image(self, model_id: str, params: ImageSegmentationParams) -> ImageSegmentationResult:
        """Segment an image. Handles get_or_load + release."""
        loaded = self.get_or_load(model_id)
        try:
            backend = self._get_backend(self._format_from_backend_name(loaded.backend_name))
            t0 = time.perf_counter()
            result = backend.segment_image(loaded, params)
            elapsed_ms = (time.perf_counter() - t0) * 1000.0
            result.total_time_ms = elapsed_ms
            logger.info("Segment %s: %d segments in %.0fms", model_id, len(result.segments), elapsed_ms)
            return result
        finally:
            self.release(model_id)

    def estimate_depth(self, model_id: str, params: DepthEstimationParams) -> DepthEstimationResult:
        """Estimate depth from an image. Handles get_or_load + release."""
        loaded = self.get_or_load(model_id)
        try:
            backend = self._get_backend(self._format_from_backend_name(loaded.backend_name))
            t0 = time.perf_counter()
            result = backend.estimate_depth(loaded, params)
            elapsed_ms = (time.perf_counter() - t0) * 1000.0
            result.total_time_ms = elapsed_ms
            logger.info("Depth %s: completed in %.0fms", model_id, elapsed_ms)
            return result
        finally:
            self.release(model_id)

    def answer_document(self, model_id: str, params: DocumentQAParams) -> DocumentQAResult:
        """Answer a question about a document. Handles get_or_load + release."""
        loaded = self.get_or_load(model_id)
        try:
            backend = self._get_backend(self._format_from_backend_name(loaded.backend_name))
            t0 = time.perf_counter()
            result = backend.answer_document(loaded, params)
            elapsed_ms = (time.perf_counter() - t0) * 1000.0
            result.total_time_ms = elapsed_ms
            logger.info("DocQA %s: completed in %.0fms", model_id, elapsed_ms)
            return result
        finally:
            self.release(model_id)

    def process_audio(self, model_id: str, params: AudioProcessingParams) -> AudioProcessingResult:
        """Process audio. Handles get_or_load + release."""
        loaded = self.get_or_load(model_id)
        try:
            backend = self._get_backend(self._format_from_backend_name(loaded.backend_name))
            t0 = time.perf_counter()
            result = backend.process_audio(loaded, params)
            elapsed_ms = (time.perf_counter() - t0) * 1000.0
            result.total_time_ms = elapsed_ms
            logger.info("Audio process %s: completed in %.0fms", model_id, elapsed_ms)
            return result
        finally:
            self.release(model_id)

    # -------------------------------------------------------------------------
    # Eviction
    # -------------------------------------------------------------------------

    def _evict_if_needed(self) -> None:
        """Evict LRU models if we're at capacity."""
        with self._global_lock:
            while len(self.loaded_models) >= self.config.max_loaded_models:
                victim = self._find_eviction_candidate()
                if victim is None:
                    logger.warning(
                        "Cannot evict: all %d loaded models have active references",
                        len(self.loaded_models),
                    )
                    break
                # Remove from dicts
                loaded = self.loaded_models.pop(victim)
                self._ref_counts.pop(victim, None)
                # Release global lock for the actual unload
                self._global_lock.release()
                try:
                    backend = self._get_backend(
                        self._format_from_backend_name(loaded.backend_name)
                    )
                    backend.unload(loaded)
                    self._release_gpu_allocations(victim)
                    logger.info("Evicted model: %s", victim)
                finally:
                    self._global_lock.acquire()

    def _find_eviction_candidate(self) -> Optional[str]:
        """
        Find the LRU model with ref_count == 0.

        Must be called while holding _global_lock.
        """
        candidate = None
        oldest_time = None

        for model_id, loaded in self.loaded_models.items():
            if self._ref_counts.get(model_id, 0) > 0:
                continue
            if oldest_time is None or loaded.last_used_at < oldest_time:
                oldest_time = loaded.last_used_at
                candidate = model_id

        return candidate

    # -------------------------------------------------------------------------
    # Idle Timeout Monitor
    # -------------------------------------------------------------------------

    def _start_idle_monitor(self) -> None:
        """Start background thread that evicts idle models."""
        self._idle_thread = threading.Thread(
            target=self._idle_monitor_loop,
            daemon=True,
            name="idle-monitor",
        )
        self._idle_thread.start()

    def _idle_monitor_loop(self) -> None:
        """Check for idle models every 60 seconds."""
        while not self._shutdown_event.wait(timeout=60.0):
            self._evict_idle_models()

    def _evict_idle_models(self) -> None:
        """Evict models that have been idle longer than idle_timeout."""
        now = datetime.now()
        to_evict = []

        with self._global_lock:
            for model_id, loaded in self.loaded_models.items():
                if self._ref_counts.get(model_id, 0) > 0:
                    continue
                idle_seconds = (now - loaded.last_used_at).total_seconds()
                if idle_seconds >= self.config.idle_timeout:
                    to_evict.append(model_id)

        # Evict outside global lock
        for model_id in to_evict:
            logger.info(
                "Evicting idle model: %s (idle %.0fs > %ds timeout)",
                model_id,
                (now - self.loaded_models[model_id].last_used_at).total_seconds()
                if model_id in self.loaded_models else 0,
                self.config.idle_timeout,
            )
            self.unload(model_id)

    # -------------------------------------------------------------------------
    # Status / Lifecycle
    # -------------------------------------------------------------------------

    def list_loaded(self) -> List[LoadedModelInfo]:
        """Return info about all currently loaded models."""
        with self._global_lock:
            return [
                LoadedModelInfo(
                    model_id=model_id,
                    backend_name=loaded.backend_name,
                    loaded_at=loaded.loaded_at,
                    last_used_at=loaded.last_used_at,
                    vram_used_bytes=loaded.vram_used_bytes,
                    ref_count=self._ref_counts.get(model_id, 0),
                )
                for model_id, loaded in self.loaded_models.items()
            ]

    def shutdown(self) -> None:
        """Graceful shutdown: stop idle monitor, unload all models."""
        logger.info("Orchestrator shutting down...")

        # Stop idle monitor
        self._shutdown_event.set()
        if self._idle_thread is not None:
            self._idle_thread.join(timeout=5.0)

        # Unload all models
        with self._global_lock:
            model_ids = list(self.loaded_models.keys())

        for model_id in model_ids:
            try:
                # Force unload even with refs (shutdown)
                with self._global_lock:
                    if model_id in self.loaded_models:
                        loaded = self.loaded_models.pop(model_id)
                        self._ref_counts.pop(model_id, None)
                    else:
                        continue
                backend = self._get_backend(
                    self._format_from_backend_name(loaded.backend_name)
                )
                backend.unload(loaded)
                self._release_gpu_allocations(model_id)
            except Exception:
                logger.error("Error unloading %s during shutdown", model_id, exc_info=True)

        logger.info("Orchestrator shutdown complete")
