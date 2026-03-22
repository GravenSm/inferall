"""
Backend Abstractions
--------------------
Abstract base classes for inference backends, plus shared data structures.

Chat backends implement: load, generate, stream, unload.
Multi-modal backends implement task-specific methods (embed, transcribe, etc.).
The orchestrator calls these through the abstract interfaces.
"""

import threading
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from datetime import datetime
from typing import Any, Iterator, List, Optional


# =============================================================================
# Chat / Text-Generation Data Structures
# =============================================================================

@dataclass
class GenerationParams:
    """Parameters for text generation."""

    max_tokens: int = 2048
    temperature: float = 0.7
    top_p: float = 0.9
    top_k: int = 50
    repetition_penalty: float = 1.1
    stop: Optional[List[str]] = None
    tools: Optional[List[dict]] = None         # OpenAI tools array
    tool_choice: Optional[Any] = None          # "auto", "none", "required", or specific
    response_format: Optional[dict] = None     # {"type": "text"|"json_object"|"json_schema"}


@dataclass
class ToolCall:
    """A tool/function call returned by the model."""

    id: str                     # e.g., "call_abc123"
    type: str = "function"
    function_name: str = ""
    function_arguments: str = ""  # JSON string


@dataclass
class GenerationResult:
    """Result from a non-streaming generation call."""

    text: str
    prompt_tokens: int
    completion_tokens: int
    finish_reason: str  # "stop", "length", or "tool_calls"
    tool_calls: Optional[List[ToolCall]] = None
    total_time_ms: Optional[float] = None
    tokens_per_second: Optional[float] = None


# =============================================================================
# Multi-Modal Data Structures
# =============================================================================

@dataclass
class EmbeddingParams:
    """Parameters for embedding generation."""

    normalize: bool = True
    truncate: bool = True


@dataclass
class EmbeddingResult:
    """Result from embedding generation."""

    embeddings: list        # List[List[float]]
    prompt_tokens: int
    model: str
    total_time_ms: Optional[float] = None


@dataclass
class TranscriptionParams:
    """Parameters for ASR."""

    language: Optional[str] = None
    response_format: str = "json"  # json, text, verbose_json
    task: str = "transcribe"       # "transcribe" or "translate" (to English)


@dataclass
class TranscriptionResult:
    """Result from ASR."""

    text: str
    language: Optional[str] = None
    duration: Optional[float] = None
    segments: Optional[list] = None  # For verbose_json
    total_time_ms: Optional[float] = None


@dataclass
class ImageGenerationParams:
    """Parameters for image generation."""

    n: int = 1
    size: str = "1024x1024"
    num_inference_steps: int = 30
    guidance_scale: float = 7.5
    negative_prompt: Optional[str] = None
    seed: Optional[int] = None


@dataclass
class ImageGenerationResult:
    """Result from image generation."""

    images: list            # List of base64-encoded strings
    revised_prompt: Optional[str] = None
    total_time_ms: Optional[float] = None


# =============================================================================
# Image-to-Image Data Structures
# =============================================================================

@dataclass
class Img2ImgParams:
    """Parameters for image-to-image editing."""

    image_b64: str = ""                    # Base64-encoded input image
    strength: float = 0.75                 # 0.0 = no change, 1.0 = full denoise
    n: int = 1
    size: Optional[str] = None            # Optional resize (None = keep input dims)
    num_inference_steps: int = 30
    guidance_scale: float = 7.5
    negative_prompt: Optional[str] = None
    seed: Optional[int] = None


@dataclass
class Img2ImgResult:
    """Result from image-to-image editing."""

    images: list            # List of base64-encoded strings
    revised_prompt: Optional[str] = None
    total_time_ms: Optional[float] = None


# =============================================================================
# Video Generation Data Structures
# =============================================================================

@dataclass
class VideoGenerationParams:
    """Parameters for text-to-video generation."""

    num_frames: int = 16
    fps: int = 8
    size: str = "512x512"
    num_inference_steps: int = 50           # Video models need more steps
    guidance_scale: float = 7.5
    negative_prompt: Optional[str] = None
    seed: Optional[int] = None
    output_format: str = "frames+mp4"      # "frames", "mp4", or "frames+mp4"


@dataclass
class VideoGenerationResult:
    """Result from video generation."""

    frames: list                            # List of base64-encoded PNG strings
    video_b64: Optional[str] = None        # Base64-encoded MP4 (if available)
    num_frames: int = 0
    fps: int = 8
    total_time_ms: Optional[float] = None


@dataclass
class TTSParams:
    """Parameters for text-to-speech."""

    voice: str = "default"
    speed: float = 1.0
    response_format: str = "wav"  # wav, mp3, opus, flac


@dataclass
class TTSResult:
    """Result from TTS. Audio bytes ready for streaming or saving."""

    audio_bytes: bytes
    content_type: str       # audio/wav, audio/mpeg, etc.
    sample_rate: int
    total_time_ms: Optional[float] = None


# =============================================================================
# Rerank Data Structures
# =============================================================================

@dataclass
class RerankParams:
    """Parameters for reranking."""

    top_n: Optional[int] = None        # Return top N results (None = all)
    return_documents: bool = False     # Echo back document text in results
    max_length: Optional[int] = None   # Max token length for truncation


@dataclass
class RerankResult:
    """Result from reranking. Results sorted by relevance_score descending."""

    results: list           # List[dict] with keys: index, relevance_score, document (optional)
    model: str
    usage: dict             # {"prompt_tokens": N}
    total_time_ms: Optional[float] = None


# =============================================================================
# Seq2Seq Data Structures
# =============================================================================

@dataclass
class Seq2SeqParams:
    """Parameters for seq2seq generation (translation, summarization)."""

    max_tokens: int = 512
    temperature: float = 1.0
    num_beams: int = 4                     # Beam search width
    source_lang: Optional[str] = None      # Source language code
    target_lang: Optional[str] = None      # Target language code


@dataclass
class Seq2SeqResult:
    """Result from seq2seq generation."""

    text: str
    prompt_tokens: int
    completion_tokens: int
    total_time_ms: Optional[float] = None
    tokens_per_second: Optional[float] = None


# =============================================================================
# Classification Data Structures
# =============================================================================

@dataclass
class ClassificationParams:
    """Parameters for classification (image, audio, zero-shot)."""

    candidate_labels: Optional[List[str]] = None  # Required for zero-shot
    top_k: int = 5
    image_b64: Optional[str] = None               # Base64-encoded image
    audio_b64: Optional[str] = None               # Base64-encoded audio


@dataclass
class ClassificationResult:
    """Result from classification. Labels sorted by score descending."""

    labels: list            # List[dict] with keys: label, score
    model: str
    pipeline_tag: Optional[str] = None
    total_time_ms: Optional[float] = None


# =============================================================================
# Object Detection Data Structures
# =============================================================================

@dataclass
class ObjectDetectionParams:
    """Parameters for object detection."""

    image_b64: str = ""
    threshold: float = 0.5
    candidate_labels: Optional[List[str]] = None  # For zero-shot


@dataclass
class ObjectDetectionResult:
    """Result from object detection."""

    detections: list     # List[dict] with keys: label, score, box: {xmin, ymin, xmax, ymax}
    model: str
    pipeline_tag: Optional[str] = None
    total_time_ms: Optional[float] = None


# =============================================================================
# Image Segmentation Data Structures
# =============================================================================

@dataclass
class ImageSegmentationParams:
    """Parameters for image segmentation."""

    image_b64: str = ""
    threshold: float = 0.5


@dataclass
class ImageSegmentationResult:
    """Result from image segmentation."""

    segments: list       # List[dict] with keys: label, score, mask_b64
    model: str
    pipeline_tag: Optional[str] = None
    total_time_ms: Optional[float] = None


# =============================================================================
# Depth Estimation Data Structures
# =============================================================================

@dataclass
class DepthEstimationParams:
    """Parameters for depth estimation."""

    image_b64: str = ""


@dataclass
class DepthEstimationResult:
    """Result from depth estimation."""

    depth_map_b64: str = ""
    width: int = 0
    height: int = 0
    model: str = ""
    total_time_ms: Optional[float] = None


# =============================================================================
# Document QA Data Structures
# =============================================================================

@dataclass
class DocumentQAParams:
    """Parameters for document question answering."""

    image_b64: str = ""
    question: str = ""


@dataclass
class DocumentQAResult:
    """Result from document QA."""

    answer: str = ""
    score: float = 0.0
    model: str = ""
    total_time_ms: Optional[float] = None


# =============================================================================
# Audio Processing Data Structures
# =============================================================================

@dataclass
class AudioProcessingParams:
    """Parameters for audio-to-audio processing."""

    audio_b64: str = ""


@dataclass
class AudioProcessingResult:
    """Result from audio processing."""

    audio_bytes: bytes = b""
    content_type: str = "audio/wav"
    sample_rate: int = 16000
    model: str = ""
    total_time_ms: Optional[float] = None


# =============================================================================
# Shared Loaded Model
# =============================================================================

@dataclass
class LoadedModel:
    """A model that has been loaded into memory and is ready for inference."""

    model_id: str
    backend_name: str           # "transformers", "llamacpp", "embedding", "vlm", etc.
    model: Any                  # The actual model object
    tokenizer: Any              # Tokenizer/processor or None
    loaded_at: datetime = field(default_factory=datetime.now)
    last_used_at: datetime = field(default_factory=datetime.now)
    vram_used_bytes: int = 0    # VRAM estimate (telemetry — for status display)

    def touch(self):
        """Update last_used_at timestamp."""
        self.last_used_at = datetime.now()


# =============================================================================
# Chat Backend (text-generation)
# =============================================================================

class BaseBackend(ABC):
    """
    Abstract base for chat/text-generation backends.

    Each backend handles a specific set of model formats.
    The orchestrator selects the backend based on ModelFormat.
    """

    @property
    @abstractmethod
    def name(self) -> str:
        """Backend identifier (e.g., 'transformers', 'llamacpp')."""
        ...

    @abstractmethod
    def load(self, record, allocation) -> LoadedModel:
        """
        Load a model into memory.

        Args:
            record: ModelRecord from the registry
            allocation: AllocationPlan from the GPU allocator

        Returns:
            LoadedModel ready for inference
        """
        ...

    @abstractmethod
    def generate(
        self,
        loaded: LoadedModel,
        messages: List[dict],
        params: GenerationParams,
    ) -> GenerationResult:
        """
        Generate a complete response (non-streaming).

        Args:
            loaded: The loaded model
            messages: Chat messages [{"role": "user", "content": "..."}]
            params: Generation parameters

        Returns:
            GenerationResult with text and token counts
        """
        ...

    @abstractmethod
    def stream(
        self,
        loaded: LoadedModel,
        messages: List[dict],
        params: GenerationParams,
        cancel: Optional[threading.Event] = None,
    ) -> Iterator[str]:
        """
        Stream tokens one at a time.

        Args:
            loaded: The loaded model
            messages: Chat messages
            params: Generation parameters
            cancel: Threading event to signal cancellation

        Yields:
            Individual tokens as strings
        """
        ...

    @abstractmethod
    def unload(self, loaded: LoadedModel) -> None:
        """
        Unload a model and free its resources.

        Args:
            loaded: The model to unload
        """
        ...


# =============================================================================
# Embedding Backend
# =============================================================================

class EmbeddingBackend(ABC):
    """ABC for embedding models (sentence-transformers, AutoModel + pooling)."""

    @property
    @abstractmethod
    def name(self) -> str: ...

    @abstractmethod
    def load(self, record, allocation) -> LoadedModel: ...

    @abstractmethod
    def embed(
        self,
        loaded: LoadedModel,
        texts: List[str],
        params: EmbeddingParams,
    ) -> EmbeddingResult:
        """Generate embeddings for a list of texts."""
        ...

    @abstractmethod
    def unload(self, loaded: LoadedModel) -> None: ...


# =============================================================================
# Vision-Language Backend
# =============================================================================

class VisionLanguageBackend(ABC):
    """ABC for VLMs that accept images + text and produce text."""

    @property
    @abstractmethod
    def name(self) -> str: ...

    @abstractmethod
    def load(self, record, allocation) -> LoadedModel: ...

    @abstractmethod
    def generate(
        self,
        loaded: LoadedModel,
        messages: List[dict],
        params: GenerationParams,
    ) -> GenerationResult:
        """Generate a response from multimodal messages (text + images)."""
        ...

    @abstractmethod
    def stream(
        self,
        loaded: LoadedModel,
        messages: List[dict],
        params: GenerationParams,
        cancel: Optional[threading.Event] = None,
    ) -> Iterator[str]:
        """Stream tokens from multimodal messages."""
        ...

    @abstractmethod
    def unload(self, loaded: LoadedModel) -> None: ...


# =============================================================================
# ASR Backend
# =============================================================================

class ASRBackend(ABC):
    """ABC for automatic speech recognition (Whisper, etc.)."""

    @property
    @abstractmethod
    def name(self) -> str: ...

    @abstractmethod
    def load(self, record, allocation) -> LoadedModel: ...

    @abstractmethod
    def transcribe(
        self,
        loaded: LoadedModel,
        audio_bytes: bytes,
        params: TranscriptionParams,
    ) -> TranscriptionResult:
        """Transcribe audio bytes to text."""
        ...

    @abstractmethod
    def unload(self, loaded: LoadedModel) -> None: ...


# =============================================================================
# Diffusion Backend
# =============================================================================

class DiffusionBackend(ABC):
    """ABC for image generation (Stable Diffusion, Flux, etc.)."""

    @property
    @abstractmethod
    def name(self) -> str: ...

    @abstractmethod
    def load(self, record, allocation) -> LoadedModel: ...

    @abstractmethod
    def generate_image(
        self,
        loaded: LoadedModel,
        prompt: str,
        params: ImageGenerationParams,
    ) -> ImageGenerationResult:
        """Generate images from a text prompt."""
        ...

    @abstractmethod
    def unload(self, loaded: LoadedModel) -> None: ...


# =============================================================================
# Image-to-Image Backend
# =============================================================================

class Img2ImgBackend(ABC):
    """ABC for image-to-image models (img2img, inpainting, ControlNet)."""

    @property
    @abstractmethod
    def name(self) -> str: ...

    @abstractmethod
    def load(self, record, allocation) -> LoadedModel: ...

    @abstractmethod
    def edit_image(
        self,
        loaded: LoadedModel,
        prompt: str,
        params: Img2ImgParams,
    ) -> Img2ImgResult:
        """Edit an image given a text prompt and input image."""
        ...

    @abstractmethod
    def unload(self, loaded: LoadedModel) -> None: ...


# =============================================================================
# Video Generation Backend
# =============================================================================

class VideoGenerationBackend(ABC):
    """ABC for text-to-video models (CogVideoX, AnimateDiff, etc.)."""

    @property
    @abstractmethod
    def name(self) -> str: ...

    @abstractmethod
    def load(self, record, allocation) -> LoadedModel: ...

    @abstractmethod
    def generate_video(
        self,
        loaded: LoadedModel,
        prompt: str,
        params: VideoGenerationParams,
    ) -> VideoGenerationResult:
        """Generate a video from a text prompt."""
        ...

    @abstractmethod
    def unload(self, loaded: LoadedModel) -> None: ...


# =============================================================================
# TTS Backend
# =============================================================================

class TTSBackend(ABC):
    """ABC for text-to-speech (Bark, XTTS, etc.)."""

    @property
    @abstractmethod
    def name(self) -> str: ...

    @abstractmethod
    def load(self, record, allocation) -> LoadedModel: ...

    @abstractmethod
    def synthesize(
        self,
        loaded: LoadedModel,
        text: str,
        params: TTSParams,
    ) -> TTSResult:
        """Synthesize speech from text."""
        ...

    @abstractmethod
    def unload(self, loaded: LoadedModel) -> None: ...


# =============================================================================
# Rerank Backend
# =============================================================================

class RerankBackend(ABC):
    """ABC for reranking models (CrossEncoder, etc.)."""

    @property
    @abstractmethod
    def name(self) -> str: ...

    @abstractmethod
    def load(self, record, allocation) -> LoadedModel: ...

    @abstractmethod
    def rerank(
        self,
        loaded: LoadedModel,
        query: str,
        documents: List[str],
        params: RerankParams,
    ) -> RerankResult:
        """Score and rank documents against a query."""
        ...

    @abstractmethod
    def unload(self, loaded: LoadedModel) -> None: ...


# =============================================================================
# Seq2Seq Backend
# =============================================================================

class Seq2SeqBackend(ABC):
    """ABC for seq2seq models (T5, mBART, NLLB, MarianMT)."""

    @property
    @abstractmethod
    def name(self) -> str: ...

    @abstractmethod
    def load(self, record, allocation) -> LoadedModel: ...

    @abstractmethod
    def generate(
        self,
        loaded: LoadedModel,
        text: str,
        params: Seq2SeqParams,
    ) -> Seq2SeqResult:
        """Generate text from input text (translate, summarize, etc.)."""
        ...

    @abstractmethod
    def unload(self, loaded: LoadedModel) -> None: ...


# =============================================================================
# Classification Backend
# =============================================================================

class ClassificationBackendABC(ABC):
    """ABC for pipeline-based models (classification, detection, segmentation, etc.)."""

    @property
    @abstractmethod
    def name(self) -> str: ...

    @abstractmethod
    def load(self, record, allocation) -> LoadedModel: ...

    @abstractmethod
    def classify(
        self,
        loaded: LoadedModel,
        text: str,
        params: ClassificationParams,
    ) -> ClassificationResult:
        """Classify input (text, image, or audio based on params)."""
        ...

    @abstractmethod
    def detect_objects(self, loaded: LoadedModel, params: ObjectDetectionParams) -> ObjectDetectionResult:
        """Detect objects in an image."""
        ...

    @abstractmethod
    def segment_image(self, loaded: LoadedModel, params: ImageSegmentationParams) -> ImageSegmentationResult:
        """Segment an image into labeled regions."""
        ...

    @abstractmethod
    def estimate_depth(self, loaded: LoadedModel, params: DepthEstimationParams) -> DepthEstimationResult:
        """Estimate depth from an image."""
        ...

    @abstractmethod
    def answer_document(self, loaded: LoadedModel, params: DocumentQAParams) -> DocumentQAResult:
        """Answer a question about a document image."""
        ...

    @abstractmethod
    def process_audio(self, loaded: LoadedModel, params: AudioProcessingParams) -> AudioProcessingResult:
        """Process audio (enhancement, conversion, etc.)."""
        ...

    @abstractmethod
    def unload(self, loaded: LoadedModel) -> None: ...
