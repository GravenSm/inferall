"""
Model Metadata
--------------
ModelRecord dataclass, ModelFormat enum, and ModelTask enum.
These are the core data structures for tracking pulled models.
"""

from dataclasses import dataclass
from datetime import datetime
from enum import Enum
from pathlib import Path
from typing import Dict, Optional


class ModelTask(Enum):
    """Inference tasks supported by the engine."""

    CHAT = "chat"                          # text-generation, conversational
    EMBEDDING = "embedding"                # feature-extraction, sentence-similarity
    VISION_LANGUAGE = "vision_language"     # image-text-to-text
    ASR = "asr"                            # automatic-speech-recognition
    DIFFUSION = "diffusion"                # text-to-image
    TTS = "tts"                            # text-to-speech
    RERANK = "rerank"                      # text-ranking
    IMAGE_TO_IMAGE = "image_to_image"      # image-to-image
    TEXT_TO_VIDEO = "text_to_video"        # text-to-video
    SEQ2SEQ = "seq2seq"                    # translation, summarization, text2text
    CLASSIFICATION = "classification"      # image/audio/zero-shot classification
    OBJECT_DETECTION = "object_detection"  # object-detection
    IMAGE_SEGMENTATION = "image_segmentation"  # image-segmentation
    DEPTH_ESTIMATION = "depth_estimation"  # depth-estimation
    DOCUMENT_QA = "document_qa"            # document-question-answering
    AUDIO_TO_AUDIO = "audio_to_audio"      # audio-to-audio


class ModelFormat(Enum):
    """Supported model formats."""

    # Chat / text-generation formats
    TRANSFORMERS = "transformers"           # Native HF (fp16/fp32/bf16)
    TRANSFORMERS_BNB_4BIT = "bnb_4bit"     # bitsandbytes 4-bit
    TRANSFORMERS_BNB_8BIT = "bnb_8bit"     # bitsandbytes 8-bit
    GPTQ = "gptq"
    AWQ = "awq"
    GGUF = "gguf"
    # Multi-modal formats
    EMBEDDING = "embedding"                # sentence-transformers / AutoModel
    VISION_LANGUAGE = "vision_language"     # AutoProcessor + model
    ASR = "asr"                            # Whisper-style
    DIFFUSION = "diffusion"                # diffusers pipeline
    TTS = "tts"                            # Bark / XTTS
    RERANK = "rerank"                      # CrossEncoder / reranker
    IMAGE_TO_IMAGE = "image_to_image"      # img2img pipeline
    TEXT_TO_VIDEO = "text_to_video"        # video diffusion pipeline
    SEQ2SEQ = "seq2seq"                    # AutoModelForSeq2SeqLM
    CLASSIFICATION = "classification"      # pipeline("*-classification")
    OLLAMA_CLOUD = "ollama_cloud"          # Remote model via Ollama cloud API


# Pipeline tag → ModelTask mapping
PIPELINE_TAG_TO_TASK: Dict[str, ModelTask] = {
    "text-generation": ModelTask.CHAT,
    "conversational": ModelTask.CHAT,
    "feature-extraction": ModelTask.EMBEDDING,
    "sentence-similarity": ModelTask.EMBEDDING,
    "image-text-to-text": ModelTask.VISION_LANGUAGE,
    "visual-question-answering": ModelTask.VISION_LANGUAGE,
    "automatic-speech-recognition": ModelTask.ASR,
    "text-to-image": ModelTask.DIFFUSION,
    "text-to-speech": ModelTask.TTS,
    "text-to-audio": ModelTask.TTS,
    "text-ranking": ModelTask.RERANK,
    "image-to-image": ModelTask.IMAGE_TO_IMAGE,
    "text-to-video": ModelTask.TEXT_TO_VIDEO,
    "translation": ModelTask.SEQ2SEQ,
    "summarization": ModelTask.SEQ2SEQ,
    "text2text-generation": ModelTask.SEQ2SEQ,
    "image-classification": ModelTask.CLASSIFICATION,
    "audio-classification": ModelTask.CLASSIFICATION,
    "zero-shot-classification": ModelTask.CLASSIFICATION,
    "zero-shot-image-classification": ModelTask.CLASSIFICATION,
    "object-detection": ModelTask.OBJECT_DETECTION,
    "zero-shot-object-detection": ModelTask.OBJECT_DETECTION,
    "image-segmentation": ModelTask.IMAGE_SEGMENTATION,
    "mask-generation": ModelTask.IMAGE_SEGMENTATION,
    "depth-estimation": ModelTask.DEPTH_ESTIMATION,
    "document-question-answering": ModelTask.DOCUMENT_QA,
    "audio-to-audio": ModelTask.AUDIO_TO_AUDIO,
}

# ModelFormat → ModelTask mapping
FORMAT_TO_TASK: Dict[ModelFormat, ModelTask] = {
    ModelFormat.TRANSFORMERS: ModelTask.CHAT,
    ModelFormat.TRANSFORMERS_BNB_4BIT: ModelTask.CHAT,
    ModelFormat.TRANSFORMERS_BNB_8BIT: ModelTask.CHAT,
    ModelFormat.GPTQ: ModelTask.CHAT,
    ModelFormat.AWQ: ModelTask.CHAT,
    ModelFormat.GGUF: ModelTask.CHAT,
    ModelFormat.EMBEDDING: ModelTask.EMBEDDING,
    ModelFormat.VISION_LANGUAGE: ModelTask.VISION_LANGUAGE,
    ModelFormat.ASR: ModelTask.ASR,
    ModelFormat.DIFFUSION: ModelTask.DIFFUSION,
    ModelFormat.TTS: ModelTask.TTS,
    ModelFormat.RERANK: ModelTask.RERANK,
    ModelFormat.IMAGE_TO_IMAGE: ModelTask.IMAGE_TO_IMAGE,
    ModelFormat.TEXT_TO_VIDEO: ModelTask.TEXT_TO_VIDEO,
    ModelFormat.SEQ2SEQ: ModelTask.SEQ2SEQ,
    ModelFormat.CLASSIFICATION: ModelTask.CLASSIFICATION,
    ModelFormat.OLLAMA_CLOUD: ModelTask.CHAT,
}


@dataclass
class ModelRecord:
    """
    A pulled model tracked in the registry.

    Identity: model_id is the primary key. Only one revision per model_id
    is stored at a time — re-pulling replaces the old revision.
    """

    model_id: str                          # HuggingFace ID (e.g., "meta-llama/Llama-3-8B-Instruct")
    revision: str                          # Git commit hash from HF
    format: ModelFormat
    local_path: Path                       # Where model files live on disk
    file_size_bytes: int                   # Total size on disk
    param_count: Optional[int]             # Estimated parameter count (for VRAM estimation)
    gguf_variant: Optional[str]            # e.g., "Q4_K_M" (GGUF only)
    trust_remote_code: bool                # Whether pulled with --trust-remote-code
    pipeline_tag: Optional[str]            # From HF metadata (e.g., "text-generation")
    pulled_at: datetime
    last_used_at: Optional[datetime] = None
    task: ModelTask = ModelTask.CHAT        # Inferred from pipeline_tag

    def to_db_row(self) -> dict:
        """Convert to a dict suitable for SQLite insertion."""
        return {
            "model_id": self.model_id,
            "revision": self.revision,
            "format": self.format.value,
            "local_path": str(self.local_path),
            "file_size_bytes": self.file_size_bytes,
            "param_count": self.param_count,
            "gguf_variant": self.gguf_variant,
            "trust_remote_code": 1 if self.trust_remote_code else 0,
            "pipeline_tag": self.pipeline_tag,
            "pulled_at": self.pulled_at.isoformat(),
            "last_used_at": self.last_used_at.isoformat() if self.last_used_at else None,
            "task": self.task.value,
        }

    @classmethod
    def from_db_row(cls, row: dict) -> "ModelRecord":
        """Construct a ModelRecord from a SQLite row dict."""
        return cls(
            model_id=row["model_id"],
            revision=row["revision"],
            format=ModelFormat(row["format"]),
            local_path=Path(row["local_path"]),
            file_size_bytes=row["file_size_bytes"],
            param_count=row["param_count"],
            gguf_variant=row["gguf_variant"],
            trust_remote_code=bool(row["trust_remote_code"]),
            pipeline_tag=row["pipeline_tag"],
            pulled_at=datetime.fromisoformat(row["pulled_at"]),
            last_used_at=(
                datetime.fromisoformat(row["last_used_at"])
                if row["last_used_at"]
                else None
            ),
            task=ModelTask(row["task"]) if row.get("task") else ModelTask.CHAT,
        )
