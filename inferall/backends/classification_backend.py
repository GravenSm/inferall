"""
Pipeline-Based Backend
-----------------------
Unified backend for pipeline()-based models: classification, object detection,
image segmentation, depth estimation, document QA, and audio processing.

Uses HuggingFace transformers pipeline() which auto-detects the correct
model class based on the pipeline_tag.
"""

import base64
import io
import logging
from io import BytesIO
from typing import List, Optional

import torch

from inferall.backends.base import (
    AudioProcessingParams,
    AudioProcessingResult,
    ClassificationBackendABC,
    ClassificationParams,
    ClassificationResult,
    DepthEstimationParams,
    DepthEstimationResult,
    DocumentQAParams,
    DocumentQAResult,
    ImageSegmentationParams,
    ImageSegmentationResult,
    LoadedModel,
    ObjectDetectionParams,
    ObjectDetectionResult,
)
from inferall.gpu.allocator import AllocationPlan
from inferall.registry.metadata import ModelRecord

logger = logging.getLogger(__name__)

# Pipeline tags that this backend handles
_SUPPORTED_TAGS = {
    "image-classification",
    "audio-classification",
    "zero-shot-classification",
    "zero-shot-image-classification",
    "object-detection",
    "zero-shot-object-detection",
    "image-segmentation",
    "mask-generation",
    "depth-estimation",
    "document-question-answering",
    "audio-to-audio",
}


class TransformersClassificationBackend(ClassificationBackendABC):
    """Unified pipeline-based backend using transformers pipeline()."""

    @property
    def name(self) -> str:
        return "classification"

    # -------------------------------------------------------------------------
    # Load
    # -------------------------------------------------------------------------

    def load(self, record: ModelRecord, allocation: AllocationPlan) -> LoadedModel:
        """Load a classification model via transformers pipeline()."""
        model_path = str(record.local_path)
        trust = record.trust_remote_code
        tag = record.pipeline_tag

        if tag not in _SUPPORTED_TAGS:
            logger.warning(
                "Unknown pipeline_tag '%s' for %s, defaulting to image-classification",
                tag, record.model_id,
            )
            tag = "image-classification"

        logger.info("Loading pipeline model %s (pipeline=%s)", record.model_id, tag)

        from transformers import pipeline

        device = self._resolve_device(allocation)

        pipe = pipeline(
            tag,
            model=model_path,
            device=device,
            trust_remote_code=trust,
        )

        logger.info("Loaded %s via pipeline('%s') on %s", record.model_id, tag, device)

        return LoadedModel(
            model_id=record.model_id,
            backend_name=self.name,
            model=pipe,
            tokenizer=tag,  # Store pipeline_tag for inference dispatch
            vram_used_bytes=allocation.estimated_vram_bytes,
        )

    # -------------------------------------------------------------------------
    # Classify
    # -------------------------------------------------------------------------

    def classify(
        self,
        loaded: LoadedModel,
        text: str,
        params: ClassificationParams,
    ) -> ClassificationResult:
        """Classify input based on the loaded pipeline type."""
        loaded.touch()

        pipe = loaded.model
        tag = loaded.tokenizer  # Pipeline tag stored during load

        if tag == "image-classification":
            return self._classify_image(pipe, params, loaded.model_id)
        elif tag == "audio-classification":
            return self._classify_audio(pipe, params, loaded.model_id)
        elif tag == "zero-shot-classification":
            return self._classify_zero_shot(pipe, text, params, loaded.model_id)
        elif tag == "zero-shot-image-classification":
            return self._classify_zero_shot_image(pipe, params, loaded.model_id)
        else:
            raise ValueError(f"Unsupported classification pipeline: {tag}")

    def _classify_image(self, pipe, params, model_id):
        """Image classification (ViT, ResNet, etc.)."""
        from PIL import Image

        if not params.image_b64:
            raise ValueError("image_b64 is required for image-classification")

        image = Image.open(io.BytesIO(base64.b64decode(params.image_b64))).convert("RGB")
        results = pipe(image, top_k=params.top_k)

        return ClassificationResult(
            labels=[{"label": r["label"], "score": float(r["score"])} for r in results],
            model=model_id,
            pipeline_tag="image-classification",
        )

    def _classify_audio(self, pipe, params, model_id):
        """Audio classification (Wav2Vec2, etc.)."""
        if not params.audio_b64:
            raise ValueError("audio_b64 is required for audio-classification")

        audio_bytes = base64.b64decode(params.audio_b64)

        # Decode to numpy array
        audio_array = self._decode_audio(audio_bytes)

        results = pipe(audio_array, top_k=params.top_k)

        return ClassificationResult(
            labels=[{"label": r["label"], "score": float(r["score"])} for r in results],
            model=model_id,
            pipeline_tag="audio-classification",
        )

    def _classify_zero_shot(self, pipe, text, params, model_id):
        """Zero-shot text classification (BART-MNLI, etc.)."""
        if not params.candidate_labels:
            raise ValueError("candidate_labels is required for zero-shot-classification")

        result = pipe(text, candidate_labels=params.candidate_labels)

        # Zero-shot returns {"labels": [...], "scores": [...]} — normalize
        labels = [
            {"label": label, "score": float(score)}
            for label, score in zip(result["labels"], result["scores"])
        ]

        if params.top_k and len(labels) > params.top_k:
            labels = labels[:params.top_k]

        return ClassificationResult(
            labels=labels,
            model=model_id,
            pipeline_tag="zero-shot-classification",
        )

    def _classify_zero_shot_image(self, pipe, params, model_id):
        """Zero-shot image classification (CLIP, etc.)."""
        from PIL import Image

        if not params.image_b64:
            raise ValueError("image_b64 is required for zero-shot-image-classification")
        if not params.candidate_labels:
            raise ValueError("candidate_labels is required for zero-shot-image-classification")

        image = Image.open(io.BytesIO(base64.b64decode(params.image_b64))).convert("RGB")
        results = pipe(image, candidate_labels=params.candidate_labels)

        return ClassificationResult(
            labels=[{"label": r["label"], "score": float(r["score"])} for r in results],
            model=model_id,
            pipeline_tag="zero-shot-image-classification",
        )

    # -------------------------------------------------------------------------
    # Object Detection
    # -------------------------------------------------------------------------

    def detect_objects(
        self, loaded: LoadedModel, params: ObjectDetectionParams,
    ) -> ObjectDetectionResult:
        """Detect objects in an image."""
        loaded.touch()
        from PIL import Image

        if not params.image_b64:
            raise ValueError("image_b64 is required for object detection")

        pipe = loaded.model
        tag = loaded.tokenizer
        image = Image.open(io.BytesIO(base64.b64decode(params.image_b64))).convert("RGB")

        if tag == "zero-shot-object-detection" and params.candidate_labels:
            results = pipe(image, candidate_labels=params.candidate_labels, threshold=params.threshold)
        else:
            results = pipe(image, threshold=params.threshold)

        detections = []
        for r in results:
            det = {"label": r["label"], "score": float(r["score"])}
            if "box" in r:
                det["box"] = {k: int(v) for k, v in r["box"].items()}
            detections.append(det)

        return ObjectDetectionResult(
            detections=detections, model=loaded.model_id, pipeline_tag=tag,
        )

    # -------------------------------------------------------------------------
    # Image Segmentation
    # -------------------------------------------------------------------------

    def segment_image(
        self, loaded: LoadedModel, params: ImageSegmentationParams,
    ) -> ImageSegmentationResult:
        """Segment an image into labeled regions."""
        loaded.touch()
        from PIL import Image

        if not params.image_b64:
            raise ValueError("image_b64 is required for image segmentation")

        pipe = loaded.model
        image = Image.open(io.BytesIO(base64.b64decode(params.image_b64))).convert("RGB")
        results = pipe(image)

        segments = []
        for r in results:
            raw_score = r.get("score")
            seg = {
                "label": r.get("label", "unknown"),
                "score": float(raw_score) if raw_score is not None else 0.0,
            }
            # Encode mask to base64 PNG if present
            mask = r.get("mask")
            if mask is not None:
                if hasattr(mask, "save"):
                    buf = BytesIO()
                    mask.save(buf, format="PNG")
                    seg["mask_b64"] = base64.b64encode(buf.getvalue()).decode()
                else:
                    import numpy as np
                    from PIL import Image as PILImage
                    mask_img = PILImage.fromarray((np.array(mask) * 255).astype("uint8"))
                    buf = BytesIO()
                    mask_img.save(buf, format="PNG")
                    seg["mask_b64"] = base64.b64encode(buf.getvalue()).decode()
            segments.append(seg)

        return ImageSegmentationResult(
            segments=segments, model=loaded.model_id, pipeline_tag=loaded.tokenizer,
        )

    # -------------------------------------------------------------------------
    # Depth Estimation
    # -------------------------------------------------------------------------

    def estimate_depth(
        self, loaded: LoadedModel, params: DepthEstimationParams,
    ) -> DepthEstimationResult:
        """Estimate depth from an image."""
        loaded.touch()
        from PIL import Image

        if not params.image_b64:
            raise ValueError("image_b64 is required for depth estimation")

        pipe = loaded.model
        image = Image.open(io.BytesIO(base64.b64decode(params.image_b64))).convert("RGB")
        result = pipe(image)

        # Pipeline returns {"depth": PIL.Image, "predicted_depth": tensor}
        depth_image = result.get("depth") if isinstance(result, dict) else result
        if hasattr(depth_image, "save"):
            buf = BytesIO()
            depth_image.save(buf, format="PNG")
            depth_b64 = base64.b64encode(buf.getvalue()).decode()
            w, h = depth_image.size
        else:
            depth_b64 = ""
            w, h = 0, 0

        return DepthEstimationResult(
            depth_map_b64=depth_b64, width=w, height=h, model=loaded.model_id,
        )

    # -------------------------------------------------------------------------
    # Document QA
    # -------------------------------------------------------------------------

    def answer_document(
        self, loaded: LoadedModel, params: DocumentQAParams,
    ) -> DocumentQAResult:
        """Answer a question about a document image."""
        loaded.touch()
        from PIL import Image

        if not params.image_b64:
            raise ValueError("image_b64 is required for document QA")
        if not params.question:
            raise ValueError("question is required for document QA")

        pipe = loaded.model
        image = Image.open(io.BytesIO(base64.b64decode(params.image_b64))).convert("RGB")
        result = pipe(image, question=params.question)

        # Pipeline may return a list or a single dict
        if isinstance(result, list):
            result = result[0] if result else {"answer": "", "score": 0.0}

        return DocumentQAResult(
            answer=result.get("answer", ""),
            score=float(result.get("score", 0.0)),
            model=loaded.model_id,
        )

    # -------------------------------------------------------------------------
    # Audio Processing
    # -------------------------------------------------------------------------

    def process_audio(
        self, loaded: LoadedModel, params: AudioProcessingParams,
    ) -> AudioProcessingResult:
        """Process audio (enhancement, conversion, etc.)."""
        loaded.touch()

        if not params.audio_b64:
            raise ValueError("audio_b64 is required for audio processing")

        pipe = loaded.model
        audio_bytes = base64.b64decode(params.audio_b64)
        audio_array = self._decode_audio(audio_bytes)

        result = pipe(audio_array)

        # Pipeline returns audio array(s) — encode back to WAV
        import numpy as np
        if isinstance(result, dict):
            output_audio = result.get("audio", result.get("sampling_rate_audio", audio_array))
            sr = result.get("sampling_rate", 16000)
        elif isinstance(result, list) and result:
            output_audio = result[0].get("audio", audio_array) if isinstance(result[0], dict) else result[0]
            sr = result[0].get("sampling_rate", 16000) if isinstance(result[0], dict) else 16000
        else:
            output_audio = audio_array
            sr = 16000

        wav_bytes = self._to_wav(np.array(output_audio, dtype=np.float32), sr)

        return AudioProcessingResult(
            audio_bytes=wav_bytes, content_type="audio/wav",
            sample_rate=sr, model=loaded.model_id,
        )

    # -------------------------------------------------------------------------
    # Unload
    # -------------------------------------------------------------------------

    def unload(self, loaded: LoadedModel) -> None:
        """Unload pipeline model and free resources."""
        logger.info("Unloading pipeline model %s", loaded.model_id)

        del loaded.model
        loaded.model = None
        loaded.tokenizer = None

        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    # -------------------------------------------------------------------------
    # Helpers
    # -------------------------------------------------------------------------

    def _resolve_device(self, allocation: AllocationPlan):
        """Determine device for pipeline() — int for GPU, -1 for CPU."""
        if allocation.gpu_ids:
            return allocation.gpu_ids[0]
        return -1

    def _decode_audio(self, audio_bytes: bytes):
        """Decode audio bytes to numpy array."""
        import numpy as np

        try:
            import soundfile as sf
            audio_array, _ = sf.read(io.BytesIO(audio_bytes))
        except Exception:
            try:
                import librosa
                audio_array, _ = librosa.load(io.BytesIO(audio_bytes), sr=None)
            except ImportError:
                raise RuntimeError(
                    "Could not decode audio. Install soundfile or librosa."
                )

        if len(audio_array.shape) > 1:
            audio_array = np.mean(audio_array, axis=1)

        return audio_array

    def _to_wav(self, audio_array, sample_rate: int) -> bytes:
        """Convert numpy audio array to WAV bytes."""
        import numpy as np

        if audio_array.dtype in (np.float32, np.float64):
            audio_array = np.clip(audio_array, -1.0, 1.0)
            audio_array = (audio_array * 32767).astype(np.int16)

        try:
            import scipy.io.wavfile as wavfile
            buf = BytesIO()
            wavfile.write(buf, sample_rate, audio_array)
            return buf.getvalue()
        except ImportError:
            raise RuntimeError("scipy is required for WAV encoding.")
