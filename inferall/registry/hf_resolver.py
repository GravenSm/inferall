"""
HuggingFace Resolver
--------------------
Handles downloading models from HuggingFace Hub and auto-detecting format.

Responsibilities:
- Format auto-detection (GGUF, GPTQ, AWQ, bitsandbytes, native transformers)
- Pipeline tag validation (advisory — block clearly incompatible, warn ambiguous)
- GGUF variant selection (default Q4_K_M, or user-specified)
- Download via huggingface_hub (snapshot_download for transformers, hf_hub_download for GGUF)
- Gated model error handling (401 → helpful message)
"""

import json
import logging
import os
import shutil
from datetime import datetime
from pathlib import Path
from typing import List, Optional, Tuple

from huggingface_hub import (
    HfApi,
    hf_hub_download,
    model_info as get_model_info,
    snapshot_download,
)
from huggingface_hub.utils import (
    GatedRepoError,
    RepositoryNotFoundError,
    RevisionNotFoundError,
    HfHubHTTPError,
)

from inferall.registry.metadata import (
    FORMAT_TO_TASK,
    PIPELINE_TAG_TO_TASK,
    ModelFormat,
    ModelRecord,
    ModelTask,
)

logger = logging.getLogger(__name__)

# Pipeline tags that warrant a warning but are allowed
_WARN_TAGS = set()

# Pipeline tags that we explicitly do NOT support (blocked unless --force)
_BLOCKED_TAGS = {
    "fill-mask",
    "token-classification",
    "question-answering",
    "image-feature-extraction",
    "video-classification",
    "table-question-answering",
}

# Default GGUF variant when multiple are available
_DEFAULT_GGUF_VARIANT = "Q4_K_M"


class UnsupportedModelError(Exception):
    """Raised when a model's pipeline tag is incompatible."""
    pass


class HFResolver:
    """
    Downloads models from HuggingFace Hub and auto-detects their format.
    """

    def __init__(self, models_dir: Path):
        self.models_dir = models_dir
        self.api = HfApi()

    def pull(
        self,
        model_id: str,
        variant: Optional[str] = None,
        trust_remote_code: bool = False,
        force: bool = False,
    ) -> ModelRecord:
        """
        Pull a model from HuggingFace Hub.

        Args:
            model_id: HuggingFace model ID (e.g., "meta-llama/Llama-3-8B-Instruct")
            variant: GGUF variant name (e.g., "Q4_K_M") or None for auto-detect
            trust_remote_code: Whether to allow custom model code
            force: Bypass pipeline tag validation

        Returns:
            ModelRecord for the successfully downloaded model

        Raises:
            UnsupportedModelError: If pipeline tag is incompatible and --force not used
            RepositoryNotFoundError: If model doesn't exist on HF
            GatedRepoError: If model is gated and user hasn't accepted terms
        """
        # Get model info from HF
        logger.info("Fetching model info for %s", model_id)
        try:
            info = get_model_info(model_id)
        except GatedRepoError:
            self._handle_gated_error(model_id)
            raise
        except RepositoryNotFoundError:
            logger.error("Model not found: %s", model_id)
            raise

        # Validate pipeline tag
        if not force:
            self._validate_pipeline_tag(info.pipeline_tag, model_id)

        # Get revision
        revision = self._get_revision(info)
        logger.info("Model %s at revision %s", model_id, revision[:12])

        # Detect format
        fmt, gguf_file = self._detect_format(model_id, info, variant)
        logger.info("Detected format: %s", fmt.value)

        # Download
        local_path = self._download(model_id, fmt, revision, gguf_file)

        # Get file size
        file_size = self._get_dir_size(local_path)

        # Try to extract param count from config
        param_count = self._extract_param_count(local_path, info)

        # Determine task from pipeline tag
        task = PIPELINE_TAG_TO_TASK.get(
            info.pipeline_tag, FORMAT_TO_TASK.get(fmt, ModelTask.CHAT)
        )

        # Build record
        record = ModelRecord(
            model_id=model_id,
            revision=revision,
            format=fmt,
            local_path=local_path,
            file_size_bytes=file_size,
            param_count=param_count,
            gguf_variant=variant if fmt == ModelFormat.GGUF else None,
            trust_remote_code=trust_remote_code,
            pipeline_tag=info.pipeline_tag,
            pulled_at=datetime.now(),
            task=task,
        )

        logger.info(
            "Pulled %s (%s, %s, %.2f GB, rev %s)",
            model_id, task.value, fmt.value, file_size / 1024**3, revision[:12],
        )
        return record

    # -------------------------------------------------------------------------
    # Pipeline Tag Validation
    # -------------------------------------------------------------------------

    def _validate_pipeline_tag(self, tag: Optional[str], model_id: str) -> None:
        """
        Validate the model's pipeline tag.

        - Known task tags (PIPELINE_TAG_TO_TASK) → allow
        - None → allow (unknown tag, default to chat)
        - text2text-generation → warn but allow
        - Clearly unsupported → block (raise UnsupportedModelError)
        - Unknown/custom → warn but allow
        """
        if tag is None or tag in PIPELINE_TAG_TO_TASK:
            return

        if tag in _WARN_TAGS:
            logger.warning(
                "Model %s has pipeline_tag '%s'. "
                "This is a seq2seq model — it may not work as expected with chat completion.",
                model_id, tag,
            )
            return

        if tag in _BLOCKED_TAGS:
            raise UnsupportedModelError(
                f"Model '{model_id}' has pipeline_tag '{tag}', which is not supported.\n"
                f"To override this check, use: inferall pull {model_id} --force"
            )

        # Unknown/custom tag — warn but allow
        logger.warning(
            "Model %s has unrecognized pipeline_tag '%s'. "
            "Proceeding anyway — many fine-tuned models have non-standard tags.",
            model_id, tag,
        )

    # -------------------------------------------------------------------------
    # Format Detection
    # -------------------------------------------------------------------------

    def _detect_format(
        self,
        model_id: str,
        info,
        variant: Optional[str],
    ) -> Tuple[ModelFormat, Optional[str]]:
        """
        Auto-detect model format.

        Priority:
        0. GGUF files ALWAYS take priority (explicit format, regardless of pipeline tag)
        1. Non-chat pipeline tags → dedicated format (EMBEDDING, ASR, etc.)
        2. HF model tags (gptq, awq)
        3. Config files (quantize_config.json, quantization_config in config.json)
        4. Default → transformers

        Returns:
            (format, gguf_filename_or_none)
        """
        siblings = info.siblings or []
        filenames = [s.rfilename for s in siblings]

        # 0. GGUF files ALWAYS take priority — a repo with .gguf is a GGUF model
        # regardless of pipeline tag (many GGUF repos have VLM/chat tags)
        if variant is not None:
            gguf_file = self._select_gguf_file(filenames, variant)
            if gguf_file:
                return ModelFormat.GGUF, gguf_file

        gguf_files = [f for f in filenames if f.endswith(".gguf")]
        if gguf_files:
            gguf_file = self._select_gguf_file(filenames, variant)
            return ModelFormat.GGUF, gguf_file

        # 1. Non-chat tasks get their dedicated format
        tag = info.pipeline_tag
        if tag in ("feature-extraction", "sentence-similarity"):
            return ModelFormat.EMBEDDING, None
        if tag in ("image-text-to-text", "visual-question-answering"):
            return ModelFormat.VISION_LANGUAGE, None
        if tag == "automatic-speech-recognition":
            return ModelFormat.ASR, None
        if tag == "text-to-image":
            return ModelFormat.DIFFUSION, None
        if tag == "image-to-image":
            return ModelFormat.IMAGE_TO_IMAGE, None
        if tag == "text-to-video":
            return ModelFormat.TEXT_TO_VIDEO, None
        if tag in ("translation", "summarization", "text2text-generation"):
            return ModelFormat.SEQ2SEQ, None
        if tag in ("image-classification", "audio-classification",
                    "zero-shot-classification", "zero-shot-image-classification",
                    "object-detection", "zero-shot-object-detection",
                    "image-segmentation", "mask-generation",
                    "depth-estimation", "document-question-answering",
                    "audio-to-audio"):
            return ModelFormat.CLASSIFICATION, None
        if tag in ("text-to-speech", "text-to-audio"):
            return ModelFormat.TTS, None
        if tag == "text-ranking":
            return ModelFormat.RERANK, None

        # 3. HF model tags
        tags = set(info.tags or [])
        if "gptq" in tags:
            return ModelFormat.GPTQ, None
        if "awq" in tags:
            return ModelFormat.AWQ, None

        # 4. Config files
        if "quantize_config.json" in filenames:
            return ModelFormat.GPTQ, None
        if "config.json" in filenames:
            quant_config = self._check_quantization_config(model_id, info)
            if quant_config == "gptq":
                return ModelFormat.GPTQ, None
            if quant_config == "awq":
                return ModelFormat.AWQ, None

        # 5. Default
        return ModelFormat.TRANSFORMERS, None

    def _check_quantization_config(self, model_id: str, info) -> Optional[str]:
        """
        Check if config.json has a quantization_config section.
        Returns 'gptq', 'awq', or None.

        Uses the HF API to read config.json without downloading the whole repo.
        """
        try:
            config_path = hf_hub_download(
                model_id, "config.json",
                revision=self._get_revision(info),
            )
            with open(config_path) as f:
                config = json.load(f)

            quant_config = config.get("quantization_config", {})
            quant_method = quant_config.get("quant_method", "").lower()

            if quant_method == "gptq":
                return "gptq"
            if quant_method == "awq":
                return "awq"
        except Exception:
            logger.debug("Could not check quantization_config for %s", model_id)

        return None

    # -------------------------------------------------------------------------
    # GGUF Variant Selection
    # -------------------------------------------------------------------------

    def _select_gguf_file(
        self,
        filenames: List[str],
        variant: Optional[str],
    ) -> Optional[str]:
        """
        Select which GGUF file to download.

        - If variant specified: find matching file
        - If not specified and multiple: pick Q4_K_M default
        - If single GGUF: use it
        """
        gguf_files = sorted([f for f in filenames if f.endswith(".gguf")])

        if not gguf_files:
            return None

        # Single file — use it
        if len(gguf_files) == 1:
            return gguf_files[0]

        # Variant specified — find match
        if variant is not None:
            variant_upper = variant.upper()
            for f in gguf_files:
                if variant_upper in f.upper():
                    return f
            # No match — log available and return first
            logger.warning(
                "GGUF variant '%s' not found. Available: %s",
                variant, ", ".join(self._extract_variant_names(gguf_files)),
            )
            return None

        # No variant specified, multiple files — pick default
        for f in gguf_files:
            if _DEFAULT_GGUF_VARIANT.upper() in f.upper():
                logger.info(
                    "Auto-selected GGUF variant: %s (default). "
                    "Available variants: %s",
                    _DEFAULT_GGUF_VARIANT,
                    ", ".join(self._extract_variant_names(gguf_files)),
                )
                return f

        # Default not found — use first file
        logger.info(
            "Default variant %s not found. Using: %s. "
            "Available: %s",
            _DEFAULT_GGUF_VARIANT, gguf_files[0],
            ", ".join(self._extract_variant_names(gguf_files)),
        )
        return gguf_files[0]

    def _extract_variant_names(self, gguf_files: List[str]) -> List[str]:
        """Extract human-readable variant names from GGUF filenames."""
        variants = []
        for f in gguf_files:
            name = Path(f).stem
            # Common patterns: model-name.Q4_K_M.gguf, model-Q4_K_M.gguf
            parts = name.split(".")
            if len(parts) > 1:
                variants.append(parts[-1])
            else:
                parts = name.split("-")
                if parts:
                    variants.append(parts[-1])
        return variants if variants else [Path(f).name for f in gguf_files]

    # -------------------------------------------------------------------------
    # Download
    # -------------------------------------------------------------------------

    def _download(
        self,
        model_id: str,
        fmt: ModelFormat,
        revision: str,
        gguf_file: Optional[str],
    ) -> Path:
        """
        Download model files to our models directory.

        - GGUF: single file download via hf_hub_download
        - Everything else: full repo via snapshot_download
        """
        # Build local path: ~/.inferall/models/<org>/<name>/
        parts = model_id.split("/")
        if len(parts) == 2:
            local_dir = self.models_dir / parts[0] / parts[1]
        else:
            local_dir = self.models_dir / model_id

        local_dir.mkdir(parents=True, exist_ok=True)

        try:
            if fmt == ModelFormat.GGUF and gguf_file:
                logger.info("Downloading GGUF file: %s", gguf_file)
                downloaded = hf_hub_download(
                    repo_id=model_id,
                    filename=gguf_file,
                    revision=revision,
                    local_dir=local_dir,
                )
                logger.info("Downloaded to: %s", downloaded)
            else:
                logger.info("Downloading full model repo: %s", model_id)
                # Ignore non-essential files (keep *.txt for tokenizer vocab/merges)
                ignore = [
                    "*.md", "*.png", "*.jpg", "*.gif",
                    "*.bin.index.json",  # We take the actual weights
                ]
                snapshot_download(
                    repo_id=model_id,
                    revision=revision,
                    local_dir=local_dir,
                    ignore_patterns=ignore,
                )
                logger.info("Downloaded to: %s", local_dir)

        except GatedRepoError:
            self._handle_gated_error(model_id)
            raise
        except Exception as e:
            logger.error("Download failed for %s: %s", model_id, e)
            raise

        return local_dir

    # -------------------------------------------------------------------------
    # Helpers
    # -------------------------------------------------------------------------

    def _get_revision(self, info) -> str:
        """Extract the commit SHA from model info."""
        return info.sha or "unknown"

    def _get_dir_size(self, path: Path) -> int:
        """Get total size of all files in a directory."""
        total = 0
        if path.is_file():
            return path.stat().st_size
        for f in path.rglob("*"):
            if f.is_file():
                total += f.stat().st_size
        return total

    def _extract_param_count(self, local_path: Path, info) -> Optional[int]:
        """
        Try to extract parameter count from model config or HF metadata.
        Returns None if not determinable.
        """
        # Try safetensors metadata or config.json
        config_path = local_path / "config.json"
        if config_path.exists():
            try:
                with open(config_path) as f:
                    config = json.load(f)
                # Some models have this directly
                if "num_parameters" in config:
                    return config["num_parameters"]
            except Exception:
                pass

        # Try HF model card metadata
        if hasattr(info, "safetensors") and info.safetensors:
            params = info.safetensors.get("total", None)
            if params:
                return params

        return None

    def _handle_gated_error(self, model_id: str):
        """Handle gated model errors with a helpful message."""
        token_path = Path.home() / ".cache" / "huggingface" / "token"
        if not token_path.exists():
            logger.error(
                "Model '%s' requires authentication. "
                "Run: inferall login",
                model_id,
            )
        else:
            logger.error(
                "Model '%s' is gated. You need to accept the license at: "
                "https://huggingface.co/%s and ensure you are logged in.",
                model_id, model_id,
            )
