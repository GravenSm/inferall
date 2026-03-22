"""
Video Generation Backend
------------------------
Handles text-to-video models (CogVideoX, AnimateDiff, ModelScope, etc.).

Uses HuggingFace diffusers DiffusionPipeline which auto-detects the
correct video pipeline class from model_index.json.

Output: base64 PNG frames + optional MP4 video.
"""

import base64
import logging
from io import BytesIO
from typing import List, Optional, Tuple

import torch

from inferall.backends.base import (
    LoadedModel,
    VideoGenerationBackend,
    VideoGenerationParams,
    VideoGenerationResult,
)
from inferall.gpu.allocator import AllocationPlan
from inferall.registry.metadata import ModelRecord

logger = logging.getLogger(__name__)


class VideoDiffusersBackend(VideoGenerationBackend):
    """Text-to-video backend using HuggingFace diffusers."""

    @property
    def name(self) -> str:
        return "video"

    # -------------------------------------------------------------------------
    # Load
    # -------------------------------------------------------------------------

    def load(self, record: ModelRecord, allocation: AllocationPlan) -> LoadedModel:
        """Load a video generation model."""
        model_path = str(record.local_path)

        logger.info("Loading video model %s", record.model_id)

        try:
            from diffusers import DiffusionPipeline
        except ImportError:
            raise RuntimeError(
                "diffusers is required for video generation models. "
                "Install with: pip install diffusers"
            )

        pipe = DiffusionPipeline.from_pretrained(
            model_path,
            torch_dtype=torch.float16,
        )

        # Video models are VRAM-heavy — prefer CPU offload when available
        if allocation.gpu_ids:
            if hasattr(pipe, "enable_model_cpu_offload"):
                logger.info("Enabling model CPU offload for %s", record.model_id)
                pipe.enable_model_cpu_offload(gpu_id=allocation.gpu_ids[0])
            else:
                pipe = pipe.to(f"cuda:{allocation.gpu_ids[0]}")

        logger.info("Loaded video model %s", record.model_id)

        return LoadedModel(
            model_id=record.model_id,
            backend_name=self.name,
            model=pipe,
            tokenizer=None,
            vram_used_bytes=allocation.estimated_vram_bytes,
        )

    # -------------------------------------------------------------------------
    # Generate Video
    # -------------------------------------------------------------------------

    def generate_video(
        self,
        loaded: LoadedModel,
        prompt: str,
        params: VideoGenerationParams,
    ) -> VideoGenerationResult:
        """Generate a video from a text prompt."""
        loaded.touch()

        pipe = loaded.model
        width, height = self._parse_size(params.size)

        # Seed for reproducibility
        generator = None
        if params.seed is not None:
            device = getattr(pipe, "device", torch.device("cpu"))
            generator = torch.Generator(device=device).manual_seed(params.seed)

        # Build pipeline kwargs
        pipe_kwargs = {
            "prompt": prompt,
            "num_frames": params.num_frames,
            "width": width,
            "height": height,
            "num_inference_steps": params.num_inference_steps,
            "guidance_scale": params.guidance_scale,
        }

        if params.negative_prompt:
            pipe_kwargs["negative_prompt"] = params.negative_prompt
        if generator:
            pipe_kwargs["generator"] = generator

        # Generate
        result = pipe(**pipe_kwargs)

        # Extract frames — different pipelines return frames differently
        pil_frames = self._extract_frames(result)

        # Convert frames to base64 PNG
        frames_b64 = []
        for frame in pil_frames:
            buffer = BytesIO()
            frame.save(buffer, format="PNG")
            frames_b64.append(base64.b64encode(buffer.getvalue()).decode("utf-8"))

        # Encode MP4 if requested
        video_b64 = None
        if "mp4" in params.output_format and pil_frames:
            video_b64 = self._encode_mp4(pil_frames, params.fps)

        return VideoGenerationResult(
            frames=frames_b64,
            video_b64=video_b64,
            num_frames=len(frames_b64),
            fps=params.fps,
        )

    # -------------------------------------------------------------------------
    # Unload
    # -------------------------------------------------------------------------

    def unload(self, loaded: LoadedModel) -> None:
        """Unload video model and free resources."""
        logger.info("Unloading video model %s", loaded.model_id)

        del loaded.model
        loaded.model = None

        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    # -------------------------------------------------------------------------
    # Helpers
    # -------------------------------------------------------------------------

    def _extract_frames(self, result) -> list:
        """Extract PIL frames from pipeline output (handles various formats)."""
        # CogVideoX, AnimateDiff: result.frames is list of lists of PIL images
        if hasattr(result, "frames") and result.frames:
            frames = result.frames
            if isinstance(frames[0], list):
                return frames[0]  # First batch
            return frames

        # ModelScope, some others: result.images
        if hasattr(result, "images") and result.images:
            images = result.images
            if isinstance(images, list):
                return images

        # Fallback: try to iterate
        logger.warning("Could not extract frames from pipeline output, trying iteration")
        try:
            return list(result)
        except TypeError:
            return []

    def _parse_size(self, size: str) -> Tuple[int, int]:
        """Parse size string like '512x512' into (width, height)."""
        parts = size.lower().split("x")
        if len(parts) != 2:
            logger.warning("Invalid size '%s', using 512x512", size)
            return 512, 512
        try:
            return int(parts[0]), int(parts[1])
        except ValueError:
            logger.warning("Invalid size '%s', using 512x512", size)
            return 512, 512

    def _encode_mp4(self, frames: list, fps: int) -> Optional[str]:
        """Encode PIL frames to base64 MP4. Returns None if encoding unavailable."""
        import numpy as np

        # Try imageio first
        try:
            import imageio.v3 as iio
            frame_arrays = [np.array(f) for f in frames]
            buffer = BytesIO()
            iio.imwrite(
                buffer, frame_arrays, extension=".mp4",
                fps=fps, codec="libx264",
            )
            return base64.b64encode(buffer.getvalue()).decode("utf-8")
        except (ImportError, Exception) as e:
            logger.debug("imageio MP4 encoding failed: %s", e)

        # Try imageio legacy API
        try:
            import imageio
            frame_arrays = [np.array(f) for f in frames]
            buffer = BytesIO()
            writer = imageio.get_writer(buffer, format="mp4", fps=fps)
            for arr in frame_arrays:
                writer.append_data(arr)
            writer.close()
            return base64.b64encode(buffer.getvalue()).decode("utf-8")
        except (ImportError, Exception) as e:
            logger.debug("imageio legacy MP4 encoding failed: %s", e)

        logger.warning(
            "MP4 encoding unavailable (install imageio[ffmpeg]). "
            "Returning frames only."
        )
        return None
