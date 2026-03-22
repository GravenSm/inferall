"""
Diffusion Backend
-----------------
Handles image generation models (Stable Diffusion, SDXL, Flux, etc.).

Uses HuggingFace diffusers library. DiffusionPipeline auto-detects
the correct pipeline class from model_index.json.
"""

import base64
import logging
from io import BytesIO
from typing import Tuple

import torch

from inferall.backends.base import (
    DiffusionBackend,
    ImageGenerationParams,
    ImageGenerationResult,
    LoadedModel,
)
from inferall.gpu.allocator import AllocationPlan
from inferall.registry.metadata import ModelRecord

logger = logging.getLogger(__name__)


class DiffusersBackend(DiffusionBackend):
    """Image generation backend using HuggingFace diffusers."""

    @property
    def name(self) -> str:
        return "diffusion"

    # -------------------------------------------------------------------------
    # Load
    # -------------------------------------------------------------------------

    def load(self, record: ModelRecord, allocation: AllocationPlan) -> LoadedModel:
        """Load a diffusion model."""
        model_path = str(record.local_path)

        logger.info("Loading diffusion model %s", record.model_id)

        try:
            from diffusers import DiffusionPipeline
        except ImportError:
            raise RuntimeError(
                "diffusers is required for image generation models. "
                "Install with: pip install diffusers Pillow"
            )

        pipe = DiffusionPipeline.from_pretrained(
            model_path,
            torch_dtype=torch.float16,
        )

        # Device placement
        if allocation.gpu_ids:
            if allocation.offload_to_cpu and hasattr(pipe, "enable_model_cpu_offload"):
                logger.info("Enabling model CPU offload for %s", record.model_id)
                pipe.enable_model_cpu_offload(gpu_id=allocation.gpu_ids[0])
            else:
                pipe = pipe.to(f"cuda:{allocation.gpu_ids[0]}")
        # else: stays on CPU

        logger.info("Loaded diffusion model %s", record.model_id)

        return LoadedModel(
            model_id=record.model_id,
            backend_name=self.name,
            model=pipe,
            tokenizer=None,
            vram_used_bytes=allocation.estimated_vram_bytes,
        )

    # -------------------------------------------------------------------------
    # Generate Image
    # -------------------------------------------------------------------------

    def generate_image(
        self,
        loaded: LoadedModel,
        prompt: str,
        params: ImageGenerationParams,
    ) -> ImageGenerationResult:
        """Generate images from a text prompt."""
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
            "num_images_per_prompt": params.n,
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

        # Convert PIL images to base64 PNG
        images_b64 = []
        for img in result.images:
            buffer = BytesIO()
            img.save(buffer, format="PNG")
            images_b64.append(base64.b64encode(buffer.getvalue()).decode("utf-8"))

        return ImageGenerationResult(images=images_b64)

    # -------------------------------------------------------------------------
    # Unload
    # -------------------------------------------------------------------------

    def unload(self, loaded: LoadedModel) -> None:
        """Unload diffusion model and free resources."""
        logger.info("Unloading diffusion model %s", loaded.model_id)

        del loaded.model
        loaded.model = None

        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    # -------------------------------------------------------------------------
    # Helpers
    # -------------------------------------------------------------------------

    def _parse_size(self, size: str) -> Tuple[int, int]:
        """Parse size string like '1024x1024' into (width, height)."""
        parts = size.lower().split("x")
        if len(parts) != 2:
            logger.warning("Invalid size '%s', using 1024x1024", size)
            return 1024, 1024
        try:
            return int(parts[0]), int(parts[1])
        except ValueError:
            logger.warning("Invalid size '%s', using 1024x1024", size)
            return 1024, 1024
