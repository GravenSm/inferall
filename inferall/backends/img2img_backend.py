"""
Image-to-Image Backend
-----------------------
Handles img2img, inpainting, and ControlNet-style models.

Uses HuggingFace diffusers AutoPipelineForImage2Image which auto-detects
the correct pipeline class from the model.
"""

import base64
import logging
from io import BytesIO
from typing import Optional, Tuple

import torch

from inferall.backends.base import (
    Img2ImgBackend,
    Img2ImgParams,
    Img2ImgResult,
    LoadedModel,
)
from inferall.gpu.allocator import AllocationPlan
from inferall.registry.metadata import ModelRecord

logger = logging.getLogger(__name__)


class Img2ImgDiffusersBackend(Img2ImgBackend):
    """Image-to-image backend using HuggingFace diffusers."""

    @property
    def name(self) -> str:
        return "img2img"

    # -------------------------------------------------------------------------
    # Load
    # -------------------------------------------------------------------------

    def load(self, record: ModelRecord, allocation: AllocationPlan) -> LoadedModel:
        """Load an img2img model."""
        model_path = str(record.local_path)

        logger.info("Loading img2img model %s", record.model_id)

        try:
            from diffusers import AutoPipelineForImage2Image
        except ImportError:
            raise RuntimeError(
                "diffusers is required for image-to-image models. "
                "Install with: pip install diffusers Pillow"
            )

        pipe = AutoPipelineForImage2Image.from_pretrained(
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

        logger.info("Loaded img2img model %s", record.model_id)

        return LoadedModel(
            model_id=record.model_id,
            backend_name=self.name,
            model=pipe,
            tokenizer=None,
            vram_used_bytes=allocation.estimated_vram_bytes,
        )

    # -------------------------------------------------------------------------
    # Edit Image
    # -------------------------------------------------------------------------

    def edit_image(
        self,
        loaded: LoadedModel,
        prompt: str,
        params: Img2ImgParams,
    ) -> Img2ImgResult:
        """Edit an image using img2img pipeline."""
        loaded.touch()

        from PIL import Image

        pipe = loaded.model

        # Decode input image from base64
        image_bytes = base64.b64decode(params.image_b64)
        input_image = Image.open(BytesIO(image_bytes)).convert("RGB")

        # Optionally resize
        if params.size:
            width, height = self._parse_size(params.size)
            input_image = input_image.resize((width, height), Image.LANCZOS)

        # Seed for reproducibility
        generator = None
        if params.seed is not None:
            device = getattr(pipe, "device", torch.device("cpu"))
            generator = torch.Generator(device=device).manual_seed(params.seed)

        # Build pipeline kwargs
        pipe_kwargs = {
            "prompt": prompt,
            "image": input_image,
            "strength": params.strength,
            "num_images_per_prompt": params.n,
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

        return Img2ImgResult(images=images_b64)

    # -------------------------------------------------------------------------
    # Unload
    # -------------------------------------------------------------------------

    def unload(self, loaded: LoadedModel) -> None:
        """Unload img2img model and free resources."""
        logger.info("Unloading img2img model %s", loaded.model_id)

        del loaded.model
        loaded.model = None

        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    # -------------------------------------------------------------------------
    # Helpers
    # -------------------------------------------------------------------------

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
