"""
Embedding Backend
-----------------
Handles embedding models (sentence-transformers, AutoModel + pooling).

Uses sentence-transformers when available for simplicity,
falls back to AutoModel + manual mean pooling otherwise.
"""

import logging
from typing import List

import torch

from inferall.backends.base import (
    EmbeddingBackend,
    EmbeddingParams,
    EmbeddingResult,
    LoadedModel,
)
from inferall.gpu.allocator import AllocationPlan
from inferall.registry.metadata import ModelRecord

logger = logging.getLogger(__name__)


class SentenceTransformersBackend(EmbeddingBackend):
    """Embedding backend using sentence-transformers or AutoModel with mean pooling."""

    @property
    def name(self) -> str:
        return "embedding"

    # -------------------------------------------------------------------------
    # Load
    # -------------------------------------------------------------------------

    def load(self, record: ModelRecord, allocation: AllocationPlan) -> LoadedModel:
        """Load an embedding model."""
        model_path = str(record.local_path)
        trust = record.trust_remote_code

        logger.info("Loading embedding model %s", record.model_id)

        # Try sentence-transformers first
        try:
            from sentence_transformers import SentenceTransformer

            device = self._resolve_device(allocation)
            model = SentenceTransformer(
                model_path,
                device=device,
                trust_remote_code=trust,
            )
            logger.info("Loaded %s via sentence-transformers on %s", record.model_id, device)

            return LoadedModel(
                model_id=record.model_id,
                backend_name=self.name,
                model=model,
                tokenizer=None,  # SentenceTransformer handles tokenization
                vram_used_bytes=allocation.estimated_vram_bytes,
            )
        except ImportError:
            logger.info("sentence-transformers not installed, using AutoModel fallback")

        # Fallback: AutoModel + AutoTokenizer
        from transformers import AutoModel, AutoTokenizer

        tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=trust)
        model = AutoModel.from_pretrained(
            model_path,
            trust_remote_code=trust,
            device_map=allocation.device_map,
            torch_dtype="auto",
        )

        logger.info("Loaded %s via AutoModel (manual pooling)", record.model_id)

        return LoadedModel(
            model_id=record.model_id,
            backend_name=self.name,
            model=model,
            tokenizer=tokenizer,
            vram_used_bytes=allocation.estimated_vram_bytes,
        )

    # -------------------------------------------------------------------------
    # Embed
    # -------------------------------------------------------------------------

    def embed(
        self,
        loaded: LoadedModel,
        texts: List[str],
        params: EmbeddingParams,
    ) -> EmbeddingResult:
        """Generate embeddings for a list of texts."""
        loaded.touch()

        # SentenceTransformer path
        if loaded.tokenizer is None:
            embeddings = loaded.model.encode(
                texts,
                normalize_embeddings=params.normalize,
                convert_to_numpy=True,
            )
            prompt_tokens = sum(
                len(loaded.model.tokenize([t])["input_ids"][0]) for t in texts
            )
            return EmbeddingResult(
                embeddings=embeddings.tolist(),
                prompt_tokens=prompt_tokens,
                model=loaded.model_id,
            )

        # AutoModel fallback path
        return self._embed_with_automodel(loaded, texts, params)

    def _embed_with_automodel(
        self,
        loaded: LoadedModel,
        texts: List[str],
        params: EmbeddingParams,
    ) -> EmbeddingResult:
        """Embed using AutoModel + mean pooling."""
        encoded = loaded.tokenizer(
            texts,
            padding=True,
            truncation=params.truncate,
            return_tensors="pt",
        )

        device = next(loaded.model.parameters()).device
        encoded = {k: v.to(device) for k, v in encoded.items()}
        prompt_tokens = encoded["input_ids"].numel()

        with torch.inference_mode():
            outputs = loaded.model(**encoded)

        # Mean pooling
        token_embeddings = outputs.last_hidden_state
        attention_mask = encoded["attention_mask"].unsqueeze(-1).expand(
            token_embeddings.size()
        ).float()
        sum_embeddings = torch.sum(token_embeddings * attention_mask, dim=1)
        sum_mask = torch.clamp(attention_mask.sum(dim=1), min=1e-9)
        embeddings = sum_embeddings / sum_mask

        if params.normalize:
            embeddings = torch.nn.functional.normalize(embeddings, p=2, dim=1)

        return EmbeddingResult(
            embeddings=embeddings.cpu().tolist(),
            prompt_tokens=prompt_tokens,
            model=loaded.model_id,
        )

    # -------------------------------------------------------------------------
    # Unload
    # -------------------------------------------------------------------------

    def unload(self, loaded: LoadedModel) -> None:
        """Unload embedding model and free resources."""
        logger.info("Unloading embedding model %s", loaded.model_id)

        del loaded.model
        del loaded.tokenizer
        loaded.model = None
        loaded.tokenizer = None

        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    # -------------------------------------------------------------------------
    # Helpers
    # -------------------------------------------------------------------------

    def _resolve_device(self, allocation: AllocationPlan) -> str:
        """Determine the device string for SentenceTransformer."""
        if allocation.gpu_ids:
            return f"cuda:{allocation.gpu_ids[0]}"
        return "cpu"
