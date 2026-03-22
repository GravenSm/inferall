"""
Rerank Backend
--------------
Handles reranking / cross-encoder models.

Uses sentence-transformers CrossEncoder when available,
falls back to AutoModelForSequenceClassification + AutoTokenizer.

Models: cross-encoder/ms-marco-MiniLM-L-6-v2, BAAI/bge-reranker-v2-m3, etc.
"""

import logging
from typing import List, Optional

import torch

from inferall.backends.base import (
    LoadedModel,
    RerankBackend,
    RerankParams,
    RerankResult,
)
from inferall.gpu.allocator import AllocationPlan
from inferall.registry.metadata import ModelRecord

logger = logging.getLogger(__name__)


class CrossEncoderRerankerBackend(RerankBackend):
    """Reranking backend using CrossEncoder or AutoModelForSequenceClassification."""

    @property
    def name(self) -> str:
        return "rerank"

    # -------------------------------------------------------------------------
    # Load
    # -------------------------------------------------------------------------

    def load(self, record: ModelRecord, allocation: AllocationPlan) -> LoadedModel:
        """Load a reranking model."""
        model_path = str(record.local_path)
        trust = record.trust_remote_code

        logger.info("Loading reranker model %s", record.model_id)

        # Try CrossEncoder first (sentence-transformers)
        try:
            from sentence_transformers import CrossEncoder

            device = self._resolve_device(allocation)
            model = CrossEncoder(
                model_path,
                device=device,
                trust_remote_code=trust,
            )
            logger.info(
                "Loaded %s via CrossEncoder on %s", record.model_id, device
            )

            return LoadedModel(
                model_id=record.model_id,
                backend_name=self.name,
                model=model,
                tokenizer=None,  # CrossEncoder handles tokenization
                vram_used_bytes=allocation.estimated_vram_bytes,
            )
        except ImportError:
            logger.info(
                "sentence-transformers not installed, using AutoModel fallback"
            )

        # Fallback: AutoModelForSequenceClassification + AutoTokenizer
        from transformers import AutoModelForSequenceClassification, AutoTokenizer

        tokenizer = AutoTokenizer.from_pretrained(
            model_path, trust_remote_code=trust
        )
        model = AutoModelForSequenceClassification.from_pretrained(
            model_path,
            trust_remote_code=trust,
            device_map=allocation.device_map,
            torch_dtype="auto",
        )

        logger.info(
            "Loaded %s via AutoModelForSequenceClassification", record.model_id
        )

        return LoadedModel(
            model_id=record.model_id,
            backend_name=self.name,
            model=model,
            tokenizer=tokenizer,
            vram_used_bytes=allocation.estimated_vram_bytes,
        )

    # -------------------------------------------------------------------------
    # Rerank
    # -------------------------------------------------------------------------

    def rerank(
        self,
        loaded: LoadedModel,
        query: str,
        documents: List[str],
        params: RerankParams,
    ) -> RerankResult:
        """Score and rank documents against a query."""
        loaded.touch()

        if not documents:
            return RerankResult(results=[], model=loaded.model_id, usage={"prompt_tokens": 0})

        # CrossEncoder path
        if loaded.tokenizer is None:
            return self._rerank_cross_encoder(loaded, query, documents, params)

        # AutoModel fallback path
        return self._rerank_automodel(loaded, query, documents, params)

    def _rerank_cross_encoder(
        self,
        loaded: LoadedModel,
        query: str,
        documents: List[str],
        params: RerankParams,
    ) -> RerankResult:
        """Rerank using CrossEncoder.predict()."""
        pairs = [(query, doc) for doc in documents]

        # CrossEncoder.predict returns raw scores (numpy array)
        scores = loaded.model.predict(pairs)

        # Estimate token count
        prompt_tokens = self._estimate_tokens_cross_encoder(loaded, query, documents)

        return self._build_result(
            scores=scores.tolist() if hasattr(scores, "tolist") else list(scores),
            documents=documents,
            params=params,
            model_id=loaded.model_id,
            prompt_tokens=prompt_tokens,
        )

    def _rerank_automodel(
        self,
        loaded: LoadedModel,
        query: str,
        documents: List[str],
        params: RerankParams,
    ) -> RerankResult:
        """Rerank using AutoModelForSequenceClassification."""
        tokenizer = loaded.tokenizer

        # Tokenize all query-document pairs
        pairs = [(query, doc) for doc in documents]
        encoded = tokenizer(
            [p[0] for p in pairs],
            [p[1] for p in pairs],
            padding=True,
            truncation=True,
            max_length=params.max_length or 512,
            return_tensors="pt",
        )

        device = next(loaded.model.parameters()).device
        encoded = {k: v.to(device) for k, v in encoded.items()}
        prompt_tokens = encoded["input_ids"].numel()

        with torch.inference_mode():
            outputs = loaded.model(**encoded)

        logits = outputs.logits

        # Handle multi-label vs single-score models
        if logits.shape[-1] == 1:
            # Single score per pair — squeeze and sigmoid
            scores = torch.sigmoid(logits.squeeze(-1))
        elif logits.shape[-1] == 2:
            # Binary classification — take probability of "relevant" class
            scores = torch.softmax(logits, dim=-1)[:, 1]
        else:
            # Multi-class — take max score
            scores = logits.max(dim=-1).values

        return self._build_result(
            scores=scores.cpu().tolist(),
            documents=documents,
            params=params,
            model_id=loaded.model_id,
            prompt_tokens=prompt_tokens,
        )

    # -------------------------------------------------------------------------
    # Unload
    # -------------------------------------------------------------------------

    def unload(self, loaded: LoadedModel) -> None:
        """Unload reranker model and free resources."""
        logger.info("Unloading reranker model %s", loaded.model_id)

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
        """Determine the device string for CrossEncoder."""
        if allocation.gpu_ids:
            return f"cuda:{allocation.gpu_ids[0]}"
        return "cpu"

    def _build_result(
        self,
        scores: list,
        documents: List[str],
        params: RerankParams,
        model_id: str,
        prompt_tokens: int,
    ) -> RerankResult:
        """Build a sorted RerankResult from raw scores."""
        # Pair scores with indices
        scored = [
            {"index": i, "relevance_score": float(score)}
            for i, score in enumerate(scores)
        ]

        # Sort by relevance descending
        scored.sort(key=lambda x: x["relevance_score"], reverse=True)

        # Apply top_n
        if params.top_n is not None and params.top_n > 0:
            scored = scored[: params.top_n]

        # Optionally include document text
        if params.return_documents:
            for item in scored:
                item["document"] = {"text": documents[item["index"]]}

        return RerankResult(
            results=scored,
            model=model_id,
            usage={"prompt_tokens": prompt_tokens},
        )

    def _estimate_tokens_cross_encoder(
        self,
        loaded: LoadedModel,
        query: str,
        documents: List[str],
    ) -> int:
        """Rough token estimate for CrossEncoder (no tokenizer exposed)."""
        # ~4 chars per token heuristic
        total_chars = len(query) * len(documents) + sum(len(d) for d in documents)
        return total_chars // 4
