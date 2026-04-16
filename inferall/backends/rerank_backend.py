"""
Rerank Backend
--------------
Handles reranking / cross-encoder models.

Three load paths are supported:
  * CrossEncoder (sentence-transformers) — standard pair-based rerankers.
  * AutoModelForSequenceClassification — plain HF classifier fallback.
  * AutoModelForCausalLM — generative rerankers (Qwen3-Reranker, etc.) that
    answer "yes" / "no" on a chat-template prompt.

Models: cross-encoder/ms-marco-MiniLM-L-6-v2, BAAI/bge-reranker-v2-m3,
Qwen/Qwen3-Reranker-8B, etc.
"""

import json
import logging
from pathlib import Path
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


def _is_generative_reranker_id(model_id: str) -> bool:
    """Static check: does this model id look like a generative reranker?

    Used as a fast-path before reading config.json.
    """
    lower = model_id.lower()
    if "reranker" not in lower:
        return False
    return any(tag in lower for tag in ("qwen", "gemma-reranker"))


def _read_architectures(local_path) -> List[str]:
    """Read `architectures` out of a model's config.json on disk.

    Returns [] when the file is absent, unreadable, malformed, or lacks an
    architectures field. Never raises.
    """
    try:
        config_path = Path(local_path) / "config.json"
    except TypeError:
        return []
    if not config_path.is_file():
        return []
    try:
        with config_path.open("r", encoding="utf-8") as f:
            cfg = json.load(f)
    except (OSError, json.JSONDecodeError):
        return []
    archs = cfg.get("architectures") or []
    return [a for a in archs if isinstance(a, str)]


def _architectures_suggest_causal_lm(archs: List[str]) -> bool:
    """True if any architecture name looks like a causal LM head class.

    Matches `*ForCausalLM` (e.g. Qwen3ForCausalLM, LlamaForCausalLM) — the
    shape `_rerank_generative` needs for yes/no token scoring. Seq2seq
    (`*ForConditionalGeneration`) is intentionally excluded: it has the
    wrong logit shape for this path.
    """
    return any(a.endswith("ForCausalLM") for a in archs)


def _detect_generative_reranker_at_load(record: ModelRecord) -> bool:
    """Load-time detection combining id heuristic with config.json architectures.

    Closes the gap where a new generative reranker family lands with an id
    that doesn't match the static heuristic: if config.json says the weights
    are a `*ForCausalLM`, we treat it as generative anyway. Falls back to
    the id check if config.json can't be read.
    """
    if _is_generative_reranker_id(record.model_id):
        return True
    return _architectures_suggest_causal_lm(_read_architectures(record.local_path))


class CrossEncoderRerankerBackend(RerankBackend):
    """Reranking backend supporting CrossEncoder, SeqClass, and generative LMs."""

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

        # Generative rerankers don't fit the CrossEncoder / SeqClass shape:
        # CrossEncoder uses plain pair tokenization (wrong for chat-template
        # models) and SeqClass creates a randomly-initialised score head on
        # top of weights that don't include one. Force-skip both. Detection
        # consults both the id heuristic and config.json's architectures
        # field (e.g. Qwen3ForCausalLM) so new generative families work
        # without an id allow-list update.
        skip_cross_encoder = _detect_generative_reranker_at_load(record)

        # Try CrossEncoder first (sentence-transformers) for standard rerankers
        try:
            if skip_cross_encoder:
                raise RuntimeError(
                    "generative reranker detected, skipping CrossEncoder"
                )

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
        except Exception as e:
            # Widened from ImportError: any CrossEncoder failure (missing dep,
            # model incompatibility, load error) falls through to transformers.
            logger.info(
                "CrossEncoder unavailable for %s (%s: %s); using transformers fallback",
                record.model_id, type(e).__name__, e,
            )

        from transformers import AutoTokenizer

        tokenizer = AutoTokenizer.from_pretrained(
            model_path, trust_remote_code=trust
        )

        if skip_cross_encoder:
            from transformers import AutoModelForCausalLM

            model = AutoModelForCausalLM.from_pretrained(
                model_path,
                trust_remote_code=trust,
                device_map=allocation.device_map,
                torch_dtype="auto",
            )
            logger.info(
                "Loaded %s via AutoModelForCausalLM (generative reranker)",
                record.model_id,
            )
        else:
            from transformers import AutoModelForSequenceClassification

            model = AutoModelForSequenceClassification.from_pretrained(
                model_path,
                trust_remote_code=trust,
                device_map=allocation.device_map,
                torch_dtype="auto",
            )
            logger.info(
                "Loaded %s via AutoModelForSequenceClassification",
                record.model_id,
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
            return RerankResult(
                results=[], model=loaded.model_id, usage={"prompt_tokens": 0}
            )

        # Empty / whitespace / None entries produce seq_len=0 tensors that
        # crash the tokenizer + attention layers (e.g. "cannot reshape tensor
        # of 0 elements into shape [N, 0, -1, 128]"). Replace with a single
        # space so tokenization always yields at least one token; original
        # docs are still echoed back via return_documents.
        sanitized = [
            doc if (doc and doc.strip()) else " "
            for doc in documents
        ]

        # CrossEncoder path
        if loaded.tokenizer is None:
            return self._rerank_cross_encoder(
                loaded, query, sanitized, params, original_documents=documents
            )

        # Generative reranker path (Qwen3-Reranker etc.)
        if self._is_generative_reranker(loaded):
            return self._rerank_generative(
                loaded, query, sanitized, params, original_documents=documents
            )

        # AutoModel seq-classification fallback
        return self._rerank_automodel(
            loaded, query, sanitized, params, original_documents=documents
        )

    def _rerank_cross_encoder(
        self,
        loaded: LoadedModel,
        query: str,
        documents: List[str],
        params: RerankParams,
        original_documents: Optional[List[str]] = None,
    ) -> RerankResult:
        """Rerank using CrossEncoder.predict()."""
        pairs = [(query, doc) for doc in documents]

        scores = loaded.model.predict(pairs)
        prompt_tokens = self._estimate_tokens_cross_encoder(loaded, query, documents)

        return self._build_result(
            scores=scores.tolist() if hasattr(scores, "tolist") else list(scores),
            documents=original_documents if original_documents is not None else documents,
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
        original_documents: Optional[List[str]] = None,
    ) -> RerankResult:
        """Rerank using AutoModelForSequenceClassification."""
        tokenizer = loaded.tokenizer

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

        if logits.shape[-1] == 1:
            scores = torch.sigmoid(logits.squeeze(-1))
        elif logits.shape[-1] == 2:
            scores = torch.softmax(logits, dim=-1)[:, 1]
        else:
            scores = logits.max(dim=-1).values

        return self._build_result(
            scores=scores.cpu().tolist(),
            documents=original_documents if original_documents is not None else documents,
            params=params,
            model_id=loaded.model_id,
            prompt_tokens=prompt_tokens,
        )

    def _rerank_generative(
        self,
        loaded: LoadedModel,
        query: str,
        documents: List[str],
        params: RerankParams,
        original_documents: Optional[List[str]] = None,
    ) -> RerankResult:
        """Rerank using a generative LM that answers yes/no to relevance.

        Designed for models like Qwen/Qwen3-Reranker-*: each document is
        formatted into a chat-template prompt, and the score is
        sigmoid(logit(yes) - logit(no)) at the last token position.
        """
        tokenizer = loaded.tokenizer
        device = next(loaded.model.parameters()).device
        max_length = params.max_length or 8192

        prefix = (
            "<|im_start|>system\n"
            "Judge whether the Document meets the requirements based on the "
            "Query and the Instruct provided. Note that the answer can only "
            'be "yes" or "no".<|im_end|>\n'
            "<|im_start|>user\n"
        )
        suffix = "<|im_end|>\n<|im_start|>assistant\n<think>\n\n</think>\n\n"
        instruction = (
            "Given a web search query, retrieve relevant passages that "
            "answer the query"
        )

        yes_id = tokenizer.convert_tokens_to_ids("yes")
        no_id = tokenizer.convert_tokens_to_ids("no")

        scores: List[float] = []
        total_tokens = 0

        for doc in documents:
            formatted = (
                prefix
                + "<Instruct>: " + instruction
                + "\n<Query>: " + query
                + "\n<Document>: " + doc
                + suffix
            )
            encoded = tokenizer(
                formatted,
                truncation=True,
                max_length=max_length,
                return_tensors="pt",
            )
            encoded = {k: v.to(device) for k, v in encoded.items()}
            total_tokens += encoded["input_ids"].numel()

            with torch.inference_mode():
                outputs = loaded.model(**encoded)

            logits = outputs.logits[0, -1, :]
            yes_score = logits[yes_id].float()
            no_score = logits[no_id].float()
            scores.append(torch.sigmoid(yes_score - no_score).item())

        return self._build_result(
            scores=scores,
            documents=original_documents if original_documents is not None else documents,
            params=params,
            model_id=loaded.model_id,
            prompt_tokens=total_tokens,
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

    def _is_generative_reranker(self, loaded: LoadedModel) -> bool:
        """Detect generative rerankers that use chat-template yes/no scoring.

        Three tiers, cheapest first:
          1. Id heuristic (Qwen*-Reranker, gemma-reranker*).
          2. `model.config.architectures` contains a `*ForCausalLM` class.
          3. Large-vocab + chat_template fallback for odd id + config.

        Called on the rerank() hot path, so `loaded.tokenizer` is always set
        here (CrossEncoder path has already returned upstream).
        """
        if _is_generative_reranker_id(loaded.model_id):
            return True

        cfg = getattr(loaded.model, "config", None)
        archs = getattr(cfg, "architectures", None) if cfg is not None else None
        if archs and _architectures_suggest_causal_lm(list(archs)):
            return True

        tok = loaded.tokenizer
        if tok is not None and getattr(tok, "chat_template", None):
            if cfg is not None and getattr(cfg, "vocab_size", 0) > 100000:
                return True
        return False

    def _build_result(
        self,
        scores: list,
        documents: List[str],
        params: RerankParams,
        model_id: str,
        prompt_tokens: int,
    ) -> RerankResult:
        """Build a sorted RerankResult from raw scores."""
        scored = [
            {"index": i, "relevance_score": float(score)}
            for i, score in enumerate(scores)
        ]

        scored.sort(key=lambda x: x["relevance_score"], reverse=True)

        if params.top_n is not None and params.top_n > 0:
            scored = scored[: params.top_n]

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
        total_chars = len(query) * len(documents) + sum(len(d) for d in documents)
        return total_chars // 4
