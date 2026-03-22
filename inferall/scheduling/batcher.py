"""
Request Batcher
----------------
Accumulates requests for the same model within a time window
and fires them as a single batched call.

Currently supports embedding batching (high value, straightforward).
Text generation batching requires continuous batching (future).
"""

import asyncio
import logging
import time
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional

logger = logging.getLogger(__name__)


@dataclass
class PendingRequest:
    """A request waiting to be batched."""

    texts: List[str]
    future: asyncio.Future
    timestamp: float = field(default_factory=time.time)


class EmbeddingBatcher:
    """
    Batches embedding requests for the same model.

    Multiple concurrent /v1/embeddings requests for the same model
    are accumulated in a buffer for up to `window_ms` or until
    `max_batch_size` texts, then fired as a single embed() call.
    """

    def __init__(
        self,
        window_ms: float = 50.0,
        max_batch_size: int = 64,
    ):
        self.window_ms = window_ms
        self.max_batch_size = max_batch_size
        # model_id → list of PendingRequest
        self._buffers: Dict[str, List[PendingRequest]] = {}
        self._timers: Dict[str, asyncio.Task] = {}
        self._lock = asyncio.Lock()

    async def submit(
        self,
        model_id: str,
        texts: List[str],
        embed_fn: Callable,
    ) -> Any:
        """
        Submit texts for embedding. May be batched with other requests.

        Args:
            model_id: The model to embed with
            texts: List of texts to embed
            embed_fn: Callable(model_id, all_texts, params) -> EmbeddingResult

        Returns:
            The slice of the EmbeddingResult for this request's texts
        """
        loop = asyncio.get_event_loop()
        future = loop.create_future()
        request = PendingRequest(texts=texts, future=future)

        async with self._lock:
            if model_id not in self._buffers:
                self._buffers[model_id] = []

            self._buffers[model_id].append(request)

            # Check if batch is full
            total_texts = sum(len(r.texts) for r in self._buffers[model_id])
            if total_texts >= self.max_batch_size:
                await self._flush(model_id, embed_fn)
            elif model_id not in self._timers or self._timers[model_id].done():
                # Start timer for this model's batch window
                self._timers[model_id] = asyncio.create_task(
                    self._timer_flush(model_id, embed_fn)
                )

        return await future

    async def _timer_flush(self, model_id: str, embed_fn: Callable) -> None:
        """Wait for the batch window, then flush."""
        await asyncio.sleep(self.window_ms / 1000.0)
        async with self._lock:
            if model_id in self._buffers and self._buffers[model_id]:
                await self._flush(model_id, embed_fn)

    async def _flush(self, model_id: str, embed_fn: Callable) -> None:
        """Flush all pending requests for a model as one batch."""
        requests = self._buffers.pop(model_id, [])
        if not requests:
            return

        # Combine all texts
        all_texts = []
        offsets = []  # (start_idx, count) for each request
        for req in requests:
            offsets.append((len(all_texts), len(req.texts)))
            all_texts.extend(req.texts)

        logger.debug(
            "Batch flush %s: %d requests, %d texts",
            model_id, len(requests), len(all_texts),
        )

        try:
            # Run the batched embed call
            from inferall.backends.base import EmbeddingParams
            loop = asyncio.get_event_loop()
            result = await loop.run_in_executor(
                None, embed_fn, model_id, all_texts, EmbeddingParams(),
            )

            # Split results back to individual requests
            for req, (start, count) in zip(requests, offsets):
                from inferall.backends.base import EmbeddingResult
                slice_result = EmbeddingResult(
                    embeddings=result.embeddings[start:start + count],
                    prompt_tokens=0,  # Can't split token counts accurately
                    model=result.model,
                    total_time_ms=result.total_time_ms,
                )
                if not req.future.done():
                    req.future.set_result(slice_result)

        except Exception as e:
            # Propagate error to all waiting requests
            for req in requests:
                if not req.future.done():
                    req.future.set_exception(e)

    def get_stats(self) -> dict:
        """Get batcher stats."""
        return {
            "pending_models": len(self._buffers),
            "pending_requests": sum(
                len(reqs) for reqs in self._buffers.values()
            ),
        }
