"""
Model Dispatcher
-----------------
Per-model request queuing with priority support.

Each model gets its own semaphore and queue. Model A being busy doesn't
block Model B. A global semaphore prevents total thread exhaustion.

This replaces the single ThreadPoolExecutor + Semaphore pattern.
"""

import asyncio
import logging
import time
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, Optional

logger = logging.getLogger(__name__)


@dataclass
class ModelQueueStats:
    """Real-time stats for a model's request queue."""

    model_id: str
    pending: int = 0
    active: int = 0
    total_served: int = 0
    total_errors: int = 0
    avg_latency_ms: float = 0.0


class ModelQueue:
    """Per-model request queue with its own concurrency semaphore."""

    def __init__(self, model_id: str, max_concurrency: int = 1, max_queue_size: int = 64):
        self.model_id = model_id
        self.semaphore = asyncio.Semaphore(max_concurrency)
        self.queue_size = max_queue_size
        self.pending = 0
        self.active = 0
        self.total_served = 0
        self.total_errors = 0
        self._latency_sum = 0.0
        self._latency_count = 0

    @property
    def avg_latency_ms(self) -> float:
        return self._latency_sum / self._latency_count if self._latency_count > 0 else 0.0

    def is_full(self) -> bool:
        return self.pending >= self.queue_size

    def record_completion(self, latency_ms: float, error: bool = False) -> None:
        self.total_served += 1
        self._latency_sum += latency_ms
        self._latency_count += 1
        if error:
            self.total_errors += 1

    def stats(self) -> ModelQueueStats:
        return ModelQueueStats(
            model_id=self.model_id,
            pending=self.pending,
            active=self.active,
            total_served=self.total_served,
            total_errors=self.total_errors,
            avg_latency_ms=self.avg_latency_ms,
        )


class ModelDispatcher:
    """
    Routes requests to per-model queues.

    Each model gets its own concurrency semaphore, so Model A being busy
    doesn't block Model B. A global semaphore caps total parallelism.
    """

    def __init__(
        self,
        max_workers: int = 4,
        max_concurrent: int = 16,
        concurrency_per_model: int = 1,
        model_queue_size: int = 64,
        admission_timeout: float = 30.0,
    ):
        self.pool = ThreadPoolExecutor(
            max_workers=max_workers,
            thread_name_prefix="inference",
        )
        self.global_semaphore = asyncio.Semaphore(max_concurrent)
        self.concurrency_per_model = concurrency_per_model
        self.model_queue_size = model_queue_size
        self.admission_timeout = admission_timeout
        self._queues: Dict[str, ModelQueue] = {}
        self._lock = asyncio.Lock()

    async def _get_queue(self, model_id: str) -> ModelQueue:
        """Get or create a per-model queue."""
        if model_id not in self._queues:
            async with self._lock:
                if model_id not in self._queues:
                    self._queues[model_id] = ModelQueue(
                        model_id=model_id,
                        max_concurrency=self.concurrency_per_model,
                        max_queue_size=self.model_queue_size,
                    )
        return self._queues[model_id]

    async def submit(
        self,
        model_id: str,
        fn: Callable,
        *args,
        priority: int = 0,
        timeout: Optional[float] = None,
    ) -> Any:
        """
        Submit a request for a specific model.

        Acquires both the model-specific semaphore and the global semaphore,
        then runs the function in the thread pool.

        Returns the function's result.
        Raises asyncio.TimeoutError if admission times out.
        Raises RuntimeError if the model's queue is full.
        """
        timeout = timeout or self.admission_timeout
        queue = await self._get_queue(model_id)

        # Check queue capacity
        if queue.is_full():
            raise RuntimeError(
                f"Model '{model_id}' queue is full ({queue.queue_size} pending). "
                "Try again later."
            )

        queue.pending += 1
        t0 = time.perf_counter()

        try:
            # Acquire both semaphores (model-specific first, then global)
            try:
                await asyncio.wait_for(
                    queue.semaphore.acquire(), timeout=timeout,
                )
            except asyncio.TimeoutError:
                raise asyncio.TimeoutError(
                    f"Timed out waiting for model '{model_id}' "
                    f"({queue.active} active, {queue.pending} pending)"
                )

            try:
                await asyncio.wait_for(
                    self.global_semaphore.acquire(), timeout=timeout,
                )
            except asyncio.TimeoutError:
                queue.semaphore.release()
                raise asyncio.TimeoutError(
                    "Timed out waiting for global inference slot"
                )

            queue.pending -= 1
            queue.active += 1

            # Execute in thread pool
            loop = asyncio.get_event_loop()
            try:
                result = await loop.run_in_executor(self.pool, fn, *args)
                elapsed_ms = (time.perf_counter() - t0) * 1000.0
                queue.record_completion(elapsed_ms)
                return result
            except Exception as e:
                elapsed_ms = (time.perf_counter() - t0) * 1000.0
                queue.record_completion(elapsed_ms, error=True)
                raise
            finally:
                queue.active -= 1
                self.global_semaphore.release()
                queue.semaphore.release()

        except (asyncio.TimeoutError, RuntimeError):
            queue.pending -= 1
            raise

    def get_stats(self) -> Dict[str, ModelQueueStats]:
        """Get stats for all model queues."""
        return {mid: q.stats() for mid, q in self._queues.items()}

    def shutdown(self) -> None:
        """Shutdown the thread pool."""
        self.pool.shutdown(wait=False)
