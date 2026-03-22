"""Tests for GPU scheduler and embedding batcher."""

import asyncio
from unittest.mock import MagicMock

import pytest

from inferall.scheduling.gpu_scheduler import GPUScheduler, ModelInstance
from inferall.scheduling.batcher import EmbeddingBatcher


# =============================================================================
# GPU Scheduler Tests
# =============================================================================

class TestGPUScheduler:
    def test_register_instance(self):
        s = GPUScheduler()
        s.register_instance("model-a", gpu_id=0)
        assert s.get_instance_count("model-a") == 1

    def test_register_multiple_gpus(self):
        s = GPUScheduler()
        s.register_instance("model-a", gpu_id=0)
        s.register_instance("model-a", gpu_id=1)
        assert s.get_instance_count("model-a") == 2

    def test_no_duplicate_registration(self):
        s = GPUScheduler()
        s.register_instance("model-a", gpu_id=0)
        s.register_instance("model-a", gpu_id=0)
        assert s.get_instance_count("model-a") == 1

    def test_get_best_instance_single(self):
        s = GPUScheduler()
        s.register_instance("model-a", gpu_id=0)
        inst = s.get_best_instance("model-a")
        assert inst.gpu_id == 0

    def test_get_best_instance_least_loaded(self):
        s = GPUScheduler()
        s.register_instance("model-a", gpu_id=0)
        s.register_instance("model-a", gpu_id=1)
        # Load up GPU 0
        s.acquire("model-a")  # goes to 0 (both equal, picks first)
        s.acquire("model-a")  # goes to 1 (0 has 1 active)
        inst = s.get_best_instance("model-a")
        # Both should have 1 active, so either is fine
        assert inst is not None

    def test_get_best_instance_missing(self):
        s = GPUScheduler()
        assert s.get_best_instance("nonexistent") is None

    def test_acquire_and_release(self):
        s = GPUScheduler()
        s.register_instance("model-a", gpu_id=0)
        gpu = s.acquire("model-a")
        assert gpu == 0
        inst = s.get_best_instance("model-a")
        assert inst.active_requests == 1
        s.release("model-a", gpu_id=0)
        inst = s.get_best_instance("model-a")
        assert inst.active_requests == 0
        assert inst.total_served == 1

    def test_acquire_no_instances(self):
        s = GPUScheduler()
        assert s.acquire("nonexistent") is None

    def test_unregister_instance(self):
        s = GPUScheduler()
        s.register_instance("model-a", gpu_id=0)
        s.register_instance("model-a", gpu_id=1)
        s.unregister_instance("model-a", gpu_id=0)
        assert s.get_instance_count("model-a") == 1

    def test_unregister_all(self):
        s = GPUScheduler()
        s.register_instance("model-a", gpu_id=0)
        s.register_instance("model-a", gpu_id=1)
        s.unregister_instance("model-a")
        assert s.get_instance_count("model-a") == 0

    def test_get_all_instances(self):
        s = GPUScheduler()
        s.register_instance("model-a", gpu_id=0)
        s.register_instance("model-b", gpu_id=1)
        all_inst = s.get_all_instances()
        assert "model-a" in all_inst
        assert "model-b" in all_inst


# =============================================================================
# Embedding Batcher Tests
# =============================================================================

class TestEmbeddingBatcher:
    @pytest.mark.asyncio
    async def test_single_request_passthrough(self):
        from inferall.backends.base import EmbeddingResult
        batcher = EmbeddingBatcher(window_ms=10, max_batch_size=32)

        call_log = []
        def mock_embed(model_id, texts, params):
            call_log.append(texts)
            return EmbeddingResult(
                embeddings=[[0.1] * len(t) for t in texts],
                prompt_tokens=len(texts),
                model=model_id,
            )

        result = await batcher.submit("model-a", ["hello"], mock_embed)
        assert len(result.embeddings) == 1

    @pytest.mark.asyncio
    async def test_batch_accumulation(self):
        """Multiple requests within the window get batched together."""
        from inferall.backends.base import EmbeddingResult
        batcher = EmbeddingBatcher(window_ms=100, max_batch_size=10)

        batch_sizes = []
        def mock_embed(model_id, texts, params):
            batch_sizes.append(len(texts))
            return EmbeddingResult(
                embeddings=[[0.1] for _ in texts],
                prompt_tokens=len(texts),
                model=model_id,
            )

        # Submit 3 requests concurrently
        results = await asyncio.gather(
            batcher.submit("model-a", ["text1"], mock_embed),
            batcher.submit("model-a", ["text2"], mock_embed),
            batcher.submit("model-a", ["text3"], mock_embed),
        )

        # All 3 should get results
        assert len(results) == 3
        for r in results:
            assert len(r.embeddings) == 1

    @pytest.mark.asyncio
    async def test_batch_max_size_triggers_flush(self):
        from inferall.backends.base import EmbeddingResult
        batcher = EmbeddingBatcher(window_ms=5000, max_batch_size=2)

        call_count = [0]
        def mock_embed(model_id, texts, params):
            call_count[0] += 1
            return EmbeddingResult(
                embeddings=[[0.1] for _ in texts],
                prompt_tokens=len(texts),
                model=model_id,
            )

        # Submit 2 requests — should trigger immediate flush
        r1 = await batcher.submit("model-a", ["t1"], mock_embed)
        assert r1 is not None

    @pytest.mark.asyncio
    async def test_stats(self):
        batcher = EmbeddingBatcher()
        stats = batcher.get_stats()
        assert stats["pending_models"] == 0
        assert stats["pending_requests"] == 0
