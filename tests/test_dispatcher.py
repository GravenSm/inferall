"""Tests for per-model request dispatcher."""

import asyncio
import time
from unittest.mock import MagicMock

import pytest

from inferall.scheduling.dispatcher import ModelDispatcher, ModelQueue, ModelQueueStats


class TestModelQueue:
    def test_initial_state(self):
        q = ModelQueue("test/model", max_concurrency=2, max_queue_size=10)
        assert q.model_id == "test/model"
        assert q.pending == 0
        assert q.active == 0
        assert q.total_served == 0
        assert not q.is_full()

    def test_is_full(self):
        q = ModelQueue("test/model", max_queue_size=2)
        q.pending = 2
        assert q.is_full()

    def test_record_completion(self):
        q = ModelQueue("test/model")
        q.record_completion(100.0)
        q.record_completion(200.0)
        assert q.total_served == 2
        assert q.avg_latency_ms == 150.0

    def test_record_error(self):
        q = ModelQueue("test/model")
        q.record_completion(100.0, error=True)
        assert q.total_errors == 1
        assert q.total_served == 1

    def test_stats(self):
        q = ModelQueue("test/model")
        q.pending = 3
        q.active = 1
        s = q.stats()
        assert isinstance(s, ModelQueueStats)
        assert s.pending == 3
        assert s.active == 1


class TestModelDispatcher:
    @pytest.fixture
    def dispatcher(self):
        d = ModelDispatcher(
            max_workers=2,
            max_concurrent=4,
            concurrency_per_model=1,
            model_queue_size=10,
            admission_timeout=5.0,
        )
        yield d
        d.shutdown()

    @pytest.mark.asyncio
    async def test_submit_sync_function(self, dispatcher):
        def add(a, b):
            return a + b
        result = await dispatcher.submit("test-model", add, 2, 3)
        assert result == 5

    @pytest.mark.asyncio
    async def test_submit_tracks_stats(self, dispatcher):
        def noop():
            return 42
        await dispatcher.submit("model-a", noop)
        await dispatcher.submit("model-a", noop)
        stats = dispatcher.get_stats()
        assert "model-a" in stats
        assert stats["model-a"].total_served == 2

    @pytest.mark.asyncio
    async def test_different_models_independent(self, dispatcher):
        """Model A and Model B should not block each other."""
        results = []

        def slow_a():
            time.sleep(0.1)
            results.append("a")
            return "a"

        def fast_b():
            results.append("b")
            return "b"

        # Submit both concurrently
        task_a = asyncio.create_task(dispatcher.submit("model-a", slow_a))
        task_b = asyncio.create_task(dispatcher.submit("model-b", fast_b))

        await asyncio.gather(task_a, task_b)

        # B should complete before A (or at least both complete)
        assert "a" in results
        assert "b" in results

    @pytest.mark.asyncio
    async def test_queue_full_raises(self, dispatcher):
        # Directly saturate the queue's pending counter
        queue = await dispatcher._get_queue("test-full")
        queue.pending = queue.queue_size  # Simulate full queue

        with pytest.raises(RuntimeError, match="queue is full"):
            await dispatcher.submit("test-full", lambda: None)

    @pytest.mark.asyncio
    async def test_exception_propagates(self, dispatcher):
        def fail():
            raise ValueError("test error")

        with pytest.raises(ValueError, match="test error"):
            await dispatcher.submit("model", fail)

    @pytest.mark.asyncio
    async def test_get_stats_empty(self, dispatcher):
        assert dispatcher.get_stats() == {}


class TestQueueStatsEndpoint:
    def test_queue_stats(self):
        from inferall.api.server import create_app
        from inferall.orchestrator import Orchestrator
        from inferall.registry.registry import ModelRegistry
        from starlette.testclient import TestClient

        orch = MagicMock(spec=Orchestrator)
        orch.list_loaded.return_value = []
        registry = MagicMock(spec=ModelRegistry)
        registry.list_all.return_value = []

        app = create_app(orchestrator=orch, registry=registry, api_key=None)
        client = TestClient(app)

        resp = client.get("/v1/queue/stats")
        assert resp.status_code == 200
        assert resp.json()["object"] == "queue.stats"
        assert "models" in resp.json()
