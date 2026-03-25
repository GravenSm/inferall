"""Tests for Fine-tuning and Batch APIs."""

import json
import sqlite3
from pathlib import Path
from unittest.mock import MagicMock

import pytest

from inferall.registry.jobs_store import FineTuningStore, BatchStore


def _make_tables(conn):
    conn.execute("""CREATE TABLE fine_tuning_jobs (
        job_id TEXT PRIMARY KEY, model TEXT NOT NULL, training_file TEXT NOT NULL,
        validation_file TEXT, hyperparameters TEXT DEFAULT '{}',
        status TEXT DEFAULT 'validating', fine_tuned_model TEXT,
        trained_tokens INTEGER, error TEXT, created_at TEXT NOT NULL,
        started_at TEXT, finished_at TEXT)""")
    conn.execute("""CREATE TABLE fine_tuning_events (
        event_id TEXT PRIMARY KEY, job_id TEXT NOT NULL, type TEXT DEFAULT 'message',
        level TEXT DEFAULT 'info', message TEXT NOT NULL, data TEXT, created_at TEXT NOT NULL)""")
    conn.execute("""CREATE TABLE batches (
        batch_id TEXT PRIMARY KEY, input_file_id TEXT NOT NULL, endpoint TEXT NOT NULL,
        completion_window TEXT DEFAULT '24h', status TEXT DEFAULT 'validating',
        output_file_id TEXT, error_file_id TEXT, request_counts TEXT DEFAULT '{}',
        errors TEXT, metadata TEXT DEFAULT '{}', created_at TEXT NOT NULL,
        started_at TEXT, completed_at TEXT, cancelled_at TEXT)""")
    conn.execute("""CREATE TABLE files (
        file_id TEXT PRIMARY KEY, filename TEXT, purpose TEXT,
        bytes INTEGER, created_at TEXT, status TEXT, local_path TEXT)""")


@pytest.fixture
def conn():
    c = sqlite3.connect(":memory:", check_same_thread=False)
    c.row_factory = sqlite3.Row
    _make_tables(c)
    return c


# =============================================================================
# Fine-tuning Store Tests
# =============================================================================

class TestFineTuningStore:
    def test_create_job(self, conn):
        store = FineTuningStore(conn)
        job = store.create_job("gpt-4", "file-abc")
        assert job["id"].startswith("ftjob-")
        assert job["status"] == "validating"
        assert job["model"] == "gpt-4"

    def test_list_jobs(self, conn):
        store = FineTuningStore(conn)
        store.create_job("m1", "f1")
        store.create_job("m2", "f2")
        assert len(store.list_jobs()) == 2

    def test_cancel_job(self, conn):
        store = FineTuningStore(conn)
        job = store.create_job("m", "f")
        assert store.cancel_job(job["id"]) is True
        assert store.get_job(job["id"])["status"] == "cancelled"

    def test_set_error(self, conn):
        store = FineTuningStore(conn)
        job = store.create_job("m", "f")
        store.set_job_error(job["id"], "test_error", "Something broke")
        updated = store.get_job(job["id"])
        assert updated["status"] == "failed"
        assert updated["error"]["code"] == "test_error"

    def test_events(self, conn):
        store = FineTuningStore(conn)
        job = store.create_job("m", "f")
        store.add_event(job["id"], "info", "Test event")
        events = store.list_events(job["id"])
        assert len(events) >= 2  # creation event + test event

    def test_checkpoints_empty(self, conn):
        store = FineTuningStore(conn)
        job = store.create_job("m", "f")
        assert store.list_checkpoints(job["id"]) == []


# =============================================================================
# Batch Store Tests
# =============================================================================

class TestBatchStore:
    def test_create_batch(self, conn):
        store = BatchStore(conn)
        batch = store.create_batch("file-input", "/v1/chat/completions")
        assert batch["id"].startswith("batch-")
        assert batch["status"] == "validating"
        assert batch["endpoint"] == "/v1/chat/completions"

    def test_list_batches(self, conn):
        store = BatchStore(conn)
        store.create_batch("f1", "/v1/embeddings")
        store.create_batch("f2", "/v1/chat/completions")
        assert len(store.list_batches()) == 2

    def test_cancel_batch(self, conn):
        store = BatchStore(conn)
        batch = store.create_batch("f", "/v1/embeddings")
        assert store.cancel_batch(batch["id"]) is True
        assert store.get_batch(batch["id"])["status"] == "cancelled"

    def test_update_counts(self, conn):
        store = BatchStore(conn)
        batch = store.create_batch("f", "/v1/embeddings")
        store.update_request_counts(batch["id"], 10, 8, 2)
        updated = store.get_batch(batch["id"])
        assert updated["request_counts"] == {"total": 10, "completed": 8, "failed": 2}

    def test_set_output(self, conn):
        store = BatchStore(conn)
        batch = store.create_batch("f", "/v1/embeddings")
        store.set_batch_output(batch["id"], "file-out", "file-err")
        updated = store.get_batch(batch["id"])
        assert updated["output_file_id"] == "file-out"
        assert updated["error_file_id"] == "file-err"


# =============================================================================
# API Endpoint Tests
# =============================================================================

@pytest.fixture
def client(tmp_path):
    from inferall.api.server import create_app
    from inferall.orchestrator import Orchestrator
    from inferall.registry.registry import ModelRegistry
    from inferall.registry.file_store import FileStore
    from starlette.testclient import TestClient

    conn = sqlite3.connect(":memory:", check_same_thread=False)
    conn.row_factory = sqlite3.Row
    _make_tables(conn)

    orch = MagicMock(spec=Orchestrator)
    orch.list_loaded.return_value = []
    registry = MagicMock(spec=ModelRegistry)
    registry.list_all.return_value = []

    ft_store = FineTuningStore(conn)
    b_store = BatchStore(conn)
    f_store = FileStore(conn)

    app = create_app(
        orchestrator=orch, registry=registry, api_key=None,
        fine_tuning_store=ft_store, batch_store=b_store,
        file_store=f_store, files_dir=tmp_path,
    )
    return TestClient(app)


class TestFineTuningEndpoints:
    def test_create_job_returns_501(self, client):
        """Fine-tuning training is not implemented — returns 501 immediately."""
        resp = client.post("/v1/fine_tuning/jobs", json={
            "model": "gpt-4", "training_file": "file-abc",
        })
        assert resp.status_code == 501
        assert "not yet implemented" in resp.json()["error"]["message"]

    def test_list_jobs_empty(self, client):
        resp = client.get("/v1/fine_tuning/jobs")
        assert resp.status_code == 200
        assert resp.json()["data"] == []

    def test_get_job_not_found(self, client):
        resp = client.get("/v1/fine_tuning/jobs/ftjob-nonexistent")
        assert resp.status_code == 404


class TestBatchEndpoints:
    def _upload_file(self, client, tmp_path):
        """Helper: upload a batch JSONL file and return file_id."""
        resp = client.post(
            "/v1/files",
            data={"purpose": "batch"},
            files={"file": ("input.jsonl", b'{"custom_id":"1","url":"/v1/embeddings","body":{"model":"m","input":"hi"}}\n', "application/jsonl")},
        )
        return resp.json()["id"]

    def test_create_batch(self, client, tmp_path):
        file_id = self._upload_file(client, tmp_path)
        resp = client.post("/v1/batches", json={
            "input_file_id": file_id,
            "endpoint": "/v1/chat/completions",
        })
        assert resp.status_code == 200
        assert resp.json()["object"] == "batch"

    def test_create_batch_invalid_endpoint(self, client):
        resp = client.post("/v1/batches", json={
            "input_file_id": "f", "endpoint": "/v1/invalid",
        })
        assert resp.status_code == 400

    def test_create_batch_file_not_found(self, client):
        resp = client.post("/v1/batches", json={
            "input_file_id": "file-nonexistent", "endpoint": "/v1/embeddings",
        })
        assert resp.status_code == 404

    def test_list_batches(self, client):
        resp = client.get("/v1/batches")
        assert resp.status_code == 200

    def test_get_batch_not_found(self, client):
        resp = client.get("/v1/batches/batch-nonexistent")
        assert resp.status_code == 404


class TestHealthJobsCapabilities:
    def test_capabilities(self, client):
        resp = client.get("/health")
        caps = resp.json()["capabilities"]
        assert caps["fine_tuning"] is True
        assert caps["batches"] is True
