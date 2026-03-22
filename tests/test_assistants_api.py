"""Tests for OpenAI Assistants API — assistants, threads, messages, runs."""

import sqlite3
from unittest.mock import MagicMock

import pytest

from inferall.registry.assistants_store import AssistantsStore


def _make_tables(conn):
    """Create the assistants tables for testing."""
    for sql in [
        """CREATE TABLE assistants (
            assistant_id TEXT PRIMARY KEY, name TEXT, model TEXT NOT NULL,
            instructions TEXT, tools TEXT DEFAULT '[]', file_ids TEXT DEFAULT '[]',
            metadata TEXT DEFAULT '{}', created_at TEXT NOT NULL)""",
        """CREATE TABLE threads (
            thread_id TEXT PRIMARY KEY, metadata TEXT DEFAULT '{}', created_at TEXT NOT NULL)""",
        """CREATE TABLE messages (
            message_id TEXT PRIMARY KEY, thread_id TEXT NOT NULL, role TEXT NOT NULL,
            content TEXT NOT NULL, file_ids TEXT DEFAULT '[]', assistant_id TEXT,
            run_id TEXT, metadata TEXT DEFAULT '{}', created_at TEXT NOT NULL)""",
        """CREATE TABLE runs (
            run_id TEXT PRIMARY KEY, thread_id TEXT NOT NULL, assistant_id TEXT NOT NULL,
            model TEXT NOT NULL, instructions TEXT, status TEXT DEFAULT 'queued',
            required_action TEXT, last_error TEXT, started_at TEXT, completed_at TEXT,
            cancelled_at TEXT, failed_at TEXT, metadata TEXT DEFAULT '{}', created_at TEXT NOT NULL)""",
    ]:
        conn.execute(sql)


@pytest.fixture
def store():
    conn = sqlite3.connect(":memory:", check_same_thread=False)
    conn.row_factory = sqlite3.Row
    _make_tables(conn)
    return AssistantsStore(conn)


# =============================================================================
# Assistants CRUD
# =============================================================================

class TestAssistantsCRUD:
    def test_create_and_get(self, store):
        a = store.create_assistant(model="gpt-4", name="Test", instructions="Be helpful")
        assert a["id"].startswith("asst-")
        assert a["object"] == "assistant"
        assert a["model"] == "gpt-4"
        assert a["name"] == "Test"

    def test_list(self, store):
        store.create_assistant(model="a")
        store.create_assistant(model="b")
        assert len(store.list_assistants()) == 2

    def test_update(self, store):
        a = store.create_assistant(model="gpt-4", name="Old")
        updated = store.update_assistant(a["id"], name="New")
        assert updated["name"] == "New"

    def test_delete(self, store):
        a = store.create_assistant(model="gpt-4")
        assert store.delete_assistant(a["id"]) is True
        assert store.get_assistant(a["id"]) is None

    def test_delete_nonexistent(self, store):
        assert store.delete_assistant("asst-nope") is False


# =============================================================================
# Threads CRUD
# =============================================================================

class TestThreadsCRUD:
    def test_create_and_get(self, store):
        t = store.create_thread()
        assert t["id"].startswith("thread-")
        assert t["object"] == "thread"

    def test_update(self, store):
        t = store.create_thread()
        updated = store.update_thread(t["id"], {"key": "value"})
        assert updated["metadata"]["key"] == "value"

    def test_delete_cascades(self, store):
        t = store.create_thread()
        store.create_message(t["id"], "user", "Hello")
        assert store.delete_thread(t["id"]) is True
        assert store.list_messages(t["id"]) == []


# =============================================================================
# Messages CRUD
# =============================================================================

class TestMessagesCRUD:
    def test_create_and_list(self, store):
        t = store.create_thread()
        store.create_message(t["id"], "user", "Hello")
        store.create_message(t["id"], "user", "World")
        msgs = store.list_messages(t["id"])
        assert len(msgs) == 2

    def test_message_content_format(self, store):
        t = store.create_thread()
        msg = store.create_message(t["id"], "user", "Hello")
        assert msg["content"][0]["type"] == "text"
        assert msg["content"][0]["text"]["value"] == "Hello"

    def test_get_message(self, store):
        t = store.create_thread()
        msg = store.create_message(t["id"], "user", "Test")
        retrieved = store.get_message(t["id"], msg["id"])
        assert retrieved["id"] == msg["id"]


# =============================================================================
# Runs CRUD
# =============================================================================

class TestRunsCRUD:
    def test_create_run(self, store):
        a = store.create_assistant(model="test-model", instructions="Be helpful")
        t = store.create_thread()
        run = store.create_run(t["id"], a["id"])
        assert run["id"].startswith("run-")
        assert run["status"] == "queued"
        assert run["model"] == "test-model"

    def test_create_run_invalid_assistant(self, store):
        t = store.create_thread()
        with pytest.raises(ValueError):
            store.create_run(t["id"], "asst-nonexistent")

    def test_update_status(self, store):
        a = store.create_assistant(model="m")
        t = store.create_thread()
        run = store.create_run(t["id"], a["id"])
        store.update_run_status(run["id"], "in_progress", started_at="2026-01-01T00:00:00+00:00")
        updated = store.get_run(t["id"], run["id"])
        assert updated["status"] == "in_progress"

    def test_set_error(self, store):
        a = store.create_assistant(model="m")
        t = store.create_thread()
        run = store.create_run(t["id"], a["id"])
        store.set_run_error(run["id"], "server_error", "Something broke")
        updated = store.get_run(t["id"], run["id"])
        assert updated["status"] == "failed"
        assert updated["last_error"]["code"] == "server_error"

    def test_list_runs(self, store):
        a = store.create_assistant(model="m")
        t = store.create_thread()
        store.create_run(t["id"], a["id"])
        store.create_run(t["id"], a["id"])
        assert len(store.list_runs(t["id"])) == 2


# =============================================================================
# API Endpoint Tests
# =============================================================================

@pytest.fixture
def client():
    from inferall.api.server import create_app
    from inferall.orchestrator import Orchestrator
    from inferall.registry.registry import ModelRegistry
    from starlette.testclient import TestClient

    orch = MagicMock(spec=Orchestrator)
    orch.list_loaded.return_value = []
    registry = MagicMock(spec=ModelRegistry)
    registry.list_all.return_value = []

    conn = sqlite3.connect(":memory:", check_same_thread=False)
    conn.row_factory = sqlite3.Row
    _make_tables(conn)
    conn.execute("""CREATE TABLE files (
        file_id TEXT PRIMARY KEY, filename TEXT, purpose TEXT,
        bytes INTEGER, created_at TEXT, status TEXT, local_path TEXT)""")
    astore = AssistantsStore(conn)

    app = create_app(
        orchestrator=orch, registry=registry, api_key=None,
        assistants_store=astore,
    )
    c = TestClient(app)
    c._astore = astore
    return c


class TestAssistantsEndpoints:
    def test_create_assistant(self, client):
        resp = client.post("/v1/assistants", json={"model": "gpt-4", "name": "Test"})
        assert resp.status_code == 200
        assert resp.json()["object"] == "assistant"

    def test_list_assistants(self, client):
        client.post("/v1/assistants", json={"model": "m"})
        resp = client.get("/v1/assistants")
        assert len(resp.json()["data"]) == 1

    def test_get_assistant(self, client):
        a = client.post("/v1/assistants", json={"model": "m"}).json()
        resp = client.get(f"/v1/assistants/{a['id']}")
        assert resp.status_code == 200

    def test_delete_assistant(self, client):
        a = client.post("/v1/assistants", json={"model": "m"}).json()
        resp = client.request("DELETE", f"/v1/assistants/{a['id']}")
        assert resp.json()["deleted"] is True


class TestThreadsEndpoints:
    def test_create_thread(self, client):
        resp = client.post("/v1/threads", json={})
        assert resp.status_code == 200
        assert resp.json()["object"] == "thread"

    def test_delete_thread(self, client):
        t = client.post("/v1/threads", json={}).json()
        resp = client.request("DELETE", f"/v1/threads/{t['id']}")
        assert resp.json()["deleted"] is True


class TestMessagesEndpoints:
    def test_create_and_list(self, client):
        t = client.post("/v1/threads", json={}).json()
        client.post(f"/v1/threads/{t['id']}/messages", json={"role": "user", "content": "Hi"})
        resp = client.get(f"/v1/threads/{t['id']}/messages")
        assert len(resp.json()["data"]) == 1

    def test_thread_not_found(self, client):
        resp = client.post("/v1/threads/thread-fake/messages", json={"role": "user", "content": "Hi"})
        assert resp.status_code == 404


class TestRunsEndpoints:
    def test_create_run(self, client):
        a = client.post("/v1/assistants", json={"model": "test-model"}).json()
        t = client.post("/v1/threads", json={}).json()
        resp = client.post(f"/v1/threads/{t['id']}/runs", json={"assistant_id": a["id"]})
        assert resp.status_code == 200
        assert resp.json()["status"] == "queued"

    def test_list_runs(self, client):
        a = client.post("/v1/assistants", json={"model": "m"}).json()
        t = client.post("/v1/threads", json={}).json()
        client.post(f"/v1/threads/{t['id']}/runs", json={"assistant_id": a["id"]})
        resp = client.get(f"/v1/threads/{t['id']}/runs")
        assert len(resp.json()["data"]) == 1

    def test_cancel_run(self, client):
        a = client.post("/v1/assistants", json={"model": "m"}).json()
        t = client.post("/v1/threads", json={}).json()
        run = client.post(f"/v1/threads/{t['id']}/runs", json={"assistant_id": a["id"]}).json()
        resp = client.post(f"/v1/threads/{t['id']}/runs/{run['id']}/cancel")
        # Run may already be failed (mock orchestrator) or cancelled
        assert resp.status_code in (200, 400)

    def test_invalid_assistant(self, client):
        t = client.post("/v1/threads", json={}).json()
        resp = client.post(f"/v1/threads/{t['id']}/runs", json={"assistant_id": "asst-fake"})
        assert resp.status_code == 404


class TestHealthAssistants:
    def test_capability(self, client):
        resp = client.get("/health")
        assert resp.json()["capabilities"]["assistants"] is True
