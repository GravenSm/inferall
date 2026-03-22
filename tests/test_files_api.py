"""Tests for OpenAI-compatible Files API."""

import sqlite3
from pathlib import Path
from unittest.mock import MagicMock

import pytest

from inferall.registry.file_store import FileStore, validate_file


# =============================================================================
# FileStore Unit Tests
# =============================================================================

@pytest.fixture
def db_conn(tmp_path):
    """Create an in-memory SQLite connection with the files table."""
    conn = sqlite3.connect(":memory:", check_same_thread=False)
    conn.row_factory = sqlite3.Row
    conn.execute("""
        CREATE TABLE files (
            file_id TEXT PRIMARY KEY,
            filename TEXT NOT NULL,
            purpose TEXT NOT NULL,
            bytes INTEGER NOT NULL,
            created_at TEXT NOT NULL,
            status TEXT NOT NULL DEFAULT 'uploaded',
            local_path TEXT NOT NULL
        )
    """)
    return conn


@pytest.fixture
def store(db_conn):
    return FileStore(db_conn)


class TestFileStoreCreate:
    def test_create_returns_file_object(self, store):
        result = store.create("data.jsonl", "fine-tune", 1024, "/tmp/data.jsonl")
        assert result["object"] == "file"
        assert result["filename"] == "data.jsonl"
        assert result["purpose"] == "fine-tune"
        assert result["bytes"] == 1024
        assert result["id"].startswith("file-")

    def test_create_persists(self, store):
        store.create("test.txt", "assistants", 100, "/tmp/test.txt")
        files = store.list_files()
        assert len(files) == 1


class TestFileStoreGet:
    def test_get_existing(self, store):
        created = store.create("test.txt", "assistants", 100, "/tmp/test.txt")
        result = store.get(created["id"])
        assert result is not None
        assert result["id"] == created["id"]

    def test_get_nonexistent(self, store):
        assert store.get("file-nonexistent") is None


class TestFileStoreList:
    def test_list_all(self, store):
        store.create("a.jsonl", "fine-tune", 100, "/tmp/a")
        store.create("b.png", "vision", 200, "/tmp/b")
        assert len(store.list_files()) == 2

    def test_list_by_purpose(self, store):
        store.create("a.jsonl", "fine-tune", 100, "/tmp/a")
        store.create("b.png", "vision", 200, "/tmp/b")
        assert len(store.list_files(purpose="fine-tune")) == 1
        assert len(store.list_files(purpose="vision")) == 1

    def test_list_empty(self, store):
        assert store.list_files() == []


class TestFileStoreDelete:
    def test_delete_existing(self, store):
        created = store.create("test.txt", "assistants", 100, "/tmp/test.txt")
        assert store.delete(created["id"]) is True
        assert store.get(created["id"]) is None

    def test_delete_nonexistent(self, store):
        assert store.delete("file-nope") is False


class TestFileStoreLocalPath:
    def test_get_local_path(self, store):
        created = store.create("test.txt", "assistants", 100, "/tmp/test.txt")
        assert store.get_local_path(created["id"]) == "/tmp/test.txt"

    def test_get_local_path_missing(self, store):
        assert store.get_local_path("file-missing") is None


# =============================================================================
# Validation Tests
# =============================================================================

class TestValidateFile:
    def test_valid_fine_tune(self):
        content = b'{"messages": [{"role": "user", "content": "hi"}]}\n'
        assert validate_file("fine-tune", "data.jsonl", content) is None

    def test_invalid_purpose(self):
        assert validate_file("invalid", "test.txt", b"") is not None

    def test_fine_tune_wrong_extension(self):
        assert validate_file("fine-tune", "data.csv", b"") is not None

    def test_fine_tune_invalid_json(self):
        content = b'{"valid": true}\nnot json\n'
        error = validate_file("fine-tune", "data.jsonl", content)
        assert error is not None
        assert "line 2" in error

    def test_vision_valid(self):
        assert validate_file("vision", "photo.png", b"PNG data") is None

    def test_vision_wrong_extension(self):
        assert validate_file("vision", "doc.pdf", b"") is not None

    def test_assistants_any_file(self):
        assert validate_file("assistants", "anything.xyz", b"data") is None


# =============================================================================
# API Endpoint Tests
# =============================================================================

@pytest.fixture
def client(tmp_path):
    from inferall.api.server import create_app
    from inferall.orchestrator import Orchestrator
    from inferall.registry.registry import ModelRegistry
    from starlette.testclient import TestClient

    orch = MagicMock(spec=Orchestrator)
    orch.list_loaded.return_value = []
    registry = MagicMock(spec=ModelRegistry)
    registry.list_all.return_value = []

    # Real file store with temp DB
    conn = sqlite3.connect(":memory:", check_same_thread=False)
    conn.row_factory = sqlite3.Row
    conn.execute("""
        CREATE TABLE files (
            file_id TEXT PRIMARY KEY,
            filename TEXT NOT NULL,
            purpose TEXT NOT NULL,
            bytes INTEGER NOT NULL,
            created_at TEXT NOT NULL,
            status TEXT NOT NULL DEFAULT 'uploaded',
            local_path TEXT NOT NULL
        )
    """)
    file_store = FileStore(conn)

    app = create_app(
        orchestrator=orch, registry=registry, api_key=None,
        file_store=file_store, files_dir=tmp_path,
    )
    return TestClient(app)


class TestUploadEndpoint:
    def test_upload_success(self, client):
        resp = client.post(
            "/v1/files",
            data={"purpose": "assistants"},
            files={"file": ("test.txt", b"Hello world", "text/plain")},
        )
        assert resp.status_code == 200
        data = resp.json()
        assert data["object"] == "file"
        assert data["filename"] == "test.txt"
        assert data["bytes"] == 11

    def test_upload_invalid_purpose(self, client):
        resp = client.post(
            "/v1/files",
            data={"purpose": "invalid"},
            files={"file": ("test.txt", b"data", "text/plain")},
        )
        assert resp.status_code == 400


class TestListEndpoint:
    def test_list_empty(self, client):
        resp = client.get("/v1/files")
        assert resp.status_code == 200
        assert resp.json()["data"] == []

    def test_list_after_upload(self, client):
        client.post(
            "/v1/files",
            data={"purpose": "assistants"},
            files={"file": ("a.txt", b"data", "text/plain")},
        )
        resp = client.get("/v1/files")
        assert len(resp.json()["data"]) == 1


class TestRetrieveEndpoint:
    def test_not_found(self, client):
        resp = client.get("/v1/files/file-nonexistent")
        assert resp.status_code == 404


class TestDeleteEndpoint:
    def test_delete_success(self, client):
        upload = client.post(
            "/v1/files",
            data={"purpose": "assistants"},
            files={"file": ("test.txt", b"data", "text/plain")},
        ).json()
        resp = client.request("DELETE", f"/v1/files/{upload['id']}")
        assert resp.status_code == 200
        assert resp.json()["deleted"] is True


class TestDownloadEndpoint:
    def test_download_content(self, client):
        upload = client.post(
            "/v1/files",
            data={"purpose": "assistants"},
            files={"file": ("test.txt", b"Hello world", "text/plain")},
        ).json()
        resp = client.get(f"/v1/files/{upload['id']}/content")
        assert resp.status_code == 200
        assert resp.content == b"Hello world"

    def test_download_not_found(self, client):
        resp = client.get("/v1/files/file-nonexistent/content")
        assert resp.status_code == 404
