"""Tests for inferall.registry.registry — SQLite registry CRUD and migrations."""

import sqlite3
from datetime import datetime
from pathlib import Path

import pytest

from inferall.registry.metadata import ModelFormat, ModelRecord, ModelTask
from inferall.registry.registry import ModelRegistry


def _make_record(model_id="test/model", **kwargs):
    """Helper to create a ModelRecord with sensible defaults."""
    defaults = dict(
        model_id=model_id,
        revision="abc123def456",
        format=ModelFormat.TRANSFORMERS,
        local_path=Path("/tmp/models/test/model"),
        file_size_bytes=1_000_000_000,
        param_count=7_000_000_000,
        gguf_variant=None,
        trust_remote_code=False,
        pipeline_tag="text-generation",
        pulled_at=datetime(2026, 1, 15, 12, 0, 0),
        last_used_at=None,
        task=ModelTask.CHAT,
    )
    defaults.update(kwargs)
    return ModelRecord(**defaults)


class TestRegistryMigrations:
    """Test schema versioning and migrations."""

    def test_fresh_db_runs_all_migrations(self, tmp_path):
        db_path = tmp_path / "test.db"
        registry = ModelRegistry(db_path)
        assert registry.get_schema_version() == 5
        registry.close()

    def test_idempotent_migrate(self, tmp_path):
        db_path = tmp_path / "test.db"
        registry = ModelRegistry(db_path)
        registry.migrate()  # should be a no-op
        assert registry.get_schema_version() == 5
        registry.close()

    def test_models_table_has_task_column(self, tmp_path):
        db_path = tmp_path / "test.db"
        registry = ModelRegistry(db_path)
        cursor = registry.conn.execute("PRAGMA table_info(models)")
        columns = {row[1] for row in cursor.fetchall()}
        assert "task" in columns
        registry.close()


class TestRegistryCRUD:
    """Test register, get, list_all, remove, update_last_used."""

    @pytest.fixture
    def registry(self, tmp_path):
        reg = ModelRegistry(tmp_path / "test.db")
        yield reg
        reg.close()

    def test_register_and_get(self, registry):
        record = _make_record()
        registry.register(record)

        retrieved = registry.get("test/model")
        assert retrieved is not None
        assert retrieved.model_id == "test/model"
        assert retrieved.format == ModelFormat.TRANSFORMERS
        assert retrieved.param_count == 7_000_000_000
        assert retrieved.task == ModelTask.CHAT

    def test_get_nonexistent_returns_none(self, registry):
        assert registry.get("does/not-exist") is None

    def test_register_upsert(self, registry):
        record = _make_record(revision="rev1")
        registry.register(record)

        updated = _make_record(revision="rev2")
        registry.register(updated)

        retrieved = registry.get("test/model")
        assert retrieved.revision == "rev2"

    def test_list_all(self, registry):
        registry.register(_make_record("org/model-a"))
        registry.register(_make_record("org/model-b"))

        records = registry.list_all()
        ids = {r.model_id for r in records}
        assert ids == {"org/model-a", "org/model-b"}

    def test_list_all_empty(self, registry):
        assert registry.list_all() == []

    def test_remove(self, registry):
        registry.register(_make_record())
        assert registry.remove("test/model") is True
        assert registry.get("test/model") is None

    def test_remove_nonexistent(self, registry):
        assert registry.remove("not/here") is False

    def test_update_last_used(self, registry):
        registry.register(_make_record())
        registry.update_last_used("test/model")

        retrieved = registry.get("test/model")
        assert retrieved.last_used_at is not None


class TestModelRecordSerialization:
    """Test to_db_row / from_db_row round-trip."""

    def test_round_trip(self):
        original = _make_record(
            last_used_at=datetime(2026, 2, 1, 10, 0, 0),
            task=ModelTask.EMBEDDING,
        )
        row = original.to_db_row()
        restored = ModelRecord.from_db_row(row)

        assert restored.model_id == original.model_id
        assert restored.format == original.format
        assert restored.task == original.task
        assert restored.local_path == original.local_path
        assert restored.pulled_at == original.pulled_at
        assert restored.last_used_at == original.last_used_at

    def test_from_db_row_missing_task_defaults_to_chat(self):
        row = _make_record().to_db_row()
        row["task"] = None  # simulate v1 row without task
        record = ModelRecord.from_db_row(row)
        assert record.task == ModelTask.CHAT

    def test_trust_remote_code_bool_coercion(self):
        row = _make_record(trust_remote_code=True).to_db_row()
        assert row["trust_remote_code"] == 1
        restored = ModelRecord.from_db_row(row)
        assert restored.trust_remote_code is True
