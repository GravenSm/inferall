"""
Model Registry
--------------
SQLite-backed registry for tracking pulled models.
Supports schema versioning with forward-only migrations.
"""

import logging
import sqlite3
from datetime import datetime
from pathlib import Path
from typing import List, Optional

from inferall.registry.metadata import ModelFormat, ModelRecord

logger = logging.getLogger(__name__)


class ModelRegistry:
    """SQLite-backed model registry with schema migrations."""

    def __init__(self, db_path: Path):
        self.db_path = db_path
        self._conn: Optional[sqlite3.Connection] = None
        self._ensure_connection()
        self.migrate()

    def _ensure_connection(self) -> sqlite3.Connection:
        """Get or create the database connection."""
        if self._conn is None:
            self.db_path.parent.mkdir(parents=True, exist_ok=True)
            self._conn = sqlite3.connect(str(self.db_path), check_same_thread=False)
            self._conn.row_factory = sqlite3.Row
            self._conn.execute("PRAGMA journal_mode=WAL")
            self._conn.execute("PRAGMA foreign_keys=ON")
        return self._conn

    @property
    def conn(self) -> sqlite3.Connection:
        return self._ensure_connection()

    # -------------------------------------------------------------------------
    # Schema Migrations
    # -------------------------------------------------------------------------

    MIGRATIONS = {}  # Populated below

    def get_schema_version(self) -> int:
        """Get the current schema version, or 0 if no schema exists."""
        try:
            cursor = self.conn.execute(
                "SELECT name FROM sqlite_master WHERE type='table' AND name='schema_version'"
            )
            if cursor.fetchone() is None:
                return 0
            cursor = self.conn.execute("SELECT MAX(version) FROM schema_version")
            row = cursor.fetchone()
            return row[0] if row[0] is not None else 0
        except sqlite3.Error:
            return 0

    def migrate(self) -> None:
        """Run any pending migrations."""
        current = self.get_schema_version()
        target = max(self.MIGRATIONS.keys()) if self.MIGRATIONS else 0

        if current >= target:
            return

        for version in range(current + 1, target + 1):
            if version not in self.MIGRATIONS:
                raise RuntimeError(
                    f"Missing migration for version {version}. "
                    f"Current: {current}, target: {target}"
                )

            migration_fn = self.MIGRATIONS[version]
            logger.info("Running migration v%d: %s", version, migration_fn.__name__)

            try:
                with self.conn:
                    migration_fn(self.conn)
                    self.conn.execute(
                        "INSERT INTO schema_version (version) VALUES (?)",
                        (version,),
                    )
            except Exception:
                logger.error("Migration v%d failed", version, exc_info=True)
                raise

        logger.info("Schema up to date at v%d", target)

    # -------------------------------------------------------------------------
    # CRUD Operations
    # -------------------------------------------------------------------------

    def register(self, record: ModelRecord) -> None:
        """
        Register or update a model record (upsert by model_id).

        This is used after a successful pull. If the model already exists,
        all fields are updated (e.g., new revision after re-pull).
        """
        row = record.to_db_row()
        columns = ", ".join(row.keys())
        placeholders = ", ".join(["?"] * len(row))
        updates = ", ".join(f"{k}=excluded.{k}" for k in row.keys() if k != "model_id")

        sql = (
            f"INSERT INTO models ({columns}) VALUES ({placeholders}) "
            f"ON CONFLICT(model_id) DO UPDATE SET {updates}"
        )

        with self.conn:
            self.conn.execute(sql, list(row.values()))
        logger.debug("Registered model: %s", record.model_id)

    def get(self, model_id: str) -> Optional[ModelRecord]:
        """
        Lookup a model by ID. Tries exact match first, then Ollama
        canonical forms so short names like 'llama3.1' resolve to
        'ollama://library/llama3.1:latest'.
        """
        # Exact match
        cursor = self.conn.execute(
            "SELECT * FROM models WHERE model_id = ?", (model_id,)
        )
        row = cursor.fetchone()
        if row is not None:
            return ModelRecord.from_db_row(dict(row))

        # Try Ollama canonical forms for short names
        for candidate in self._ollama_candidates(model_id):
            cursor = self.conn.execute(
                "SELECT * FROM models WHERE model_id = ?", (candidate,)
            )
            row = cursor.fetchone()
            if row is not None:
                return ModelRecord.from_db_row(dict(row))

        return None

    @staticmethod
    def _ollama_candidates(model_id: str) -> list:
        """Generate possible Ollama registry IDs for a short name."""
        # Already fully qualified
        if model_id.startswith("ollama://"):
            return []

        candidates = []

        # Strip ollama:// if somehow partially present
        clean = model_id.removeprefix("ollama://")

        # Parse namespace/model:tag
        if "/" in clean:
            parts = clean.split("/", 1)
            namespace, rest = parts[0], parts[1]
        else:
            namespace, rest = "library", clean

        if ":" not in rest:
            rest_with_tag = rest + ":latest"
        else:
            rest_with_tag = rest

        candidates.append(f"ollama://{namespace}/{rest_with_tag}")

        # Also try without :latest if tag was added
        if ":" not in clean:
            candidates.append(f"ollama://{namespace}/{rest}")

        return candidates

    def list_all(self) -> List[ModelRecord]:
        """List all registered models."""
        cursor = self.conn.execute("SELECT * FROM models ORDER BY pulled_at DESC")
        return [ModelRecord.from_db_row(dict(row)) for row in cursor.fetchall()]

    def remove(self, model_id: str) -> bool:
        """
        Remove a model record. Returns True if the record existed.
        Tries exact match, then Ollama canonical forms.
        File deletion is handled separately by the caller.
        """
        # Try exact match first
        with self.conn:
            cursor = self.conn.execute(
                "DELETE FROM models WHERE model_id = ?", (model_id,)
            )
        if cursor.rowcount > 0:
            logger.debug("Removed model from registry: %s", model_id)
            return True

        # Try Ollama candidates
        for candidate in self._ollama_candidates(model_id):
            with self.conn:
                cursor = self.conn.execute(
                    "DELETE FROM models WHERE model_id = ?", (candidate,)
                )
            if cursor.rowcount > 0:
                logger.debug("Removed model from registry: %s", candidate)
                return True

        return False

    def update_last_used(self, model_id: str) -> None:
        """Update the last_used_at timestamp for LRU eviction."""
        now = datetime.now().isoformat()
        with self.conn:
            self.conn.execute(
                "UPDATE models SET last_used_at = ? WHERE model_id = ?",
                (now, model_id),
            )

    # -------------------------------------------------------------------------
    # Lifecycle
    # -------------------------------------------------------------------------

    def close(self) -> None:
        """Close the database connection."""
        if self._conn is not None:
            self._conn.close()
            self._conn = None


# =============================================================================
# Migration Functions
# =============================================================================

def _migration_v1_initial_schema(conn: sqlite3.Connection) -> None:
    """v1: Create the initial schema — schema_version + models tables."""
    conn.execute("""
        CREATE TABLE IF NOT EXISTS schema_version (
            version INTEGER PRIMARY KEY
        )
    """)
    conn.execute("""
        CREATE TABLE IF NOT EXISTS models (
            model_id TEXT PRIMARY KEY,
            revision TEXT NOT NULL,
            format TEXT NOT NULL,
            local_path TEXT NOT NULL,
            file_size_bytes INTEGER NOT NULL,
            param_count INTEGER,
            gguf_variant TEXT,
            trust_remote_code INTEGER NOT NULL DEFAULT 0,
            pipeline_tag TEXT,
            pulled_at TEXT NOT NULL,
            last_used_at TEXT
        )
    """)


def _migration_v2_add_task_column(conn: sqlite3.Connection) -> None:
    """v2: Add task column for multi-modal support."""
    conn.execute("ALTER TABLE models ADD COLUMN task TEXT DEFAULT 'chat'")


def _migration_v3_create_files_table(conn: sqlite3.Connection) -> None:
    """v3: Create files table for the Files API."""
    conn.execute("""
        CREATE TABLE IF NOT EXISTS files (
            file_id TEXT PRIMARY KEY,
            filename TEXT NOT NULL,
            purpose TEXT NOT NULL,
            bytes INTEGER NOT NULL,
            created_at TEXT NOT NULL,
            status TEXT NOT NULL DEFAULT 'uploaded',
            local_path TEXT NOT NULL
        )
    """)


def _migration_v4_create_assistants_tables(conn: sqlite3.Connection) -> None:
    """v4: Create tables for Assistants API (assistants, threads, messages, runs)."""
    conn.execute("""
        CREATE TABLE IF NOT EXISTS assistants (
            assistant_id TEXT PRIMARY KEY,
            name TEXT,
            model TEXT NOT NULL,
            instructions TEXT,
            tools TEXT NOT NULL DEFAULT '[]',
            file_ids TEXT NOT NULL DEFAULT '[]',
            metadata TEXT NOT NULL DEFAULT '{}',
            created_at TEXT NOT NULL
        )
    """)
    conn.execute("""
        CREATE TABLE IF NOT EXISTS threads (
            thread_id TEXT PRIMARY KEY,
            metadata TEXT NOT NULL DEFAULT '{}',
            created_at TEXT NOT NULL
        )
    """)
    conn.execute("""
        CREATE TABLE IF NOT EXISTS messages (
            message_id TEXT PRIMARY KEY,
            thread_id TEXT NOT NULL,
            role TEXT NOT NULL,
            content TEXT NOT NULL,
            file_ids TEXT NOT NULL DEFAULT '[]',
            assistant_id TEXT,
            run_id TEXT,
            metadata TEXT NOT NULL DEFAULT '{}',
            created_at TEXT NOT NULL
        )
    """)
    conn.execute("CREATE INDEX IF NOT EXISTS idx_messages_thread ON messages(thread_id, created_at)")
    conn.execute("""
        CREATE TABLE IF NOT EXISTS runs (
            run_id TEXT PRIMARY KEY,
            thread_id TEXT NOT NULL,
            assistant_id TEXT NOT NULL,
            model TEXT NOT NULL,
            instructions TEXT,
            status TEXT NOT NULL DEFAULT 'queued',
            required_action TEXT,
            last_error TEXT,
            started_at TEXT,
            completed_at TEXT,
            cancelled_at TEXT,
            failed_at TEXT,
            metadata TEXT NOT NULL DEFAULT '{}',
            created_at TEXT NOT NULL
        )
    """)
    conn.execute("CREATE INDEX IF NOT EXISTS idx_runs_thread ON runs(thread_id, created_at)")


def _migration_v5_create_jobs_tables(conn: sqlite3.Connection) -> None:
    """v5: Create tables for Fine-tuning and Batch APIs."""
    conn.execute("""
        CREATE TABLE IF NOT EXISTS fine_tuning_jobs (
            job_id TEXT PRIMARY KEY,
            model TEXT NOT NULL,
            training_file TEXT NOT NULL,
            validation_file TEXT,
            hyperparameters TEXT NOT NULL DEFAULT '{}',
            status TEXT NOT NULL DEFAULT 'validating',
            fine_tuned_model TEXT,
            trained_tokens INTEGER,
            error TEXT,
            created_at TEXT NOT NULL,
            started_at TEXT,
            finished_at TEXT
        )
    """)
    conn.execute("""
        CREATE TABLE IF NOT EXISTS fine_tuning_events (
            event_id TEXT PRIMARY KEY,
            job_id TEXT NOT NULL,
            type TEXT NOT NULL DEFAULT 'message',
            level TEXT NOT NULL DEFAULT 'info',
            message TEXT NOT NULL,
            data TEXT,
            created_at TEXT NOT NULL
        )
    """)
    conn.execute("CREATE INDEX IF NOT EXISTS idx_ft_events_job ON fine_tuning_events(job_id, created_at)")
    conn.execute("""
        CREATE TABLE IF NOT EXISTS batches (
            batch_id TEXT PRIMARY KEY,
            input_file_id TEXT NOT NULL,
            endpoint TEXT NOT NULL,
            completion_window TEXT NOT NULL DEFAULT '24h',
            status TEXT NOT NULL DEFAULT 'validating',
            output_file_id TEXT,
            error_file_id TEXT,
            request_counts TEXT NOT NULL DEFAULT '{}',
            errors TEXT,
            metadata TEXT NOT NULL DEFAULT '{}',
            created_at TEXT NOT NULL,
            started_at TEXT,
            completed_at TEXT,
            cancelled_at TEXT
        )
    """)


# Register migrations
ModelRegistry.MIGRATIONS = {
    1: _migration_v1_initial_schema,
    2: _migration_v2_add_task_column,
    3: _migration_v3_create_files_table,
    4: _migration_v4_create_assistants_tables,
    5: _migration_v5_create_jobs_tables,
}
