"""
File Store
----------
CRUD operations for the Files API. Stores metadata in SQLite
(shared with ModelRegistry) and file content on disk.
"""

import json
import logging
import sqlite3
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import List, Optional

logger = logging.getLogger(__name__)

# Valid purpose values
_VALID_PURPOSES = {"fine-tune", "assistants", "batch", "batch_output", "vision"}

# File extensions by purpose
_FINE_TUNE_EXTENSIONS = {".jsonl"}
_BATCH_EXTENSIONS = {".jsonl"}
_VISION_EXTENSIONS = {".png", ".jpg", ".jpeg", ".gif", ".webp"}


class FileStore:
    """File metadata storage backed by SQLite."""

    def __init__(self, conn: sqlite3.Connection):
        self.conn = conn

    def create(
        self,
        filename: str,
        purpose: str,
        size_bytes: int,
        local_path: str,
    ) -> dict:
        """Create a new file record."""
        file_id = f"file-{uuid.uuid4().hex[:24]}"
        created_at = datetime.now(timezone.utc).isoformat()

        with self.conn:
            self.conn.execute(
                "INSERT INTO files (file_id, filename, purpose, bytes, created_at, status, local_path) "
                "VALUES (?, ?, ?, ?, ?, ?, ?)",
                (file_id, filename, purpose, size_bytes, created_at, "uploaded", local_path),
            )

        logger.debug("Created file: %s (%s, %d bytes)", file_id, purpose, size_bytes)
        return self._build_file_object(
            file_id, filename, purpose, size_bytes, created_at, "uploaded",
        )

    def get(self, file_id: str) -> Optional[dict]:
        """Get file metadata by ID."""
        cursor = self.conn.execute(
            "SELECT * FROM files WHERE file_id = ?", (file_id,)
        )
        row = cursor.fetchone()
        if row is None:
            return None
        return self._row_to_file_object(dict(row))

    def list_files(self, purpose: Optional[str] = None) -> List[dict]:
        """List all files, optionally filtered by purpose."""
        if purpose:
            cursor = self.conn.execute(
                "SELECT * FROM files WHERE purpose = ? ORDER BY created_at DESC",
                (purpose,),
            )
        else:
            cursor = self.conn.execute(
                "SELECT * FROM files ORDER BY created_at DESC"
            )
        return [self._row_to_file_object(dict(row)) for row in cursor.fetchall()]

    def delete(self, file_id: str) -> bool:
        """Delete a file record. Returns True if it existed."""
        with self.conn:
            cursor = self.conn.execute(
                "DELETE FROM files WHERE file_id = ?", (file_id,)
            )
        return cursor.rowcount > 0

    def get_local_path(self, file_id: str) -> Optional[str]:
        """Get the local file path for a file."""
        cursor = self.conn.execute(
            "SELECT local_path FROM files WHERE file_id = ?", (file_id,)
        )
        row = cursor.fetchone()
        return row[0] if row else None

    # -------------------------------------------------------------------------
    # Helpers
    # -------------------------------------------------------------------------

    @staticmethod
    def _build_file_object(
        file_id: str, filename: str, purpose: str,
        size_bytes: int, created_at: str, status: str,
    ) -> dict:
        """Build an OpenAI-compatible file object."""
        # Convert ISO timestamp to Unix timestamp
        try:
            ts = int(datetime.fromisoformat(created_at).timestamp())
        except (ValueError, TypeError):
            ts = 0

        return {
            "id": file_id,
            "object": "file",
            "bytes": size_bytes,
            "created_at": ts,
            "filename": filename,
            "purpose": purpose,
            "status": status,
        }

    @staticmethod
    def _row_to_file_object(row: dict) -> dict:
        """Convert a database row to a file object."""
        return FileStore._build_file_object(
            file_id=row["file_id"],
            filename=row["filename"],
            purpose=row["purpose"],
            size_bytes=row["bytes"],
            created_at=row["created_at"],
            status=row["status"],
        )


def validate_file(purpose: str, filename: str, content: bytes) -> Optional[str]:
    """
    Validate a file upload. Returns error message or None if valid.
    """
    if purpose not in _VALID_PURPOSES:
        return f"Invalid purpose '{purpose}'. Must be one of: {', '.join(sorted(_VALID_PURPOSES))}"

    ext = Path(filename).suffix.lower()

    if purpose == "fine-tune":
        if ext not in _FINE_TUNE_EXTENSIONS:
            return f"Fine-tune files must be .jsonl (got '{ext}')"
        # Validate each line is valid JSON
        for i, line in enumerate(content.decode("utf-8", errors="replace").splitlines()):
            line = line.strip()
            if not line:
                continue
            try:
                json.loads(line)
            except json.JSONDecodeError as e:
                return f"Invalid JSON on line {i + 1}: {e}"

    elif purpose == "batch":
        if ext not in _BATCH_EXTENSIONS:
            return f"Batch files must be .jsonl (got '{ext}')"

    elif purpose == "vision":
        if ext not in _VISION_EXTENSIONS:
            return f"Vision files must be an image ({', '.join(_VISION_EXTENSIONS)}), got '{ext}'"

    return None
