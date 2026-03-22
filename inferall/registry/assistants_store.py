"""
Assistants Store
-----------------
CRUD for Assistants, Threads, Messages, and Runs.
All data stored in SQLite (shared with ModelRegistry).
"""

import json
import logging
import sqlite3
import uuid
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional

logger = logging.getLogger(__name__)


def _gen_id(prefix: str) -> str:
    return f"{prefix}-{uuid.uuid4().hex[:24]}"


def _now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def _iso_to_unix(iso: str) -> int:
    try:
        return int(datetime.fromisoformat(iso).timestamp())
    except (ValueError, TypeError):
        return 0


class AssistantsStore:
    """CRUD store for Assistants API entities."""

    def __init__(self, conn: sqlite3.Connection):
        self.conn = conn

    # =====================================================================
    # Assistants
    # =====================================================================

    def create_assistant(
        self, model: str, name: str = None, instructions: str = None,
        tools: list = None, file_ids: list = None, metadata: dict = None,
    ) -> dict:
        aid = _gen_id("asst")
        now = _now_iso()
        with self.conn:
            self.conn.execute(
                "INSERT INTO assistants (assistant_id, name, model, instructions, tools, file_ids, metadata, created_at) "
                "VALUES (?,?,?,?,?,?,?,?)",
                (aid, name, model, instructions,
                 json.dumps(tools or []), json.dumps(file_ids or []),
                 json.dumps(metadata or {}), now),
            )
        return self.get_assistant(aid)

    def get_assistant(self, assistant_id: str) -> Optional[dict]:
        cur = self.conn.execute("SELECT * FROM assistants WHERE assistant_id=?", (assistant_id,))
        row = cur.fetchone()
        return self._to_assistant(dict(row)) if row else None

    def list_assistants(self, limit: int = 20, order: str = "desc") -> list:
        direction = "DESC" if order == "desc" else "ASC"
        cur = self.conn.execute(
            f"SELECT * FROM assistants ORDER BY created_at {direction} LIMIT ?", (limit,)
        )
        return [self._to_assistant(dict(r)) for r in cur.fetchall()]

    def update_assistant(self, assistant_id: str, **kwargs) -> Optional[dict]:
        existing = self.get_assistant(assistant_id)
        if not existing:
            return None
        updates = []
        values = []
        for key in ("name", "model", "instructions"):
            if key in kwargs and kwargs[key] is not None:
                updates.append(f"{key}=?")
                values.append(kwargs[key])
        for key in ("tools", "file_ids", "metadata"):
            if key in kwargs and kwargs[key] is not None:
                updates.append(f"{key}=?")
                values.append(json.dumps(kwargs[key]))
        if not updates:
            return existing
        values.append(assistant_id)
        with self.conn:
            self.conn.execute(
                f"UPDATE assistants SET {','.join(updates)} WHERE assistant_id=?", values
            )
        return self.get_assistant(assistant_id)

    def delete_assistant(self, assistant_id: str) -> bool:
        with self.conn:
            cur = self.conn.execute("DELETE FROM assistants WHERE assistant_id=?", (assistant_id,))
        return cur.rowcount > 0

    def _to_assistant(self, row: dict) -> dict:
        return {
            "id": row["assistant_id"], "object": "assistant",
            "created_at": _iso_to_unix(row["created_at"]),
            "name": row["name"], "model": row["model"],
            "instructions": row["instructions"],
            "tools": json.loads(row["tools"]),
            "file_ids": json.loads(row["file_ids"]),
            "metadata": json.loads(row["metadata"]),
        }

    # =====================================================================
    # Threads
    # =====================================================================

    def create_thread(self, metadata: dict = None) -> dict:
        tid = _gen_id("thread")
        now = _now_iso()
        with self.conn:
            self.conn.execute(
                "INSERT INTO threads (thread_id, metadata, created_at) VALUES (?,?,?)",
                (tid, json.dumps(metadata or {}), now),
            )
        return self.get_thread(tid)

    def get_thread(self, thread_id: str) -> Optional[dict]:
        cur = self.conn.execute("SELECT * FROM threads WHERE thread_id=?", (thread_id,))
        row = cur.fetchone()
        return self._to_thread(dict(row)) if row else None

    def update_thread(self, thread_id: str, metadata: dict) -> Optional[dict]:
        existing = self.get_thread(thread_id)
        if not existing:
            return None
        with self.conn:
            self.conn.execute(
                "UPDATE threads SET metadata=? WHERE thread_id=?",
                (json.dumps(metadata), thread_id),
            )
        return self.get_thread(thread_id)

    def delete_thread(self, thread_id: str) -> bool:
        with self.conn:
            self.conn.execute("DELETE FROM messages WHERE thread_id=?", (thread_id,))
            self.conn.execute("DELETE FROM runs WHERE thread_id=?", (thread_id,))
            cur = self.conn.execute("DELETE FROM threads WHERE thread_id=?", (thread_id,))
        return cur.rowcount > 0

    def _to_thread(self, row: dict) -> dict:
        return {
            "id": row["thread_id"], "object": "thread",
            "created_at": _iso_to_unix(row["created_at"]),
            "metadata": json.loads(row["metadata"]),
        }

    # =====================================================================
    # Messages
    # =====================================================================

    def create_message(
        self, thread_id: str, role: str, content: str,
        file_ids: list = None, metadata: dict = None,
        assistant_id: str = None, run_id: str = None,
    ) -> dict:
        mid = _gen_id("msg")
        now = _now_iso()
        # Store content as JSON array of content blocks
        content_json = json.dumps([{"type": "text", "text": {"value": content}}])
        with self.conn:
            self.conn.execute(
                "INSERT INTO messages (message_id, thread_id, role, content, file_ids, assistant_id, run_id, metadata, created_at) "
                "VALUES (?,?,?,?,?,?,?,?,?)",
                (mid, thread_id, role, content_json,
                 json.dumps(file_ids or []), assistant_id, run_id,
                 json.dumps(metadata or {}), now),
            )
        return self.get_message(thread_id, mid)

    def get_message(self, thread_id: str, message_id: str) -> Optional[dict]:
        cur = self.conn.execute(
            "SELECT * FROM messages WHERE message_id=? AND thread_id=?",
            (message_id, thread_id),
        )
        row = cur.fetchone()
        return self._to_message(dict(row)) if row else None

    def list_messages(self, thread_id: str, limit: int = 20, order: str = "desc") -> list:
        direction = "DESC" if order == "desc" else "ASC"
        cur = self.conn.execute(
            f"SELECT * FROM messages WHERE thread_id=? ORDER BY created_at {direction} LIMIT ?",
            (thread_id, limit),
        )
        return [self._to_message(dict(r)) for r in cur.fetchall()]

    def _to_message(self, row: dict) -> dict:
        return {
            "id": row["message_id"], "object": "thread.message",
            "created_at": _iso_to_unix(row["created_at"]),
            "thread_id": row["thread_id"], "role": row["role"],
            "content": json.loads(row["content"]),
            "file_ids": json.loads(row["file_ids"]),
            "assistant_id": row["assistant_id"],
            "run_id": row["run_id"],
            "metadata": json.loads(row["metadata"]),
        }

    # =====================================================================
    # Runs
    # =====================================================================

    def create_run(
        self, thread_id: str, assistant_id: str,
        model: str = None, instructions: str = None, metadata: dict = None,
    ) -> dict:
        # Resolve model from assistant if not overridden
        assistant = self.get_assistant(assistant_id)
        if not assistant:
            raise ValueError(f"Assistant '{assistant_id}' not found")
        if not model:
            model = assistant["model"]
        if not instructions:
            instructions = assistant.get("instructions")

        rid = _gen_id("run")
        now = _now_iso()
        with self.conn:
            self.conn.execute(
                "INSERT INTO runs (run_id, thread_id, assistant_id, model, instructions, status, metadata, created_at) "
                "VALUES (?,?,?,?,?,?,?,?)",
                (rid, thread_id, assistant_id, model, instructions,
                 "queued", json.dumps(metadata or {}), now),
            )
        return self.get_run(thread_id, rid)

    def get_run(self, thread_id: str, run_id: str) -> Optional[dict]:
        cur = self.conn.execute(
            "SELECT * FROM runs WHERE run_id=? AND thread_id=?", (run_id, thread_id),
        )
        row = cur.fetchone()
        return self._to_run(dict(row)) if row else None

    def list_runs(self, thread_id: str, limit: int = 20, order: str = "desc") -> list:
        direction = "DESC" if order == "desc" else "ASC"
        cur = self.conn.execute(
            f"SELECT * FROM runs WHERE thread_id=? ORDER BY created_at {direction} LIMIT ?",
            (thread_id, limit),
        )
        return [self._to_run(dict(r)) for r in cur.fetchall()]

    def update_run_status(self, run_id: str, status: str, **kwargs) -> None:
        sets = ["status=?"]
        vals = [status]
        for key in ("started_at", "completed_at", "cancelled_at", "failed_at"):
            if key in kwargs:
                sets.append(f"{key}=?")
                vals.append(kwargs[key])
        vals.append(run_id)
        with self.conn:
            self.conn.execute(f"UPDATE runs SET {','.join(sets)} WHERE run_id=?", vals)

    def set_run_required_action(self, run_id: str, tool_calls_json: str) -> None:
        with self.conn:
            self.conn.execute(
                "UPDATE runs SET status='requires_action', required_action=? WHERE run_id=?",
                (tool_calls_json, run_id),
            )

    def set_run_error(self, run_id: str, code: str, message: str) -> None:
        error_json = json.dumps({"code": code, "message": message})
        with self.conn:
            self.conn.execute(
                "UPDATE runs SET status='failed', last_error=?, failed_at=? WHERE run_id=?",
                (error_json, _now_iso(), run_id),
            )

    def _to_run(self, row: dict) -> dict:
        result = {
            "id": row["run_id"], "object": "thread.run",
            "created_at": _iso_to_unix(row["created_at"]),
            "thread_id": row["thread_id"],
            "assistant_id": row["assistant_id"],
            "model": row["model"],
            "instructions": row["instructions"],
            "status": row["status"],
            "required_action": json.loads(row["required_action"]) if row.get("required_action") else None,
            "last_error": json.loads(row["last_error"]) if row.get("last_error") else None,
            "started_at": _iso_to_unix(row["started_at"]) if row.get("started_at") else None,
            "completed_at": _iso_to_unix(row["completed_at"]) if row.get("completed_at") else None,
            "cancelled_at": _iso_to_unix(row["cancelled_at"]) if row.get("cancelled_at") else None,
            "failed_at": _iso_to_unix(row["failed_at"]) if row.get("failed_at") else None,
            "metadata": json.loads(row["metadata"]),
        }
        return result


# =========================================================================
# Run Execution
# =========================================================================

def execute_run(run_id: str, thread_id: str, store: AssistantsStore, orchestrator) -> None:
    """
    Execute a run — called in a thread pool.
    Reads thread messages, sends to model, appends response.
    """
    from inferall.backends.base import GenerationParams

    try:
        # Mark in_progress
        store.update_run_status(run_id, "in_progress", started_at=_now_iso())

        # Check if cancelled
        run = store.get_run(thread_id, run_id)
        if not run or run["status"] == "cancelled":
            return

        # Get assistant
        assistant = store.get_assistant(run["assistant_id"])
        if not assistant:
            store.set_run_error(run_id, "server_error", "Assistant not found")
            return

        # Build messages from thread
        thread_messages = store.list_messages(thread_id, limit=100, order="asc")
        messages = []

        # System prompt from instructions
        instructions = run.get("instructions") or assistant.get("instructions")
        if instructions:
            messages.append({"role": "system", "content": instructions})

        # Thread messages
        for msg in thread_messages:
            content_blocks = msg.get("content", [])
            text = ""
            for block in content_blocks:
                if block.get("type") == "text":
                    text += block.get("text", {}).get("value", "")
            messages.append({"role": msg["role"], "content": text})

        # Build params
        tools = assistant.get("tools", [])
        params = GenerationParams(
            max_tokens=4096,
            tools=tools if tools else None,
        )

        # Run inference
        model = run["model"]
        result = orchestrator.generate(model, messages, params)

        # Check if cancelled during inference
        current = store.get_run(thread_id, run_id)
        if current and current["status"] == "cancelled":
            return

        # Handle tool calls
        if result.tool_calls:
            tool_calls_data = [
                {
                    "id": tc.id,
                    "type": tc.type,
                    "function": {
                        "name": tc.function_name,
                        "arguments": tc.function_arguments,
                    },
                }
                for tc in result.tool_calls
            ]
            store.set_run_required_action(
                run_id,
                json.dumps({"type": "submit_tool_outputs", "submit_tool_outputs": {"tool_calls": tool_calls_data}}),
            )
            return

        # Append assistant response to thread
        store.create_message(
            thread_id=thread_id,
            role="assistant",
            content=result.text,
            assistant_id=run["assistant_id"],
            run_id=run_id,
        )

        # Mark completed
        store.update_run_status(run_id, "completed", completed_at=_now_iso())

    except Exception as e:
        logger.error("Run %s failed: %s", run_id, e, exc_info=True)
        try:
            store.set_run_error(run_id, "server_error", str(e))
        except Exception:
            pass
