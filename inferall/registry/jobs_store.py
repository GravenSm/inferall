"""
Jobs Store
----------
CRUD for Fine-tuning jobs and Batch processing jobs.
"""

import json
import logging
import sqlite3
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import List, Optional

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


# =========================================================================
# Fine-tuning Store
# =========================================================================

class FineTuningStore:
    """CRUD for fine-tuning jobs and events."""

    def __init__(self, conn: sqlite3.Connection):
        self.conn = conn

    def create_job(self, model: str, training_file: str,
                   validation_file: str = None, hyperparameters: dict = None) -> dict:
        jid = _gen_id("ftjob")
        now = _now_iso()
        with self.conn:
            self.conn.execute(
                "INSERT INTO fine_tuning_jobs (job_id, model, training_file, validation_file, "
                "hyperparameters, status, created_at) VALUES (?,?,?,?,?,?,?)",
                (jid, model, training_file, validation_file,
                 json.dumps(hyperparameters or {}), "validating", now),
            )
        self.add_event(jid, "info", "Fine-tuning job created")
        return self.get_job(jid)

    def get_job(self, job_id: str) -> Optional[dict]:
        cur = self.conn.execute("SELECT * FROM fine_tuning_jobs WHERE job_id=?", (job_id,))
        row = cur.fetchone()
        return self._to_job(dict(row)) if row else None

    def list_jobs(self, limit: int = 20, after: str = None) -> list:
        if after:
            cur = self.conn.execute(
                "SELECT * FROM fine_tuning_jobs WHERE created_at < "
                "(SELECT created_at FROM fine_tuning_jobs WHERE job_id=?) "
                "ORDER BY created_at DESC LIMIT ?", (after, limit),
            )
        else:
            cur = self.conn.execute(
                "SELECT * FROM fine_tuning_jobs ORDER BY created_at DESC LIMIT ?", (limit,),
            )
        return [self._to_job(dict(r)) for r in cur.fetchall()]

    def cancel_job(self, job_id: str) -> bool:
        job = self.get_job(job_id)
        if not job or job["status"] not in ("validating", "queued", "running"):
            return False
        with self.conn:
            self.conn.execute(
                "UPDATE fine_tuning_jobs SET status='cancelled' WHERE job_id=?", (job_id,),
            )
        self.add_event(job_id, "info", "Job cancelled")
        return True

    def update_job_status(self, job_id: str, status: str, **kwargs) -> None:
        sets = ["status=?"]
        vals = [status]
        for key in ("started_at", "finished_at", "fine_tuned_model", "trained_tokens"):
            if key in kwargs:
                sets.append(f"{key}=?")
                vals.append(kwargs[key])
        vals.append(job_id)
        with self.conn:
            self.conn.execute(f"UPDATE fine_tuning_jobs SET {','.join(sets)} WHERE job_id=?", vals)

    def set_job_error(self, job_id: str, code: str, message: str) -> None:
        error_json = json.dumps({"code": code, "message": message})
        with self.conn:
            self.conn.execute(
                "UPDATE fine_tuning_jobs SET status='failed', error=?, finished_at=? WHERE job_id=?",
                (error_json, _now_iso(), job_id),
            )
        self.add_event(job_id, "error", message)

    def add_event(self, job_id: str, level: str, message: str, data: dict = None) -> dict:
        eid = _gen_id("ftevent")
        now = _now_iso()
        with self.conn:
            self.conn.execute(
                "INSERT INTO fine_tuning_events (event_id, job_id, type, level, message, data, created_at) "
                "VALUES (?,?,?,?,?,?,?)",
                (eid, job_id, "message", level, message, json.dumps(data) if data else None, now),
            )
        return {"id": eid, "object": "fine_tuning.job.event", "created_at": _iso_to_unix(now),
                "level": level, "message": message, "data": data}

    def list_events(self, job_id: str, limit: int = 20) -> list:
        cur = self.conn.execute(
            "SELECT * FROM fine_tuning_events WHERE job_id=? ORDER BY created_at DESC LIMIT ?",
            (job_id, limit),
        )
        return [self._to_event(dict(r)) for r in cur.fetchall()]

    def list_checkpoints(self, job_id: str, limit: int = 10) -> list:
        return []  # Stub — no checkpoint support yet

    def _to_job(self, row: dict) -> dict:
        return {
            "id": row["job_id"], "object": "fine_tuning.job",
            "model": row["model"], "training_file": row["training_file"],
            "validation_file": row["validation_file"],
            "hyperparameters": json.loads(row["hyperparameters"]),
            "status": row["status"],
            "fine_tuned_model": row["fine_tuned_model"],
            "trained_tokens": row["trained_tokens"],
            "error": json.loads(row["error"]) if row.get("error") else None,
            "created_at": _iso_to_unix(row["created_at"]),
            "started_at": _iso_to_unix(row["started_at"]) if row.get("started_at") else None,
            "finished_at": _iso_to_unix(row["finished_at"]) if row.get("finished_at") else None,
        }

    def _to_event(self, row: dict) -> dict:
        return {
            "id": row["event_id"], "object": "fine_tuning.job.event",
            "created_at": _iso_to_unix(row["created_at"]),
            "level": row["level"], "message": row["message"],
            "data": json.loads(row["data"]) if row.get("data") else None,
        }


# =========================================================================
# Batch Store
# =========================================================================

class BatchStore:
    """CRUD for batch processing jobs."""

    def __init__(self, conn: sqlite3.Connection):
        self.conn = conn

    def create_batch(self, input_file_id: str, endpoint: str,
                     completion_window: str = "24h", metadata: dict = None) -> dict:
        bid = _gen_id("batch")
        now = _now_iso()
        with self.conn:
            self.conn.execute(
                "INSERT INTO batches (batch_id, input_file_id, endpoint, completion_window, "
                "status, request_counts, metadata, created_at) VALUES (?,?,?,?,?,?,?,?)",
                (bid, input_file_id, endpoint, completion_window,
                 "validating", json.dumps({"total": 0, "completed": 0, "failed": 0}),
                 json.dumps(metadata or {}), now),
            )
        return self.get_batch(bid)

    def get_batch(self, batch_id: str) -> Optional[dict]:
        cur = self.conn.execute("SELECT * FROM batches WHERE batch_id=?", (batch_id,))
        row = cur.fetchone()
        return self._to_batch(dict(row)) if row else None

    def list_batches(self, limit: int = 20, after: str = None) -> list:
        if after:
            cur = self.conn.execute(
                "SELECT * FROM batches WHERE created_at < "
                "(SELECT created_at FROM batches WHERE batch_id=?) "
                "ORDER BY created_at DESC LIMIT ?", (after, limit),
            )
        else:
            cur = self.conn.execute(
                "SELECT * FROM batches ORDER BY created_at DESC LIMIT ?", (limit,),
            )
        return [self._to_batch(dict(r)) for r in cur.fetchall()]

    def cancel_batch(self, batch_id: str) -> bool:
        batch = self.get_batch(batch_id)
        if not batch or batch["status"] not in ("validating", "in_progress"):
            return False
        with self.conn:
            self.conn.execute(
                "UPDATE batches SET status='cancelled', cancelled_at=? WHERE batch_id=?",
                (_now_iso(), batch_id),
            )
        return True

    def update_batch_status(self, batch_id: str, status: str, **kwargs) -> None:
        sets = ["status=?"]
        vals = [status]
        for key in ("started_at", "completed_at", "cancelled_at"):
            if key in kwargs:
                sets.append(f"{key}=?")
                vals.append(kwargs[key])
        vals.append(batch_id)
        with self.conn:
            self.conn.execute(f"UPDATE batches SET {','.join(sets)} WHERE batch_id=?", vals)

    def update_request_counts(self, batch_id: str, total: int, completed: int, failed: int) -> None:
        with self.conn:
            self.conn.execute(
                "UPDATE batches SET request_counts=? WHERE batch_id=?",
                (json.dumps({"total": total, "completed": completed, "failed": failed}), batch_id),
            )

    def set_batch_output(self, batch_id: str, output_file_id: str, error_file_id: str = None) -> None:
        with self.conn:
            self.conn.execute(
                "UPDATE batches SET output_file_id=?, error_file_id=? WHERE batch_id=?",
                (output_file_id, error_file_id, batch_id),
            )

    def _to_batch(self, row: dict) -> dict:
        return {
            "id": row["batch_id"], "object": "batch",
            "input_file_id": row["input_file_id"],
            "endpoint": row["endpoint"],
            "completion_window": row["completion_window"],
            "status": row["status"],
            "output_file_id": row["output_file_id"],
            "error_file_id": row["error_file_id"],
            "request_counts": json.loads(row["request_counts"]),
            "errors": json.loads(row["errors"]) if row.get("errors") else None,
            "metadata": json.loads(row["metadata"]),
            "created_at": _iso_to_unix(row["created_at"]),
            "started_at": _iso_to_unix(row["started_at"]) if row.get("started_at") else None,
            "completed_at": _iso_to_unix(row["completed_at"]) if row.get("completed_at") else None,
            "cancelled_at": _iso_to_unix(row["cancelled_at"]) if row.get("cancelled_at") else None,
        }


# =========================================================================
# Execution Functions
# =========================================================================

def execute_fine_tuning_job(job_id: str, store: FineTuningStore) -> None:
    """Execute a fine-tuning job (stub — training not yet implemented)."""
    try:
        store.update_job_status(job_id, "running", started_at=_now_iso())
        store.add_event(job_id, "info", "Fine-tuning job started")
        store.add_event(job_id, "warn",
            "Training backend not yet implemented. "
            "Requires LoRA/QLoRA + PEFT integration. "
            "The API structure is ready for when training is added."
        )
        store.set_job_error(job_id, "training_not_implemented",
            "Fine-tuning training backend is not yet implemented.")
    except Exception as e:
        logger.error("Fine-tuning job %s error: %s", job_id, e, exc_info=True)
        try:
            store.set_job_error(job_id, "server_error", str(e))
        except Exception:
            pass


_MAX_BATCH_REQUESTS = 10000  # Maximum requests per batch job
_BATCH_THROTTLE_SECONDS = 0.1  # Pause between requests to avoid monopolizing

def execute_batch(
    batch_id: str, store: BatchStore, file_store, orchestrator, files_dir,
) -> None:
    """Execute a batch job — process JSONL input through inference endpoints."""
    import time as time_mod
    from inferall.backends.base import GenerationParams, EmbeddingParams

    try:
        store.update_batch_status(batch_id, "in_progress", started_at=_now_iso())

        batch = store.get_batch(batch_id)
        if not batch:
            return

        # Read input file
        local_path = file_store.get_local_path(batch["input_file_id"])
        if not local_path:
            store.update_batch_status(batch_id, "failed")
            return

        input_path = Path(local_path)
        if not input_path.exists():
            store.update_batch_status(batch_id, "failed")
            return

        lines = input_path.read_text().strip().splitlines()
        total = len(lines)

        # Enforce batch size limit
        if total > _MAX_BATCH_REQUESTS:
            store.update_batch_status(batch_id, "failed")
            logger.error("Batch %s rejected: %d requests exceeds limit of %d",
                         batch_id, total, _MAX_BATCH_REQUESTS)
            return

        completed = 0
        failed = 0
        output_lines = []
        error_lines = []

        store.update_request_counts(batch_id, total, 0, 0)

        for line in lines:
            # Check if cancelled
            current = store.get_batch(batch_id)
            if current and current["status"] == "cancelled":
                return

            try:
                request = json.loads(line)
            except json.JSONDecodeError:
                failed += 1
                error_lines.append(json.dumps({
                    "custom_id": None, "error": {"message": "Invalid JSON line"},
                }))
                continue

            custom_id = request.get("custom_id", "")
            url = request.get("url", "")
            body = request.get("body", {})

            # Throttle to avoid monopolizing inference workers
            time_mod.sleep(_BATCH_THROTTLE_SECONDS)

            try:
                response_body = _dispatch_batch_request(url, body, orchestrator)
                output_lines.append(json.dumps({
                    "id": _gen_id("bresp"),
                    "custom_id": custom_id,
                    "response": {"status_code": 200, "body": response_body},
                }))
                completed += 1
            except Exception as e:
                failed += 1
                error_lines.append(json.dumps({
                    "id": _gen_id("bresp"),
                    "custom_id": custom_id,
                    "response": {"status_code": 500, "body": {"error": str(e)}},
                }))

            store.update_request_counts(batch_id, total, completed, failed)

        # Write output file
        output_file_id = None
        if output_lines:
            output_content = "\n".join(output_lines).encode()
            fdir = Path(files_dir) / _gen_id("file")
            fdir.mkdir(parents=True, exist_ok=True)
            out_path = fdir / "output.jsonl"
            out_path.write_bytes(output_content)
            result = file_store.create("output.jsonl", "batch_output", len(output_content), str(out_path))
            output_file_id = result["id"]

        # Write error file
        error_file_id = None
        if error_lines:
            error_content = "\n".join(error_lines).encode()
            fdir = Path(files_dir) / _gen_id("file")
            fdir.mkdir(parents=True, exist_ok=True)
            err_path = fdir / "errors.jsonl"
            err_path.write_bytes(error_content)
            result = file_store.create("errors.jsonl", "batch_output", len(error_content), str(err_path))
            error_file_id = result["id"]

        store.set_batch_output(batch_id, output_file_id, error_file_id)
        store.update_batch_status(batch_id, "completed", completed_at=_now_iso())

    except Exception as e:
        logger.error("Batch %s error: %s", batch_id, e, exc_info=True)
        try:
            store.update_batch_status(batch_id, "failed")
        except Exception:
            pass


def _dispatch_batch_request(url: str, body: dict, orchestrator) -> dict:
    """Dispatch a single batch request to the appropriate orchestrator method."""
    from inferall.backends.base import GenerationParams, EmbeddingParams

    if url == "/v1/chat/completions":
        messages = body.get("messages", [])
        params = GenerationParams(
            max_tokens=body.get("max_tokens", 2048),
            temperature=body.get("temperature", 0.7),
            top_p=body.get("top_p", 0.9),
            stop=body.get("stop"),
        )
        result = orchestrator.generate(body["model"], messages, params)
        return {
            "choices": [{"index": 0, "message": {"role": "assistant", "content": result.text},
                         "finish_reason": result.finish_reason}],
            "usage": {"prompt_tokens": result.prompt_tokens,
                      "completion_tokens": result.completion_tokens,
                      "total_tokens": result.prompt_tokens + result.completion_tokens},
        }

    elif url == "/v1/embeddings":
        texts = body.get("input", [])
        if isinstance(texts, str):
            texts = [texts]
        params = EmbeddingParams()
        result = orchestrator.embed(body["model"], texts, params)
        return {
            "data": [{"index": i, "embedding": e} for i, e in enumerate(result.embeddings)],
            "usage": {"prompt_tokens": result.prompt_tokens, "total_tokens": result.prompt_tokens},
        }

    elif url == "/v1/completions":
        prompt = body.get("prompt", "")
        if isinstance(prompt, list):
            prompt = prompt[0]
        messages = [{"role": "user", "content": prompt}]
        params = GenerationParams(
            max_tokens=body.get("max_tokens", 2048),
            temperature=body.get("temperature", 0.7),
        )
        result = orchestrator.generate(body["model"], messages, params)
        return {
            "choices": [{"text": result.text, "index": 0, "finish_reason": result.finish_reason}],
            "usage": {"prompt_tokens": result.prompt_tokens,
                      "completion_tokens": result.completion_tokens,
                      "total_tokens": result.prompt_tokens + result.completion_tokens},
        }

    else:
        raise ValueError(f"Unsupported batch endpoint: {url}")
