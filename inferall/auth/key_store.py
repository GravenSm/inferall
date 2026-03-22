"""
API Key Store
--------------
SQLite-backed multi-key management with rate limiting and priority.

Keys stored as SHA-256 hashes. Rate limiting uses a sliding window
counter on the usage_log table.
"""

import hashlib
import logging
import secrets
import sqlite3
import time
from dataclasses import dataclass
from datetime import datetime, timezone
from typing import List, Optional

logger = logging.getLogger(__name__)


@dataclass
class KeyInfo:
    """Information about an API key."""

    key_hash: str
    name: str
    priority: int           # 0=normal, 1=high, 2=critical
    rate_limit_rpm: int     # requests per minute
    rate_limit_rpd: int     # requests per day
    created_at: float
    enabled: bool


class KeyStore:
    """SQLite-backed API key management with rate limiting."""

    def __init__(self, db_path: str):
        self.conn = sqlite3.connect(str(db_path), check_same_thread=False)
        self.conn.row_factory = sqlite3.Row
        self.conn.execute("PRAGMA journal_mode=WAL")
        self._ensure_tables()

    def _ensure_tables(self):
        self.conn.execute("""
            CREATE TABLE IF NOT EXISTS api_keys (
                key_hash TEXT PRIMARY KEY,
                name TEXT NOT NULL,
                key_prefix TEXT NOT NULL,
                priority INTEGER NOT NULL DEFAULT 0,
                rate_limit_rpm INTEGER NOT NULL DEFAULT 60,
                rate_limit_rpd INTEGER NOT NULL DEFAULT 10000,
                created_at REAL NOT NULL,
                enabled INTEGER NOT NULL DEFAULT 1
            )
        """)
        self.conn.execute("""
            CREATE TABLE IF NOT EXISTS usage_log (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                key_hash TEXT NOT NULL,
                timestamp REAL NOT NULL,
                model_id TEXT,
                tokens INTEGER DEFAULT 0,
                endpoint TEXT
            )
        """)
        self.conn.execute(
            "CREATE INDEX IF NOT EXISTS idx_usage_key_ts ON usage_log(key_hash, timestamp)"
        )
        self.conn.commit()

    def create_key(
        self, name: str, priority: int = 0,
        rate_limit_rpm: int = 60, rate_limit_rpd: int = 10000,
    ) -> str:
        """
        Create a new API key. Returns the plaintext key (shown only once).
        """
        raw_key = f"me-{secrets.token_hex(32)}"
        key_hash = hashlib.sha256(raw_key.encode()).hexdigest()
        key_prefix = raw_key[:12]

        with self.conn:
            self.conn.execute(
                "INSERT INTO api_keys (key_hash, name, key_prefix, priority, "
                "rate_limit_rpm, rate_limit_rpd, created_at, enabled) "
                "VALUES (?,?,?,?,?,?,?,?)",
                (key_hash, name, key_prefix, priority,
                 rate_limit_rpm, rate_limit_rpd, time.time(), 1),
            )

        logger.info("Created API key '%s' (prefix: %s, priority: %d)", name, key_prefix, priority)
        return raw_key

    def validate_key(self, raw_key: str) -> Optional[KeyInfo]:
        """Validate a key and return its info, or None if invalid."""
        key_hash = hashlib.sha256(raw_key.encode()).hexdigest()
        cursor = self.conn.execute(
            "SELECT * FROM api_keys WHERE key_hash = ? AND enabled = 1",
            (key_hash,),
        )
        row = cursor.fetchone()
        if row is None:
            return None

        return KeyInfo(
            key_hash=row["key_hash"],
            name=row["name"],
            priority=row["priority"],
            rate_limit_rpm=row["rate_limit_rpm"],
            rate_limit_rpd=row["rate_limit_rpd"],
            created_at=row["created_at"],
            enabled=bool(row["enabled"]),
        )

    def check_rate_limit(self, key_info: KeyInfo) -> Optional[str]:
        """
        Check if a key is rate limited.
        Returns error message if limited, None if OK.
        """
        now = time.time()

        # Check RPM (requests in last 60 seconds)
        cursor = self.conn.execute(
            "SELECT COUNT(*) FROM usage_log WHERE key_hash = ? AND timestamp > ?",
            (key_info.key_hash, now - 60),
        )
        rpm_count = cursor.fetchone()[0]
        if rpm_count >= key_info.rate_limit_rpm:
            return f"Rate limit exceeded: {rpm_count}/{key_info.rate_limit_rpm} requests per minute"

        # Check RPD (requests in last 24 hours)
        cursor = self.conn.execute(
            "SELECT COUNT(*) FROM usage_log WHERE key_hash = ? AND timestamp > ?",
            (key_info.key_hash, now - 86400),
        )
        rpd_count = cursor.fetchone()[0]
        if rpd_count >= key_info.rate_limit_rpd:
            return f"Rate limit exceeded: {rpd_count}/{key_info.rate_limit_rpd} requests per day"

        return None

    def log_usage(
        self, key_hash: str, model_id: str = None,
        tokens: int = 0, endpoint: str = None,
    ):
        """Log a request for rate limiting and analytics."""
        with self.conn:
            self.conn.execute(
                "INSERT INTO usage_log (key_hash, timestamp, model_id, tokens, endpoint) "
                "VALUES (?,?,?,?,?)",
                (key_hash, time.time(), model_id, tokens, endpoint),
            )

    def list_keys(self) -> List[dict]:
        """List all keys (without hashes, just metadata)."""
        cursor = self.conn.execute(
            "SELECT key_prefix, name, priority, rate_limit_rpm, rate_limit_rpd, "
            "created_at, enabled FROM api_keys ORDER BY created_at DESC"
        )
        return [
            {
                "key_prefix": row["key_prefix"],
                "name": row["name"],
                "priority": row["priority"],
                "rate_limit_rpm": row["rate_limit_rpm"],
                "rate_limit_rpd": row["rate_limit_rpd"],
                "created_at": row["created_at"],
                "enabled": bool(row["enabled"]),
            }
            for row in cursor.fetchall()
        ]

    def revoke_key(self, key_prefix: str) -> bool:
        """Revoke a key by its prefix. Returns True if found."""
        with self.conn:
            cursor = self.conn.execute(
                "UPDATE api_keys SET enabled = 0 WHERE key_prefix = ?",
                (key_prefix,),
            )
        return cursor.rowcount > 0

    def get_usage(self, key_prefix: str, hours: int = 24) -> dict:
        """Get usage stats for a key prefix."""
        cursor = self.conn.execute(
            "SELECT key_hash FROM api_keys WHERE key_prefix = ?", (key_prefix,)
        )
        row = cursor.fetchone()
        if not row:
            return {"error": "Key not found"}

        key_hash = row["key_hash"]
        since = time.time() - (hours * 3600)

        cursor = self.conn.execute(
            "SELECT COUNT(*) as requests, COALESCE(SUM(tokens), 0) as total_tokens "
            "FROM usage_log WHERE key_hash = ? AND timestamp > ?",
            (key_hash, since),
        )
        row = cursor.fetchone()
        return {
            "requests": row["requests"],
            "total_tokens": row["total_tokens"],
            "period_hours": hours,
        }

    def cleanup_old_logs(self, days: int = 30):
        """Remove usage logs older than N days."""
        cutoff = time.time() - (days * 86400)
        with self.conn:
            cursor = self.conn.execute(
                "DELETE FROM usage_log WHERE timestamp < ?", (cutoff,),
            )
        if cursor.rowcount > 0:
            logger.debug("Cleaned up %d old usage log entries", cursor.rowcount)

    def close(self):
        self.conn.close()
