"""Tests for multi-key auth — key store, rate limiting, middleware."""

import time
from unittest.mock import MagicMock

import pytest

from inferall.auth.key_store import KeyStore


@pytest.fixture
def store(tmp_path):
    s = KeyStore(str(tmp_path / "auth.db"))
    yield s
    s.close()


class TestKeyCreation:
    def test_create_key(self, store):
        key = store.create_key("user1")
        assert key.startswith("me-")
        assert len(key) > 30

    def test_validate_key(self, store):
        key = store.create_key("user1", priority=1)
        info = store.validate_key(key)
        assert info is not None
        assert info.name == "user1"
        assert info.priority == 1
        assert info.enabled is True

    def test_invalid_key(self, store):
        assert store.validate_key("me-invalid") is None

    def test_revoked_key_fails(self, store):
        key = store.create_key("user1")
        info = store.validate_key(key)
        prefix = key[:12]
        store.revoke_key(prefix)
        assert store.validate_key(key) is None


class TestRateLimiting:
    def test_under_limit(self, store):
        key = store.create_key("user1", rate_limit_rpm=10)
        info = store.validate_key(key)
        for _ in range(5):
            store.log_usage(info.key_hash)
        assert store.check_rate_limit(info) is None

    def test_over_rpm_limit(self, store):
        key = store.create_key("user1", rate_limit_rpm=3)
        info = store.validate_key(key)
        for _ in range(3):
            store.log_usage(info.key_hash)
        error = store.check_rate_limit(info)
        assert error is not None
        assert "per minute" in error


class TestKeyManagement:
    def test_list_keys(self, store):
        store.create_key("a")
        store.create_key("b")
        keys = store.list_keys()
        assert len(keys) == 2

    def test_revoke_key(self, store):
        key = store.create_key("user1")
        prefix = key[:12]
        assert store.revoke_key(prefix) is True
        keys = store.list_keys()
        assert any(not k["enabled"] for k in keys)

    def test_revoke_nonexistent(self, store):
        assert store.revoke_key("me-nonexist") is False

    def test_usage_stats(self, store):
        key = store.create_key("user1")
        info = store.validate_key(key)
        store.log_usage(info.key_hash, tokens=100, endpoint="/v1/chat/completions")
        store.log_usage(info.key_hash, tokens=50, endpoint="/v1/embeddings")
        prefix = key[:12]
        usage = store.get_usage(prefix)
        assert usage["requests"] == 2
        assert usage["total_tokens"] == 150


class TestMiddleware:
    def test_no_auth_passthrough(self):
        """When no auth configured, all requests pass."""
        from inferall.auth.middleware import create_auth_middleware
        import asyncio

        middleware = create_auth_middleware(key_store=None, single_api_key=None)
        # Just verify it creates without error
        assert callable(middleware)

    def test_multi_key_validates(self, store):
        from inferall.auth.middleware import create_auth_middleware
        middleware = create_auth_middleware(key_store=store)
        assert callable(middleware)
