"""
Auth Middleware
---------------
FastAPI middleware for multi-key authentication with rate limiting.

Backward compatible: if no KeyStore is configured, falls back to
the simple single-key comparison.
"""

import logging
from typing import Optional

from fastapi import Request
from fastapi.responses import JSONResponse

from inferall.auth.key_store import KeyInfo, KeyStore

logger = logging.getLogger(__name__)


def create_auth_middleware(
    key_store: Optional[KeyStore] = None,
    single_api_key: Optional[str] = None,
    skip_paths: set = None,
):
    """
    Create an auth middleware function.

    If key_store is provided, uses multi-key auth with rate limiting.
    If single_api_key is provided, uses simple string comparison.
    If neither, no auth required.
    """
    skip = skip_paths or {"/health", "/v1/queue/stats"}

    async def middleware(request: Request, call_next):
        # Skip auth for certain paths
        if request.url.path in skip:
            return await call_next(request)

        # No auth configured
        if key_store is None and single_api_key is None:
            return await call_next(request)

        # Extract token
        auth_header = request.headers.get("authorization", "")
        if not auth_header.startswith("Bearer "):
            return JSONResponse(
                status_code=401,
                content={"error": {
                    "message": "Missing or invalid API key. Provide via: Authorization: Bearer <key>",
                    "type": "authentication_error",
                    "code": "invalid_api_key",
                }},
            )
        token = auth_header[7:]

        # Multi-key auth
        if key_store is not None:
            key_info = key_store.validate_key(token)
            if key_info is None:
                # Fall through to single-key check if configured
                if single_api_key and token == single_api_key:
                    request.state.key_info = None
                    request.state.priority = 0
                    return await call_next(request)
                return JSONResponse(
                    status_code=401,
                    content={"error": {
                        "message": "Invalid API key.",
                        "type": "authentication_error",
                        "code": "invalid_api_key",
                    }},
                )

            # Check rate limit
            rate_error = key_store.check_rate_limit(key_info)
            if rate_error:
                return JSONResponse(
                    status_code=429,
                    content={"error": {
                        "message": rate_error,
                        "type": "rate_limit_error",
                        "code": "rate_limit_exceeded",
                    }},
                    headers={"Retry-After": "60"},
                )

            # Attach key info to request state
            request.state.key_info = key_info
            request.state.priority = key_info.priority

            # Log usage for rate limiting
            key_store.log_usage(
                key_hash=key_info.key_hash,
                endpoint=request.url.path,
            )

            return await call_next(request)

        # Single-key auth (backward compatible)
        if token != single_api_key:
            return JSONResponse(
                status_code=401,
                content={"error": {
                    "message": "Invalid API key.",
                    "type": "authentication_error",
                    "code": "invalid_api_key",
                }},
            )

        request.state.key_info = None
        request.state.priority = 0
        return await call_next(request)

    return middleware
