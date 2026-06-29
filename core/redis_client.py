from __future__ import annotations

import logging

import redis

from core.env import settings

LOGGER = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Global module-level pools (initialised once on first import)
# ---------------------------------------------------------------------------
# redis-py stores ``decode_responses`` on the ConnectionPool, so we need
# TWO pools – one for binary clients and one for text/string clients.
# ---------------------------------------------------------------------------

_binary_pool: redis.ConnectionPool | None = None
_text_pool: redis.ConnectionPool | None = None


def _build_pool(*, decode_responses: bool) -> redis.ConnectionPool:
    """Construct a ``ConnectionPool`` from application settings.

    Supports both local Redis Stack (host/port/db/password) and
    Upstash Redis (URL with ``rediss://`` scheme).
    """
    url: str | None = settings.UPSTASH_REDIS_URL or None

    if url:
        return redis.ConnectionPool.from_url(
            url,
            decode_responses=decode_responses,
            socket_keepalive=True,
            socket_connect_timeout=5,
            socket_timeout=10,
            retry_on_timeout=True,
            max_connections=32,
        )

    return redis.ConnectionPool(
        host=settings.REDIS_HOST,
        port=settings.REDIS_PORT,
        db=settings.REDIS_DB,
        password=settings.REDIS_PASSWORD or None,
        decode_responses=decode_responses,
        socket_keepalive=True,
        socket_connect_timeout=5,
        socket_timeout=10,
        retry_on_timeout=True,
        max_connections=32,
    )


def _ensure_pools() -> None:
    global _binary_pool, _text_pool
    if _binary_pool is None:
        _binary_pool = _build_pool(decode_responses=False)
    if _text_pool is None:
        _text_pool = _build_pool(decode_responses=True)


def get_binary_client() -> redis.Redis:
    """Return a Redis client with ``decode_responses=False``.

    Intended for the vector store (raw embedding bytes).
    """
    _ensure_pools()
    return redis.Redis(connection_pool=_binary_pool)


def get_text_client() -> redis.Redis:
    """Return a Redis client with ``decode_responses=True``.

    Intended for the rate limiter, chat history, and general KV ops.
    """
    _ensure_pools()
    return redis.Redis(connection_pool=_text_pool)


def ping_binary() -> bool:
    """Check connectivity using the binary client pool."""
    try:
        return bool(get_binary_client().ping())
    except Exception as exc:
        LOGGER.warning("Redis binary client ping failed: %s", exc)
        return False


def ping_text() -> bool:
    """Check connectivity using the text client pool."""
    try:
        return bool(get_text_client().ping())
    except Exception as exc:
        LOGGER.warning("Redis text client ping failed: %s", exc)
        return False
