from __future__ import annotations

import time
import logging

import redis
from fastapi import HTTPException, Request

from core.env import settings
from core.redis_client import get_text_client

LOGGER = logging.getLogger(__name__)


class RateLimitExceeded(HTTPException):
    """Raised when a client has exceeded the allowed request rate."""

    def __init__(self, detail: str = "Rate limit exceeded") -> None:
        super().__init__(status_code=429, detail=detail)


def _client_ip(request: Request) -> str:
    """Extract the client IP from the request, respecting common proxies."""
    forwarded = request.headers.get("X-Forwarded-For")
    if forwarded:
        return forwarded.split(",")[0].strip()
    client = request.client
    return client.host if client is not None else "unknown"


class RedisRateLimiter:
    """Dual-window sliding-window rate limiter backed by Redis."""

    def __init__(
        self,
        client: redis.Redis,
        *,
        short_window_seconds: int = 60,
        short_max_requests: int = 5,
        long_window_seconds: int = 86400,
        long_max_requests: int = 30,
        key_prefix: str = "ratelimit:ip:",
    ) -> None:
        self._client = client
        self._short_window = short_window_seconds
        self._short_max = short_max_requests
        self._long_window = long_window_seconds
        self._long_max = long_max_requests
        self._prefix = key_prefix

    def check(self, ip: str) -> None:
        now = time.time()
        short_key = f"{self._prefix}{ip}:short"
        long_key = f"{self._prefix}{ip}:long"

        self._trim(short_key, now - self._short_window)
        self._trim(long_key, now - self._long_window)

        short_count = self._client.zcard(short_key) or 0
        long_count = self._client.zcard(long_key) or 0

        if short_count >= self._short_max or long_count >= self._long_max:
            LOGGER.warning(
                "Rate limit exceeded for %s: short=%d/%d long=%d/%d",
                ip,
                short_count,
                self._short_max,
                long_count,
                self._long_max,
            )
            raise RateLimitExceeded(
                "Rate limit exceeded. Usage is limited due to the application's "
                "nature as portfolio."
            )

        pipeline = self._client.pipeline(transaction=False)
        pipeline.zadd(short_key, {str(now): now})
        pipeline.zadd(long_key, {str(now): now})
        pipeline.expire(short_key, self._short_window * 2)
        pipeline.expire(long_key, self._long_window * 2)
        pipeline.execute()

    def _trim(self, key: str, cutoff: float) -> None:
        self._client.zremrangebyscore(key, "-inf", cutoff)


# ---- Lazily-initialised global rate-limiter client ---------------------------
# One client (from the shared text pool) for the lifetime of the process.
_rate_limiter_client: redis.Redis | None = None


def _get_rate_limiter_client() -> redis.Redis:
    global _rate_limiter_client
    if _rate_limiter_client is None:
        _rate_limiter_client = get_text_client()
    return _rate_limiter_client


# ---- FastAPI dependency factory ----------------------------------------------


def get_redis_rate_limiter(request: Request) -> RedisRateLimiter | None:
    """FastAPI dependency that builds a ``RedisRateLimiter`` and checks the
    caller's IP.

    Uses the shared text-mode client pool instead of creating a new
    connection per request.
    """
    try:
        client = _get_rate_limiter_client()
        limiter = RedisRateLimiter(
            client,
            short_window_seconds=settings.RATE_LIMIT_SHORT_WINDOW_SECONDS,
            short_max_requests=settings.RATE_LIMIT_SHORT_MAX_REQUESTS,
            long_window_seconds=settings.RATE_LIMIT_LONG_WINDOW_SECONDS,
            long_max_requests=settings.RATE_LIMIT_LONG_MAX_REQUESTS,
        )
        ip = _client_ip(request)
        limiter.check(ip)
        return limiter
    except RateLimitExceeded:
        raise
    except Exception:
        LOGGER.warning("Redis unreachable – rate limiter bypassed.")
        return None
