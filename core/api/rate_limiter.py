from __future__ import annotations

import time
import logging

import redis
from fastapi import HTTPException, Request

from core.env import settings

LOGGER = logging.getLogger(__name__)


class RateLimitExceeded(HTTPException):
    """Raised when a client has exceeded the allowed request rate."""

    def __init__(self, detail: str = "Rate limit exceeded") -> None:
        super().__init__(status_code=429, detail=detail)


def _client_ip(request: Request) -> str:
    """Extract the client IP from the request, respecting common proxies."""
    forwarded = request.headers.get("X-Forwarded-For")
    if forwarded:
        # Take the first IP in the chain
        return forwarded.split(",")[0].strip()
    # Fall back to direct remote address
    client = request.client
    return client.host if client is not None else "unknown"


class RedisRateLimiter:
    """Dual-window sliding-window rate limiter backed by Redis.

    Two windows are tracked per client IP:
      - Short window (e.g. 5 requests / 60 seconds)
      - Long window  (e.g. 30 requests / 86400 seconds = 1 day)

    Uses sorted sets in Redis with timestamps as scores so old entries
    can be trimmed efficiently.
    """

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

    # ---- public API ----------------------------------------------------------

    def check(self, ip: str) -> None:
        """Increment counters and raise ``RateLimitExceeded`` if either window
        is over the limit.

        Must be called synchronously inside a FastAPI endpoint or dependency.
        """
        now = time.time()

        short_key = f"{self._prefix}{ip}:short"
        long_key = f"{self._prefix}{ip}:long"

        # Trim old entries (sliding window cleanup)
        self._trim(short_key, now - self._short_window)
        self._trim(long_key, now - self._long_window)

        # Count current requests in each window
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

        # Record this request in both windows
        pipeline = self._client.pipeline(transaction=False)
        pipeline.zadd(short_key, {str(now): now})
        pipeline.zadd(long_key, {str(now): now})
        pipeline.expire(short_key, self._short_window * 2)  # safety TTL
        pipeline.expire(long_key, self._long_window * 2)
        pipeline.execute()

    # ---- internals -----------------------------------------------------------

    def _trim(self, key: str, cutoff: float) -> None:
        """Remove entries older than *cutoff* from the sorted set."""
        self._client.zremrangebyscore(key, "-inf", cutoff)


# ---- FastAPI dependency factory ----------------------------------------------


def get_redis_rate_limiter(request: Request) -> RedisRateLimiter:
    """FastAPI dependency that builds a ``RedisRateLimiter`` and checks the
    caller's IP.

    Usage::

        @router.post("/chat")
        async def chat(
            body: ChatRequest,
            svc: RagService = Depends(get_rag_service),
            _: None = Depends(get_redis_rate_limiter),
        ) -> ChatResponse: ...
    """
    # Build a lightweight Redis client just for rate limiting.
    # We don't reuse RagService's store because rate limiting has a
    # different key space and should never block the main RAG pipeline.
    client = redis.Redis(
        host=settings.REDIS_HOST,
        port=settings.REDIS_PORT,
        db=settings.REDIS_DB,
        password=settings.REDIS_PASSWORD or None,
        decode_responses=True,
    )
    try:
        client.ping()
    except Exception:
        LOGGER.warning("Redis unreachable – rate limiter bypassed.")
        return None  # type: ignore[return-value]

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
