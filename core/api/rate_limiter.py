from __future__ import annotations

import time
import logging

import redis
from fastapi import HTTPException, Request

from core.env import settings
from core.redis_client import get_text_client

LOGGER = logging.getLogger(__name__)

# Lua script: atomically trim both sorted sets, check limits,
# add current timestamp + set expiry if under limit, return pass/fail.
# Returns 1 if allowed, 0 if rate-limited.
_RATE_LIMIT_SCRIPT = """
local short_key = KEYS[1]
local long_key = KEYS[2]
local short_window = tonumber(ARGV[1])
local long_window = tonumber(ARGV[2])
local short_max = tonumber(ARGV[3])
local long_max = tonumber(ARGV[4])
local now = tonumber(ARGV[5])
local short_ttl = tonumber(ARGV[6])
local long_ttl = tonumber(ARGV[7])

redis.call("ZREMRANGEBYSCORE", short_key, "-inf", now - short_window)
redis.call("ZREMRANGEBYSCORE", long_key, "-inf", now - long_window)

local short_count = redis.call("ZCARD", short_key) or 0
local long_count = redis.call("ZCARD", long_key) or 0

if short_count >= short_max or long_count >= long_max then
    return 0
end

redis.call("ZADD", short_key, now, now)
redis.call("ZADD", long_key, now, now)
redis.call("EXPIRE", short_key, short_ttl)
redis.call("EXPIRE", long_key, long_ttl)

return 1
"""


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
    """Dual-window sliding-window rate limiter backed by Redis.

    Uses a single Lua script (registered via SCRIPT LOAD / EVALSHA) to
    atomically trim, check, and update both windows in 1 round-trip.
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
        self._script_sha: str | None = None

    def check(self, ip: str) -> None:
        now = time.time()
        short_key = f"{self._prefix}{ip}:short"
        long_key = f"{self._prefix}{ip}:long"

        if self._script_sha is None:
            self._script_sha = self._client.script_load(_RATE_LIMIT_SCRIPT)

        allowed = self._client.evalsha(
            self._script_sha,
            2,
            short_key,
            long_key,
            self._short_window,
            self._long_window,
            self._short_max,
            self._long_max,
            now,
            self._short_window * 2,
            self._long_window * 2,
        )

        if allowed == 0:
            LOGGER.warning(
                "Rate limit exceeded for %s: short_max=%d long_max=%d",
                ip,
                self._short_max,
                self._long_max,
            )
            raise RateLimitExceeded(
                "Rate limit exceeded. Usage is limited due to the application's "
                "nature as portfolio."
            )


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
