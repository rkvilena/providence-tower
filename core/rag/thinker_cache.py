from __future__ import annotations

import json
import logging

from core.env import settings
from core.rag.schema import ThinkerState
from core.redis_client import get_text_client

LOGGER = logging.getLogger(__name__)

_KEY_PREFIX = "rag:thinker:cache:"


class ThinkerCache:
    """Redis-backed cache for Thinker LLM results.

    Keys are deterministic fingerprints (see ``ThinkerNode._cache_key``) that
    already include the query, chunk IDs, model, prompt version, and context
    budget config — so any change to those inputs naturally misses and
    re-populates, providing invalidation without explicit purges.
    """

    def __init__(self, client=None) -> None:
        if client is not None:
            self.client = client
        else:
            self.client = get_text_client()

    def get(self, key: str) -> ThinkerState | None:
        raw = self.client.get(_KEY_PREFIX + key)
        if not raw:
            return None
        try:
            return ThinkerState.model_validate_json(raw)
        except Exception as exc:
            LOGGER.warning("Thinker cache entry invalid, ignoring: %s", exc)
            return None

    def put(self, key: str, thinker_state: ThinkerState) -> None:
        payload = thinker_state.model_dump_json()
        ttl = settings.THINKER_CACHE_TTL_SECONDS
        if ttl > 0:
            self.client.set(_KEY_PREFIX + key, payload, ex=ttl)
        else:
            self.client.set(_KEY_PREFIX + key, payload)
