from __future__ import annotations

import json
import secrets
import string

import redis

from core.env import settings
from core.rag.schema import HistoryTurn


_NANOID_ALPHABET = string.ascii_letters + string.digits


def generate_session_id(*, length: int = 12) -> str:
    if length < 1:
        raise ValueError("length must be >= 1")
    return "".join(secrets.choice(_NANOID_ALPHABET) for _ in range(length))


class RedisHistoryStore:
    def __init__(
        self,
        *,
        host: str = settings.REDIS_HOST,
        port: int = settings.REDIS_PORT,
        db: int = settings.REDIS_DB,
        password: str | None = settings.REDIS_PASSWORD or None,
        key_prefix: str = "rag:session:",
    ) -> None:
        self.key_prefix = key_prefix
        self.client = redis.Redis(
            host=host,
            port=port,
            db=db,
            password=password,
            decode_responses=True,
        )

    def ping(self) -> bool:
        return bool(self.client.ping())

    def load_history(self, session_id: str) -> list[HistoryTurn]:
        key = self._history_key(session_id)
        raw_items = self.client.lrange(key, 0, -1)
        history: list[HistoryTurn] = []
        for raw in raw_items:
            try:
                obj = json.loads(raw)
            except Exception:
                continue
            if not isinstance(obj, dict):
                continue
            q = str(obj.get("q", "")).strip()
            a = str(obj.get("a", "")).strip()
            if not q and not a:
                continue
            history.append(HistoryTurn(q=q, a=a))
        return history

    def append_turn(
        self,
        session_id: str,
        *,
        question: str,
        answer: str,
        history_window: int,
        ttl_seconds: int,
    ) -> None:
        if history_window < 1:
            raise ValueError("history_window must be >= 1")
        if ttl_seconds < 1:
            raise ValueError("ttl_seconds must be >= 1")

        key = self._history_key(session_id)
        payload = json.dumps({"q": str(question), "a": str(answer)}, ensure_ascii=False)
        pipeline = self.client.pipeline(transaction=False)
        pipeline.rpush(key, payload)
        pipeline.ltrim(key, -history_window, -1)
        pipeline.expire(key, ttl_seconds)
        pipeline.execute()

    def clear_history(self, session_id: str) -> None:
        self.client.delete(self._history_key(session_id))

    def _history_key(self, session_id: str) -> str:
        return f"{self.key_prefix}{session_id}:history"
