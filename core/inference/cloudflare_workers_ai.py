"""Cloudflare Workers AI REST client for RAG-time inference.

Only query embeddings and reranking run in the cloud. Corpus preparation keeps
using the local BGE model, so this module must not import PyTorch,
SentenceTransformers, or any other local model dependency.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any, Sequence

import requests

from core.env import settings

CLOUDFLARE_API_BASE = "https://api.cloudflare.com/client/v4/accounts"
CLOUDFLARE_EMBEDDING_MODEL_ID = "@cf/baai/bge-small-en-v1.5"
CLOUDFLARE_RERANK_MODEL_ID = "@cf/baai/bge-reranker-base"
EMBEDDING_DIMENSIONS = 384
DEFAULT_TIMEOUT_SECONDS = 30.0
_MAX_ERROR_DETAIL_CHARS = 500


class CloudflareWorkersAIError(RuntimeError):
    """Base error for Cloudflare Workers AI inference failures."""


class CloudflareConfigurationError(CloudflareWorkersAIError):
    """Raised when the Cloudflare account credentials are missing."""


class CloudflareAPIError(CloudflareWorkersAIError):
    """Raised when Cloudflare rejects a request or returns invalid data."""


@dataclass(frozen=True)
class RerankScore:
    """Score assigned to one candidate text, identified by its input index."""

    index: int
    score: float


def _require_setting(value: str | None, env_name: str) -> str:
    if value is None or not str(value).strip():
        raise CloudflareConfigurationError(
            "Cloudflare Workers AI is not configured: set "
            f"{env_name} in the environment before using RAG inference."
        )
    return str(value).strip()


def _as_float_list(vector: Any, *, model_id: str) -> list[float]:
    if not isinstance(vector, list) or not vector:
        raise CloudflareAPIError(
            f"Cloudflare Workers AI model {model_id} returned an empty or "
            "non-list embedding vector."
        )
    try:
        values = [float(value) for value in vector]
    except (TypeError, ValueError) as exc:
        raise CloudflareAPIError(
            f"Cloudflare Workers AI model {model_id} returned a non-numeric "
            "embedding vector."
        ) from exc
    if any(not math.isfinite(value) for value in values):
        raise CloudflareAPIError(
            f"Cloudflare Workers AI model {model_id} returned a non-finite "
            "embedding vector."
        )
    return values


class CloudflareWorkersAIClient:
    """Account-scoped client for the documented Cloudflare Workers AI REST API."""

    def __init__(
        self,
        *,
        account_id: str | None = None,
        auth_token: str | None = None,
        timeout: float = DEFAULT_TIMEOUT_SECONDS,
        session: requests.Session | None = None,
    ) -> None:
        self._account_id = (
            settings.CLOUDFLARE_ACCOUNT_ID if account_id is None else account_id
        )
        self._auth_token = (
            settings.CLOUDFLARE_AUTH_TOKEN if auth_token is None else auth_token
        )
        self._timeout = timeout
        # Credentials stay optional until a call is made so tests and non-RAG
        # startup paths never need a live Cloudflare connection.
        self._session = session if session is not None else requests.Session()

    def embed_query(self, text: str) -> list[float]:
        """Return the mean-pooled BGE embedding for a single query text."""
        return self.embed_texts([text])[0]

    def embed_texts(self, texts: Sequence[str]) -> list[list[float]]:
        """Embed one or more texts with BGE small and validate the payload."""
        items = list(texts)
        if not items:
            return []

        result = self._run(
            CLOUDFLARE_EMBEDDING_MODEL_ID,
            {"text": items, "pooling": "mean"},
        )
        data = result.get("data") if isinstance(result, dict) else None
        if not isinstance(data, list) or len(data) != len(items):
            raise CloudflareAPIError(
                "Cloudflare Workers AI embedding response did not contain one "
                f"vector per input text ({len(items)} expected)."
            )

        vectors: list[list[float]] = []
        for vector in data:
            values = _as_float_list(
                vector, model_id=CLOUDFLARE_EMBEDDING_MODEL_ID
            )
            if len(values) != EMBEDDING_DIMENSIONS:
                raise CloudflareAPIError(
                    "Cloudflare Workers AI embedding dimension mismatch: "
                    f"expected {EMBEDDING_DIMENSIONS}, got {len(values)}. "
                    "The local BGE corpus and the Cloudflare model must share "
                    "the same embedding space."
                )
            vectors.append(values)
        return vectors

    def rerank(self, query: str, texts: Sequence[str]) -> list[RerankScore]:
        """Rerank candidate texts for a query using the BGE reranker model."""
        items = list(texts)
        if not items:
            return []

        result = self._run(
            CLOUDFLARE_RERANK_MODEL_ID,
            {
                "query": query,
                "contexts": [{"text": text} for text in items],
            },
        )
        entries = result
        if isinstance(result, dict):
            entries = (
                result.get("response")
                or result.get("results")
                or result.get("matches")
                or result.get("data")
            )
        if not isinstance(entries, list) or not entries:
            raise CloudflareAPIError(
                "Cloudflare Workers AI rerank response did not contain a "
                "non-empty result list."
            )

        scores: list[RerankScore] = []
        seen: set[int] = set()
        for entry in entries:
            if not isinstance(entry, dict):
                raise CloudflareAPIError(
                    "Cloudflare Workers AI rerank response contained a "
                    "malformed context entry."
                )
            raw_index = entry.get("id", entry.get("index"))
            raw_score = entry.get("score")
            if isinstance(raw_index, bool) or not isinstance(raw_index, int):
                raise CloudflareAPIError(
                    "Cloudflare Workers AI rerank response contained a "
                    "context entry without a valid index."
                )
            if raw_index < 0 or raw_index >= len(items) or raw_index in seen:
                raise CloudflareAPIError(
                    "Cloudflare Workers AI rerank response referenced an "
                    f"unknown or duplicate context index {raw_index}."
                )
            try:
                score = float(raw_score)
            except (TypeError, ValueError) as exc:
                raise CloudflareAPIError(
                    "Cloudflare Workers AI rerank response contained a "
                    "non-numeric score."
                ) from exc
            if not math.isfinite(score):
                raise CloudflareAPIError(
                    "Cloudflare Workers AI rerank response contained a "
                    "non-finite score."
                )
            seen.add(raw_index)
            scores.append(RerankScore(index=raw_index, score=score))
        return scores

    def _run(self, model_id: str, payload: dict[str, Any]) -> Any:
        account_id = _require_setting(
            None if not self._account_id else self._account_id,
            "CLOUDFLARE_ACCOUNT_ID",
        )
        auth_token = _require_setting(
            None if not self._auth_token else self._auth_token,
            "CLOUDFLARE_AUTH_TOKEN",
        )
        url = f"{CLOUDFLARE_API_BASE}/{account_id}/ai/run/{model_id}"

        try:
            response = self._session.post(
                url,
                json=payload,
                headers={
                    "Authorization": f"Bearer {auth_token}",
                    "Content-Type": "application/json",
                },
                timeout=self._timeout,
            )
        except requests.RequestException as exc:
            raise CloudflareAPIError(
                f"Cloudflare Workers AI request for {model_id} failed: "
                f"{type(exc).__name__}: {exc}"
            ) from exc

        if response.status_code >= 400:
            detail = self._error_detail(response)
            raise CloudflareAPIError(
                f"Cloudflare Workers AI returned HTTP {response.status_code} "
                f"for {model_id}: {detail}"
            )

        try:
            body = response.json()
        except ValueError as exc:
            raise CloudflareAPIError(
                f"Cloudflare Workers AI returned a non-JSON response "
                f"(HTTP {response.status_code}) for {model_id}."
            ) from exc

        return self._unwrap_envelope(body, model_id=model_id)

    @staticmethod
    def _unwrap_envelope(body: Any, *, model_id: str) -> Any:
        if not isinstance(body, dict):
            raise CloudflareAPIError(
                f"Cloudflare Workers AI returned an unexpected response "
                f"envelope for {model_id}."
            )
        errors = body.get("errors")
        if errors:
            raise CloudflareAPIError(
                f"Cloudflare Workers AI reported an error for {model_id}: "
                f"{CloudflareWorkersAIClient._format_messages(errors)}"
            )
        if body.get("success") is not True:
            raise CloudflareAPIError(
                f"Cloudflare Workers AI reported an unsuccessful response "
                f"for {model_id}."
            )
        if "result" not in body or body["result"] is None:
            raise CloudflareAPIError(
                f"Cloudflare Workers AI response for {model_id} did not "
                "include a result payload."
            )
        return body["result"]

    @staticmethod
    def _error_detail(response: Any) -> str:
        try:
            body = response.json()
        except ValueError:
            return f"HTTP response without a JSON error body (status {response.status_code})."
        if isinstance(body, dict) and body.get("errors"):
            return CloudflareWorkersAIClient._format_messages(body["errors"])
        return f"unexpected error response (status {response.status_code})."

    @staticmethod
    def _format_messages(errors: Any) -> str:
        if isinstance(errors, list):
            messages = []
            for error in errors:
                if isinstance(error, dict) and error.get("message"):
                    messages.append(str(error["message"]))
                elif isinstance(error, str):
                    messages.append(error)
            if messages:
                text = "; ".join(messages)
                return text[:_MAX_ERROR_DETAIL_CHARS]
        return str(errors)[:_MAX_ERROR_DETAIL_CHARS]
