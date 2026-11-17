"""Unit tests for the shared Cloudflare Workers AI REST client.

Run from the project root:
    python -m pytest test/unit -q
"""

from __future__ import annotations

import pytest

from core.inference.cloudflare_workers_ai import (
    CLOUDFLARE_EMBEDDING_MODEL_ID,
    CLOUDFLARE_RERANK_MODEL_ID,
    EMBEDDING_DIMENSIONS,
    CloudflareAPIError,
    CloudflareConfigurationError,
    CloudflareWorkersAIClient,
)

TOKEN = "super-secret-cloudflare-token"


def embedding_vector(value: float = 0.01) -> list[float]:
    return [value] * EMBEDDING_DIMENSIONS


class FakeResponse:
    def __init__(self, body, status_code: int = 200) -> None:
        self._body = body
        self.status_code = status_code

    def json(self):
        return self._body


class FakeSession:
    def __init__(self, response: FakeResponse) -> None:
        self.response = response
        self.calls: list[dict] = []

    def post(self, url, json=None, headers=None, timeout=None):
        self.calls.append(
            {"url": url, "json": json, "headers": headers, "timeout": timeout}
        )
        return self.response


def make_client(response: FakeResponse, **kwargs) -> CloudflareWorkersAIClient:
    return CloudflareWorkersAIClient(
        account_id=kwargs.pop("account_id", "acct-123"),
        auth_token=kwargs.pop("auth_token", TOKEN),
        session=FakeSession(response),
        **kwargs,
    )


def test_embed_query_sends_documented_request():
    session = FakeSession(
        FakeResponse(
            {
                "result": {"shape": [1, EMBEDDING_DIMENSIONS], "data": [embedding_vector()]},
                "success": True,
                "errors": [],
                "messages": [],
            }
        )
    )
    client = CloudflareWorkersAIClient(
        account_id="acct-123", auth_token=TOKEN, session=session
    )

    vector = client.embed_query("Who is Marco?")

    assert len(vector) == EMBEDDING_DIMENSIONS
    assert vector == embedding_vector()
    [call] = session.calls
    assert call["url"] == (
        "https://api.cloudflare.com/client/v4/accounts/acct-123/ai/run/"
        f"{CLOUDFLARE_EMBEDDING_MODEL_ID}"
    )
    assert call["json"] == {"text": ["Who is Marco?"], "pooling": "mean"}
    assert call["headers"]["Authorization"] == f"Bearer {TOKEN}"
    assert 0 < call["timeout"] <= 60


def test_embed_texts_returns_one_vector_per_text():
    session = FakeSession(
        FakeResponse(
            {
                "result": {
                    "shape": [2, EMBEDDING_DIMENSIONS],
                    "data": [embedding_vector(0.1), embedding_vector(0.2)],
                },
                "success": True,
                "errors": [],
            }
        )
    )
    client = CloudflareWorkersAIClient(
        account_id="acct-123", auth_token=TOKEN, session=session
    )

    vectors = client.embed_texts(["a", "b"])

    assert vectors == [embedding_vector(0.1), embedding_vector(0.2)]
    assert session.calls[0]["json"]["text"] == ["a", "b"]


@pytest.mark.parametrize(
    "data",
    [
        [embedding_vector(0.1)[:-1]],  # wrong dimensions
        "not-a-list",
        [[0.1, "bad"]],  # non-numeric
        [],
    ],
)
def test_embed_query_rejects_malformed_payloads(data):
    client = make_client(
        FakeResponse({"result": {"data": data}, "success": True, "errors": []})
    )

    with pytest.raises(CloudflareAPIError):
        client.embed_query("query")


def test_embed_query_rejects_vector_count_mismatch():
    client = make_client(
        FakeResponse(
            {
                "result": {"data": [embedding_vector(), embedding_vector()]},
                "success": True,
                "errors": [],
            }
        )
    )

    with pytest.raises(CloudflareAPIError):
        client.embed_query("query")


def test_missing_credentials_fail_at_call_time():
    # Constructing the client must stay safe when credentials are absent.
    client = CloudflareWorkersAIClient(
        account_id="",
        auth_token="",
        session=FakeSession(FakeResponse({})),
    )

    with pytest.raises(CloudflareConfigurationError, match="CLOUDFLARE_ACCOUNT_ID"):
        client.embed_query("query")


def test_api_error_envelope_is_actionable_and_hides_token():
    client = make_client(
        FakeResponse(
            {
                "result": None,
                "success": False,
                "errors": [{"code": 1000, "message": "Invalid API token"}],
            }
        )
    )

    with pytest.raises(CloudflareAPIError) as excinfo:
        client.embed_query("query")

    message = str(excinfo.value)
    assert "Invalid API token" in message
    assert TOKEN not in message


def test_http_error_includes_api_error_messages():
    client = make_client(
        FakeResponse(
            {"success": False, "errors": [{"message": "Authentication error"}]},
            status_code=401,
        )
    )

    with pytest.raises(CloudflareAPIError) as excinfo:
        client.embed_query("query")

    assert "401" in str(excinfo.value)
    assert "Authentication error" in str(excinfo.value)
    assert TOKEN not in str(excinfo.value)


def test_non_json_response_raises_api_error():
    class NonJsonResponse:
        status_code = 200

        def json(self):
            raise ValueError("not json")

    client = CloudflareWorkersAIClient(
        account_id="acct-123",
        auth_token=TOKEN,
        session=FakeSession(NonJsonResponse()),
    )

    with pytest.raises(CloudflareAPIError):
        client.embed_query("query")


def test_rerank_sends_documented_schema_and_maps_scores():
    session = FakeSession(
        FakeResponse(
            {
                "result": {
                    "response": [
                        {"id": 1, "score": 0.91},
                        {"id": 0, "score": 0.12},
                    ]
                },
                "success": True,
                "errors": [],
            }
        )
    )
    client = CloudflareWorkersAIClient(
        account_id="acct-123", auth_token=TOKEN, session=session
    )

    scores = client.rerank("query", ["first", "second"])

    [call] = session.calls
    assert call["url"] == (
        "https://api.cloudflare.com/client/v4/accounts/acct-123/ai/run/"
        f"{CLOUDFLARE_RERANK_MODEL_ID}"
    )
    assert call["json"] == {
        "query": "query",
        "contexts": [{"text": "first"}, {"text": "second"}],
    }
    assert [(score.index, score.score) for score in scores] == [(1, 0.91), (0, 0.12)]


def test_rerank_accepts_flat_result_list():
    client = make_client(
        FakeResponse(
            {
                "result": [{"id": 0, "score": 0.5}],
                "success": True,
                "errors": [],
            }
        )
    )

    assert [(score.index, score.score) for score in client.rerank("q", ["x"])] == [
        (0, 0.5)
    ]


def test_rerank_empty_contexts_skip_the_api_call():
    session = FakeSession(FakeResponse({}))
    client = CloudflareWorkersAIClient(
        account_id="acct-123", auth_token=TOKEN, session=session
    )

    assert client.rerank("query", []) == []
    assert session.calls == []


@pytest.mark.parametrize(
    "result",
    [
        [],
        [{"id": 5, "score": 0.5}],  # index outside the candidate list
        [{"id": 0, "score": "bad"}],
        [{"id": 0, "score": 0.5}, {"id": 0, "score": 0.7}],  # duplicate index
        [{"score": 0.5}],
    ],
)
def test_rerank_rejects_malformed_results(result):
    client = make_client(
        FakeResponse({"result": result, "success": True, "errors": []})
    )

    with pytest.raises(CloudflareAPIError):
        client.rerank("query", ["first", "second"])
