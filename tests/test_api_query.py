from contextlib import contextmanager
from types import SimpleNamespace

from fastapi.testclient import TestClient

import knowledge_rag.api.main as api_main
from knowledge_rag.api.main import app
from knowledge_rag.retrieval import SearchResult


client = TestClient(app)


@contextmanager
def fake_connection():
    """Provide a harmless stand-in for the database connection context."""

    yield SimpleNamespace(
        execute=lambda *args, **kwargs: None,
    )


def fake_search_results() -> list[SearchResult]:
    """Return deterministic synthetic retrieval results."""

    return [
        SearchResult(
            source_path="Reference One.md",
            title="Reference One",
            note_type="reference",
            topic=["privacy-demo"],
            ai_access="allowed",
            chunk_index=0,
            heading_path="Reference One",
            content="Synthetic reference content.",
            distance=0.1,
        ),
        SearchResult(
            source_path="Study One.md",
            title="Study One",
            note_type="study",
            topic=["azure"],
            ai_access="allowed",
            chunk_index=0,
            heading_path="Study One",
            content="Synthetic study content.",
            distance=0.2,
        ),
    ]


def test_query_returns_semantic_results(monkeypatch) -> None:
    monkeypatch.setattr(
        api_main,
        "get_connection",
        fake_connection,
    )

    monkeypatch.setattr(
        api_main,
        "get_embedding_provider",
        lambda: SimpleNamespace(),
    )

    monkeypatch.setattr(
        api_main,
        "semantic_search",
        lambda *args, **kwargs: fake_search_results(),
    )

    response = client.post(
        "/query",
        json={
            "query": "privacy",
        },
    )

    assert response.status_code == 200

    body = response.json()

    assert body["results"][0]["source_path"] == "Reference One.md"
    assert body["results"][0]["title"] == "Reference One"
    assert body["results"][0]["ai_access"] == "allowed"
    assert body["results"][0]["content"] == "Synthetic reference content."


def test_query_passes_top_k_to_retrieval(monkeypatch) -> None:
    captured: dict[str, object] = {}

    def fake_semantic_search(*args, **kwargs):
        captured.update(kwargs)
        return fake_search_results()[:1]

    monkeypatch.setattr(
        api_main,
        "get_connection",
        fake_connection,
    )

    monkeypatch.setattr(
        api_main,
        "get_embedding_provider",
        lambda: SimpleNamespace(),
    )

    monkeypatch.setattr(
        api_main,
        "semantic_search",
        fake_semantic_search,
    )

    response = client.post(
        "/query",
        json={
            "query": "privacy",
            "top_k": 1,
        },
    )

    assert response.status_code == 200
    assert captured["limit"] == 1


def test_query_passes_note_type_filter(monkeypatch) -> None:
    captured: dict[str, object] = {}

    def fake_semantic_search(*args, **kwargs):
        captured.update(kwargs)

        return [
            result
            for result in fake_search_results()
            if result.note_type == "reference"
        ]

    monkeypatch.setattr(
        api_main,
        "get_connection",
        fake_connection,
    )

    monkeypatch.setattr(
        api_main,
        "get_embedding_provider",
        lambda: SimpleNamespace(),
    )

    monkeypatch.setattr(
        api_main,
        "semantic_search",
        fake_semantic_search,
    )

    response = client.post(
        "/query",
        json={
            "query": "privacy",
            "note_type": "reference",
        },
    )

    assert response.status_code == 200
    assert captured["note_type"] == "reference"

    body = response.json()

    assert all(
        result["note_type"] == "reference"
        for result in body["results"]
    )


def test_query_passes_topic_filter(monkeypatch) -> None:
    captured: dict[str, object] = {}

    def fake_semantic_search(*args, **kwargs):
        captured.update(kwargs)

        return [
            result
            for result in fake_search_results()
            if "privacy-demo" in result.topic
        ]

    monkeypatch.setattr(
        api_main,
        "get_connection",
        fake_connection,
    )

    monkeypatch.setattr(
        api_main,
        "get_embedding_provider",
        lambda: SimpleNamespace(),
    )

    monkeypatch.setattr(
        api_main,
        "semantic_search",
        fake_semantic_search,
    )

    response = client.post(
        "/query",
        json={
            "query": "privacy",
            "topic": "privacy-demo",
        },
    )

    assert response.status_code == 200
    assert captured["topic"] == "privacy-demo"

    body = response.json()

    assert len(body["results"]) == 1
    assert body["results"][0]["source_path"] == "Reference One.md"


def test_query_returns_empty_results(monkeypatch) -> None:
    monkeypatch.setattr(
        api_main,
        "get_connection",
        fake_connection,
    )

    monkeypatch.setattr(
        api_main,
        "get_embedding_provider",
        lambda: SimpleNamespace(),
    )

    monkeypatch.setattr(
        api_main,
        "semantic_search",
        lambda *args, **kwargs: [],
    )

    response = client.post(
        "/query",
        json={
            "query": "nothing matches",
        },
    )

    assert response.status_code == 200
    assert response.json() == {"results": []}


def test_query_rejects_missing_query() -> None:
    response = client.post(
        "/query",
        json={},
    )

    assert response.status_code == 422


def test_query_rejects_empty_query() -> None:
    response = client.post(
        "/query",
        json={
            "query": "",
        },
    )

    assert response.status_code == 422


def test_query_rejects_top_k_below_minimum() -> None:
    response = client.post(
        "/query",
        json={
            "query": "privacy",
            "top_k": 0,
        },
    )

    assert response.status_code == 422


def test_query_rejects_top_k_above_maximum() -> None:
    response = client.post(
        "/query",
        json={
            "query": "privacy",
            "top_k": 26,
        },
    )

    assert response.status_code == 422


def test_query_handles_database_failure(monkeypatch) -> None:
    def failing_connection():
        raise RuntimeError("database unavailable")

    monkeypatch.setattr(
        api_main,
        "get_connection",
        failing_connection,
    )

    response = client.post(
        "/query",
        json={
            "query": "privacy",
        },
    )

    assert response.status_code == 503
    assert response.json() == {
        "detail": "Query service temporarily unavailable."
    }


def test_query_handles_provider_failure(monkeypatch) -> None:
    def failing_provider():
        raise RuntimeError("provider failure")

    monkeypatch.setattr(
        api_main,
        "get_embedding_provider",
        failing_provider,
    )

    response = client.post(
        "/query",
        json={
            "query": "privacy",
        },
    )

    assert response.status_code == 503
    assert response.json() == {
        "detail": "Query service temporarily unavailable."
    }


def test_query_handles_retrieval_failure(monkeypatch) -> None:
    monkeypatch.setattr(
        api_main,
        "get_connection",
        fake_connection,
    )

    monkeypatch.setattr(
        api_main,
        "get_embedding_provider",
        lambda: SimpleNamespace(),
    )

    def failing_search(*args, **kwargs):
        raise RuntimeError("retrieval failure")

    monkeypatch.setattr(
        api_main,
        "semantic_search",
        failing_search,
    )

    response = client.post(
        "/query",
        json={
            "query": "privacy",
        },
    )

    assert response.status_code == 503
    assert response.json() == {
        "detail": "Query service temporarily unavailable."
    }


def test_health_reports_database_available(monkeypatch) -> None:
    monkeypatch.setattr(
        api_main,
        "get_connection",
        fake_connection,
    )

    response = client.get("/health")

    assert response.status_code == 200
    assert response.json() == {
        "status": "ok",
        "database": "ok",
    }


def test_health_handles_database_failure(monkeypatch) -> None:
    def failing_connection():
        raise RuntimeError("database unavailable")

    monkeypatch.setattr(
        api_main,
        "get_connection",
        failing_connection,
    )

    response = client.get("/health")

    assert response.status_code == 503
    assert response.json() == {
        "detail": "Service dependency unavailable."
    }