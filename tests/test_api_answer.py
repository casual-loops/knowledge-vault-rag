from contextlib import contextmanager
from types import SimpleNamespace

from fastapi.testclient import TestClient

import knowledge_rag.api.main as api_main
from knowledge_rag.api.main import app


client = TestClient(app)


@contextmanager
def fake_connection():
    yield SimpleNamespace(
        execute=lambda *args, **kwargs: None,
    )


class FakeGenerationProvider:
    @property
    def is_external(self) -> bool:
        return False

    def generate(
        self,
        *,
        prompt: str,
    ) -> str:
        return "Synthetic grounded answer"


def fake_search(*args, **kwargs):
    return [
        SimpleNamespace(
            source_path="Reference.md",
            title="Reference",
            note_type="reference",
            topic="demo",
            ai_access="allowed",
            chunk_index=0,
            heading_path="Reference",
            content="Synthetic source content.",
            distance=0.12,
        )
    ]


def test_answer_returns_grounded_response(monkeypatch) -> None:
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
        "get_generation_provider",
        lambda: FakeGenerationProvider(),
    )

    monkeypatch.setattr(
        api_main,
        "semantic_search",
        fake_search,
    )

    response = client.post(
        "/answer",
        json={
            "query": "Synthetic query",
            "top_k": 5,
        },
    )

    assert response.status_code == 200

    body = response.json()

    assert body["answer"] == "Synthetic grounded answer"

    assert body["citations"] == [
        {
            "citation_id": "S1",
            "source_path": "Reference.md",
            "title": "Reference",
            "heading_path": "Reference",
            "chunk_index": 0,
        }
    ]

    assert body["sources"] == [
        {
            "source_path": "Reference.md",
            "title": "Reference",
            "heading_path": "Reference",
            "chunk_index": 0,
            "content": "Synthetic source content.",
            "ai_access": "allowed",
        }
    ]


def test_answer_preserves_query_controls(monkeypatch) -> None:
    captured: dict[str, object] = {}

    def capturing_search(
        conn,
        provider,
        *,
        query,
        limit,
        note_type,
        topic,
    ):
        captured.update(
            {
                "query": query,
                "limit": limit,
                "note_type": note_type,
                "topic": topic,
            }
        )

        return []

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
        "get_generation_provider",
        lambda: FakeGenerationProvider(),
    )

    monkeypatch.setattr(
        api_main,
        "semantic_search",
        capturing_search,
    )

    response = client.post(
        "/answer",
        json={
            "query": "Synthetic query",
            "top_k": 7,
            "note_type": "reference",
            "topic": "demo",
        },
    )

    assert response.status_code == 200

    assert captured == {
        "query": "Synthetic query",
        "limit": 7,
        "note_type": "reference",
        "topic": "demo",
    }


def test_answer_handles_service_failure(monkeypatch) -> None:
    def failing_provider():
        raise RuntimeError("generation unavailable")

    monkeypatch.setattr(
        api_main,
        "get_generation_provider",
        failing_provider,
    )

    response = client.post(
        "/answer",
        json={
            "query": "Synthetic query",
        },
    )

    assert response.status_code == 503
    assert response.json() == {
        "detail": "Answer service temporarily unavailable."
    }


def test_answer_rejects_invalid_request() -> None:
    response = client.post(
        "/answer",
        json={
            "query": "",
        },
    )

    assert response.status_code == 422