from contextlib import contextmanager
from types import SimpleNamespace

from fastapi.testclient import TestClient

import knowledge_rag.api.main as api_main
from knowledge_rag.api.main import app
from knowledge_rag.generation import GenerationProvider
from knowledge_rag.grounded_generation import generate_grounded_answer
from knowledge_rag.retrieval_policy import RetrievedChunk


client = TestClient(app)


class ExternalFakeProvider(GenerationProvider):
    def __init__(self) -> None:
        self.prompts: list[str] = []

    @property
    def is_external(self) -> bool:
        return True

    def generate(
        self,
        *,
        prompt: str,
    ) -> str:
        self.prompts.append(prompt)
        return "Synthetic external answer"


class FailingGenerationProvider(GenerationProvider):
    @property
    def is_external(self) -> bool:
        return False

    def generate(
        self,
        *,
        prompt: str,
    ) -> str:
        raise RuntimeError(
            "provider failed with secret=test-secret "
            "and note=Sensitive private note content"
        )


@contextmanager
def fake_connection():
    yield SimpleNamespace(
        execute=lambda *args, **kwargs: None,
    )


def test_empty_retrieval_context_returns_empty_grounded_result() -> None:
    provider = ExternalFakeProvider()

    result = generate_grounded_answer(
        query="Synthetic query",
        chunks=[],
        provider=provider,
    )

    assert result.answer == ""
    assert result.sources == []
    assert result.citations == []
    assert provider.prompts == []


def test_mixed_context_sends_only_allowed_content_external() -> None:
    provider = ExternalFakeProvider()

    chunks = [
        RetrievedChunk(
            source_path="Allowed.md",
            content="Allowed synthetic content.",
            ai_access="allowed",
            metadata={
                "title": "Allowed",
                "heading_path": "Allowed",
                "chunk_index": 0,
            },
        ),
        RetrievedChunk(
            source_path="Local.md",
            content="Sensitive local-only content.",
            ai_access="local-only",
            metadata={
                "title": "Local",
                "heading_path": "Local",
                "chunk_index": 0,
            },
        ),
    ]

    result = generate_grounded_answer(
        query="Synthetic query",
        chunks=chunks,
        provider=provider,
    )

    assert result.answer == "Synthetic external answer"
    assert len(provider.prompts) == 1

    prompt = provider.prompts[0]

    assert "Allowed synthetic content." in prompt
    assert "Sensitive local-only content." not in prompt

    assert [source.source_path for source in result.sources] == [
        "Allowed.md"
    ]

    assert [citation.source_path for citation in result.citations] == [
        "Allowed.md"
    ]


def test_answer_endpoint_returns_empty_result_for_empty_retrieval(
    monkeypatch,
) -> None:
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
        lambda: ExternalFakeProvider(),
    )

    monkeypatch.setattr(
        api_main,
        "semantic_search",
        lambda *args, **kwargs: [],
    )

    response = client.post(
        "/answer",
        json={
            "query": "Synthetic query",
        },
    )

    assert response.status_code == 200
    assert response.json() == {
        "answer": "",
        "citations": [],
        "sources": [],
    }


def test_answer_provider_failure_does_not_expose_sensitive_details(
    monkeypatch,
) -> None:
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
        lambda: FailingGenerationProvider(),
    )

    monkeypatch.setattr(
        api_main,
        "semantic_search",
        lambda *args, **kwargs: [
            SimpleNamespace(
                source_path="Sensitive.md",
                title="Sensitive",
                note_type="reference",
                topic="private",
                ai_access="allowed",
                chunk_index=0,
                heading_path="Sensitive",
                content="Sensitive private note content",
                distance=0.1,
            )
        ],
    )

    response = client.post(
        "/answer",
        json={
            "query": "Synthetic query",
        },
    )

    assert response.status_code == 503

    body = response.json()

    assert body == {
        "detail": "Answer service temporarily unavailable."
    }

    response_text = response.text

    assert "test-secret" not in response_text
    assert "Sensitive private note content" not in response_text