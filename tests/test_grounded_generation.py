from knowledge_rag.generation import GenerationProvider
from knowledge_rag.grounded_generation import (
    build_generation_context,
    generate_grounded_answer,
)
from knowledge_rag.retrieval_policy import RetrievedChunk


class FakeGenerationProvider(GenerationProvider):
    def __init__(self, *, is_external: bool) -> None:
        self._is_external = is_external
        self.prompts: list[str] = []

    @property
    def is_external(self) -> bool:
        return self._is_external

    def generate(
        self,
        *,
        prompt: str,
    ) -> str:
        self.prompts.append(prompt)
        return "Synthetic grounded answer"


def make_chunk(
    *,
    source_path: str,
    content: str,
    ai_access: str,
) -> RetrievedChunk:
    return RetrievedChunk(
        source_path=source_path,
        content=content,
        ai_access=ai_access,
        metadata={},
    )


def test_build_generation_context_includes_chunks() -> None:
    chunks = [
        make_chunk(
            source_path="Allowed One.md",
            content="First synthetic chunk.",
            ai_access="allowed",
        ),
        make_chunk(
            source_path="Allowed Two.md",
            content="Second synthetic chunk.",
            ai_access="allowed",
        ),
    ]

    context, selected_chunks = build_generation_context(chunks)

    assert "Source: Allowed One.md" in context
    assert "First synthetic chunk." in context
    assert "Source: Allowed Two.md" in context
    assert "Second synthetic chunk." in context
    assert selected_chunks == chunks


def test_build_generation_context_respects_character_limit() -> None:
    chunks = [
        make_chunk(
            source_path="One.md",
            content="A" * 100,
            ai_access="allowed",
        ),
        make_chunk(
            source_path="Two.md",
            content="B" * 100,
            ai_access="allowed",
        ),
    ]

    context, selected_chunks = build_generation_context(
        chunks,
        max_characters=130,
    )

    assert "One.md" in context
    assert "Two.md" not in context
    assert len(selected_chunks) == 1
    assert selected_chunks[0].source_path == "One.md"


def test_external_provider_filters_local_only_chunks() -> None:
    provider = FakeGenerationProvider(
        is_external=True,
    )

    chunks = [
        make_chunk(
            source_path="Allowed.md",
            content="Allowed synthetic content.",
            ai_access="allowed",
        ),
        make_chunk(
            source_path="Local.md",
            content="Local-only synthetic content.",
            ai_access="local-only",
        ),
    ]

    result = generate_grounded_answer(
        query="Synthetic query",
        chunks=chunks,
        provider=provider,
    )

    assert len(result.sources) == 1
    assert result.sources[0].source_path == "Allowed.md"

    prompt = provider.prompts[0]

    assert "Allowed synthetic content." in prompt
    assert "Local-only synthetic content." not in prompt


def test_local_provider_can_use_local_only_chunks() -> None:
    provider = FakeGenerationProvider(
        is_external=False,
    )

    chunks = [
        make_chunk(
            source_path="Local.md",
            content="Local-only synthetic content.",
            ai_access="local-only",
        ),
    ]

    result = generate_grounded_answer(
        query="Synthetic query",
        chunks=chunks,
        provider=provider,
    )

    assert len(result.sources) == 1
    assert result.sources[0].source_path == "Local.md"

    prompt = provider.prompts[0]

    assert "Local-only synthetic content." in prompt


def test_grounded_answer_returns_answer_and_sources() -> None:
    provider = FakeGenerationProvider(
        is_external=False,
    )

    chunks = [
        make_chunk(
            source_path="Allowed.md",
            content="Synthetic context.",
            ai_access="allowed",
        ),
    ]

    result = generate_grounded_answer(
        query="Synthetic query",
        chunks=chunks,
        provider=provider,
    )

    assert result.answer == "Synthetic grounded answer"
    assert result.sources == chunks


def test_grounded_prompt_contains_query() -> None:
    provider = FakeGenerationProvider(
        is_external=False,
    )

    chunks = [
        make_chunk(
            source_path="Allowed.md",
            content="Synthetic context.",
            ai_access="allowed",
        ),
    ]

    generate_grounded_answer(
        query="What is the synthetic answer?",
        chunks=chunks,
        provider=provider,
    )

    prompt = provider.prompts[0]

    assert "What is the synthetic answer?" in prompt


def test_external_provider_skips_generation_when_no_chunks_are_allowed() -> None:
    provider = FakeGenerationProvider(
        is_external=True,
    )

    chunks = [
        make_chunk(
            source_path="Local.md",
            content="Local-only synthetic content.",
            ai_access="local-only",
        ),
    ]

    result = generate_grounded_answer(
        query="Synthetic query",
        chunks=chunks,
        provider=provider,
    )

    assert result.answer == ""
    assert result.sources == []
    assert provider.prompts == []