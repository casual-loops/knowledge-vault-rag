from dataclasses import dataclass

from knowledge_rag.generation import GenerationProvider
from knowledge_rag.retrieval_policy import (
    RetrievedChunk,
    filter_external_generation_context,
)


@dataclass(frozen=True, slots=True)
class GroundedAnswer:
    """Generated answer plus the source chunks used to produce it."""

    answer: str
    sources: list[RetrievedChunk]


def build_generation_context(
    chunks: list[RetrievedChunk],
    *,
    max_characters: int = 6000,
) -> tuple[str, list[RetrievedChunk]]:
    """Build bounded generation context and return the chunks used."""

    selected_parts: list[str] = []
    selected_chunks: list[RetrievedChunk] = []
    current_length = 0

    for chunk in chunks:
        block = (
            f"Source: {chunk.source_path}\n"
            f"{chunk.content.strip()}"
        )

        separator_length = 2 if selected_parts else 0
        next_length = current_length + separator_length + len(block)

        if next_length > max_characters:
            break

        selected_parts.append(block)
        selected_chunks.append(chunk)
        current_length = next_length

    return "\n\n".join(selected_parts), selected_chunks


def generate_grounded_answer(
    *,
    query: str,
    chunks: list[RetrievedChunk],
    provider: GenerationProvider,
    max_context_characters: int = 6000,
) -> GroundedAnswer:
    """Generate an answer grounded in retrieved source chunks."""

    usable_chunks = chunks

    if provider.is_external:
        usable_chunks = filter_external_generation_context(chunks)

    if not usable_chunks:
        return GroundedAnswer(
            answer="",
            sources=[],
        )

    context, selected_chunks = build_generation_context(
        usable_chunks,
        max_characters=max_context_characters,
    )

    if not selected_chunks:
        return GroundedAnswer(
            answer="",
            sources=[],
        )

    prompt = (
        "Answer the user query using only the provided context.\n\n"
        f"User query:\n{query}\n\n"
        f"Context:\n{context}"
    )

    answer = provider.generate(
        prompt=prompt,
    )

    return GroundedAnswer(
        answer=answer,
        sources=selected_chunks,
    )