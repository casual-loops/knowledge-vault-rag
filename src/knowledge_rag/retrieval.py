from dataclasses import dataclass
from typing import Any

from psycopg import Connection

from knowledge_rag.embeddings import EmbeddingProvider


@dataclass(frozen=True, slots=True)
class SearchResult:
    """One retrieval result with source metadata and retrieval score."""

    source_path: str
    title: str
    note_type: str | None
    topic: Any
    ai_access: str
    chunk_index: int
    heading_path: str | None
    content: str
    score: float
    score_type: str


def semantic_search(
    conn: Connection,
    provider: EmbeddingProvider,
    query: str,
    limit: int = 5,
    note_type: str | None = None,
    topic: str | None = None,
) -> list[SearchResult]:
    """Return nearest chunks by cosine distance with optional metadata filters."""

    query_vector = provider.embed(query)

    rows = conn.execute(
    """
    SELECT
        d.source_path,
        d.title,
        d.note_type,
        d.metadata -> 'topic' AS topic,
        d.ai_access,
        c.chunk_index,
        c.heading_path,
        c.content,
        c.embedding <=> %s::vector AS distance
    FROM document_chunks c
    JOIN documents d
        ON d.document_id = c.document_id
    WHERE c.embedding IS NOT NULL
      AND (%s::text IS NULL OR d.note_type = %s::text)
      AND (
            %s::text IS NULL
            OR d.metadata -> 'topic' ? %s::text
            OR d.metadata ->> 'topic' = %s::text
          )
    ORDER BY c.embedding <=> %s::vector
    LIMIT %s;
    """,
    (
        query_vector,
        note_type,
        note_type,
        topic,
        topic,
        topic,
        query_vector,
        limit,
    ),
).fetchall()

    return [
    SearchResult(
        source_path=row[0],
        title=row[1],
        note_type=row[2],
        topic=row[3],
        ai_access=row[4],
        chunk_index=row[5],
        heading_path=row[6],
        content=row[7],
        score=1.0 - float(row[8]),
        score_type="semantic",
    )
    for row in rows
    ]


def lexical_search(
    conn: Connection,
    query: str,
    limit: int = 5,
    note_type: str | None = None,
    topic: str | None = None,
) -> list[SearchResult]:
    """Return chunks ranked by PostgreSQL full-text relevance."""

    rows = conn.execute(
        """
        SELECT
            d.source_path,
            d.title,
            d.note_type,
            d.metadata -> 'topic' AS topic,
            d.ai_access,
            c.chunk_index,
            c.heading_path,
            c.content,
            ts_rank_cd(
                to_tsvector(
                    'english',
                    concat_ws(
                        ' ',
                        d.title,
                        c.heading_path,
                        c.content
                    )
                ),
                plainto_tsquery('english', %s)
            ) AS rank
        FROM document_chunks c
        JOIN documents d
            ON d.document_id = c.document_id
        WHERE
            to_tsvector(
                'english',
                concat_ws(
                    ' ',
                    d.title,
                    c.heading_path,
                    c.content
                )
            ) @@ plainto_tsquery('english', %s)
          AND (%s::text IS NULL OR d.note_type = %s::text)
          AND (
                %s::text IS NULL
                OR d.metadata -> 'topic' ? %s::text
                OR d.metadata ->> 'topic' = %s::text
              )
        ORDER BY rank DESC
        LIMIT %s;
        """,
        (
            query,
            query,
            note_type,
            note_type,
            topic,
            topic,
            topic,
            limit,
        ),
    ).fetchall()

    return [
    SearchResult(
        source_path=row[0],
        title=row[1],
        note_type=row[2],
        topic=row[3],
        ai_access=row[4],
        chunk_index=row[5],
        heading_path=row[6],
        content=row[7],
        score=float(row[8]),
        score_type="lexical",
    )
    for row in rows
    ]


@dataclass(frozen=True, slots=True)
class HybridSearchResult:
    """One hybrid retrieval result produced from fused rankings."""

    source_path: str
    title: str
    note_type: str | None
    topic: Any
    ai_access: str
    chunk_index: int
    heading_path: str | None
    content: str
    score: float
    score_type: str = "hybrid"


def hybrid_search(
    semantic_results: list[SearchResult],
    lexical_results: list[SearchResult],
    *,
    limit: int = 5,
    rrf_k: int = 60,
) -> list[HybridSearchResult]:
    """Fuse semantic and lexical rankings with reciprocal rank fusion."""

    fused: dict[tuple[str, int], dict[str, Any]] = {}

    def add_results(results: list[SearchResult]) -> None:
        for rank, result in enumerate(results, start=1):
            key = (
                result.source_path,
                result.chunk_index,
            )

            contribution = 1.0 / (rrf_k + rank)

            if key not in fused:
                fused[key] = {
                    "result": result,
                    "score": 0.0,
                }

            fused[key]["score"] += contribution

    add_results(semantic_results)
    add_results(lexical_results)

    ranked = sorted(
        fused.values(),
        key=lambda item: (
            -item["score"],
            item["result"].source_path,
            item["result"].chunk_index,
        ),
    )

    return [
        HybridSearchResult(
            source_path=item["result"].source_path,
            title=item["result"].title,
            note_type=item["result"].note_type,
            topic=item["result"].topic,
            ai_access=item["result"].ai_access,
            chunk_index=item["result"].chunk_index,
            heading_path=item["result"].heading_path,
            content=item["result"].content,
            score=item["score"],
        )
        for item in ranked[:limit]
    ]