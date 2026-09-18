from psycopg import Connection

from knowledge_rag.embedding_pipeline import (
    ChunkForEmbedding,
    EmbeddedChunk,
)
from knowledge_rag.embeddings import EmbeddingProvider


def load_chunks_missing_embeddings(
    conn: Connection,
    limit: int = 100,
) -> list[ChunkForEmbedding]:
    """Load externally eligible chunks that do not yet have embeddings."""

    rows = conn.execute(
        """
        SELECT
            c.chunk_id,
            c.content,
            d.ai_access
        FROM document_chunks c
        JOIN documents d
            ON d.document_id = c.document_id
        WHERE c.embedding IS NULL
          AND d.is_active = TRUE
          AND d.ai_access = 'allowed'
        ORDER BY c.chunk_id
        LIMIT %s;
        """,
        (limit,),
    ).fetchall()

    return [
        ChunkForEmbedding(
            chunk_id=row[0],
            content=row[1],
            ai_access=row[2],
        )
        for row in rows
    ]


def persist_embeddings(
    conn: Connection,
    embedded_chunks: list[EmbeddedChunk],
) -> None:
    """Persist generated vectors to pgvector."""

    for chunk in embedded_chunks:
        conn.execute(
            """
            UPDATE document_chunks
            SET embedding = %s
            WHERE chunk_id = %s;
            """,
            (
                chunk.vector,
                chunk.chunk_id,
            ),
        )


def embed_missing_chunks(
    conn: Connection,
    provider: EmbeddingProvider,
    limit: int = 100,
) -> int:
    """Generate and persist embeddings for eligible chunks without vectors."""

    chunks = load_chunks_missing_embeddings(
        conn,
        limit=limit,
    )

    embedded_chunks = []

    for chunk in chunks:
        embedded_chunks.append(
            EmbeddedChunk(
                chunk_id=chunk.chunk_id,
                vector=provider.embed(chunk.content),
            )
        )

    persist_embeddings(conn, embedded_chunks)

    return len(embedded_chunks)
