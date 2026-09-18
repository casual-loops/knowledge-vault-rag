import psycopg
from psycopg.types.json import Jsonb

from knowledge_rag.config import settings
from knowledge_rag.embeddings import DeterministicEmbeddingProvider
from knowledge_rag.retrieval import semantic_search


def create_test_tables(conn: psycopg.Connection) -> None:
    """Create temporary tables compatible with semantic retrieval tests."""

    conn.execute(
        """
        CREATE TEMP TABLE documents (
            document_id BIGSERIAL PRIMARY KEY,
            document_uuid UUID NOT NULL UNIQUE,
            source_path TEXT NOT NULL UNIQUE,
            title TEXT NOT NULL,
            note_type TEXT,
            created_date DATE,
            status TEXT,
            ai_access TEXT NOT NULL DEFAULT 'local-only',
            metadata JSONB NOT NULL DEFAULT '{}'::jsonb,
            content_hash TEXT NOT NULL,
            is_active BOOLEAN NOT NULL DEFAULT TRUE,
            indexed_at TIMESTAMPTZ NOT NULL DEFAULT NOW()
        );
        """
    )

    conn.execute(
        """
        CREATE TEMP TABLE document_chunks (
            chunk_id BIGSERIAL PRIMARY KEY,
            document_id BIGINT NOT NULL,
            chunk_index INTEGER NOT NULL,
            heading_path TEXT,
            content TEXT NOT NULL,
            token_estimate INTEGER,
            embedding VECTOR(1536),
            UNIQUE (document_id, chunk_index)
        );
        """
    )


def insert_document_with_chunk(
    conn: psycopg.Connection,
    provider: DeterministicEmbeddingProvider,
    source_path: str,
    title: str,
    note_type: str,
    topic: str,
    content: str,
    is_active: bool = True,
) -> None:
    """Insert one synthetic document and one embedded chunk."""

    row = conn.execute(
        """
        INSERT INTO documents (
            document_uuid,
            source_path,
            title,
            note_type,
            ai_access,
            metadata,
            content_hash,
            is_active
        )
        VALUES (
            gen_random_uuid(),
            %s,
            %s,
            %s,
            'allowed',
            %s,
            'test-hash',
            %s
        )
        RETURNING document_id;
        """,
        (
            source_path,
            title,
            note_type,
            Jsonb({"topic": [topic]}),
            is_active,
        ),
    ).fetchone()

    assert row is not None
    document_id = row[0]

    vector = provider.embed(content)

    conn.execute(
        """
        INSERT INTO document_chunks (
            document_id,
            chunk_index,
            heading_path,
            content,
            embedding
        )
        VALUES (%s, 0, %s, %s, %s);
        """,
        (
            document_id,
            title,
            content,
            vector,
        ),
    )


def test_semantic_search_respects_top_k_limit() -> None:
    provider = DeterministicEmbeddingProvider(dimensions=1536)

    with psycopg.connect(settings.database_url) as conn:
        create_test_tables(conn)

        insert_document_with_chunk(
            conn,
            provider,
            "Reference One.md",
            "Reference One",
            "reference",
            "privacy-demo",
            "Synthetic privacy reference content.",
        )

        insert_document_with_chunk(
            conn,
            provider,
            "Study One.md",
            "Study One",
            "study",
            "azure",
            "Synthetic Azure study content.",
        )

        results = semantic_search(
            conn,
            provider,
            query="privacy",
            limit=1,
        )

    assert len(results) == 1


def test_semantic_search_filters_by_note_type() -> None:
    provider = DeterministicEmbeddingProvider(dimensions=1536)

    with psycopg.connect(settings.database_url) as conn:
        create_test_tables(conn)

        insert_document_with_chunk(
            conn,
            provider,
            "Reference One.md",
            "Reference One",
            "reference",
            "privacy-demo",
            "Synthetic reference content.",
        )

        insert_document_with_chunk(
            conn,
            provider,
            "Study One.md",
            "Study One",
            "study",
            "privacy-demo",
            "Synthetic study content.",
        )

        results = semantic_search(
            conn,
            provider,
            query="privacy",
            limit=10,
            note_type="reference",
        )

    assert len(results) == 1
    assert results[0].note_type == "reference"
    assert results[0].source_path == "Reference One.md"


def test_semantic_search_filters_by_topic() -> None:
    provider = DeterministicEmbeddingProvider(dimensions=1536)

    with psycopg.connect(settings.database_url) as conn:
        create_test_tables(conn)

        insert_document_with_chunk(
            conn,
            provider,
            "Privacy Note.md",
            "Privacy Note",
            "reference",
            "privacy-demo",
            "Synthetic privacy content.",
        )

        insert_document_with_chunk(
            conn,
            provider,
            "Azure Note.md",
            "Azure Note",
            "reference",
            "azure",
            "Synthetic Azure content.",
        )

        results = semantic_search(
            conn,
            provider,
            query="example",
            limit=10,
            topic="privacy-demo",
        )

    assert len(results) == 1
    assert results[0].source_path == "Privacy Note.md"
    assert results[0].topic == ["privacy-demo"]


def test_semantic_search_excludes_inactive_documents() -> None:
    provider = DeterministicEmbeddingProvider(dimensions=1536)

    with psycopg.connect(settings.database_url) as conn:
        create_test_tables(conn)
        insert_document_with_chunk(
            conn,
            provider,
            "Inactive Note.md",
            "Inactive Note",
            "reference",
            "synthetic",
            "Synthetic content from an inactive note.",
            is_active=False,
        )

        results = semantic_search(
            conn,
            provider,
            query="synthetic",
        )

    assert results == []
