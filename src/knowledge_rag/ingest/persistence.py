from pathlib import Path
from enum import StrEnum

from psycopg import Connection
from psycopg.types.json import Jsonb

from knowledge_rag.config import settings
from knowledge_rag.ingest.chunker import chunk_markdown
from knowledge_rag.ingest.identity import content_hash, document_identity
from knowledge_rag.ingest.markdown import load_markdown
from knowledge_rag.policy import (
    evaluate_note_policy,
    note_type_is_excluded,
    path_is_excluded,
)


class PersistResult(StrEnum):
    INDEXED = "indexed"
    UNCHANGED = "unchanged"
    EXCLUDED = "excluded"


def persist_note(
    conn: Connection,
    vault_path: Path,
    note_path: Path,
) -> PersistResult:
    document = load_markdown(note_path)

    if path_is_excluded(
        vault_path,
        note_path,
        settings.excluded_paths,
    ):
        return PersistResult.EXCLUDED

    if note_type_is_excluded(
        document.metadata,
        settings.excluded_note_types,
    ):
        return PersistResult.EXCLUDED

    policy = evaluate_note_policy(document.metadata)

    if not policy.may_index:
        return PersistResult.EXCLUDED

    source_path = note_path.resolve().relative_to(vault_path.resolve()).as_posix()
    document_uuid = document_identity(vault_path, note_path)
    digest = content_hash(document.content, document.metadata)

    existing = conn.execute(
        """
        SELECT document_id, content_hash
        FROM documents
        WHERE source_path = %s;
        """,
        (source_path,),
    ).fetchone()

    if existing is not None and existing[1] == digest:
        return PersistResult.UNCHANGED

    metadata = document.metadata

    row = conn.execute(
        """
        INSERT INTO documents (
            document_uuid,
            source_path,
            title,
            note_type,
            created_date,
            status,
            ai_access,
            metadata,
            content_hash,
            indexed_at
        )
        VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, NOW())
        ON CONFLICT (source_path)
        DO UPDATE SET
            title = EXCLUDED.title,
            note_type = EXCLUDED.note_type,
            created_date = EXCLUDED.created_date,
            status = EXCLUDED.status,
            ai_access = EXCLUDED.ai_access,
            metadata = EXCLUDED.metadata,
            content_hash = EXCLUDED.content_hash,
            indexed_at = NOW()
        RETURNING document_id;
        """,
        (
            document_uuid,
            source_path,
            document.title,
            metadata.get("type"),
            metadata.get("created"),
            metadata.get("status"),
            policy.ai_access.value,
            Jsonb(metadata),
            digest,
        ),
    ).fetchone()

    if row is None:
        raise RuntimeError(f"Failed to persist document: {source_path}")

    document_id = row[0]

    conn.execute(
        """
        DELETE FROM document_chunks
        WHERE document_id = %s;
        """,
        (document_id,),
    )

    chunks = chunk_markdown(document.content)

    for chunk in chunks:
        conn.execute(
            """
            INSERT INTO document_chunks (
                document_id,
                chunk_index,
                heading_path,
                content
            )
            VALUES (%s, %s, %s, %s);
            """,
            (
                document_id,
                chunk.index,
                chunk.heading_path,
                chunk.content,
            ),
        )

    return PersistResult.INDEXED
