from enum import StrEnum
from pathlib import Path

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
    REACTIVATED = "reactivated"


def deactivate_document(
    conn: Connection,
    source_path: str,
) -> bool:
    """Mark an active document inactive without deleting its stored data."""
    result = conn.execute(
        """
        UPDATE documents
        SET is_active = FALSE
        WHERE source_path = %s
          AND is_active = TRUE;
        """,
        (source_path,),
    )
    return result.rowcount == 1


def reconcile_inactive_documents(
    conn: Connection,
    discovered_source_paths: set[str],
) -> int:
    """Mark active documents missing from a complete vault scan inactive."""
    active_paths = {
        row[0]
        for row in conn.execute(
            "SELECT source_path FROM documents WHERE is_active = TRUE;"
        ).fetchall()
    }
    missing_paths = sorted(active_paths - discovered_source_paths)

    if not missing_paths:
        return 0

    result = conn.execute(
        """
        UPDATE documents
        SET is_active = FALSE
        WHERE source_path = ANY(%s)
          AND is_active = TRUE;
        """,
        (missing_paths,),
    )
    return result.rowcount


def persist_note(
    conn: Connection,
    vault_path: Path,
    note_path: Path,
) -> PersistResult:
    source_path = note_path.resolve().relative_to(vault_path.resolve()).as_posix()

    if path_is_excluded(
        vault_path,
        note_path,
        settings.excluded_paths,
    ):
        deactivate_document(conn, source_path)
        return PersistResult.EXCLUDED

    document = load_markdown(note_path)

    if note_type_is_excluded(
        document.metadata,
        settings.excluded_note_types,
    ):
        deactivate_document(conn, source_path)
        return PersistResult.EXCLUDED

    policy = evaluate_note_policy(document.metadata)

    if not policy.may_index:
        deactivate_document(conn, source_path)
        return PersistResult.EXCLUDED

    document_uuid = document_identity(vault_path, note_path)
    digest = content_hash(document.content, document.metadata)

    existing = conn.execute(
        """
        SELECT document_id, content_hash, is_active
        FROM documents
        WHERE source_path = %s;
        """,
        (source_path,),
    ).fetchone()

    if existing is not None and existing[1] == digest:
        if existing[2]:
            return PersistResult.UNCHANGED

        conn.execute(
            """
            UPDATE documents
            SET is_active = TRUE,
                indexed_at = NOW()
            WHERE document_id = %s;
            """,
            (existing[0],),
        )
        return PersistResult.REACTIVATED

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
            is_active,
            indexed_at
        )
        VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, TRUE, NOW())
        ON CONFLICT (source_path)
        DO UPDATE SET
            title = EXCLUDED.title,
            note_type = EXCLUDED.note_type,
            created_date = EXCLUDED.created_date,
            status = EXCLUDED.status,
            ai_access = EXCLUDED.ai_access,
            metadata = EXCLUDED.metadata,
            content_hash = EXCLUDED.content_hash,
            is_active = TRUE,
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
