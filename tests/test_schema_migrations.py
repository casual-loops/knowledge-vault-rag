from pathlib import Path

import psycopg

from knowledge_rag.config import settings
from scripts.apply_schema import SCHEMA_FILENAMES


def test_schema_runner_applies_active_state_migration_in_order() -> None:
    assert SCHEMA_FILENAMES == (
        "001_extensions.sql",
        "002_schema.sql",
        "003_add_ai_access.sql",
        "004_add_document_active_state.sql",
    )


def test_active_state_migration_preserves_existing_documents() -> None:
    migration = Path("sql/004_add_document_active_state.sql").read_text(
        encoding="utf-8"
    )

    with psycopg.connect(settings.database_url) as conn:
        conn.execute(
            """
            CREATE TEMP TABLE documents (
                document_id BIGSERIAL PRIMARY KEY,
                source_path TEXT NOT NULL UNIQUE
            );
            """
        )
        conn.execute(
            "INSERT INTO documents (source_path) VALUES ('Existing.md');"
        )
        conn.execute(migration)
        row = conn.execute(
            """
            SELECT is_active
            FROM documents
            WHERE source_path = 'Existing.md';
            """
        ).fetchone()

    assert row is not None
    assert row[0] is True
