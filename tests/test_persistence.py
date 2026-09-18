from pathlib import Path
from shutil import copy2
from tempfile import TemporaryDirectory

import psycopg

from knowledge_rag.config import settings
from knowledge_rag.embedding_store import embed_missing_chunks
from knowledge_rag.embeddings import DeterministicEmbeddingProvider
from knowledge_rag.ingest.persistence import (
    PersistResult,
    persist_note,
    reconcile_inactive_documents,
)


def create_test_tables(conn: psycopg.Connection) -> None:
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


def test_persist_note_inserts_skips_and_updates() -> None:
    test_root = Path(".test-temp")
    test_root.mkdir(exist_ok=True)

    sample_note = Path(
        "examples/sample-vault/20 Learning/Azure RBAC.md"
    )

    with TemporaryDirectory(dir=test_root) as temp_dir:
        vault_path = Path(temp_dir)
        note_dir = vault_path / "20 Learning"
        note_dir.mkdir()

        note_path = note_dir / "Azure RBAC.md"
        copy2(sample_note, note_path)

        with psycopg.connect(settings.database_url) as conn:
            create_test_tables(conn)

            first_result = persist_note(
                conn,
                vault_path,
                note_path,
            )

            second_result = persist_note(
                conn,
                vault_path,
                note_path,
            )

            document_count = conn.execute(
                """
                SELECT COUNT(*)
                FROM documents;
                """
            ).fetchone()

            original_chunk_count = conn.execute(
                """
                SELECT COUNT(*)
                FROM document_chunks;
                """
            ).fetchone()

            note_path.write_text(
                note_path.read_text(encoding="utf-8")
                + "\n\n"
                + "## Persistence Test\n\n"
                + "Changed content for automated testing.\n",
                encoding="utf-8",
            )

            third_result = persist_note(
                conn,
                vault_path,
                note_path,
            )

            updated_document_count = conn.execute(
                """
                SELECT COUNT(*)
                FROM documents;
                """
            ).fetchone()

            updated_chunks = conn.execute(
                """
                SELECT heading_path, content
                FROM document_chunks
                ORDER BY chunk_index;
                """
            ).fetchall()

            assert first_result is PersistResult.INDEXED
            assert second_result is PersistResult.UNCHANGED
            assert third_result is PersistResult.INDEXED

            assert document_count is not None
            assert document_count[0] == 1

            assert original_chunk_count is not None
            assert original_chunk_count[0] > 0

            assert updated_document_count is not None
            assert updated_document_count[0] == 1

            assert any(
                heading == "Persistence Test"
                and "Changed content for automated testing." in content
                for heading, content in updated_chunks
            )

def test_excluded_ai_access_is_not_persisted(tmp_path: Path) -> None:
    vault_path = tmp_path / "vault"
    vault_path.mkdir()

    note_path = vault_path / "Excluded.md"
    note_path.write_text(
        """---
type: note
ai_access: exclude
---

# Excluded

This note must not be indexed.
""",
        encoding="utf-8",
    )

    with psycopg.connect(settings.database_url) as conn:
        create_test_tables(conn)

        result = persist_note(conn, vault_path, note_path)

        count = conn.execute(
            "SELECT COUNT(*) FROM documents;"
        ).fetchone()

    assert result is PersistResult.EXCLUDED
    assert count is not None
    assert count[0] == 0


def test_excluded_note_type_is_not_persisted(
    tmp_path: Path,
    monkeypatch,
) -> None:
    vault_path = tmp_path / "vault"
    vault_path.mkdir()

    note_path = vault_path / "Journal.md"
    note_path.write_text(
        """---
type: journal
ai_access: allowed
---

# Journal

This note is excluded by type.
""",
        encoding="utf-8",
    )

    monkeypatch.setattr(
        settings,
        "excluded_note_types",
        ("journal",),
    )

    with psycopg.connect(settings.database_url) as conn:
        create_test_tables(conn)

        result = persist_note(conn, vault_path, note_path)

        count = conn.execute(
            "SELECT COUNT(*) FROM documents;"
        ).fetchone()

    assert result is PersistResult.EXCLUDED
    assert count is not None
    assert count[0] == 0


def test_excluded_path_is_not_persisted(
    tmp_path: Path,
    monkeypatch,
) -> None:
    vault_path = tmp_path / "vault"
    private_dir = vault_path / "Private"
    private_dir.mkdir(parents=True)

    note_path = private_dir / "Secret.md"
    note_path.write_text(
        """---
type: note
ai_access: allowed
---

# Secret

This note is excluded by path.
""",
        encoding="utf-8",
    )

    monkeypatch.setattr(
        settings,
        "excluded_paths",
        ("Private",),
    )

    with psycopg.connect(settings.database_url) as conn:
        create_test_tables(conn)

        result = persist_note(conn, vault_path, note_path)

        count = conn.execute(
            "SELECT COUNT(*) FROM documents;"
        ).fetchone()

    assert result is PersistResult.EXCLUDED
    assert count is not None
    assert count[0] == 0

def test_local_only_policy_is_persisted(tmp_path: Path) -> None:
    vault_path = tmp_path / "vault"
    vault_path.mkdir()

    note_path = vault_path / "Local Only.md"
    note_path.write_text(
        """---
type: note
ai_access: local-only
---

# Local Only

This note may be indexed locally but must not leave the system.
""",
        encoding="utf-8",
    )

    with psycopg.connect(settings.database_url) as conn:
        create_test_tables(conn)

        result = persist_note(conn, vault_path, note_path)

        row = conn.execute(
            """
            SELECT ai_access
            FROM documents
            WHERE source_path = %s;
            """,
            ("Local Only.md",),
        ).fetchone()

    assert result is PersistResult.INDEXED
    assert row is not None
    assert row[0] == "local-only"

def test_allowed_policy_is_persisted(tmp_path: Path) -> None:
    vault_path = tmp_path / "vault"
    vault_path.mkdir()

    note_path = vault_path / "Allowed.md"
    note_path.write_text(
        """---
type: note
ai_access: allowed
---

# Allowed

This note may be used with external model providers.
""",
        encoding="utf-8",
    )

    with psycopg.connect(settings.database_url) as conn:
        create_test_tables(conn)

        result = persist_note(conn, vault_path, note_path)

        row = conn.execute(
            """
            SELECT ai_access
            FROM documents
            WHERE source_path = %s;
            """,
            ("Allowed.md",),
        ).fetchone()

    assert result is PersistResult.INDEXED
    assert row is not None
    assert row[0] == "allowed"


def test_metadata_only_changes_update_existing_document(tmp_path: Path) -> None:
    vault_path = tmp_path / "vault"
    vault_path.mkdir()

    note_path = vault_path / "Identity.md"
    note_path.write_text(
        """---
type: reference
status: draft
ai_access: local-only
topic:
  - identity
---

# Identity

The body remains unchanged while frontmatter changes.
""",
        encoding="utf-8",
    )

    with psycopg.connect(settings.database_url) as conn:
        create_test_tables(conn)

        first_result = persist_note(conn, vault_path, note_path)

        note_path.write_text(
            """---
type: guide
status: active
ai_access: allowed
topic:
  - access-control
---

# Identity

The body remains unchanged while frontmatter changes.
""",
            encoding="utf-8",
        )

        second_result = persist_note(conn, vault_path, note_path)
        third_result = persist_note(conn, vault_path, note_path)

        rows = conn.execute(
            """
            SELECT note_type, status, ai_access, metadata
            FROM documents
            WHERE source_path = %s;
            """,
            ("Identity.md",),
        ).fetchall()

    assert first_result is PersistResult.INDEXED
    assert second_result is PersistResult.INDEXED
    assert third_result is PersistResult.UNCHANGED
    assert rows == [
        (
            "guide",
            "active",
            "allowed",
            {
                "type": "guide",
                "status": "active",
                "ai_access": "allowed",
                "topic": ["access-control"],
            },
        )
    ]

def test_changed_document_chunks_are_reembedded(tmp_path: Path) -> None:
    vault_path = tmp_path / "vault"
    vault_path.mkdir()

    note_path = vault_path / "Allowed.md"
    note_path.write_text(
        """---
type: reference
ai_access: allowed
---

# Allowed

Initial content.
""",
        encoding="utf-8",
    )

    provider = DeterministicEmbeddingProvider(dimensions=1536)

    with psycopg.connect(settings.database_url) as conn:
        create_test_tables(conn)

        first_result = persist_note(
            conn,
            vault_path,
            note_path,
        )

        first_embedded_count = embed_missing_chunks(
            conn,
            provider,
        )

        first_row = conn.execute(
            """
            SELECT chunk_id, embedding IS NOT NULL
            FROM document_chunks;
            """
        ).fetchone()

        assert first_result is PersistResult.INDEXED
        assert first_embedded_count > 0
        assert first_row is not None
        assert first_row[1] is True

        original_chunk_id = first_row[0]

        note_path.write_text(
            """---
type: reference
ai_access: allowed
---

# Allowed

Changed content that requires a new embedding.
""",
            encoding="utf-8",
        )

        second_result = persist_note(
            conn,
            vault_path,
            note_path,
        )

        changed_row = conn.execute(
            """
            SELECT chunk_id, embedding IS NULL
            FROM document_chunks;
            """
        ).fetchone()

        assert second_result is PersistResult.INDEXED
        assert changed_row is not None
        assert changed_row[0] != original_chunk_id
        assert changed_row[1] is True

        second_embedded_count = embed_missing_chunks(
            conn,
            provider,
        )

        final_row = conn.execute(
            """
            SELECT embedding IS NOT NULL
            FROM document_chunks;
            """
        ).fetchone()

        assert second_embedded_count > 0
        assert final_row is not None
        assert final_row[0] is True


class FailingEmbeddingProvider:
    """Embedding provider that fails if ineligible content reaches it."""

    def embed(self, text: str) -> list[float]:
        raise AssertionError("Inactive documents must not be embedded.")


def test_inactive_allowed_document_is_not_embedded(tmp_path: Path) -> None:
    vault_path = tmp_path / "vault"
    vault_path.mkdir()
    note_path = vault_path / "Inactive.md"
    note_path.write_text(
        """---
type: note
ai_access: allowed
---

# Inactive

This content must not reach an embedding provider.
""",
        encoding="utf-8",
    )

    with psycopg.connect(settings.database_url) as conn:
        create_test_tables(conn)
        assert persist_note(conn, vault_path, note_path) is PersistResult.INDEXED
        conn.execute(
            "UPDATE documents SET is_active = FALSE WHERE source_path = 'Inactive.md';"
        )

        embedded_count = embed_missing_chunks(conn, FailingEmbeddingProvider())

    assert embedded_count == 0


def test_persist_note_reactivates_unchanged_document_without_rebuilding_chunks(
    tmp_path: Path,
) -> None:
    vault_path = tmp_path / "vault"
    vault_path.mkdir()
    note_path = vault_path / "Restored.md"
    note_path.write_text(
        """---
type: note
ai_access: allowed
---

# Restored

This document returns with unchanged content and metadata.
""",
        encoding="utf-8",
    )

    provider = DeterministicEmbeddingProvider(dimensions=1536)

    with psycopg.connect(settings.database_url) as conn:
        create_test_tables(conn)
        assert persist_note(conn, vault_path, note_path) is PersistResult.INDEXED
        assert embed_missing_chunks(conn, provider) == 1

        original = conn.execute(
            """
            SELECT d.document_id, c.chunk_id, c.embedding
            FROM documents d
            JOIN document_chunks c ON c.document_id = d.document_id
            WHERE d.source_path = 'Restored.md';
            """
        ).fetchone()
        assert original is not None

        conn.execute(
            "UPDATE documents SET is_active = FALSE WHERE source_path = 'Restored.md';"
        )

        result = persist_note(conn, vault_path, note_path)
        restored = conn.execute(
            """
            SELECT d.document_id, c.chunk_id, c.embedding, d.is_active
            FROM documents d
            JOIN document_chunks c ON c.document_id = d.document_id
            WHERE d.source_path = 'Restored.md';
            """
        ).fetchone()

    assert result is PersistResult.REACTIVATED
    assert restored is not None
    assert restored[0] == original[0]
    assert restored[1] == original[1]
    assert restored[2] == original[2]
    assert restored[3] is True


def test_excluded_ai_access_deactivates_existing_document(tmp_path: Path) -> None:
    vault_path = tmp_path / "vault"
    vault_path.mkdir()
    note_path = vault_path / "Policy.md"
    note_path.write_text(
        """---
type: note
ai_access: allowed
---

# Policy

This note starts as externally eligible.
""",
        encoding="utf-8",
    )

    with psycopg.connect(settings.database_url) as conn:
        create_test_tables(conn)
        assert persist_note(conn, vault_path, note_path) is PersistResult.INDEXED

        note_path.write_text(
            """---
type: note
ai_access: exclude
---

# Policy

This note starts as externally eligible.
""",
            encoding="utf-8",
        )
        result = persist_note(conn, vault_path, note_path)
        stored = conn.execute(
            "SELECT is_active FROM documents WHERE source_path = 'Policy.md';"
        ).fetchone()

    assert result is PersistResult.EXCLUDED
    assert stored == (False,)


def test_excluded_path_deactivates_existing_document(
    tmp_path: Path,
    monkeypatch,
) -> None:
    vault_path = tmp_path / "vault"
    note_path = vault_path / "Private" / "Policy.md"
    note_path.parent.mkdir(parents=True)
    note_path.write_text(
        """---
type: note
ai_access: allowed
---

# Policy

This note is later excluded by path policy.
""",
        encoding="utf-8",
    )

    with psycopg.connect(settings.database_url) as conn:
        create_test_tables(conn)
        assert persist_note(conn, vault_path, note_path) is PersistResult.INDEXED

        monkeypatch.setattr(settings, "excluded_paths", ("Private",))
        result = persist_note(conn, vault_path, note_path)
        stored = conn.execute(
            "SELECT is_active FROM documents WHERE source_path = 'Private/Policy.md';"
        ).fetchone()

    assert result is PersistResult.EXCLUDED
    assert stored == (False,)


def test_excluded_note_type_deactivates_existing_document(
    tmp_path: Path,
    monkeypatch,
) -> None:
    vault_path = tmp_path / "vault"
    vault_path.mkdir()
    note_path = vault_path / "Policy.md"
    note_path.write_text(
        """---
type: note
ai_access: allowed
---

# Policy

This note is later excluded by note type.
""",
        encoding="utf-8",
    )

    with psycopg.connect(settings.database_url) as conn:
        create_test_tables(conn)
        assert persist_note(conn, vault_path, note_path) is PersistResult.INDEXED

        monkeypatch.setattr(settings, "excluded_note_types", ("note",))
        result = persist_note(conn, vault_path, note_path)
        stored = conn.execute(
            "SELECT is_active FROM documents WHERE source_path = 'Policy.md';"
        ).fetchone()

    assert result is PersistResult.EXCLUDED
    assert stored == (False,)


def test_reconciliation_marks_only_missing_active_documents_inactive() -> None:
    with psycopg.connect(settings.database_url) as conn:
        create_test_tables(conn)
        conn.execute(
            """
            INSERT INTO documents (
                document_uuid, source_path, title, ai_access,
                metadata, content_hash, is_active
            )
            VALUES
                (gen_random_uuid(), 'Present.md', 'Present', 'allowed', '{}', 'one', TRUE),
                (gen_random_uuid(), 'Missing.md', 'Missing', 'allowed', '{}', 'two', TRUE),
                (gen_random_uuid(), 'Old.md', 'Old', 'allowed', '{}', 'three', FALSE);
            """
        )
        changed = reconcile_inactive_documents(conn, {"Present.md"})
        rows = conn.execute(
            "SELECT source_path, is_active FROM documents ORDER BY source_path;"
        ).fetchall()

    assert changed == 1
    assert rows == [
        ("Missing.md", False),
        ("Old.md", False),
        ("Present.md", True),
    ]
