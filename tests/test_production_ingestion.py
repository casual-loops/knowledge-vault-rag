from contextlib import nullcontext

import psycopg
import pytest

from knowledge_rag.config import settings
from knowledge_rag.embeddings import DeterministicEmbeddingProvider
from knowledge_rag.ingest.persistence import PersistResult
from knowledge_rag.ingestion_config import IngestionConfig


def create_production_test_tables(conn: psycopg.Connection) -> None:
    """Create the production workflow's minimal temporary database schema."""
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


def test_run_production_ingestion_coordinates_successful_pipeline(
    tmp_path,
    monkeypatch,
) -> None:
    from knowledge_rag import production_ingestion

    vault_path = (tmp_path / "production-vault").resolve()
    vault_path.mkdir()
    indexed_note = vault_path / "Indexed.md"
    unchanged_note = vault_path / "Unchanged.md"
    indexed_note.write_text("# Indexed", encoding="utf-8")
    unchanged_note.write_text("# Unchanged", encoding="utf-8")
    config = IngestionConfig(
        vault_path=vault_path,
        production=True,
        production_opt_in=True,
    )

    class FakeConnection:
        def transaction(self):
            return nullcontext()

    connection = FakeConnection()
    persisted: list[tuple[object, object, object]] = []

    monkeypatch.setattr(
        production_ingestion,
        "discover_markdown_files",
        lambda path: [indexed_note, unchanged_note],
    )

    def fake_persist_note(conn, vault, note):
        persisted.append((conn, vault, note))
        if note == indexed_note:
            return PersistResult.INDEXED
        return PersistResult.UNCHANGED

    monkeypatch.setattr(
        production_ingestion,
        "persist_note",
        fake_persist_note,
    )

    reconciled_paths: list[set[str]] = []

    def fake_reconcile_inactive_documents(conn, paths):
        assert conn is connection
        reconciled_paths.append(paths)
        return 1

    monkeypatch.setattr(
        production_ingestion,
        "reconcile_inactive_documents",
        fake_reconcile_inactive_documents,
    )

    embedding_counts = iter((2, 1, 0))
    embedding_calls: list[tuple[object, object, int]] = []

    def fake_embed_missing_chunks(conn, provider, limit):
        embedding_calls.append((conn, provider, limit))
        return next(embedding_counts)

    monkeypatch.setattr(
        production_ingestion,
        "embed_missing_chunks",
        fake_embed_missing_chunks,
    )

    provider = object()
    summary = production_ingestion.run_production_ingestion(
        connection,
        config,
        provider,
        embedding_batch_size=25,
    )

    assert persisted == [
        (connection, vault_path, indexed_note),
        (connection, vault_path, unchanged_note),
    ]
    assert reconciled_paths == [{"Indexed.md", "Unchanged.md"}]
    assert embedding_calls == [
        (connection, provider, 25),
        (connection, provider, 25),
        (connection, provider, 25),
    ]
    assert summary.discovered == 2
    assert summary.indexed == 1
    assert summary.unchanged == 1
    assert summary.inactivated == 1
    assert summary.embedded == 3
    assert summary.failures == ()
    assert summary.reconciliation_completed is True
    assert summary.complete is True


def test_run_production_ingestion_isolates_note_failure_and_continues_embedding(
    tmp_path,
    monkeypatch,
) -> None:
    from knowledge_rag import production_ingestion

    vault_path = (tmp_path / "production-vault").resolve()
    vault_path.mkdir()
    broken_note = vault_path / "Broken.md"
    healthy_note = vault_path / "Healthy.md"
    broken_note.write_text("# Broken", encoding="utf-8")
    healthy_note.write_text("# Healthy", encoding="utf-8")
    config = IngestionConfig(
        vault_path=vault_path,
        production=True,
        production_opt_in=True,
    )

    class FakeConnection:
        def transaction(self):
            return nullcontext()

    processed: list[str] = []
    reconciliation_called = False
    embedding_called = False

    monkeypatch.setattr(
        production_ingestion,
        "discover_markdown_files",
        lambda path: [broken_note, healthy_note],
    )

    def fake_persist_note(conn, vault, note):
        processed.append(note.name)
        if note == broken_note:
            raise ValueError("PRIVATE_NOTE_MARKER")
        return PersistResult.INDEXED

    monkeypatch.setattr(
        production_ingestion,
        "persist_note",
        fake_persist_note,
    )

    def fake_reconcile_inactive_documents(conn, paths):
        nonlocal reconciliation_called
        reconciliation_called = True
        return 0

    monkeypatch.setattr(
        production_ingestion,
        "reconcile_inactive_documents",
        fake_reconcile_inactive_documents,
    )

    def fake_embed_missing_chunks(conn, provider, limit):
        nonlocal embedding_called
        embedding_called = True
        return 0

    monkeypatch.setattr(
        production_ingestion,
        "embed_missing_chunks",
        fake_embed_missing_chunks,
    )

    summary = production_ingestion.run_production_ingestion(
        FakeConnection(),
        config,
        object(),
    )

    assert processed == ["Broken.md", "Healthy.md"]
    assert reconciliation_called is False
    assert embedding_called is True
    assert summary.indexed == 1
    assert summary.reconciliation_completed is False
    assert summary.failures == (
        production_ingestion.NoteFailure("Broken.md", "ValueError"),
    )
    assert summary.complete is False
    assert "PRIVATE_NOTE_MARKER" not in repr(summary)


def test_run_production_ingestion_reports_embedding_failure_and_retries_later(
    tmp_path,
    monkeypatch,
) -> None:
    from knowledge_rag import production_ingestion

    vault_path = (tmp_path / "production-vault").resolve()
    vault_path.mkdir()
    note_path = vault_path / "Indexed.md"
    note_path.write_text("# Indexed", encoding="utf-8")
    config = IngestionConfig(
        vault_path=vault_path,
        production=True,
        production_opt_in=True,
    )

    class FakeConnection:
        def transaction(self):
            return nullcontext()

    connection = FakeConnection()
    monkeypatch.setattr(
        production_ingestion,
        "discover_markdown_files",
        lambda path: [note_path],
    )
    monkeypatch.setattr(
        production_ingestion,
        "persist_note",
        lambda conn, vault, note: PersistResult.INDEXED,
    )
    monkeypatch.setattr(
        production_ingestion,
        "reconcile_inactive_documents",
        lambda conn, paths: 0,
    )

    def fail_embed_missing_chunks(conn, provider, limit):
        raise RuntimeError("PRIVATE_CHUNK_MARKER")

    monkeypatch.setattr(
        production_ingestion,
        "embed_missing_chunks",
        fail_embed_missing_chunks,
    )

    failed = production_ingestion.run_production_ingestion(
        connection,
        config,
        object(),
    )

    assert failed.indexed == 1
    assert failed.embedding_error == "RuntimeError"
    assert failed.complete is False
    assert "PRIVATE_CHUNK_MARKER" not in repr(failed)

    embedding_counts = iter((1, 0))
    monkeypatch.setattr(
        production_ingestion,
        "embed_missing_chunks",
        lambda conn, provider, limit: next(embedding_counts),
    )

    retry = production_ingestion.run_production_ingestion(
        connection,
        config,
        object(),
    )

    assert retry.embedded == 1
    assert retry.embedding_error is None
    assert retry.complete is True


def test_run_production_ingestion_aborts_before_persistence_on_discovery_failure(
    tmp_path,
    monkeypatch,
) -> None:
    from knowledge_rag import production_ingestion

    vault_path = (tmp_path / "production-vault").resolve()
    vault_path.mkdir()
    config = IngestionConfig(
        vault_path=vault_path,
        production=True,
        production_opt_in=True,
    )

    class FakeConnection:
        def transaction(self):
            return nullcontext()

    persisted = False

    def fail_discovery(path):
        raise PermissionError("vault discovery denied")

    def fake_persist_note(conn, vault, note):
        nonlocal persisted
        persisted = True
        return PersistResult.INDEXED

    monkeypatch.setattr(
        production_ingestion,
        "discover_markdown_files",
        fail_discovery,
    )
    monkeypatch.setattr(
        production_ingestion,
        "persist_note",
        fake_persist_note,
    )

    with pytest.raises(PermissionError):
        production_ingestion.run_production_ingestion(
            FakeConnection(),
            config,
            object(),
        )

    assert persisted is False


def test_run_production_ingestion_sanitizes_escaping_symlink(
    tmp_path,
    monkeypatch,
) -> None:
    from knowledge_rag import production_ingestion

    vault_path = (tmp_path / "production-vault").resolve()
    vault_path.mkdir()
    outside_note = tmp_path / "private-outside.md"
    outside_note.write_text("# Private marker", encoding="utf-8")
    escaped_note = vault_path / "Escape.md"

    try:
        escaped_note.symlink_to(outside_note)
    except OSError as exc:
        pytest.skip(f"Symlinks unavailable: {type(exc).__name__}")

    config = IngestionConfig(
        vault_path=vault_path,
        production=True,
        production_opt_in=True,
    )

    class FakeConnection:
        def transaction(self):
            return nullcontext()

    reconciliation_called = False
    persistence_called = False

    def fake_reconcile(conn, paths):
        nonlocal reconciliation_called
        reconciliation_called = True
        return 0

    def fake_persist(conn, vault, note):
        nonlocal persistence_called
        persistence_called = True
        return PersistResult.INDEXED

    monkeypatch.setattr(
        production_ingestion,
        "reconcile_inactive_documents",
        fake_reconcile,
    )
    monkeypatch.setattr(
        production_ingestion,
        "persist_note",
        fake_persist,
    )
    monkeypatch.setattr(
        production_ingestion,
        "embed_missing_chunks",
        lambda conn, provider, limit: 0,
    )

    summary = production_ingestion.run_production_ingestion(
        FakeConnection(),
        config,
        object(),
    )

    assert persistence_called is False
    assert reconciliation_called is False
    assert summary.failures == (
        production_ingestion.NoteFailure("Escape.md", "ValueError"),
    )
    assert summary.complete is False
    assert str(vault_path) not in repr(summary)
    assert str(outside_note) not in repr(summary)


def test_production_ingestion_preserves_embedded_document_on_recovery(
    tmp_path,
) -> None:
    """A missing note becomes inactive and restores without re-embedding."""
    from knowledge_rag import production_ingestion

    vault_path = (tmp_path / "production-vault").resolve()
    vault_path.mkdir()
    allowed_note = vault_path / "Allowed.md"
    local_note = vault_path / "Local.md"
    allowed_content = """---
type: reference
ai_access: allowed
---

# Allowed

Synthetic externally eligible content.
"""
    allowed_note.write_text(allowed_content, encoding="utf-8")
    local_note.write_text(
        """---
type: journal
ai_access: local-only
---

# Local

Synthetic local-only content.
""",
        encoding="utf-8",
    )
    config = IngestionConfig(
        vault_path=vault_path,
        production=True,
        production_opt_in=True,
    )
    provider = DeterministicEmbeddingProvider(dimensions=1536)

    with psycopg.connect(settings.database_url) as conn:
        create_production_test_tables(conn)

        first_run = production_ingestion.run_production_ingestion(
            conn,
            config,
            provider,
        )
        original = conn.execute(
            """
            SELECT d.document_id, c.chunk_id, c.embedding
            FROM documents d
            JOIN document_chunks c ON c.document_id = d.document_id
            WHERE d.source_path = 'Allowed.md';
            """
        ).fetchone()
        assert original is not None

        initial_documents = conn.execute(
            """
            SELECT source_path, is_active
            FROM documents
            ORDER BY source_path;
            """
        ).fetchall()
        initial_embeddings = conn.execute(
            """
            SELECT d.source_path, c.embedding IS NOT NULL
            FROM documents d
            JOIN document_chunks c ON c.document_id = d.document_id
            ORDER BY d.source_path;
            """
        ).fetchall()

        allowed_note.unlink()
        missing_run = production_ingestion.run_production_ingestion(
            conn,
            config,
            provider,
        )
        missing_state = conn.execute(
            """
            SELECT is_active
            FROM documents
            WHERE source_path = 'Allowed.md';
            """
        ).fetchone()

        allowed_note.write_text(allowed_content, encoding="utf-8")
        restored_run = production_ingestion.run_production_ingestion(
            conn,
            config,
            provider,
        )
        restored = conn.execute(
            """
            SELECT d.document_id, c.chunk_id, c.embedding, d.is_active
            FROM documents d
            JOIN document_chunks c ON c.document_id = d.document_id
            WHERE d.source_path = 'Allowed.md';
            """
        ).fetchone()
        documents = conn.execute(
            """
            SELECT source_path, is_active
            FROM documents
            ORDER BY source_path;
            """
        ).fetchall()
        embedding_rows = conn.execute(
            """
            SELECT d.source_path, c.embedding IS NOT NULL
            FROM documents d
            JOIN document_chunks c ON c.document_id = d.document_id
            ORDER BY d.source_path;
            """
        ).fetchall()

    assert first_run.embedded == 1
    assert first_run.complete is True
    assert initial_documents == [
        ("Allowed.md", True),
        ("Local.md", True),
    ]
    assert initial_embeddings == [
        ("Allowed.md", True),
        ("Local.md", False),
    ]
    assert missing_run.inactivated == 1
    assert missing_state == (False,)
    assert restored_run.reactivated == 1
    assert restored_run.embedded == 0
    assert restored is not None
    assert restored[0] == original[0]
    assert restored[1] == original[1]
    assert restored[2] == original[2]
    assert restored[3] is True
    assert documents == [
        ("Allowed.md", True),
        ("Local.md", True),
    ]
    assert embedding_rows == [
        ("Allowed.md", True),
        ("Local.md", False),
    ]


def test_production_ingestion_marks_former_path_inactive_after_rename(
    tmp_path,
) -> None:
    """A rename creates a current document and preserves the former record."""
    from knowledge_rag import production_ingestion

    vault_path = (tmp_path / "production-vault").resolve()
    vault_path.mkdir()
    original_note = vault_path / "Original.md"
    renamed_note = vault_path / "Renamed.md"
    original_note.write_text(
        """---
type: reference
ai_access: allowed
---

# Original

Synthetic rename coverage.
""",
        encoding="utf-8",
    )
    config = IngestionConfig(
        vault_path=vault_path,
        production=True,
        production_opt_in=True,
    )
    provider = DeterministicEmbeddingProvider(dimensions=1536)

    with psycopg.connect(settings.database_url) as conn:
        create_production_test_tables(conn)
        production_ingestion.run_production_ingestion(
            conn,
            config,
            provider,
        )

        original_note.rename(renamed_note)
        renamed_run = production_ingestion.run_production_ingestion(
            conn,
            config,
            provider,
        )
        lifecycle_rows = conn.execute(
            """
            SELECT source_path, is_active
            FROM documents
            ORDER BY source_path;
            """
        ).fetchall()

    assert renamed_run.indexed == 1
    assert renamed_run.inactivated == 1
    assert lifecycle_rows == [
        ("Original.md", False),
        ("Renamed.md", True),
    ]
