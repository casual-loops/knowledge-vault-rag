# Production Vault Indexing Workflow Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox syntax for tracking.

**Goal:** Build a safe, resumable command that ingests the configured private vault, marks missing notes inactive, and embeds every eligible active chunk.

**Architecture:** scripts/ingest_production_vault.py remains a thin entry point. A new knowledge_rag.production_ingestion module coordinates discovery, persistence, reconciliation, embedding, and the aggregate result. Persistence owns document lifecycle changes, while retrieval and embedding queries enforce active state.

**Tech Stack:** Python 3.12+, PostgreSQL 17, Psycopg 3, pgvector, python-frontmatter, pytest, Ruff

**Spec:** docs/superpowers/specs/2026-09-17-production-vault-indexing-design.md

## Global Constraints

1. Merge Issue #63 first, then rebase issue-36-production-vault-indexing onto the updated main branch.
2. Preserve the metadata-aware digest interface delivered by Issue #63.
3. Production ingestion requires KNOWLEDGE_RAG_ENABLE_PRODUCTION_INGESTION=true.
4. Missing and newly excluded documents become inactive and are never hard deleted.
5. Inactive documents retain metadata, chunks, and embeddings.
6. Only active documents with ai_access equal to allowed may reach the embedding provider.
7. Failures may expose only a vault-relative note reference and exception type.
8. Output must not contain note bodies, chunk content, embeddings, absolute paths, or frontmatter values.
9. The first real vault indexing run waits until Issues #36 through #39 are complete.
10. Run tests with python -m pytest and lint with python -m ruff check .
11. The pre-change baseline is 158 passing tests.

---

### Task 1: Add recoverable document lifecycle state

**Files:**
1. Modify: sql/002_schema.sql
2. Create: sql/004_add_document_active_state.sql
3. Create: tests/test_schema_migrations.py
4. Modify: tests/test_persistence.py
5. Modify: tests/test_retrieval.py

**Interfaces:**
1. Consumes: Existing documents table.
2. Produces: documents.is_active BOOLEAN NOT NULL DEFAULT TRUE.

- [ ] **Step 1: Write the failing migration test**

~~~python
from pathlib import Path

import psycopg

from knowledge_rag.config import settings


def test_active_state_migration_preserves_existing_documents() -> None:
    migration = Path(
        "sql/004_add_document_active_state.sql"
    ).read_text(encoding="utf-8")

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
~~~

- [ ] **Step 2: Verify that the test fails**

Run: python -m pytest tests/test_schema_migrations.py -v

Expected: FAIL because sql/004_add_document_active_state.sql does not exist.

- [ ] **Step 3: Add the migration**

~~~sql
ALTER TABLE documents
ADD COLUMN IF NOT EXISTS is_active BOOLEAN NOT NULL DEFAULT TRUE;
~~~

- [ ] **Step 4: Update all schema definitions**

Add this field after content_hash in sql/002_schema.sql and in the temporary documents tables in tests/test_persistence.py and tests/test_retrieval.py:

~~~sql
is_active BOOLEAN NOT NULL DEFAULT TRUE,
~~~

- [ ] **Step 5: Verify the schema change**

Run: python -m pytest tests/test_schema_migrations.py tests/test_persistence.py tests/test_retrieval.py -v

Expected: PASS.

- [ ] **Step 6: Commit**

~~~powershell
git add sql/002_schema.sql sql/004_add_document_active_state.sql tests/test_schema_migrations.py tests/test_persistence.py tests/test_retrieval.py
git commit -m "feat: add recoverable document lifecycle state"
~~~

### Task 2: Implement activation, policy deactivation, and reconciliation

**Files:**
1. Modify: src/knowledge_rag/ingest/persistence.py
2. Modify: tests/test_persistence.py

**Interfaces:**
1. Consumes: documents.is_active and the Issue #63 digest.
2. Produces: PersistResult.REACTIVATED.
3. Produces: deactivate_document(conn: Connection, source_path: str) -> bool.
4. Produces: reconcile_inactive_documents(conn: Connection, discovered_source_paths: set[str]) -> int.

- [ ] **Step 1: Write a failing reactivation test**

Create an allowed synthetic note, persist and embed it, set its documents row to is_active = FALSE, and call persist_note again. Assert:

~~~python
assert result is PersistResult.REACTIVATED
assert restored_document_id == original_document_id
assert restored_chunk_id == original_chunk_id
assert restored_embedding == original_embedding
assert restored_is_active is True
~~~

- [ ] **Step 2: Write a failing policy-deactivation test**

Persist an allowed note, rewrite only ai_access as exclude, call persist_note, and assert:

~~~python
assert result is PersistResult.EXCLUDED
assert stored_is_active is False
~~~

- [ ] **Step 3: Write the failing reconciliation test**

~~~python
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
~~~

- [ ] **Step 4: Verify the tests fail**

Run: python -m pytest tests/test_persistence.py -k "reactivated or deactivates or reconciliation" -v

Expected: FAIL because the lifecycle interfaces do not exist.

- [ ] **Step 5: Add lifecycle helpers**

~~~python
class PersistResult(StrEnum):
    INDEXED = "indexed"
    UNCHANGED = "unchanged"
    EXCLUDED = "excluded"
    REACTIVATED = "reactivated"


def deactivate_document(
    conn: Connection,
    source_path: str,
) -> bool:
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
~~~

- [ ] **Step 6: Update persist_note state transitions**

Compute source_path before policy checks. Call deactivate_document before every policy-driven EXCLUDED return.

Load active state with the existing row:

~~~python
existing = conn.execute(
    """
    SELECT document_id, content_hash, is_active
    FROM documents
    WHERE source_path = %s;
    """,
    (source_path,),
).fetchone()
~~~

Keep the Issue #63 digest statement unchanged. Replace the unchanged return:

~~~python
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
~~~

Add is_active to the insert columns, use TRUE in VALUES, and set is_active = TRUE in the conflict update.

- [ ] **Step 7: Verify and commit**

Run: python -m pytest tests/test_persistence.py -v

Expected: PASS.

~~~powershell
git add src/knowledge_rag/ingest/persistence.py tests/test_persistence.py
git commit -m "feat: reconcile inactive vault documents"
~~~

### Task 3: Exclude inactive documents from embedding and retrieval

**Files:**
1. Modify: src/knowledge_rag/embedding_store.py
2. Modify: src/knowledge_rag/retrieval.py
3. Modify: tests/test_persistence.py
4. Modify: tests/test_retrieval.py
5. Modify: tests/test_lexical_search.py

**Interfaces:**
1. Consumes: documents.is_active.
2. Produces: Active-state enforcement for embedding, semantic retrieval, lexical retrieval, and therefore hybrid retrieval.

- [ ] **Step 1: Write failing eligibility tests**

Add an embedding provider that raises AssertionError if called. Persist an allowed note, mark it inactive, call embed_missing_chunks, and assert the returned count is zero.

Extend the semantic test helper with is_active: bool = True. Insert an inactive embedded note and assert semantic_search returns an empty list.

Add this lexical assertion:

~~~python
def test_lexical_search_requires_active_documents() -> None:
    conn: Any = FakeConnection([])
    lexical_search(conn, query="synthetic")
    assert conn.executed_sql is not None
    assert "d.is_active = TRUE" in conn.executed_sql
~~~

- [ ] **Step 2: Verify that the tests fail**

Run: python -m pytest tests/test_persistence.py::test_inactive_allowed_document_is_not_embedded tests/test_retrieval.py::test_semantic_search_excludes_inactive_documents tests/test_lexical_search.py::test_lexical_search_requires_active_documents -v

Expected: FAIL because the SQL has no active-state filters.

- [ ] **Step 3: Add the SQL predicates**

Embedding:

~~~sql
WHERE c.embedding IS NULL
  AND d.is_active = TRUE
  AND d.ai_access = 'allowed'
~~~

Semantic retrieval:

~~~sql
WHERE c.embedding IS NOT NULL
  AND d.is_active = TRUE
~~~

Lexical retrieval, before optional metadata filters:

~~~sql
AND d.is_active = TRUE
~~~

- [ ] **Step 4: Verify retrieval and embedding behavior**

Run: python -m pytest tests/test_persistence.py tests/test_retrieval.py tests/test_lexical_search.py tests/test_hybrid_search.py tests/test_hybrid_retrieval_regressions.py -v

Expected: PASS. Hybrid retrieval requires no direct edit because its inputs are filtered.

- [ ] **Step 5: Commit**

~~~powershell
git add src/knowledge_rag/embedding_store.py src/knowledge_rag/retrieval.py tests/test_persistence.py tests/test_retrieval.py tests/test_lexical_search.py
git commit -m "fix: exclude inactive documents from RAG processing"
~~~

### Task 4: Build the production ingestion orchestrator

**Files:**
1. Create: src/knowledge_rag/production_ingestion.py
2. Create: tests/test_production_ingestion.py

**Interfaces:**
1. Consumes: IngestionConfig, EmbeddingProvider, persist_note, reconcile_inactive_documents, and embed_missing_chunks.
2. Produces: NoteFailure(reference: str, error_type: str).
3. Produces: ProductionIngestionSummary.
4. Produces: run_production_ingestion(conn, config, provider, embedding_batch_size=100).

- [ ] **Step 1: Write the failing success-path test**

Use a fake connection whose transaction method returns contextlib.nullcontext. Discover two temporary notes. Return INDEXED for one and UNCHANGED for the other. Make reconciliation return 1 and embedding batches return 2, 1, then 0.

Assert:

~~~python
assert summary.discovered == 2
assert summary.indexed == 1
assert summary.unchanged == 1
assert summary.inactivated == 1
assert summary.embedded == 3
assert summary.failures == ()
assert summary.reconciliation_completed is True
assert summary.complete is True
~~~

- [ ] **Step 2: Verify the import failure**

Run: python -m pytest tests/test_production_ingestion.py -v

Expected: FAIL because knowledge_rag.production_ingestion does not exist.

- [ ] **Step 3: Create the orchestrator types**

~~~python
from dataclasses import dataclass

from psycopg import Connection

from knowledge_rag.embedding_store import embed_missing_chunks
from knowledge_rag.embeddings import EmbeddingProvider
from knowledge_rag.ingest.markdown import discover_markdown_files
from knowledge_rag.ingest.persistence import (
    PersistResult,
    persist_note,
    reconcile_inactive_documents,
)
from knowledge_rag.ingestion_config import IngestionConfig
from knowledge_rag.logging_utils import safe_note_reference


@dataclass(frozen=True, slots=True)
class NoteFailure:
    reference: str
    error_type: str


@dataclass(frozen=True, slots=True)
class ProductionIngestionSummary:
    discovered: int
    indexed: int
    unchanged: int
    excluded: int
    reactivated: int
    inactivated: int
    embedded: int
    failures: tuple[NoteFailure, ...]
    reconciliation_completed: bool
    embedding_error: str | None

    @property
    def complete(self) -> bool:
        return (
            not self.failures
            and self.reconciliation_completed
            and self.embedding_error is None
        )
~~~

- [ ] **Step 4: Implement run_production_ingestion**

~~~python
def run_production_ingestion(
    conn: Connection,
    config: IngestionConfig,
    provider: EmbeddingProvider,
    *,
    embedding_batch_size: int = 100,
) -> ProductionIngestionSummary:
    note_paths = discover_markdown_files(config.vault_path)
    discovered_paths = {
        safe_note_reference(config.vault_path, path)
        for path in note_paths
    }
    counts = {result: 0 for result in PersistResult}
    failures: list[NoteFailure] = []

    for note_path in note_paths:
        try:
            with conn.transaction():
                result = persist_note(
                    conn,
                    config.vault_path,
                    note_path,
                )
            counts[result] += 1
        except Exception as exc:
            failures.append(
                NoteFailure(
                    safe_note_reference(config.vault_path, note_path),
                    type(exc).__name__,
                )
            )

    reconciliation_completed = not failures
    inactivated = 0

    if reconciliation_completed:
        with conn.transaction():
            inactivated = reconcile_inactive_documents(
                conn,
                discovered_paths,
            )

    embedded = 0
    embedding_error: str | None = None

    try:
        while True:
            with conn.transaction():
                count = embed_missing_chunks(
                    conn,
                    provider,
                    limit=embedding_batch_size,
                )
            if count == 0:
                break
            embedded += count
    except Exception as exc:
        embedding_error = type(exc).__name__

    return ProductionIngestionSummary(
        discovered=len(note_paths),
        indexed=counts[PersistResult.INDEXED],
        unchanged=counts[PersistResult.UNCHANGED],
        excluded=counts[PersistResult.EXCLUDED],
        reactivated=counts[PersistResult.REACTIVATED],
        inactivated=inactivated,
        embedded=embedded,
        failures=tuple(failures),
        reconciliation_completed=reconciliation_completed,
        embedding_error=embedding_error,
    )
~~~

Use imports from the modules named in the Interfaces section.

- [ ] **Step 5: Verify and commit**

Run: python -m pytest tests/test_production_ingestion.py -v

Expected: PASS.

~~~powershell
git add src/knowledge_rag/production_ingestion.py tests/test_production_ingestion.py
git commit -m "feat: orchestrate production vault indexing"
~~~

### Task 5: Verify failure isolation and resumability

**Files:**
1. Modify: tests/test_production_ingestion.py
2. Modify only if evidence requires it: src/knowledge_rag/production_ingestion.py

**Interfaces:**
1. Consumes: run_production_ingestion.
2. Produces: Verified note isolation, reconciliation suppression, embedding recovery, and discovery abort behavior.

- [ ] **Step 1: Add the note-failure test**

Make discovery return Broken.md and Healthy.md. Make persist_note raise ValueError containing a private marker for Broken.md and return INDEXED for Healthy.md. Make the reconciliation fake record whether it ran. Make embedding set embedding_called to true and return zero.

Assert:

~~~python
assert processed == ["Broken.md", "Healthy.md"]
assert reconciliation_called is False
assert embedding_called is True
assert summary.indexed == 1
assert summary.reconciliation_completed is False
assert summary.failures == (
    NoteFailure("Broken.md", "ValueError"),
)
assert summary.complete is False
assert "private marker" not in repr(summary)
~~~

- [ ] **Step 2: Add embedding and discovery failure tests**

Make embed_missing_chunks raise RuntimeError containing a private chunk marker after successful note processing. Assert indexed equals 1, embedding_error equals RuntimeError, complete is false, and the private marker is absent from repr(summary).

Then replace the embedding fake with an iterator that returns 1 and then 0. Run the workflow again and assert:

~~~python
assert retry.embedded == 1
assert retry.embedding_error is None
assert retry.complete is True
~~~

This proves that a later run resumes work left with a null embedding.

Make discover_markdown_files raise PermissionError. Use pytest.raises(PermissionError) and assert persist_note was never called.

- [ ] **Step 3: Verify and commit**

Run: python -m pytest tests/test_production_ingestion.py -v

Expected: PASS.

~~~powershell
git add src/knowledge_rag/production_ingestion.py tests/test_production_ingestion.py
git commit -m "test: verify production ingestion recovery behavior"
~~~

### Task 6: Convert the production script into a safe entry point

**Files:**
1. Modify: scripts/ingest_production_vault.py
2. Modify: tests/test_ingest_production_vault.py

**Interfaces:**
1. Consumes: Configuration, database, provider, and orchestrator factories.
2. Produces: format_summary(summary: ProductionIngestionSummary) -> str.
3. Produces: main() -> int.

- [ ] **Step 1: Write failing runner tests**

Create a complete summary and assert main returns 0, passes the connection, config, and provider to run_production_ingestion, and prints complete: true.

Create an incomplete summary with NoteFailure("Broken.md", "ValueError"). Assert main returns 1, prints the safe reference and error type, and omits the absolute temporary vault path.

- [ ] **Step 2: Verify failure**

Run: python -m pytest tests/test_ingest_production_vault.py -v

Expected: FAIL because the script still performs direct note iteration.

- [ ] **Step 3: Implement the entry point**

~~~python
def format_summary(summary: ProductionIngestionSummary) -> str:
    lines = [
        f"complete: {str(summary.complete).lower()}",
        f"discovered: {summary.discovered}",
        f"indexed: {summary.indexed}",
        f"unchanged: {summary.unchanged}",
        f"excluded: {summary.excluded}",
        f"reactivated: {summary.reactivated}",
        f"inactivated: {summary.inactivated}",
        f"embedded: {summary.embedded}",
        (
            "reconciliation_completed: "
            f"{str(summary.reconciliation_completed).lower()}"
        ),
    ]

    for failure in summary.failures:
        lines.append(
            f"failure: {failure.reference}: {failure.error_type}"
        )

    if summary.embedding_error is not None:
        lines.append(
            f"embedding_failure: {summary.embedding_error}"
        )

    return "\n".join(lines)


def main() -> int:
    config = get_production_ingestion_config()
    provider = get_embedding_provider()

    with get_connection() as conn:
        summary = run_production_ingestion(
            conn,
            config,
            provider,
        )

    print(format_summary(summary))
    return 0 if summary.complete else 1


if __name__ == "__main__":
    raise SystemExit(main())
~~~

Import get_embedding_provider, ProductionIngestionSummary, and run_production_ingestion. Configuration must be loaded before provider or database creation.

- [ ] **Step 4: Verify and commit**

Run: python -m pytest tests/test_ingest_production_vault.py -v

Expected: PASS.

~~~powershell
git add scripts/ingest_production_vault.py tests/test_ingest_production_vault.py
git commit -m "feat: run complete production indexing pipeline"
~~~

### Task 7: Add database-backed lifecycle coverage and complete verification

**Files:**
1. Modify: tests/test_production_ingestion.py
2. Modify implementation files only if the integration test demonstrates a defect.

**Interfaces:**
1. Consumes: Complete production workflow.
2. Produces: Database-backed proof of indexing, embedding, inactivity, reactivation, and privacy enforcement.

- [ ] **Step 1: Add the lifecycle integration test**

Create updated temporary documents and document_chunks tables. Create Allowed.md with ai_access allowed and Local.md with ai_access local-only. Run with DeterministicEmbeddingProvider.

Assert both documents are active and only Allowed.md has an embedding. Record the Allowed.md document_id, chunk_id, and embedding. Delete Allowed.md, rerun, and assert it is inactive. Restore identical content, rerun, and assert the same document_id, chunk_id, and embedding are active again.

Add a separate renamed-note test. Index Original.md, rename it to Renamed.md, and rerun the workflow. Assert:

~~~python
assert lifecycle_rows == [
    ("Original.md", False),
    ("Renamed.md", True),
]
~~~

Final assertions for the restoration test:

~~~python
assert documents == [
    ("Allowed.md", True),
    ("Local.md", True),
]
assert embedding_rows == [
    ("Allowed.md", True),
    ("Local.md", False),
]
~~~

- [ ] **Step 2: Run focused workflow tests**

Run: python -m pytest tests/test_production_ingestion.py -v

Expected: PASS.

- [ ] **Step 3: Run all affected tests**

Run: python -m pytest tests/test_ingestion_config.py tests/test_ingest_production_vault.py tests/test_production_ingestion.py tests/test_persistence.py tests/test_schema_migrations.py tests/test_retrieval.py tests/test_lexical_search.py tests/test_hybrid_search.py tests/test_hybrid_retrieval_regressions.py -v

Expected: PASS.

- [ ] **Step 4: Run full verification**

Run: python -m pytest

Expected: All 158 baseline tests plus Issue #63 and Issue #36 tests pass.

Run: python -m ruff check .

Expected: No lint errors.

- [ ] **Step 5: Review privacy and scope**

Run: git diff origin/main...HEAD

Confirm:

1. No note body, frontmatter, embedding, or absolute path logging exists.
2. Active-state filters exist in semantic, lexical, and embedding SQL.
3. Reconciliation requires zero note failures.
4. No real vault path, hostname, credential, or private identifier appears.
5. No Issue #37, #38, #39, or performance-refactor behavior was added.

- [ ] **Step 6: Commit final integration coverage**

~~~powershell
git add tests/test_production_ingestion.py
git commit -m "test: cover production indexing lifecycle"
~~~

- [ ] **Step 7: Verify branch state**

Run: git status

Expected: Clean working tree.

Run: git log --oneline origin/main..HEAD

Expected: Design, plan, implementation, and focused test commits are present.
