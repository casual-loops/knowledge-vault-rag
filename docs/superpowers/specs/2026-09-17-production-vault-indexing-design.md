# Production Vault Indexing Workflow Design

Status: Approved for implementation planning
Date: 2026-09-17
Issue: #36, Implement safe production vault indexing workflow
Dependency: #63, Detect metadata-only note changes during ingestion

## Context

The repository already provides validated production ingestion configuration, Markdown discovery, policy-aware note persistence, chunk generation, embedding generation, and semantic, lexical, and hybrid retrieval. The current production runner only iterates through discovered notes and prints each persistence result.

Issue #36 must turn those pieces into a safe, resumable production workflow. The workflow must ingest the configured private vault replica only after explicit opt-in, preserve stable document identity, process changed content incrementally, skip unchanged content, enforce note policies before persistence or external-provider use, complete embedding generation, reconcile removed notes without destructive deletion, and report a concise operational summary.

## Goals

1. Provide one production command that completes ingestion, reconciliation, and embedding.
2. Preserve indexed documents, chunks, and embeddings when a note disappears.
3. Prevent inactive or policy-excluded content from appearing in retrieval or being sent to an external embedding provider.
4. Continue processing independent notes after a note-specific failure.
5. Skip reconciliation when discovery or note processing is incomplete.
6. Make interrupted runs safely resumable.
7. Produce operator-readable output without note bodies or frontmatter values.
8. Provide deterministic coverage using synthetic production-like fixtures.

## Non-goals

1. Dry-run behavior and audit reporting, which belong to Issue #37.
2. Formal production privacy preflight, which belongs to Issue #38.
3. The production indexing runbook, which belongs to Issue #39.
4. Metadata-aware change hashing itself, which belongs to Issue #63 and must be merged first.
5. Embedding batching, bulk persistence, connection pooling, or retrieval index optimization.
6. Automatic hard deletion of inactive records.
7. Automatic identity preservation across a file rename.

## Approved decisions

1. Missing notes are marked inactive rather than deleted.
2. The production command completes the full ingestion and embedding pipeline.
3. A note-specific failure does not stop independent notes.
4. Any discovery or note-processing failure suppresses missing-note reconciliation.
5. The workflow uses a dedicated orchestrator module.
6. Issue #63 is a required dependency because current content-only hashing can miss metadata-only policy changes.

## Architecture

The script at `scripts/ingest_production_vault.py` remains a thin process entry point. A new module at `src/knowledge_rag/production_ingestion.py` coordinates the workflow and returns an immutable structured result.

The runner is responsible for:

1. Loading and validating production configuration.
2. Opening the database connection.
3. Creating the configured embedding provider.
4. Calling the production ingestion orchestrator.
5. Printing the aggregate summary.
6. Returning a nonzero process status when the run is incomplete.

The orchestrator is responsible for:

1. Discovering Markdown files.
2. Processing each note in an isolated transaction.
3. Recording sanitized note failures and continuing.
4. Reconciling missing documents only after a complete successful scan.
5. Embedding all eligible chunks in bounded batches.
6. Returning a `ProductionIngestionSummary`.

Persistence remains responsible for document and chunk state transitions. Retrieval remains responsible for excluding inactive documents. The embedding store remains responsible for selecting only externally eligible chunks.

## Data flow

1. Validate the absolute production vault path and explicit opt-in before database or provider activity.
2. Discover the complete Markdown file set.
3. Convert every discovered path to its normalized vault-relative source path.
4. Process each note:
   1. Evaluate path, note-type, and `ai_access` policy.
   2. Deactivate an existing document if its current policy excludes it.
   3. Insert a new document and chunks when the source path is new.
   4. Update the document and rebuild chunks when its change identity differs.
   5. Return unchanged without rebuilding chunks when active content and normalized metadata are unchanged.
   6. Reactivate an inactive document when the same source path returns.
5. If discovery and all note operations succeed, mark active stored documents inactive when their source paths are absent from the discovered path set.
6. Generate embeddings for active chunks whose documents have `ai_access = 'allowed'` and whose embedding is null.
7. Repeat bounded embedding batches until no eligible chunks remain.
8. Return and print the final summary.

## Document lifecycle

Add `documents.is_active BOOLEAN NOT NULL DEFAULT TRUE`.

The base schema will include the field for new installations. A numbered migration will add it to existing installations with a default of true, preserving current behavior.

Lifecycle rules:

1. New documents are active.
2. Updated documents are active.
3. A previously inactive document found at the same source path becomes active.
4. A restored document with unchanged content and metadata reuses existing chunks and embeddings.
5. A restored document with changed content follows the normal update and reembedding path.
6. A document absent from a complete successful scan becomes inactive.
7. A document newly excluded by path, note type, or `ai_access` becomes inactive.
8. Inactive documents retain their metadata, chunks, and embeddings.
9. A renamed note creates a new active document at the new source path. The former source path becomes inactive after reconciliation.

Hard deletion and automatic cross-path identity matching are intentionally excluded.

## Retrieval and external-provider enforcement

Both `semantic_search()` and `lexical_search()` must require `documents.is_active = TRUE`. Hybrid retrieval inherits the constraint because it fuses semantic and lexical results.

`load_chunks_missing_embeddings()` must require:

1. `documents.is_active = TRUE`
2. `documents.ai_access = 'allowed'`
3. `document_chunks.embedding IS NULL`

Inactive, local-only, and excluded content must never be submitted to the external embedding provider.

Issue #63 must update the change identity to include normalized metadata that affects persistence, retrieval, or policy. Without that fix, a metadata-only transition from `allowed` to `local-only` could leave stale policy in the database.

## Transactions and failure handling

Each note is processed in its own database transaction.

If one note fails:

1. Only that note transaction rolls back.
2. The failure is recorded using a vault-relative reference and a safe error category.
3. Remaining notes continue.
4. Missing-note reconciliation is skipped.
5. Successfully processed notes remain committed and may be embedded.
6. The final process status is nonzero.

Reconciliation uses a separate transaction and runs only when discovery and all note-processing operations succeed.

Embedding begins after note processing and any permitted reconciliation. Embeddings are processed in bounded batches. If the provider or database fails during embedding:

1. Completed ingestion and reconciliation remain committed.
2. Completed embedding batches remain committed.
3. The failure is recorded without chunk content.
4. The process status is nonzero.
5. A later run resumes from eligible chunks whose embeddings remain null.

A discovery failure aborts the workflow before note processing and reconciliation.

## Structured summary

`ProductionIngestionSummary` will be an immutable dataclass containing:

1. Discovered note count
2. Indexed note count
3. Unchanged note count
4. Excluded note count
5. Reactivated note count
6. Newly inactive document count
7. Generated embedding count
8. Sanitized note failures
9. Reconciliation status
10. Optional embedding-stage failure
11. Overall completion status

The normal console summary will contain aggregate counts. Failure details may contain vault-relative note references and safe error categories. It must not include note bodies, chunk content, embeddings, absolute vault paths, or frontmatter values.

## Expected code changes

1. Add `src/knowledge_rag/production_ingestion.py`.
2. Update `scripts/ingest_production_vault.py` to call the orchestrator.
3. Update `src/knowledge_rag/ingest/persistence.py` for activation and policy-driven deactivation.
4. Add a focused reconciliation function near the document persistence layer.
5. Update `src/knowledge_rag/embedding_store.py` to exclude inactive documents.
6. Update `src/knowledge_rag/retrieval.py` to exclude inactive documents.
7. Update `sql/002_schema.sql` for new installations.
8. Add `sql/004_add_document_active_state.sql` for existing installations.
9. Update affected temporary test schemas.
10. Expand production runner, persistence, embedding, and retrieval tests.
11. Add focused orchestrator tests using synthetic production-like fixtures.

## Verification scenarios

1. A new eligible note is persisted, chunked, and embedded.
2. A local-only note is persisted but never sent to the embedding provider.
3. An excluded note is not searchable.
4. An unchanged note does not rebuild chunks or regenerate embeddings.
5. A changed note receives new chunks and embeddings.
6. A missing note becomes inactive after a complete successful scan.
7. An inactive note is absent from semantic, lexical, and hybrid retrieval.
8. A restored note reactivates without creating a duplicate document.
9. A restored unchanged note retains its chunks and embeddings.
10. A renamed note produces one inactive former record and one active current record.
11. A newly excluded note deactivates its previous searchable record.
12. A note failure does not stop independent notes and suppresses reconciliation.
13. Successfully processed notes may still be embedded after another note fails.
14. An embedding failure preserves ingestion and resumes on a later run.
15. Summary output contains no note bodies or frontmatter values.
16. A metadata-only `ai_access` change is persisted before embedding eligibility is evaluated, using the behavior implemented by Issue #63.
17. The complete test suite remains green. The pre-change baseline is 158 passing tests.

## Rollout

1. Complete and merge Issue #63.
2. Implement Issue #36 using test-driven development.
3. Apply the lifecycle migration to the development database.
4. Run the full automated suite.
5. Run the production workflow only against synthetic fixtures during development.
6. Complete Issue #37 dry-run behavior.
7. Complete Issue #38 privacy preflight.
8. Use Issue #39 to document the first real production indexing procedure.

The first real vault indexing run must not occur until Issues #36 through #39 are complete.
