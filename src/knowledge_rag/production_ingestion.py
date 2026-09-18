"""Production vault ingestion workflow orchestration."""

from dataclasses import dataclass
from pathlib import Path

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
    """A sanitized note-specific production ingestion failure."""

    reference: str
    error_type: str


@dataclass(frozen=True, slots=True)
class ProductionIngestionSummary:
    """Aggregate result of one production vault indexing run."""

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
        """Whether ingestion, reconciliation, and embedding all completed."""
        return (
            not self.failures
            and self.reconciliation_completed
            and self.embedding_error is None
        )


def run_production_ingestion(
    conn: Connection,
    config: IngestionConfig,
    provider: EmbeddingProvider,
    *,
    embedding_batch_size: int = 100,
) -> ProductionIngestionSummary:
    """Ingest, reconcile, and embed the configured production vault."""
    note_paths = discover_markdown_files(config.vault_path)
    discovered_paths: set[str] = set()
    counts = {result: 0 for result in PersistResult}
    failures: list[NoteFailure] = []

    for note_path in note_paths:
        fallback_reference = _unresolved_note_reference(
            config.vault_path,
            note_path,
        )

        try:
            reference = safe_note_reference(
                config.vault_path,
                note_path,
            )
        except Exception as exc:  # noqa: BLE001
            failures.append(
                NoteFailure(
                    fallback_reference,
                    type(exc).__name__,
                )
            )
            continue

        discovered_paths.add(reference)

        try:
            with conn.transaction():
                result = persist_note(
                    conn,
                    config.vault_path,
                    note_path,
                )
            counts[result] += 1
        except Exception as exc:  # noqa: BLE001
            failures.append(
                NoteFailure(
                    reference,
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
    except Exception as exc:  # noqa: BLE001
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


def _unresolved_note_reference(
    vault_path: Path,
    note_path: Path,
) -> str:
    """Return a non-resolving relative reference for invalid path reporting."""
    try:
        return note_path.absolute().relative_to(
            vault_path.absolute()
        ).as_posix()
    except (OSError, ValueError):
        return note_path.name or "<invalid-note-path>"
