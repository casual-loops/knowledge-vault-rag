"""Read-only production vault ingestion audit."""

from dataclasses import dataclass
from enum import StrEnum
from pathlib import Path
from typing import Any, Protocol

from knowledge_rag.config import settings
from knowledge_rag.ingest.identity import content_hash
from knowledge_rag.ingest.markdown import (
    discover_markdown_files,
    load_markdown,
)
from knowledge_rag.ingestion_config import IngestionConfig
from knowledge_rag.logging_utils import safe_note_reference
from knowledge_rag.policy import (
    evaluate_note_policy,
    note_type_is_excluded,
    path_is_excluded,
)


class ReadOnlyDatabaseConnection(Protocol):
    """Minimal database interface required by dry-run evaluation."""

    def execute(
        self,
        *args: Any,
        **kwargs: Any,
    ) -> Any:
        ...


class DryRunAction(StrEnum):
    """Possible actions reported by a production dry-run."""

    INDEX = "would-index"
    UPDATE = "would-update"
    SKIP = "unchanged"
    EXCLUDE = "excluded"


class ExternalEligibility(StrEnum):
    """External-provider eligibility reported without provider use."""

    ELIGIBLE = "external-eligible"
    LOCAL_ONLY = "local-only"
    EXCLUDED = "excluded"


@dataclass(frozen=True, slots=True)
class DryRunNoteResult:
    """Sanitized audit result for one discovered note."""

    reference: str
    action: DryRunAction
    external_eligibility: ExternalEligibility


@dataclass(frozen=True, slots=True)
class DryRunFailure:
    """Sanitized note-specific dry-run failure."""

    reference: str
    error_type: str


@dataclass(frozen=True, slots=True)
class ProductionDryRunSummary:
    """Aggregate result of a read-only production ingestion audit."""

    discovered: int
    would_index: int
    would_update: int
    unchanged: int
    excluded: int
    external_eligible: int
    local_only: int
    results: tuple[DryRunNoteResult, ...]
    failures: tuple[DryRunFailure, ...]

    @property
    def complete(self) -> bool:
        """Whether every discovered note was evaluated successfully."""

        return not self.failures


def run_production_dry_run(
    conn: ReadOnlyDatabaseConnection,
    config: IngestionConfig,
) -> ProductionDryRunSummary:
    """Evaluate production ingestion without changing persistent state."""

    note_paths = discover_markdown_files(
        config.vault_path,
    )

    results: list[DryRunNoteResult] = []
    failures: list[DryRunFailure] = []

    for note_path in note_paths:
        try:
            reference = safe_note_reference(
                config.vault_path,
                note_path,
            )
        except Exception as exc:  # noqa: BLE001
            failures.append(
                DryRunFailure(
                    reference=note_path.name or "<invalid-note-path>",
                    error_type=type(exc).__name__,
                )
            )
            continue

        try:
            result = _evaluate_note(
                conn,
                config,
                note_path,
                reference,
            )
            results.append(result)
        except Exception as exc:  # noqa: BLE001
            failures.append(
                DryRunFailure(
                    reference=reference,
                    error_type=type(exc).__name__,
                )
            )

    return _build_summary(
        discovered=len(note_paths),
        results=results,
        failures=failures,
    )


def _evaluate_note(
    conn: ReadOnlyDatabaseConnection,
    config: IngestionConfig,
    note_path: Path,
    reference: str,
) -> DryRunNoteResult:
    """Evaluate one note without performing any persistent writes."""

    if path_is_excluded(
        config.vault_path,
        note_path,
        settings.excluded_paths,
    ):
        return DryRunNoteResult(
            reference=reference,
            action=DryRunAction.EXCLUDE,
            external_eligibility=ExternalEligibility.EXCLUDED,
        )

    document = load_markdown(
        note_path,
    )

    if note_type_is_excluded(
        document.metadata,
        settings.excluded_note_types,
    ):
        return DryRunNoteResult(
            reference=reference,
            action=DryRunAction.EXCLUDE,
            external_eligibility=ExternalEligibility.EXCLUDED,
        )

    policy = evaluate_note_policy(
        document.metadata,
    )

    if not policy.may_index:
        return DryRunNoteResult(
            reference=reference,
            action=DryRunAction.EXCLUDE,
            external_eligibility=ExternalEligibility.EXCLUDED,
        )

    eligibility = (
        ExternalEligibility.ELIGIBLE
        if policy.may_use_external_embeddings
        else ExternalEligibility.LOCAL_ONLY
    )

    digest = content_hash(
        document.content,
        document.metadata,
    )

    existing = conn.execute(
        """
        SELECT content_hash, is_active
        FROM documents
        WHERE source_path = %s;
        """,
        (reference,),
    ).fetchone()

    if existing is None:
        action = DryRunAction.INDEX
    elif existing[0] != digest or not existing[1]:
        action = DryRunAction.UPDATE
    else:
        action = DryRunAction.SKIP

    return DryRunNoteResult(
        reference=reference,
        action=action,
        external_eligibility=eligibility,
    )


def _build_summary(
    *,
    discovered: int,
    results: list[DryRunNoteResult],
    failures: list[DryRunFailure],
) -> ProductionDryRunSummary:
    """Build aggregate counts from sanitized per-note results."""

    return ProductionDryRunSummary(
        discovered=discovered,
        would_index=sum(
            result.action is DryRunAction.INDEX
            for result in results
        ),
        would_update=sum(
            result.action is DryRunAction.UPDATE
            for result in results
        ),
        unchanged=sum(
            result.action is DryRunAction.SKIP
            for result in results
        ),
        excluded=sum(
            result.action is DryRunAction.EXCLUDE
            for result in results
        ),
        external_eligible=sum(
            result.external_eligibility
            is ExternalEligibility.ELIGIBLE
            for result in results
        ),
        local_only=sum(
            result.external_eligibility
            is ExternalEligibility.LOCAL_ONLY
            for result in results
        ),
        results=tuple(results),
        failures=tuple(failures),
    )