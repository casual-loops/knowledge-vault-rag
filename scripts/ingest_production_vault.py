import argparse

from knowledge_rag.db import get_connection
from knowledge_rag.embedding_factory import get_embedding_provider
from knowledge_rag.ingestion_config_factory import (
    get_production_ingestion_config,
)
from knowledge_rag.production_dry_run import (
    ProductionDryRunSummary,
    run_production_dry_run,
)
from knowledge_rag.production_ingestion import (
    ProductionIngestionSummary,
    run_production_ingestion,
)


def parse_args() -> argparse.Namespace:
    """Parse production ingestion command-line arguments."""

    parser = argparse.ArgumentParser(
        description="Run production vault ingestion.",
    )

    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Audit the production vault without making changes.",
    )

    return parser.parse_args()


def format_summary(
    summary: ProductionIngestionSummary,
) -> str:
    """Format a sanitized aggregate production ingestion result."""

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


def format_dry_run_summary(
    summary: ProductionDryRunSummary,
) -> str:
    """Format a sanitized production dry-run report."""

    lines = [
        "mode: dry-run",
        f"complete: {str(summary.complete).lower()}",
        f"discovered: {summary.discovered}",
        f"would-index: {summary.would_index}",
        f"would-update: {summary.would_update}",
        f"unchanged: {summary.unchanged}",
        f"excluded: {summary.excluded}",
        f"external-eligible: {summary.external_eligible}",
        f"local-only: {summary.local_only}",
    ]

    for result in summary.results:
        lines.append(
            
                f"note: {result.reference}: "
                f"{result.action.value}: "
                f"{result.external_eligibility.value}"
            
        )

    for failure in summary.failures:
        lines.append(
            f"failure: {failure.reference}: {failure.error_type}"
        )

    return "\n".join(lines)


def main() -> int:
    """Run production ingestion or a read-only dry-run."""

    args = parse_args()
    config = get_production_ingestion_config()

    if args.dry_run:
        with get_connection() as conn:
            summary = run_production_dry_run(
                conn,
                config,
            )

        print(
            format_dry_run_summary(
                summary,
            )
        )

        return 0 if summary.complete else 1

    provider = get_embedding_provider()

    with get_connection() as conn:
        summary = run_production_ingestion(
            conn,
            config,
            provider,
        )

    print(
        format_summary(
            summary,
        )
    )

    return 0 if summary.complete else 1