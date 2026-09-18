from knowledge_rag.db import get_connection
from knowledge_rag.embedding_factory import get_embedding_provider
from knowledge_rag.ingestion_config_factory import (
    get_production_ingestion_config,
)
from knowledge_rag.production_ingestion import (
    ProductionIngestionSummary,
    run_production_ingestion,
)


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


def main() -> int:
    """Run the complete production ingestion workflow."""
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
