from knowledge_rag.db import get_connection
from knowledge_rag.ingest.markdown import discover_markdown_files
from knowledge_rag.ingest.persistence import persist_note
from knowledge_rag.ingestion_config_factory import (
    get_production_ingestion_config,
)
from knowledge_rag.logging_utils import safe_note_reference


def main() -> None:
    config = get_production_ingestion_config()
    vault_path = config.vault_path

    with get_connection() as conn:
        for note_path in discover_markdown_files(
            vault_path,
        ):
            result = persist_note(
                conn,
                vault_path,
                note_path,
            )

            reference = safe_note_reference(
                vault_path,
                note_path,
            )

            print(
                f"{result.value}: {reference}"
            )


if __name__ == "__main__":
    main()