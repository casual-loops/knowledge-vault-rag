import os
from pathlib import Path

from knowledge_rag.ingestion_config import (
    IngestionConfig,
    validate_ingestion_config,
)

PRODUCTION_VAULT_PATH_ENV = "KNOWLEDGE_RAG_PRODUCTION_VAULT_PATH"
PRODUCTION_OPT_IN_ENV = "KNOWLEDGE_RAG_ENABLE_PRODUCTION_INGESTION"


def get_production_ingestion_config() -> IngestionConfig:
    """Build validated production ingestion configuration."""

    raw_path = os.getenv(
        PRODUCTION_VAULT_PATH_ENV,
    )

    if not raw_path:
        raise ValueError(
            "Production vault path is not configured"
        )

    opt_in = (
        os.getenv(
            PRODUCTION_OPT_IN_ENV,
            "",
        ).strip().lower()
        == "true"
    )

    config = IngestionConfig(
        vault_path=Path(raw_path),
        production=True,
        production_opt_in=opt_in,
    )

    validate_ingestion_config(
        config,
    )

    return config