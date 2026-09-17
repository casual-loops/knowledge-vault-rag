from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True, slots=True)
class IngestionConfig:
    """Configuration for a vault ingestion run."""

    vault_path: Path
    production: bool
    production_opt_in: bool


def validate_ingestion_config(
    config: IngestionConfig,
) -> None:
    """Validate ingestion configuration before vault access."""

    if not config.vault_path.is_absolute():
        raise ValueError(
            "Vault path must be absolute"
        )

    if not config.vault_path.exists():
        raise ValueError(
            "Vault path does not exist"
        )

    if not config.vault_path.is_dir():
        raise ValueError(
            "Vault path must reference a directory"
        )

    if config.production and not config.production_opt_in:
        raise ValueError(
            "Production ingestion requires explicit opt-in"
        )