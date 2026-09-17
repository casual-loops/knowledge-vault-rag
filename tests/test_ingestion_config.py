from pathlib import Path

import pytest

from knowledge_rag.ingestion_config import (
    IngestionConfig,
    validate_ingestion_config,
)
from knowledge_rag.ingestion_config_factory import (
    PRODUCTION_OPT_IN_ENV,
    PRODUCTION_VAULT_PATH_ENV,
    get_production_ingestion_config,
)


def test_validate_ingestion_config_accepts_existing_absolute_directory(
    tmp_path,
) -> None:
    config = IngestionConfig(
        vault_path=tmp_path,
        production=False,
        production_opt_in=False,
    )

    validate_ingestion_config(
        config,
    )


def test_validate_ingestion_config_rejects_relative_path(
    tmp_path,
    monkeypatch,
) -> None:
    monkeypatch.chdir(
        tmp_path,
    )

    config = IngestionConfig(
        vault_path=Path("relative-vault"),
        production=False,
        production_opt_in=False,
    )

    with pytest.raises(
        ValueError,
        match="Vault path must be absolute",
    ):
        validate_ingestion_config(
            config,
        )


def test_validate_ingestion_config_rejects_missing_path(
    tmp_path,
) -> None:
    config = IngestionConfig(
        vault_path=tmp_path / "missing",
        production=False,
        production_opt_in=False,
    )

    with pytest.raises(
        ValueError,
        match="Vault path does not exist",
    ):
        validate_ingestion_config(
            config,
        )


def test_validate_ingestion_config_rejects_file_path(
    tmp_path,
) -> None:
    file_path = tmp_path / "vault.txt"

    file_path.write_text(
        "not a directory",
        encoding="utf-8",
    )

    config = IngestionConfig(
        vault_path=file_path,
        production=False,
        production_opt_in=False,
    )

    with pytest.raises(
        ValueError,
        match="Vault path must reference a directory",
    ):
        validate_ingestion_config(
            config,
        )


def test_validate_ingestion_config_requires_production_opt_in(
    tmp_path,
) -> None:
    config = IngestionConfig(
        vault_path=tmp_path,
        production=True,
        production_opt_in=False,
    )

    with pytest.raises(
        ValueError,
        match="Production ingestion requires explicit opt-in",
    ):
        validate_ingestion_config(
            config,
        )


def test_get_production_ingestion_config_requires_path(
    monkeypatch,
) -> None:
    monkeypatch.delenv(
        "KNOWLEDGE_RAG_PRODUCTION_VAULT_PATH",
        raising=False,
    )

    monkeypatch.setenv(
        "KNOWLEDGE_RAG_ENABLE_PRODUCTION_INGESTION",
        "true",
    )

    with pytest.raises(
        ValueError,
        match="Production vault path is not configured",
    ):
        get_production_ingestion_config()


def test_get_production_ingestion_config_requires_opt_in(
    tmp_path,
    monkeypatch,
) -> None:
    monkeypatch.setenv(
        "KNOWLEDGE_RAG_PRODUCTION_VAULT_PATH",
        str(tmp_path),
    )

    monkeypatch.setenv(
        "KNOWLEDGE_RAG_ENABLE_PRODUCTION_INGESTION",
        "false",
    )

    with pytest.raises(
        ValueError,
        match="Production ingestion requires explicit opt-in",
    ):
        get_production_ingestion_config()


def test_get_production_ingestion_config_accepts_valid_configuration(
    tmp_path,
    monkeypatch,
) -> None:
    monkeypatch.setenv(
        "KNOWLEDGE_RAG_PRODUCTION_VAULT_PATH",
        str(tmp_path),
    )

    monkeypatch.setenv(
        "KNOWLEDGE_RAG_ENABLE_PRODUCTION_INGESTION",
        "true",
    )

    config = get_production_ingestion_config()

    assert config.vault_path == tmp_path
    assert config.production is True
    assert config.production_opt_in is True


def test_production_config_fails_when_path_is_missing(
    monkeypatch,
) -> None:
    monkeypatch.delenv(
        PRODUCTION_VAULT_PATH_ENV,
        raising=False,
    )

    monkeypatch.setenv(
        PRODUCTION_OPT_IN_ENV,
        "true",
    )

    with pytest.raises(
        ValueError,
        match="Production vault path is not configured",
    ):
        get_production_ingestion_config()


def test_ingestion_config_rejects_relative_path(
    tmp_path,
) -> None:
    relative_path = Path("relative-vault")

    config = IngestionConfig(
        vault_path=relative_path,
        production=True,
        production_opt_in=True,
    )

    with pytest.raises(
        ValueError,
        match="Vault path must be absolute",
    ):
        validate_ingestion_config(
            config,
        )


def test_ingestion_config_rejects_nonexistent_path(
    tmp_path,
) -> None:
    missing_path = (
        tmp_path
        / "missing-vault"
    ).resolve()

    config = IngestionConfig(
        vault_path=missing_path,
        production=True,
        production_opt_in=True,
    )

    with pytest.raises(
        ValueError,
        match="Vault path does not exist",
    ):
        validate_ingestion_config(
            config,
        )


def test_production_config_requires_explicit_opt_in(
    tmp_path,
    monkeypatch,
) -> None:
    vault_path = (
        tmp_path
        / "vault"
    )

    vault_path.mkdir()

    monkeypatch.setenv(
        PRODUCTION_VAULT_PATH_ENV,
        str(
            vault_path.resolve()
        ),
    )

    monkeypatch.setenv(
        PRODUCTION_OPT_IN_ENV,
        "false",
    )

    with pytest.raises(
        ValueError,
        match="Production ingestion requires explicit opt-in",
    ):
        get_production_ingestion_config()


def test_production_config_succeeds_with_valid_configuration(
    tmp_path,
    monkeypatch,
) -> None:
    vault_path = (
        tmp_path
        / "vault"
    )

    vault_path.mkdir()

    monkeypatch.setenv(
        PRODUCTION_VAULT_PATH_ENV,
        str(
            vault_path.resolve()
        ),
    )

    monkeypatch.setenv(
        PRODUCTION_OPT_IN_ENV,
        "true",
    )

    config = get_production_ingestion_config()

    assert config.vault_path == vault_path.resolve()
    assert config.production is True
    assert config.production_opt_in is True


def test_ingestion_config_rejects_file_path(
    tmp_path,
) -> None:
    file_path = (
        tmp_path
        / "vault.md"
    )

    file_path.write_text(
        "not a directory",
        encoding="utf-8",
    )

    config = IngestionConfig(
        vault_path=file_path.resolve(),
        production=True,
        production_opt_in=True,
    )

    with pytest.raises(
        ValueError,
        match="Vault path must reference a directory",
    ):
        validate_ingestion_config(
            config,
        )


def test_opt_in_is_case_insensitive(
    tmp_path,
    monkeypatch,
) -> None:
    vault_path = (
        tmp_path
        / "vault"
    )

    vault_path.mkdir()

    monkeypatch.setenv(
        PRODUCTION_VAULT_PATH_ENV,
        str(
            vault_path.resolve()
        ),
    )

    monkeypatch.setenv(
        PRODUCTION_OPT_IN_ENV,
        "TRUE",
    )

    config = get_production_ingestion_config()

    assert config.production_opt_in is True