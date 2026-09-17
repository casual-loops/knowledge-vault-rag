from pathlib import Path

import scripts.ingest_production_vault as runner
from knowledge_rag.ingest.persistence import PersistResult
from knowledge_rag.ingestion_config import IngestionConfig


def test_production_runner_uses_configured_vault_path(
    tmp_path,
    monkeypatch,
) -> None:
    vault_path = (
        tmp_path
        / "production-vault"
    ).resolve()

    vault_path.mkdir()

    note_path = (
        vault_path
        / "Note.md"
    )

    note_path.write_text(
        "# Synthetic note",
        encoding="utf-8",
    )

    config = IngestionConfig(
        vault_path=vault_path,
        production=True,
        production_opt_in=True,
    )

    monkeypatch.setattr(
        runner,
        "get_production_ingestion_config",
        lambda: config,
    )

    monkeypatch.setattr(
        runner,
        "discover_markdown_files",
        lambda path: [note_path],
    )

    persisted: list[
        tuple[object, Path, Path]
    ] = []

    def fake_persist_note(
        conn,
        received_vault_path: Path,
        received_note_path: Path,
    ) -> PersistResult:
        persisted.append(
            (
                conn,
                received_vault_path,
                received_note_path,
            )
        )

        return PersistResult.INDEXED

    monkeypatch.setattr(
        runner,
        "persist_note",
        fake_persist_note,
    )

    class FakeConnection:
        def __enter__(self):
            return self

        def __exit__(
            self,
            exc_type,
            exc,
            traceback,
        ) -> None:
            return None

    connection = FakeConnection()

    monkeypatch.setattr(
        runner,
        "get_connection",
        lambda: connection,
    )

    monkeypatch.setattr(
        runner,
        "safe_note_reference",
        lambda vault, note: "Note.md",
    )

    runner.main()

    assert persisted == [
        (
            connection,
            vault_path,
            note_path,
        )
    ]