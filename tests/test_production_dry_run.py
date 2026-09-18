from pathlib import Path

from knowledge_rag import production_dry_run
from knowledge_rag.config import settings
from knowledge_rag.ingestion_config import IngestionConfig
from knowledge_rag.production_dry_run import (
    DryRunAction,
    ExternalEligibility,
)


class FakeCursor:
    def __init__(self, row):
        self.row = row

    def fetchone(self):
        return self.row


class ReadOnlyConnection:
    def __init__(self, rows=None):
        self.rows = rows or {}
        self.statements: list[str] = []

    def execute(
        self,
        query: str,
        params=None,
    ):
        normalized = " ".join(
            query.split()
        ).upper()

        self.statements.append(
            normalized
        )

        assert normalized.startswith("SELECT")
        assert not any(
            keyword in normalized
            for keyword in (
                "INSERT ",
                "UPDATE ",
                "DELETE ",
            )
        )

        source_path = (
            params[0]
            if params
            else None
        )

        return FakeCursor(
            self.rows.get(
                source_path,
            )
        )


def make_config(
    vault_path: Path,
) -> IngestionConfig:
    return IngestionConfig(
        vault_path=vault_path.resolve(),
        production=True,
        production_opt_in=True,
    )


def write_note(
    path: Path,
    *,
    ai_access: str | None = "allowed",
    note_type: str = "reference",
    body: str = "Synthetic content.",
) -> None:
    metadata = [
        "---",
        f"type: {note_type}",
    ]

    if ai_access is not None:
        metadata.append(
            f"ai_access: {ai_access}"
        )

    metadata.extend(
        [
            "---",
            "",
            f"# {path.stem}",
            "",
            body,
        ]
    )

    path.write_text(
        "\n".join(metadata),
        encoding="utf-8",
    )


def test_dry_run_reports_new_allowed_note(
    tmp_path,
) -> None:
    vault_path = (
        tmp_path
        / "vault"
    )

    vault_path.mkdir()

    note_path = (
        vault_path
        / "New.md"
    )

    write_note(
        note_path,
        ai_access="allowed",
    )

    conn = ReadOnlyConnection()

    summary = production_dry_run.run_production_dry_run(
        conn,
        make_config(vault_path),
    )

    assert summary.discovered == 1
    assert summary.would_index == 1
    assert summary.would_update == 0
    assert summary.unchanged == 0
    assert summary.excluded == 0
    assert summary.external_eligible == 1
    assert summary.local_only == 0

    assert summary.results[0].action is DryRunAction.INDEX
    assert (
        summary.results[0].external_eligibility
        is ExternalEligibility.ELIGIBLE
    )


def test_dry_run_reports_changed_existing_note(
    tmp_path,
) -> None:
    vault_path = (
        tmp_path
        / "vault"
    )

    vault_path.mkdir()

    note_path = (
        vault_path
        / "Changed.md"
    )

    write_note(
        note_path,
        body="Current content.",
    )

    conn = ReadOnlyConnection(
        rows={
            "Changed.md": (
                "old-content-hash",
                True,
            )
        }
    )

    summary = production_dry_run.run_production_dry_run(
        conn,
        make_config(vault_path),
    )

    assert summary.would_update == 1
    assert summary.results[0].action is DryRunAction.UPDATE


def test_dry_run_reports_unchanged_existing_note(
    tmp_path,
) -> None:
    vault_path = (
        tmp_path
        / "vault"
    )

    vault_path.mkdir()

    note_path = (
        vault_path
        / "Stable.md"
    )

    write_note(
        note_path,
    )

    document = production_dry_run.load_markdown(
        note_path,
    )

    digest = production_dry_run.content_hash(
        document.content,
        document.metadata,
    )

    conn = ReadOnlyConnection(
        rows={
            "Stable.md": (
                digest,
                True,
            )
        }
    )

    summary = production_dry_run.run_production_dry_run(
        conn,
        make_config(vault_path),
    )

    assert summary.unchanged == 1
    assert summary.results[0].action is DryRunAction.SKIP


def test_dry_run_reports_inactive_existing_note_as_update(
    tmp_path,
) -> None:
    vault_path = (
        tmp_path
        / "vault"
    )

    vault_path.mkdir()

    note_path = (
        vault_path
        / "Inactive.md"
    )

    write_note(
        note_path,
    )

    document = production_dry_run.load_markdown(
        note_path,
    )

    digest = production_dry_run.content_hash(
        document.content,
        document.metadata,
    )

    conn = ReadOnlyConnection(
        rows={
            "Inactive.md": (
                digest,
                False,
            )
        }
    )

    summary = production_dry_run.run_production_dry_run(
        conn,
        make_config(vault_path),
    )

    assert summary.would_update == 1
    assert summary.results[0].action is DryRunAction.UPDATE


def test_dry_run_excludes_ai_access_exclude(
    tmp_path,
) -> None:
    vault_path = (
        tmp_path
        / "vault"
    )

    vault_path.mkdir()

    note_path = (
        vault_path
        / "Excluded.md"
    )

    write_note(
        note_path,
        ai_access="exclude",
    )

    conn = ReadOnlyConnection()

    summary = production_dry_run.run_production_dry_run(
        conn,
        make_config(vault_path),
    )

    assert summary.excluded == 1

    result = summary.results[0]

    assert result.action is DryRunAction.EXCLUDE
    assert (
        result.external_eligibility
        is ExternalEligibility.EXCLUDED
    )

    assert conn.statements == []


def test_dry_run_reports_local_only_without_external_eligibility(
    tmp_path,
) -> None:
    vault_path = (
        tmp_path
        / "vault"
    )

    vault_path.mkdir()

    note_path = (
        vault_path
        / "Local.md"
    )

    write_note(
        note_path,
        ai_access="local-only",
    )

    conn = ReadOnlyConnection()

    summary = production_dry_run.run_production_dry_run(
        conn,
        make_config(vault_path),
    )

    assert summary.would_index == 1
    assert summary.external_eligible == 0
    assert summary.local_only == 1

    assert (
        summary.results[0].external_eligibility
        is ExternalEligibility.LOCAL_ONLY
    )


def test_dry_run_defaults_missing_ai_access_to_local_only(
    tmp_path,
) -> None:
    vault_path = (
        tmp_path
        / "vault"
    )

    vault_path.mkdir()

    note_path = (
        vault_path
        / "MissingPolicy.md"
    )

    write_note(
        note_path,
        ai_access=None,
    )

    conn = ReadOnlyConnection()

    summary = production_dry_run.run_production_dry_run(
        conn,
        make_config(vault_path),
    )

    assert summary.would_index == 1
    assert summary.external_eligible == 0
    assert summary.local_only == 1


def test_dry_run_respects_path_exclusions(
    tmp_path,
    monkeypatch,
) -> None:
    vault_path = (
        tmp_path
        / "vault"
    )

    excluded_dir = (
        vault_path
        / "Private"
    )

    excluded_dir.mkdir(
        parents=True,
    )

    note_path = (
        excluded_dir
        / "Hidden.md"
    )

    write_note(
        note_path,
    )

    monkeypatch.setattr(
        settings,
        "excluded_paths",
        ("Private",),
    )

    conn = ReadOnlyConnection()

    summary = production_dry_run.run_production_dry_run(
        conn,
        make_config(vault_path),
    )

    assert summary.excluded == 1
    assert conn.statements == []


def test_dry_run_respects_note_type_exclusions(
    tmp_path,
    monkeypatch,
) -> None:
    vault_path = (
        tmp_path
        / "vault"
    )

    vault_path.mkdir()

    note_path = (
        vault_path
        / "Journal.md"
    )

    write_note(
        note_path,
        note_type="journal",
    )

    monkeypatch.setattr(
        settings,
        "excluded_note_types",
        ("journal",),
    )

    conn = ReadOnlyConnection()

    summary = production_dry_run.run_production_dry_run(
        conn,
        make_config(vault_path),
    )

    assert summary.excluded == 1
    assert conn.statements == []


def test_dry_run_never_executes_database_writes(
    tmp_path,
) -> None:
    vault_path = (
        tmp_path
        / "vault"
    )

    vault_path.mkdir()

    write_note(
        vault_path
        / "Allowed.md",
        ai_access="allowed",
    )

    write_note(
        vault_path
        / "Local.md",
        ai_access="local-only",
    )

    conn = ReadOnlyConnection()

    production_dry_run.run_production_dry_run(
        conn,
        make_config(vault_path),
    )

    assert conn.statements

    assert all(
        statement.startswith("SELECT")
        for statement in conn.statements
    )