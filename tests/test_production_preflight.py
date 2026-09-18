from pathlib import Path

from knowledge_rag import production_preflight
from knowledge_rag.config import settings
from knowledge_rag.ingestion_config import IngestionConfig


def make_config(
    vault_path: Path,
    *,
    production: bool = True,
    production_opt_in: bool = True,
) -> IngestionConfig:
    return IngestionConfig(
        vault_path=vault_path.resolve(),
        production=production,
        production_opt_in=production_opt_in,
    )


def write_note(
    path: Path,
    *,
    ai_access: str | None = "allowed",
    note_type: str = "reference",
    body: str = "Synthetic content.",
) -> None:
    lines = [
        "---",
        f"type: {note_type}",
    ]

    if ai_access is not None:
        lines.append(
            f"ai_access: {ai_access}"
        )

    lines.extend(
        [
            "---",
            "",
            f"# {path.stem}",
            "",
            body,
        ]
    )

    path.write_text(
        "\n".join(lines),
        encoding="utf-8",
    )


def test_preflight_passes_safe_configuration(
    tmp_path,
    monkeypatch,
) -> None:
    vault_path = tmp_path / "vault"
    vault_path.mkdir()

    write_note(
        vault_path / "Allowed.md",
        ai_access="allowed",
    )

    write_note(
        vault_path / "Local.md",
        ai_access="local-only",
    )

    monkeypatch.setattr(
        settings,
        "excluded_paths",
        (),
    )

    monkeypatch.setattr(
        settings,
        "excluded_note_types",
        (),
    )

    summary = production_preflight.run_production_preflight(
        make_config(vault_path),
    )

    assert summary.passed is True
    assert summary.discovered == 2
    assert summary.explicit_allowed == 1
    assert summary.explicit_local_only == 1
    assert summary.failures == ()


def test_preflight_rejects_non_production_configuration(
    tmp_path,
) -> None:
    vault_path = tmp_path / "vault"
    vault_path.mkdir()

    summary = production_preflight.run_production_preflight(
        make_config(
            vault_path,
            production=False,
        )
    )

    assert summary.passed is False

    assert any(
        failure.error_type == "ProductionModeRequired"
        for failure in summary.failures
    )


def test_preflight_requires_production_opt_in(
    tmp_path,
) -> None:
    vault_path = tmp_path / "vault"
    vault_path.mkdir()

    summary = production_preflight.run_production_preflight(
        make_config(
            vault_path,
            production_opt_in=False,
        )
    )

    assert summary.passed is False

    assert any(
        failure.error_type == "ProductionOptInRequired"
        for failure in summary.failures
    )


def test_preflight_rejects_invalid_vault_path(
    tmp_path,
) -> None:
    vault_path = (
        tmp_path
        / "missing"
    ).resolve()

    summary = production_preflight.run_production_preflight(
        IngestionConfig(
            vault_path=vault_path,
            production=True,
            production_opt_in=True,
        )
    )

    assert summary.passed is False

    assert any(
        failure.error_type == "InvalidVaultPath"
        for failure in summary.failures
    )


def test_preflight_rejects_unsafe_path_exclusion(
    tmp_path,
    monkeypatch,
) -> None:
    vault_path = tmp_path / "vault"
    vault_path.mkdir()

    monkeypatch.setattr(
        settings,
        "excluded_paths",
        ("../outside",),
    )

    summary = production_preflight.run_production_preflight(
        make_config(vault_path),
    )

    assert summary.passed is False

    assert any(
        failure.error_type == "UnsafePathExclusion"
        for failure in summary.failures
    )


def test_preflight_rejects_invalid_note_type_exclusion(
    tmp_path,
    monkeypatch,
) -> None:
    vault_path = tmp_path / "vault"
    vault_path.mkdir()

    monkeypatch.setattr(
        settings,
        "excluded_note_types",
        ("",),
    )

    summary = production_preflight.run_production_preflight(
        make_config(vault_path),
    )

    assert summary.passed is False

    assert any(
        failure.error_type == "InvalidNoteTypeExclusion"
        for failure in summary.failures
    )


def test_preflight_validates_allowed_external_eligibility(
    tmp_path,
) -> None:
    vault_path = tmp_path / "vault"
    vault_path.mkdir()

    write_note(
        vault_path / "Allowed.md",
        ai_access="allowed",
    )

    summary = production_preflight.run_production_preflight(
        make_config(vault_path),
    )

    assert summary.passed is True
    assert summary.explicit_allowed == 1


def test_preflight_validates_local_only_is_not_external(
    tmp_path,
) -> None:
    vault_path = tmp_path / "vault"
    vault_path.mkdir()

    write_note(
        vault_path / "Local.md",
        ai_access="local-only",
    )

    summary = production_preflight.run_production_preflight(
        make_config(vault_path),
    )

    assert summary.passed is True
    assert summary.explicit_local_only == 1


def test_preflight_validates_explicit_exclude(
    tmp_path,
) -> None:
    vault_path = tmp_path / "vault"
    vault_path.mkdir()

    write_note(
        vault_path / "Excluded.md",
        ai_access="exclude",
    )

    summary = production_preflight.run_production_preflight(
        make_config(vault_path),
    )

    assert summary.passed is True
    assert summary.explicit_excluded == 1


def test_preflight_missing_ai_access_falls_back_conservatively(
    tmp_path,
) -> None:
    vault_path = tmp_path / "vault"
    vault_path.mkdir()

    write_note(
        vault_path / "Missing.md",
        ai_access=None,
    )

    summary = production_preflight.run_production_preflight(
        make_config(vault_path),
    )

    assert summary.passed is True
    assert summary.conservative_fallback == 1


def test_preflight_malformed_ai_access_falls_back_conservatively(
    tmp_path,
) -> None:
    vault_path = tmp_path / "vault"
    vault_path.mkdir()

    write_note(
        vault_path / "Malformed.md",
        ai_access="definitely-not-valid",
    )

    summary = production_preflight.run_production_preflight(
        make_config(vault_path),
    )

    assert summary.passed is True
    assert summary.conservative_fallback == 1


def test_preflight_does_not_expose_note_body_or_frontmatter(
    tmp_path,
) -> None:
    vault_path = tmp_path / "vault"
    vault_path.mkdir()

    private_marker = "PRIVATE_BODY_MARKER"
    frontmatter_marker = "PRIVATE_FRONTMATTER_MARKER"

    note_path = vault_path / "Private.md"

    note_path.write_text(
        (
            "---\n"
            "type: reference\n"
            "ai_access: allowed\n"
            f"secret: {frontmatter_marker}\n"
            "---\n\n"
            "# Private\n\n"
            f"{private_marker}\n"
        ),
        encoding="utf-8",
    )

    summary = production_preflight.run_production_preflight(
        make_config(vault_path),
    )

    rendered = repr(summary)

    assert private_marker not in rendered
    assert frontmatter_marker not in rendered