from pathlib import Path

import scripts.ingest_production_vault as runner
from knowledge_rag.ingestion_config import IngestionConfig
from knowledge_rag.production_ingestion import (
    NoteFailure,
    ProductionIngestionSummary,
)


def make_summary(
    *,
    failures: tuple[NoteFailure, ...] = (),
) -> ProductionIngestionSummary:
    return ProductionIngestionSummary(
        discovered=3,
        indexed=1,
        unchanged=1,
        excluded=0,
        reactivated=0,
        inactivated=1,
        embedded=2,
        failures=failures,
        reconciliation_completed=not failures,
        embedding_error=None,
    )


def test_production_runner_completes_pipeline_and_returns_zero(
    tmp_path: Path,
    monkeypatch,
    capsys,
) -> None:
    vault_path = (tmp_path / "production-vault").resolve()
    config = IngestionConfig(
        vault_path=vault_path,
        production=True,
        production_opt_in=True,
    )
    provider = object()
    calls: list[str] = []

    monkeypatch.setattr(
        runner,
        "get_production_ingestion_config",
        lambda: calls.append("config") or config,
    )
    monkeypatch.setattr(
        runner,
        "get_embedding_provider",
        lambda: calls.append("provider") or provider,
    )

    class FakeConnection:
        def __enter__(self):
            calls.append("connection")
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

    def fake_run(
        received_connection,
        received_config,
        received_provider,
    ) -> ProductionIngestionSummary:
        calls.append("orchestrator")
        assert received_connection is connection
        assert received_config is config
        assert received_provider is provider
        return make_summary()

    monkeypatch.setattr(
        runner,
        "run_production_ingestion",
        fake_run,
    )

    assert runner.main() == 0
    assert calls == [
        "config",
        "provider",
        "connection",
        "orchestrator",
    ]
    assert "complete: true" in capsys.readouterr().out


def test_production_runner_reports_safe_incomplete_failure(
    tmp_path: Path,
    monkeypatch,
    capsys,
) -> None:
    vault_path = (tmp_path / "private-vault").resolve()
    config = IngestionConfig(
        vault_path=vault_path,
        production=True,
        production_opt_in=True,
    )
    incomplete = make_summary(
        failures=(NoteFailure("Broken.md", "ValueError"),),
    )

    monkeypatch.setattr(
        runner,
        "get_production_ingestion_config",
        lambda: config,
    )
    monkeypatch.setattr(runner, "get_embedding_provider", object)

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

    monkeypatch.setattr(
        runner,
        "get_connection",
        FakeConnection,
    )
    monkeypatch.setattr(
        runner,
        "run_production_ingestion",
        lambda conn, received_config, provider: incomplete,
    )

    assert runner.main() == 1

    output = capsys.readouterr().out
    assert "complete: false" in output
    assert "failure: Broken.md: ValueError" in output
    assert str(vault_path) not in output


def test_production_runner_loads_configuration_before_other_factories(
    monkeypatch,
) -> None:
    calls: list[str] = []

    def fail_config() -> IngestionConfig:
        calls.append("config")
        raise ValueError("Production ingestion is disabled")

    monkeypatch.setattr(
        runner,
        "get_production_ingestion_config",
        fail_config,
    )
    monkeypatch.setattr(
        runner,
        "get_embedding_provider",
        lambda: calls.append("provider"),
    )
    monkeypatch.setattr(
        runner,
        "get_connection",
        lambda: calls.append("connection"),
    )

    try:
        runner.main()
    except ValueError as exc:
        assert str(exc) == "Production ingestion is disabled"
    else:
        raise AssertionError("Expected production configuration failure")

    assert calls == ["config"]


def test_production_runner_dry_run_never_loads_embedding_provider(
    tmp_path: Path,
    monkeypatch,
    capsys,
) -> None:
    from knowledge_rag.production_dry_run import (
        ProductionDryRunSummary,
    )

    vault_path = (
        tmp_path
        / "production-vault"
    ).resolve()

    config = IngestionConfig(
        vault_path=vault_path,
        production=True,
        production_opt_in=True,
    )

    calls: list[str] = []

    monkeypatch.setattr(
        runner,
        "parse_args",
        lambda: type(
            "Args",
            (),
            {
                "dry_run": True,
            },
        )(),
    )

    monkeypatch.setattr(
        runner,
        "get_production_ingestion_config",
        lambda: calls.append("config") or config,
    )

    def fail_provider():
        raise AssertionError(
            "Embedding provider must not load during dry-run"
        )

    monkeypatch.setattr(
        runner,
        "get_embedding_provider",
        fail_provider,
    )

    class FakeConnection:
        def __enter__(self):
            calls.append("connection")
            return self

        def __exit__(
            self,
            exc_type,
            exc,
            traceback,
        ) -> None:
            return None

    monkeypatch.setattr(
        runner,
        "get_connection",
        FakeConnection,
    )

    summary = ProductionDryRunSummary(
        discovered=1,
        would_index=1,
        would_update=0,
        unchanged=0,
        excluded=0,
        external_eligible=1,
        local_only=0,
        results=(),
        failures=(),
    )

    monkeypatch.setattr(
        runner,
        "run_production_dry_run",
        lambda conn, received_config: (
            calls.append("dry-run")
            or summary
        ),
    )

    assert runner.main() == 0

    assert calls == [
        "config",
        "connection",
        "dry-run",
    ]

    output = capsys.readouterr().out

    assert "mode: dry-run" in output
    assert "would-index: 1" in output


def test_production_runner_preflight_uses_no_database_or_provider(
    tmp_path: Path,
    monkeypatch,
    capsys,
) -> None:
    from knowledge_rag.production_preflight import (
        ProductionPreflightSummary,
    )

    vault_path = (
        tmp_path
        / "production-vault"
    ).resolve()

    vault_path.mkdir()

    config = IngestionConfig(
        vault_path=vault_path,
        production=True,
        production_opt_in=True,
    )

    calls: list[str] = []

    monkeypatch.setattr(
        runner,
        "parse_args",
        lambda: type(
            "Args",
            (),
            {
                "preflight": True,
                "dry_run": False,
            },
        )(),
    )

    monkeypatch.setattr(
        runner,
        "get_production_ingestion_config",
        lambda: calls.append("config") or config,
    )

    monkeypatch.setattr(
        runner,
        "get_connection",
        lambda: (_ for _ in ()).throw(
            AssertionError(
                "Database must not open during preflight"
            )
        ),
    )

    monkeypatch.setattr(
        runner,
        "get_embedding_provider",
        lambda: (_ for _ in ()).throw(
            AssertionError(
                "Provider must not load during preflight"
            )
        ),
    )

    summary = ProductionPreflightSummary(
        discovered=1,
        excluded_by_path=0,
        excluded_by_type=0,
        explicit_allowed=1,
        explicit_local_only=0,
        explicit_excluded=0,
        conservative_fallback=0,
        failures=(),
    )

    monkeypatch.setattr(
        runner,
        "run_production_preflight",
        lambda received_config: (
            calls.append("preflight")
            or summary
        ),
    )

    assert runner.main() == 0

    assert calls == [
        "config",
        "preflight",
    ]

    output = capsys.readouterr().out

    assert "mode: preflight" in output
    assert "passed: true" in output