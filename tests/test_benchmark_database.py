from __future__ import annotations

import pytest

from knowledge_rag.benchmark_database import (
    collect_environment_metadata,
    count_benchmark_rows,
    reset_benchmark_tables,
    validate_benchmark_target,
)


@pytest.mark.parametrize(
    ("benchmark_url", "allow_reset", "message"),
    [
        (None, True, "BENCHMARK_DATABASE_URL is required"),
        (
            "postgresql://user:secret@db:5432/knowledge_rag_benchmark",
            False,
            "explicit reset authorization is required",
        ),
        (
            "postgresql://user:other@db:5432/knowledge_rag",
            True,
            "benchmark database must differ",
        ),
        (
            "postgresql://user:secret@db:5432/performance",
            True,
            "database name must contain benchmark",
        ),
    ],
)
def test_validate_benchmark_target_rejects_unsafe_configuration(
    benchmark_url: str | None,
    allow_reset: bool,
    message: str,
) -> None:
    with pytest.raises(ValueError, match=message) as exc_info:
        validate_benchmark_target(
            "postgresql://app:private@db:5432/knowledge_rag",
            benchmark_url,
            allow_reset,
        )

    rendered = str(exc_info.value)
    assert "private" not in rendered
    assert "secret" not in rendered


def test_validate_benchmark_target_accepts_dedicated_database() -> None:
    validate_benchmark_target(
        "postgresql://app:private@db:5432/knowledge_rag",
        "postgresql://user:secret@db/knowledge_rag_benchmark",
        True,
    )


def test_validate_benchmark_target_normalizes_localhost_identity() -> None:
    with pytest.raises(ValueError, match="benchmark database must differ"):
        validate_benchmark_target(
            "postgresql://app:private@localhost:5432/knowledge_rag",
            "postgresql://user:secret@127.0.0.1/knowledge_rag",
            True,
        )


@pytest.mark.parametrize(
    "application_url,benchmark_url",
    [
        (
            "postgresql://app:pw@app-token.example/app-db-token",
            "postgresql://bad%zz:secret@benchmark-token.example/benchmark-db-token",
        ),
        (
            "postgresql://bad%zz:secret@app-token.example/app-db-token",
            "postgresql://user:pw@benchmark-token.example/benchmark-db-token",
        ),
    ],
)
def test_validate_benchmark_target_sanitizes_malformed_conninfo(
    application_url: str,
    benchmark_url: str,
) -> None:
    with pytest.raises(ValueError, match="^invalid database target configuration$") as exc_info:
        validate_benchmark_target(application_url, benchmark_url, True)

    rendered = str(exc_info.value)
    assert exc_info.value.__cause__ is None
    assert exc_info.value.__context__ is None
    for token in ("app-token", "benchmark-token", "app-db-token", "benchmark-db-token", "secret"):
        assert token not in rendered


@pytest.mark.parametrize("port", ["alpha", "0", "-1", "65536"])
def test_validate_benchmark_target_rejects_invalid_ports(port: str) -> None:
    with pytest.raises(ValueError, match="^invalid database target configuration$"):
        validate_benchmark_target(
            "postgresql://app:pw@db:5432/knowledge_rag",
            f"postgresql://user:pw@benchmark.example:{port}/knowledge_rag_benchmark",
            True,
        )


def test_validate_benchmark_target_rejects_invalid_application_port() -> None:
    with pytest.raises(ValueError, match="^invalid database target configuration$"):
        validate_benchmark_target(
            "postgresql://app:pw@db:0/knowledge_rag",
            "postgresql://user:pw@benchmark.example:5432/knowledge_rag_benchmark",
            True,
        )


class _Cursor:
    def __init__(self, row: tuple[object, ...]) -> None:
        self.row = row

    def fetchone(self) -> tuple[object, ...]:
        return self.row


class _Connection:
    def __init__(self, rows: list[tuple[object, ...]]) -> None:
        self.rows = iter(rows)
        self.sql: list[str] = []

    def execute(self, query: str) -> _Cursor:
        self.sql.append(" ".join(query.split()))
        return _Cursor(next(self.rows, ()))


def test_reset_benchmark_tables_only_truncates_application_tables() -> None:
    conn = _Connection([])

    reset_benchmark_tables(conn)

    assert conn.sql == [
        "TRUNCATE TABLE document_chunks, documents RESTART IDENTITY CASCADE;"
    ]


def test_count_benchmark_rows_returns_active_documents_and_chunks() -> None:
    conn = _Connection([(12,), (34,)])

    assert count_benchmark_rows(conn) == (12, 34)
    assert "is_active = TRUE" in conn.sql[0]
    assert "JOIN documents" in conn.sql[1]


def test_collect_environment_metadata_contains_only_approved_fields(monkeypatch: pytest.MonkeyPatch) -> None:
    class _Platform:
        @staticmethod
        def python_version() -> str:
            return "3.13.7"

        @staticmethod
        def system() -> str:
            return "Linux"

    monkeypatch.setattr("knowledge_rag.benchmark_database.platform", _Platform)
    monkeypatch.setattr("knowledge_rag.benchmark_database.os.cpu_count", lambda: 8)
    monkeypatch.setattr(
        "knowledge_rag.benchmark_database.os.sysconf",
        lambda name: {"SC_PAGE_SIZE": 4096, "SC_PHYS_PAGES": 1024}[name],
    )
    conn = _Connection([("PostgreSQL 16.4",), ("0.8.5",)])

    metadata = collect_environment_metadata(conn)

    assert set(metadata.__slots__) == {
        "python_version",
        "operating_system",
        "logical_cpu_count",
        "total_memory_bytes",
        "postgresql_version",
        "pgvector_version",
    }
    assert metadata.python_version == "3.13.7"
    assert metadata.operating_system == "Linux"
    assert metadata.logical_cpu_count == 8
    assert metadata.total_memory_bytes == 4_194_304
    assert metadata.postgresql_version == "PostgreSQL 16.4"
    assert metadata.pgvector_version == "0.8.5"
    assert all("url" not in field.lower() for field in metadata.__slots__)


def test_collect_environment_metadata_uses_none_when_memory_is_unavailable(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr("knowledge_rag.benchmark_database.platform.python_version", lambda: "3.13.7")
    monkeypatch.setattr("knowledge_rag.benchmark_database.platform.system", lambda: "Windows")
    monkeypatch.setattr("knowledge_rag.benchmark_database.os.cpu_count", lambda: None)
    conn = _Connection([("PostgreSQL 16.4",), (None,)])

    metadata = collect_environment_metadata(conn)

    assert metadata.total_memory_bytes is None
    assert metadata.logical_cpu_count is None
    assert metadata.pgvector_version == "unavailable"
