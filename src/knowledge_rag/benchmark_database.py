"""Safety checks and database helpers for the performance benchmark."""

from __future__ import annotations

import os
import platform
from typing import Any

from psycopg import Connection, ProgrammingError
from psycopg.conninfo import conninfo_to_dict

from knowledge_rag.benchmark_models import EnvironmentMetadata


def validate_benchmark_target(
    application_url: str,
    benchmark_url: str | None,
    allow_reset: bool,
) -> None:
    """Validate a destructive benchmark target without opening a connection."""

    if benchmark_url is None or not benchmark_url.strip():
        raise ValueError("BENCHMARK_DATABASE_URL is required")
    if not allow_reset:
        raise ValueError("explicit reset authorization is required")

    application_identity = _safe_database_identity(application_url)
    benchmark_identity = _safe_database_identity(benchmark_url)
    if application_identity is None or benchmark_identity is None:
        raise ValueError("invalid database target configuration")
    benchmark_name = benchmark_identity[2]

    if application_identity == benchmark_identity:
        raise ValueError("benchmark database must differ from application database")
    if "benchmark" not in benchmark_name.lower():
        raise ValueError("database name must contain benchmark")


def reset_benchmark_tables(conn: Connection) -> None:
    """Clear only the application tables used by a benchmark database."""

    conn.execute(
        """
        TRUNCATE TABLE document_chunks, documents
        RESTART IDENTITY CASCADE;
        """
    )


def count_benchmark_rows(conn: Connection) -> tuple[int, int]:
    """Return active document and active-document chunk counts."""

    documents = conn.execute(
        "SELECT count(*) FROM documents WHERE is_active = TRUE;"
    ).fetchone()[0]
    chunks = conn.execute(
        """
        SELECT count(*)
        FROM document_chunks c
        JOIN documents d ON d.document_id = c.document_id
        WHERE d.is_active = TRUE;
        """
    ).fetchone()[0]
    return int(documents), int(chunks)


def collect_environment_metadata(conn: Connection) -> EnvironmentMetadata:
    """Collect only the approved public environment fields."""

    postgresql_version = conn.execute("SHOW server_version;").fetchone()[0]
    vector_row = conn.execute(
        "SELECT extversion FROM pg_extension WHERE extname = 'vector';"
    ).fetchone()
    pgvector_version = vector_row[0] if vector_row and vector_row[0] else "unavailable"

    system = platform.system()
    return EnvironmentMetadata(
        python_version=platform.python_version(),
        operating_system=system,
        logical_cpu_count=os.cpu_count(),
        total_memory_bytes=_total_memory_bytes(system),
        postgresql_version=str(postgresql_version),
        pgvector_version=str(pgvector_version),
    )


def _database_identity(url: str) -> tuple[str, str, str]:
    return _database_identity_from_info(conninfo_to_dict(url))


def _safe_database_identity(url: str) -> tuple[str, str, str] | None:
    try:
        return _database_identity(url)
    except (ProgrammingError, ValueError):
        return None


def _database_identity_from_info(info: dict[str, Any]) -> tuple[str, str, str]:
    host = str(info.get("host") or "localhost").strip().lower()
    if host in {"127.0.0.1", "::1"}:
        host = "localhost"
    port_value = str(info.get("port") or "5432")
    if not port_value.isdigit() or not 1 <= int(port_value) <= 65535:
        raise ValueError("invalid port")
    port = str(int(port_value))
    database = str(info.get("dbname") or "").strip()
    if not database:
        raise ValueError("missing database name")
    return host, port, database


def _total_memory_bytes(system: str) -> int | None:
    if system.lower() != "linux":
        return None
    try:
        page_size = os.sysconf("SC_PAGE_SIZE")
        page_count = os.sysconf("SC_PHYS_PAGES")
        if not isinstance(page_size, int) or not isinstance(page_count, int):
            return None
        if page_size < 0 or page_count < 0:
            return None
        return page_size * page_count
    except (OSError, ValueError):
        return None
