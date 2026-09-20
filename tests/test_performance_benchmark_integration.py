"""PostgreSQL lifecycle coverage for the small synthetic benchmark workload."""

from __future__ import annotations

import json
from collections import Counter
from contextlib import contextmanager, nullcontext

import psycopg
import pytest

from knowledge_rag.benchmark_corpus import (
    BenchmarkWorkload,
    PerformanceQuery,
    generate_synthetic_vault,
    modify_synthetic_vault,
    workload_to_dict,
)
from knowledge_rag.benchmark_database import count_benchmark_rows, reset_benchmark_tables
from knowledge_rag.benchmark_models import BenchmarkReport, TierBenchmarkResult, report_to_dict
from knowledge_rag.config import settings
from knowledge_rag.embeddings import DeterministicEmbeddingProvider
from knowledge_rag.performance_benchmark import (
    run_embedding_stage,
    run_indexing_stage,
    run_retrieval_latency,
)


def _assert_private_values_absent(payload, private_values):
    """Inspect decoded strings, including dictionary keys, without JSON escaping."""
    if isinstance(payload, str):
        assert all(value not in payload for value in private_values)
    elif isinstance(payload, dict):
        for key, value in payload.items():
            _assert_private_values_absent(key, private_values)
            _assert_private_values_absent(value, private_values)
    elif isinstance(payload, (tuple, list)):
        for value in payload:
            _assert_private_values_absent(value, private_values)


class _ObservedConnectionFactory:
    def __init__(self, connection):
        self.connection = connection
        self.entries = 0
        self.exits = 0

    @contextmanager
    def __call__(self):
        # Preserve the session: temporary tables must shadow permanent tables.
        with nullcontext(self.connection) as connection:
            self.entries += 1
            try:
                yield connection
            finally:
                self.exits += 1


def test_privacy_check_rejects_nested_multiline_body():
    body = '# Synthetic note\nA "quoted" observation.\nSecond paragraph.'
    payload = json.loads(json.dumps({"tiers": [{"leaked_body": body}]}))
    with pytest.raises(AssertionError):
        _assert_private_values_absent(payload, (body,))
    _assert_private_values_absent({"tiers": [{"count": 30}]}, (body,))


@pytest.mark.parametrize("failing_call", (1, 2))
def test_observed_connection_balances_exit_on_materialization_failure(monkeypatch, failing_call):
    from knowledge_rag import performance_benchmark

    connection = object()
    factory = _ObservedConnectionFactory(connection)
    executions = 0

    def failing_results(**kwargs):
        nonlocal executions
        assert kwargs["conn"] is connection
        executions += 1
        yield object()
        if executions == failing_call:
            raise ValueError("synthetic materialization failure")

    monkeypatch.setattr(performance_benchmark, "run_retrieval", failing_results)
    with pytest.raises(ValueError, match="synthetic materialization failure"):
        run_retrieval_latency(
            conn_factory=factory,
            provider=DeterministicEmbeddingProvider(1536),
            workload=_small_workload(),
            validate_approved_workload=False,
        )
    assert (factory.entries, factory.exits) == (failing_call, failing_call)


def test_observed_retrieval_counts_warmups_but_records_only_measured_samples(monkeypatch):
    from knowledge_rag import performance_benchmark

    connection = object()
    factory = _ObservedConnectionFactory(connection)
    workload = _small_workload()
    executions = Counter()

    def results(**kwargs):
        assert kwargs["conn"] is connection
        executions[kwargs["query"]] += 1
        yield object()

    monkeypatch.setattr(performance_benchmark, "run_retrieval", results)
    queries, _ = run_retrieval_latency(
        conn_factory=factory,
        provider=DeterministicEmbeddingProvider(1536),
        workload=workload,
        validate_approved_workload=False,
    )
    assert executions == Counter({query.query: 3 for query in workload.queries})
    assert (factory.entries, factory.exits) == (9, 9)
    assert all(result.timing.sample_count == len(result.timing.samples_ns) == 2 for result in queries)


def test_small_mutation_selects_first_manifest_note(tmp_path):
    workload = _small_workload()
    manifest = generate_synthetic_vault(tmp_path, workload, note_count=10)
    changed = modify_synthetic_vault(tmp_path, manifest, workload.changed_fraction)
    assert manifest.paths[0] == "synthetic/0000/note-00000.md"
    assert changed == (manifest.paths[0],)


def _create_benchmark_test_tables(conn: psycopg.Connection) -> None:
    """Create the production-shaped temporary tables used by the lifecycle test."""
    conn.execute(
        """
        CREATE TEMP TABLE documents (
            document_id BIGSERIAL PRIMARY KEY,
            document_uuid UUID NOT NULL UNIQUE,
            source_path TEXT NOT NULL UNIQUE,
            title TEXT NOT NULL,
            note_type TEXT,
            created_date DATE,
            status TEXT,
            ai_access TEXT NOT NULL DEFAULT 'local-only',
            metadata JSONB NOT NULL DEFAULT '{}'::jsonb,
            content_hash TEXT NOT NULL,
            is_active BOOLEAN NOT NULL DEFAULT TRUE,
            indexed_at TIMESTAMPTZ NOT NULL DEFAULT NOW()
        );
        """
    )
    conn.execute(
        """
        CREATE TEMP TABLE document_chunks (
            chunk_id BIGSERIAL PRIMARY KEY,
            document_id BIGINT NOT NULL,
            chunk_index INTEGER NOT NULL,
            heading_path TEXT,
            content TEXT NOT NULL,
            token_estimate INTEGER,
            embedding VECTOR(1536),
            UNIQUE (document_id, chunk_index)
        );
        """
    )


@pytest.fixture
def benchmark_connection() -> psycopg.Connection:
    """Keep one connection open so temporary tables shadow permanent tables."""
    with psycopg.connect(settings.database_url) as conn:
        _create_benchmark_test_tables(conn)
        yield conn


def _small_workload() -> BenchmarkWorkload:
    return BenchmarkWorkload(
        version=1,
        seed=14_069,
        tiers=(10,),
        chunks_per_note=3,
        changed_fraction=0.1,
        warmup_iterations=1,
        measured_iterations=2,
        queries=(
            PerformanceQuery(
                "semantic-small", "semantic", "capacity planning for distributed monitoring", None, None
            ),
            PerformanceQuery("lexical-small", "lexical", "monitoring alert threshold", None, None),
            PerformanceQuery(
                "hybrid-small", "hybrid", "private knowledge retrieval architecture", None, None
            ),
        ),
    )


def test_small_workload_exercises_real_postgresql_benchmark_lifecycle(
    benchmark_connection: psycopg.Connection, tmp_path, monkeypatch
) -> None:
    def forbid_hosted_provider(*args, **kwargs):
        pytest.fail("benchmark must not instantiate a hosted provider")

    monkeypatch.setattr("knowledge_rag.embeddings.OpenAI", forbid_hosted_provider)
    workload = _small_workload()
    provider = DeterministicEmbeddingProvider(dimensions=1536)
    vault = tmp_path / "synthetic-vault"
    conn_factory = _ObservedConnectionFactory(benchmark_connection)
    from knowledge_rag import performance_benchmark

    real_retrieval = performance_benchmark.run_retrieval
    executions = Counter()

    def observed_retrieval(**kwargs):
        assert kwargs["conn"] is benchmark_connection
        executions[kwargs["query"]] += 1
        return real_retrieval(**kwargs)

    monkeypatch.setattr(performance_benchmark, "run_retrieval", observed_retrieval)

    reset_benchmark_tables(benchmark_connection)
    manifest = generate_synthetic_vault(vault, workload, note_count=10)
    clean = run_indexing_stage(benchmark_connection, vault, "clean")
    initial_embedding = run_embedding_stage(benchmark_connection, provider)
    assert benchmark_connection.execute(
        "SELECT count(*) FROM document_chunks GROUP BY document_id;"
    ).fetchall() == [(3,)] * 10
    unchanged = run_indexing_stage(benchmark_connection, vault, "unchanged")
    changed_paths = modify_synthetic_vault(vault, manifest, workload.changed_fraction)
    changed = run_indexing_stage(
        benchmark_connection, vault, "changed", chunks_per_note=workload.chunks_per_note
    )
    changed_embedding = run_embedding_stage(benchmark_connection, provider)
    queries, modes = run_retrieval_latency(
        conn_factory=conn_factory,
        provider=provider,
        workload=workload,
        validate_approved_workload=False,
    )

    assert (clean.indexed, clean.observed_chunks) == (10, 30)
    assert (clean.discovered, clean.affected_chunks) == (10, 30)
    assert (initial_embedding.generated, initial_embedding.persisted) == (30, 30)
    assert benchmark_connection.execute(
        "SELECT count(*), min(vector_dims(embedding)), max(vector_dims(embedding)) "
        "FROM document_chunks WHERE embedding IS NOT NULL;"
    ).fetchone() == (30, 1536, 1536)
    assert (unchanged.unchanged, unchanged.affected_chunks) == (10, 0)
    assert unchanged.indexed == 0
    assert changed_paths == (manifest.paths[0],)
    assert (changed.indexed, changed.affected_chunks) == (1, 3)
    assert changed.unchanged == 9
    assert (changed_embedding.generated, changed_embedding.persisted) == (3, 3)
    assert count_benchmark_rows(benchmark_connection) == (10, 30)
    assert benchmark_connection.execute(
        "SELECT count(*) FROM document_chunks GROUP BY document_id;"
    ).fetchall() == [(3,)] * 10
    persisted_chunks = benchmark_connection.execute(
        "SELECT content, embedding::text FROM document_chunks;"
    ).fetchall()
    for content, vector in persisted_chunks:
        assert json.loads(vector) == pytest.approx(provider.embed(content), abs=1e-7)
        assert provider.embed(content) == DeterministicEmbeddingProvider(1536).embed(content)
    for result, expected in ((initial_embedding, 30), (changed_embedding, 3)):
        assert result.loaded == expected
        assert result.generation_rate.count == expected
        assert result.persistence_rate.count == expected
        assert result.total_rate.count == expected
    for result, expected in ((clean, 30), (unchanged, 0), (changed, 3)):
        assert result.document_rate.count == 10
        assert result.chunk_rate.count == expected
    assert {result.retrieval_mode for result in queries} == {"semantic", "lexical", "hybrid"}
    assert all(result.timing.sample_count == 2 for result in queries)
    assert all(len(result.timing.samples_ns) == 2 for result in queries)
    assert executions == Counter({query.query: 3 for query in workload.queries})
    assert (conn_factory.entries, conn_factory.exits) == (9, 9)
    assert not benchmark_connection.closed
    assert all(result.result_count > 0 for result in queries)
    assert {result.retrieval_mode for result in modes} == {"semantic", "lexical", "hybrid"}
    assert all(result.timing.sample_count == 2 for result in modes)

    # Exercise the public serializer with real database-produced stage results.
    # This focused lifecycle does not run the separate judged quality phase.
    report = BenchmarkReport(
        complete=True,
        schema_version=1,
        workload_version=workload.version,
        git_commit=None,
        executed_at_utc="2026-09-18T00:00:00Z",
        environment=None,
        workload=workload_to_dict(workload),
        quality=None,
        tiers=(TierBenchmarkResult(
            note_count=10,
            expected_chunks=manifest.chunk_count,
            observed_chunks=clean.observed_chunks,
            clean_indexing=clean,
            initial_embedding=initial_embedding,
            unchanged_indexing=unchanged,
            changed_indexing=changed,
            changed_embedding=changed_embedding,
            queries=queries,
            modes=modes,
        ),),
    )
    payload = json.loads(json.dumps(report_to_dict(report)))
    private_values = (
        "Revision note:", settings.database_url, str(tmp_path.resolve()),
        *(content for content, _ in persisted_chunks),
        *((vault / path).read_text(encoding="utf-8") for path in manifest.paths),
    )
    _assert_private_values_absent(payload, private_values)
