from __future__ import annotations

from itertools import count
from pathlib import Path

import pytest

from knowledge_rag.embedding_pipeline import ChunkForEmbedding
from knowledge_rag.ingest.persistence import PersistResult


class ConnectionContext:
    def __init__(self, connection, events):
        self.connection = connection
        self.events = events

    def __enter__(self):
        self.events.append("open")
        return self.connection

    def __exit__(self, exc_type, exc_value, traceback):
        self.events.append("close")
        return False


class FakeConnection:
    def __init__(self) -> None:
        self.transaction_calls = 0
        self.active_transaction: int | None = None
        self.completed_transactions: list[int] = []

    def transaction(self):
        self.transaction_calls += 1
        transaction_id = self.transaction_calls
        connection = self

        class Transaction:
            def __enter__(self):
                connection.active_transaction = transaction_id
                return self

            def __exit__(self, exc_type, exc_value, traceback):
                connection.completed_transactions.append(transaction_id)
                connection.active_transaction = None
                return False

        return Transaction()


def test_run_indexing_stage_records_counts_and_duration(monkeypatch, tmp_path) -> None:
    from knowledge_rag import performance_benchmark

    vault = tmp_path / "vault"
    vault.mkdir()
    notes = [vault / "one.md", vault / "two.md", vault / "three.md"]
    results = iter((PersistResult.INDEXED, PersistResult.UNCHANGED, PersistResult.EXCLUDED))
    clock = iter((1_000, 2_001)).__next__
    connection = FakeConnection()
    reconciled: list[set[str]] = []

    monkeypatch.setattr(performance_benchmark, "discover_markdown_files", lambda _: notes)
    monkeypatch.setattr(
        performance_benchmark,
        "persist_note",
        lambda conn, root, note: next(results),
    )
    monkeypatch.setattr(
        performance_benchmark,
        "reconcile_inactive_documents",
        lambda conn, paths: reconciled.append(paths) or 2,
    )
    monkeypatch.setattr(
        performance_benchmark,
        "count_benchmark_rows",
        lambda conn: (2, 6),
    )

    result = performance_benchmark.run_indexing_stage(
        connection,
        vault,
        "clean",
        chunks_per_note=3,
        clock_ns=clock,
    )

    assert result.discovered == 3
    assert result.indexed == 1
    assert result.unchanged == 1
    assert result.excluded == 1
    assert result.reactivated == 0
    assert result.inactivated == 2
    assert result.observed_documents == 2
    assert result.observed_chunks == 6
    assert result.affected_chunks == 6
    assert result.document_rate.duration_ns == 1_001
    assert result.chunk_rate.count == 6
    assert reconciled == [{"one.md", "two.md", "three.md"}]
    assert connection.transaction_calls == 4


def test_run_indexing_stage_changed_uses_indexed_chunks(monkeypatch, tmp_path) -> None:
    from knowledge_rag import performance_benchmark

    vault = tmp_path / "vault"
    vault.mkdir()
    note = vault / "changed.md"
    ticks = iter((10, 20)).__next__
    monkeypatch.setattr(performance_benchmark, "discover_markdown_files", lambda _: [note])
    monkeypatch.setattr(performance_benchmark, "persist_note", lambda *args: PersistResult.INDEXED)
    monkeypatch.setattr(performance_benchmark, "reconcile_inactive_documents", lambda *args: 0)
    monkeypatch.setattr(performance_benchmark, "count_benchmark_rows", lambda _: (1, 3))

    result = performance_benchmark.run_indexing_stage(
        FakeConnection(), vault, "changed", chunks_per_note=3, clock_ns=ticks
    )

    assert result.affected_chunks == 3


def test_run_indexing_stage_rejects_empty_or_failed_scans(monkeypatch, tmp_path) -> None:
    from knowledge_rag import performance_benchmark

    vault = tmp_path / "vault"
    vault.mkdir()
    monkeypatch.setattr(performance_benchmark, "discover_markdown_files", lambda _: [])
    with pytest.raises(ValueError, match="discovered notes"):
        performance_benchmark.run_indexing_stage(FakeConnection(), vault, "clean", clock_ns=lambda: 1)

    note = vault / "bad.md"
    monkeypatch.setattr(performance_benchmark, "discover_markdown_files", lambda _: [note])
    monkeypatch.setattr(performance_benchmark, "persist_note", lambda *args: (_ for _ in ()).throw(RuntimeError("secret body")))
    with pytest.raises(RuntimeError, match="secret body"):
        performance_benchmark.run_indexing_stage(FakeConnection(), vault, "clean", clock_ns=iter((1, 2)).__next__)


class FakeProvider:
    def __init__(self) -> None:
        self.texts: list[str] = []

    def embed(self, text: str) -> list[float]:
        self.texts.append(text)
        return [float(len(text))]


def test_run_embedding_stage_measures_each_batch(monkeypatch) -> None:
    from knowledge_rag import performance_benchmark

    batches = [
        [ChunkForEmbedding(index, f"text-{index}", "allowed") for index in range(100)],
        [ChunkForEmbedding(index, f"text-{index}", "allowed") for index in range(100, 200)],
        [ChunkForEmbedding(index, f"text-{index}", "allowed") for index in range(200, 250)],
        [],
    ]
    load_limits: list[int] = []
    persisted: list[tuple[int | None, list[object]]] = []
    connection = FakeConnection()

    def fake_load(conn, limit):
        assert conn is connection
        load_limits.append(limit)
        return batches.pop(0)

    monkeypatch.setattr(
        performance_benchmark,
        "load_chunks_missing_embeddings",
        fake_load,
    )

    provider = FakeProvider()

    def fake_persist(conn, chunks):
        assert conn is connection
        assert conn.active_transaction is not None
        persisted.append((conn.active_transaction, chunks))

    monkeypatch.setattr(
        performance_benchmark,
        "persist_embeddings",
        fake_persist,
    )
    clock_values = iter(
        (
            0,
            10, 20, 30, 60, 70, 100,
            110, 130, 140, 180, 190, 230,
            240, 270, 280, 330, 340, 390,
            400, 450,
            500,
        )
    )

    result = performance_benchmark.run_embedding_stage(
        connection, provider, batch_size=100, clock_ns=clock_values.__next__
    )

    assert (result.loaded, result.generated, result.persisted) == (250, 250, 250)
    assert load_limits == [100, 100, 100, 100]
    assert result.load_duration_ns == 110
    assert result.generation_rate.duration_ns == 120
    assert result.persistence_rate.duration_ns == 120
    assert result.total_rate.duration_ns == 500
    assert provider.texts == [f"text-{index}" for index in range(250)]
    assert [transaction_id for transaction_id, _ in persisted] == [1, 2, 3]
    assert connection.completed_transactions == [1, 2, 3]
    assert [
        [chunk.chunk_id for chunk in chunks]
        for _, chunks in persisted
    ] == [
        list(range(100)),
        list(range(100, 200)),
        list(range(200, 250)),
    ]
    assert [
        [chunk.vector for chunk in chunks]
        for _, chunks in persisted
    ] == [
        [[float(len(f"text-{index}"))] for index in range(100)],
        [[float(len(f"text-{index}"))] for index in range(100, 200)],
        [[float(len(f"text-{index}"))] for index in range(200, 250)],
    ]


def test_run_embedding_stage_propagates_provider_failure(monkeypatch) -> None:
    from knowledge_rag import performance_benchmark

    monkeypatch.setattr(
        performance_benchmark,
        "load_chunks_missing_embeddings",
        lambda *args: [ChunkForEmbedding(1, "private", "allowed")],
    )
    monkeypatch.setattr(performance_benchmark, "persist_embeddings", lambda *args: pytest.fail("must not persist"))

    class BrokenProvider:
        def embed(self, text: str) -> list[float]:
            raise RuntimeError("provider failed")

    with pytest.raises(RuntimeError, match="provider failed"):
        performance_benchmark.run_embedding_stage(
            FakeConnection(), BrokenProvider(), clock_ns=count(1).__next__
        )


def test_run_embedding_stage_rejects_zero_vector_workload(monkeypatch) -> None:
    from knowledge_rag import performance_benchmark

    monkeypatch.setattr(performance_benchmark, "load_chunks_missing_embeddings", lambda *args: [])
    with pytest.raises(ValueError, match="no chunks"):
        performance_benchmark.run_embedding_stage(
            FakeConnection(), FakeProvider(), clock_ns=count(1).__next__
        )


def test_run_retrieval_latency_excludes_warmups(monkeypatch) -> None:
    from knowledge_rag import performance_benchmark
    from knowledge_rag.benchmark_corpus import BenchmarkWorkload, PerformanceQuery

    workload = BenchmarkWorkload(
        version=1,
        seed=7,
        tiers=(100,),
        chunks_per_note=3,
        changed_fraction=0.1,
        warmup_iterations=2,
        measured_iterations=3,
        queries=(PerformanceQuery("q1", "semantic", "query", None, None),),
    )
    events: list[str] = []
    connections = iter(FakeConnection() for _ in range(5))
    calls: list[tuple[str, object]] = []
    materialized_results: list[object] = []

    def fake_connection_factory():
        return ConnectionContext(next(connections), events)

    def fake_retrieval(*, conn, embedding_provider, query, limit, note_type, topic, retrieval_mode):
        calls.append((query, conn))
        result = object()

        def results():
            materialized_results.append(result)
            yield result

        return results()

    monkeypatch.setattr(performance_benchmark, "run_retrieval", fake_retrieval)
    clock = iter((100, 110, 200, 220, 300, 330)).__next__

    queries, modes = performance_benchmark.run_retrieval_latency(
        conn_factory=fake_connection_factory,
        provider=object(),
        workload=workload,
        clock_ns=clock,
        validate_approved_workload=False,
    )

    assert len(calls) == 5
    assert events == ["open", "close"] * 5
    assert len({id(connection) for _, connection in calls}) == 5
    assert len(materialized_results) == 5
    assert queries[0].timing.samples_ns == (10, 20, 30)
    assert queries[0].timing.sample_count == 3
    assert queries[0].result_count == 1
    assert modes[0].retrieval_mode == "semantic"
    assert modes[0].timing.samples_ns == (10, 20, 30)


def test_run_retrieval_latency_summarizes_all_approved_modes(monkeypatch) -> None:
    from knowledge_rag import performance_benchmark
    from knowledge_rag.benchmark_corpus import BenchmarkWorkload, PerformanceQuery

    workload = BenchmarkWorkload(
        version=1,
        seed=7,
        tiers=(100,),
        chunks_per_note=3,
        changed_fraction=0.1,
        warmup_iterations=0,
        measured_iterations=2,
        queries=tuple(
            PerformanceQuery(f"{mode}-{index}", mode, "query", None, None)
            for mode in ("semantic", "lexical", "hybrid")
            for index in range(3)
        ),
    )
    events: list[str] = []
    monkeypatch.setattr(performance_benchmark, "run_retrieval", lambda **kwargs: [])
    clock = iter(
        value
        for operation in range(18)
        for value in (operation * 10, operation * 10 + 5)
    ).__next__

    queries, modes = performance_benchmark.run_retrieval_latency(
        conn_factory=lambda: ConnectionContext(object(), events),
        provider=object(),
        workload=workload,
        clock_ns=clock,
    )

    assert len(queries) == 9
    assert events == ["open", "close"] * 18
    assert [(mode.retrieval_mode, mode.timing.sample_count) for mode in modes] == [
        ("semantic", 6),
        ("lexical", 6),
        ("hybrid", 6),
    ]
    assert all(mode.timing.samples_ns == (5,) * 6 for mode in modes)


@pytest.mark.parametrize(
    "mode_counts",
    [
        (3, 3, 2),
        (4, 2, 3),
        (3, 3, 4),
    ],
)
def test_run_retrieval_latency_rejects_invalid_approved_mode_distribution(
    monkeypatch,
    mode_counts,
) -> None:
    from knowledge_rag import performance_benchmark
    from knowledge_rag.benchmark_corpus import BenchmarkWorkload, PerformanceQuery

    queries = tuple(
        PerformanceQuery(f"{mode}-{index}", mode, "query", None, None)
        for mode, query_count in zip(("semantic", "lexical", "hybrid"), mode_counts)
        for index in range(query_count)
    )
    workload = BenchmarkWorkload(
        version=1,
        seed=7,
        tiers=(100,),
        chunks_per_note=3,
        changed_fraction=0.1,
        warmup_iterations=0,
        measured_iterations=1,
        queries=queries,
    )
    monkeypatch.setattr(performance_benchmark, "run_retrieval", lambda **kwargs: [])

    with pytest.raises(ValueError, match="approved retrieval workload"):
        performance_benchmark.run_retrieval_latency(
            conn_factory=lambda: ConnectionContext(object(), []),
            provider=object(),
            workload=workload,
            clock_ns=count(1).__next__,
        )


@pytest.mark.parametrize("invalid_case", ("duplicate_query_id", "unknown_extra_mode"))
def test_run_retrieval_latency_rejects_invalid_approved_queries(invalid_case) -> None:
    from knowledge_rag import performance_benchmark
    from knowledge_rag.benchmark_corpus import BenchmarkWorkload, PerformanceQuery

    queries = [
        PerformanceQuery(f"{mode}-{index}", mode, "query", None, None)
        for mode in ("semantic", "lexical", "hybrid")
        for index in range(3)
    ]
    if invalid_case == "duplicate_query_id":
        queries[-1] = PerformanceQuery("semantic-0", "hybrid", "query", None, None)
    else:
        queries.append(PerformanceQuery("extra", "unsupported", "query", None, None))
    workload = BenchmarkWorkload(
        version=1,
        seed=7,
        tiers=(100,),
        chunks_per_note=3,
        changed_fraction=0.1,
        warmup_iterations=0,
        measured_iterations=1,
        queries=tuple(queries),
    )

    with pytest.raises(ValueError, match="approved retrieval workload"):
        performance_benchmark.run_retrieval_latency(
            conn_factory=lambda: pytest.fail("invalid workload must not connect"),
            provider=object(),
            workload=workload,
            clock_ns=count(1).__next__,
        )


def _write_evaluation_dataset(path: Path) -> None:
    path.write_text(
        '{"version": 1, "cases": [{"id": "case-1", "category": "semantic", '
        '"query": "query", "retrieval_mode": "semantic", "filters": {}, '
        '"relevant_chunks": [{"source_path": "A.md", "chunk_index": 0, "relevance": 3}]}]}',
        encoding="utf-8",
    )


def test_run_quality_baseline_records_evaluation_and_passing_thresholds(monkeypatch, tmp_path) -> None:
    from knowledge_rag import performance_benchmark

    dataset = tmp_path / "evaluation.json"
    thresholds = tmp_path / "thresholds.json"
    _write_evaluation_dataset(dataset)
    thresholds.write_text(
        '{"mode_thresholds": {"semantic": {"mean_precision_at_k": 1.0, '
        '"mean_recall_at_k": 1.0, "mean_reciprocal_rank": 1.0}}, '
        '"case_thresholds": {"minimum_recall_at_k": 1.0, "minimum_reciprocal_rank": 1.0}}',
        encoding="utf-8",
    )
    events: list[str] = []
    monkeypatch.setattr(
        performance_benchmark,
        "run_retrieval",
        lambda **kwargs: [type("Result", (), {"source_path": "A.md", "chunk_index": 0})()],
    )

    result = performance_benchmark.run_quality_baseline(
        conn_factory=lambda: ConnectionContext(object(), events),
        provider=object(),
        dataset_path=dataset,
        thresholds_path=thresholds,
    )

    assert result.passed is True
    assert result.failures == ()
    assert result.evaluation == {
        "cases": [
            {
                "result": {
                    "case_id": "case-1",
                    "category": "semantic",
                    "retrieval_mode": "semantic",
                    "k": 5,
                    "retrieved_count": 1,
                    "relevant_count": 1,
                    "metrics": {
                        "precision_at_k": 1.0,
                        "recall_at_k": 1.0,
                        "reciprocal_rank": 1.0,
                    },
                },
                "ranking": [{"source_path": "A.md", "chunk_index": 0, "rank": 1}],
            }
        ],
        "summaries": [
            {
                "retrieval_mode": "semantic",
                "case_count": 1,
                "mean_precision_at_k": 1.0,
                "mean_recall_at_k": 1.0,
                "mean_reciprocal_rank": 1.0,
            }
        ],
    }
    assert events == ["open", "close"]


def test_run_quality_baseline_records_public_threshold_failures(monkeypatch, tmp_path) -> None:
    from knowledge_rag import performance_benchmark

    dataset = tmp_path / "evaluation.json"
    thresholds = tmp_path / "thresholds.json"
    _write_evaluation_dataset(dataset)
    thresholds.write_text(
        '{"mode_thresholds": {}, "case_thresholds": '
        '{"minimum_recall_at_k": 1.0, "minimum_reciprocal_rank": 1.0}}',
        encoding="utf-8",
    )
    monkeypatch.setattr(performance_benchmark, "run_retrieval", lambda **kwargs: [])

    result = performance_benchmark.run_quality_baseline(
        conn_factory=lambda: ConnectionContext(object(), []),
        provider=object(),
        dataset_path=dataset,
        thresholds_path=thresholds,
    )

    assert result.passed is False
    assert result.failures == (
        {"scope": "case", "identifier": "case-1", "metric": "recall_at_k", "actual": 0.0, "minimum": 1.0},
        {"scope": "case", "identifier": "case-1", "metric": "reciprocal_rank", "actual": 0.0, "minimum": 1.0},
    )


def test_retrieval_primary_failure_survives_connection_exit_failure(monkeypatch) -> None:
    from knowledge_rag import performance_benchmark
    from knowledge_rag.benchmark_corpus import BenchmarkWorkload, PerformanceQuery

    class ExitFailure:
        def __enter__(self):
            return object()

        def __exit__(self, *args):
            raise OSError("postgresql://secret@close")

    workload = BenchmarkWorkload(1, 1, (100,), 3, 0.1, 0, 1, (PerformanceQuery("q", "semantic", "q", None, None),))
    monkeypatch.setattr(performance_benchmark, "run_retrieval", lambda **kwargs: (_ for _ in ()).throw(ValueError("primary retrieval")))

    with pytest.raises(ValueError, match="primary retrieval") as error:
        performance_benchmark.run_retrieval_latency(
            conn_factory=ExitFailure, provider=object(), workload=workload, validate_approved_workload=False
        )
    assert "secret" not in str(error.value)


def test_retrieval_standalone_exit_failure_is_sanitized(monkeypatch) -> None:
    from knowledge_rag import performance_benchmark
    from knowledge_rag.benchmark_corpus import BenchmarkWorkload, PerformanceQuery

    class ExitFailure:
        def __enter__(self):
            return object()

        def __exit__(self, *args):
            raise OSError("postgresql://secret@close")

    workload = BenchmarkWorkload(1, 1, (100,), 3, 0.1, 0, 1, (PerformanceQuery("q", "semantic", "q", None, None),))
    monkeypatch.setattr(performance_benchmark, "run_retrieval", lambda **kwargs: [])

    with pytest.raises(performance_benchmark.BenchmarkStageError, match=r"^retrieval_latency: OSError$") as error:
        performance_benchmark.run_retrieval_latency(
            conn_factory=ExitFailure, provider=object(), workload=workload, validate_approved_workload=False
        )
    assert "secret" not in str(error.value)


def test_quality_primary_failure_survives_connection_exit_failure(monkeypatch, tmp_path) -> None:
    from knowledge_rag import performance_benchmark

    class ExitFailure:
        def __enter__(self):
            return object()

        def __exit__(self, *args):
            raise OSError("postgresql://secret@close")

    dataset = tmp_path / "evaluation.json"
    thresholds = tmp_path / "thresholds.json"
    _write_evaluation_dataset(dataset)
    thresholds.write_text('{"mode_thresholds": {}, "case_thresholds": {}}', encoding="utf-8")
    monkeypatch.setattr(performance_benchmark, "run_retrieval", lambda **kwargs: (_ for _ in ()).throw(ValueError("primary quality")))

    with pytest.raises(ValueError, match="primary quality") as error:
        performance_benchmark.run_quality_baseline(
            conn_factory=ExitFailure, provider=object(), dataset_path=dataset, thresholds_path=thresholds
        )
    assert "secret" not in str(error.value)
