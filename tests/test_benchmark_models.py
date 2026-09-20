import json
import math

import pytest

from knowledge_rag.benchmark_models import (
    BenchmarkReport,
    EmbeddingBenchmarkResult,
    EnvironmentMetadata,
    IndexingBenchmarkResult,
    ModeLatencyResult,
    QualityBaselineResult,
    QueryLatencyResult,
    RateMeasurement,
    TierBenchmarkResult,
    TimingDistribution,
    calculate_rate,
    report_to_dict,
    summarize_timings,
)


def test_summarize_timings_uses_median_and_nearest_rank_p95() -> None:
    samples = tuple(value * 1_000_000 for value in range(1, 21))

    result = summarize_timings(samples)

    assert result.samples_ns == samples
    assert result.sample_count == 20
    assert result.median_ms == 10.5
    assert result.p95_ms == 19.0


def test_calculate_rate_rejects_zero_duration() -> None:
    with pytest.raises(ValueError, match="duration_ns must be positive"):
        calculate_rate(10, 0)


def test_calculate_rate_returns_operations_per_second() -> None:
    result = calculate_rate(250, 2_000_000_000)

    assert result.count == 250
    assert result.duration_ns == 2_000_000_000
    assert result.per_second == 125.0


@pytest.mark.parametrize("samples", [(1.0, 2), (True, 2), (math.nan,), (math.inf,)])
def test_summarize_timings_rejects_non_exact_integer_samples(samples) -> None:
    with pytest.raises(ValueError):
        summarize_timings(samples)


@pytest.mark.parametrize("count,duration", [(True, 1), (1, True), (1.0, 1), (1, 1.0), (-1, 1), (1, 0)])
def test_calculate_rate_rejects_invalid_integer_inputs(count, duration) -> None:
    with pytest.raises(ValueError):
        calculate_rate(count, duration)


def test_report_serialization_is_json_safe_and_private() -> None:
    report = BenchmarkReport(
        complete=True,
        schema_version=1,
        workload_version=1,
        git_commit=None,
        executed_at_utc="2026-09-18T00:00:00Z",
        environment=EnvironmentMetadata(
            python_version="3.12",
            operating_system="Linux",
            logical_cpu_count=4,
            total_memory_bytes=1024,
            postgresql_version="16",
            pgvector_version="0.8",
        ),
        workload={
            "version": 1,
            "seed": 14069,
            "tiers": [100, 1000, 10000],
            "chunks_per_note": 3,
            "changed_fraction": 0.1,
            "warmup_iterations": 5,
            "measured_iterations": 30,
            "queries": [],
        },
        quality=None,
        tiers=(),
    )

    payload = report_to_dict(report)
    encoded = json.dumps(payload, sort_keys=True)
    round_tripped = json.loads(encoded)

    forbidden = (
        "database_url",
        "database_name",
        "postgresql://",
        "hostname",
        "db.internal",
        "port",
        "username",
        "password",
        "ip_address",
        "filesystem_path",
        "/home/user/vault",
        "environment_variables",
        "DATABASE_URL",
        "note_body",
        "private note body",
        "exception_text",
        "raw exception text",
    )
    assert all(token not in encoded for token in forbidden)
    assert payload["workload"]["tiers"] == [100, 1000, 10000]
    assert round_tripped["complete"] is True
    assert round_tripped["tiers"] == []
    assert round_tripped["quality"] is None


def test_report_serialization_rejects_sensitive_values_in_safe_string_fields() -> None:
    report = BenchmarkReport(
        complete=False,
        schema_version=1,
        workload_version=1,
        git_commit="not-a-real-commit",
        executed_at_utc="2026-09-18T00:00:00Z",
        environment=EnvironmentMetadata(
            python_version="192.168.12.34",
            operating_system="db.internal",
            logical_cpu_count=1,
            total_memory_bytes=1,
            postgresql_version="FATAL:passwordauthenticationfailed",
            pgvector_version="localhost",
        ),
        workload={
            "name": "192.168.12.34",
            "scenario": "db.internal",
            "query_id": "localhost",
            "retrieval_mode": "10.0.0.1",
            "mode": "FATAL:passwordauthenticationfailed",
            "status": "Traceback",
            "stage": "vault.internal",
        },
        quality=None,
        tiers=(),
        failure_stage="db.internal",
        failure_type="FATAL:passwordauthenticationfailed",
    )

    with pytest.raises(ValueError):
        report_to_dict(report)

def test_report_serialization_rejects_unknown_workload_keys() -> None:
    report = BenchmarkReport(
        complete=True, schema_version=1, workload_version=1, git_commit=None,
        executed_at_utc="2026-09-18T00:00:00Z", environment=None,
        workload={"unknown": 1}, quality=None, tiers=(),
    )
    with pytest.raises(ValueError, match="invalid workload schema"):
        report_to_dict(report)


def test_quality_schema_round_trips_public_case_and_failure_fields() -> None:
    quality = QualityBaselineResult(
        passed=False,
        evaluation={
            "cases": [{
                "result": {
                    "case_id": "case-1", "category": "basic",
                    "retrieval_mode": "semantic", "k": 5,
                    "retrieved_count": 2, "relevant_count": 1,
                    "metrics": {"precision_at_k": 0.5, "recall_at_k": 1.0, "reciprocal_rank": 1.0},
                },
                "ranking": [{"source_path": "20 Learning/Note.md", "chunk_index": 0, "rank": 1}],
            }],
            "summaries": [{
                "retrieval_mode": "semantic", "case_count": 1,
                "mean_precision_at_k": 0.5, "mean_recall_at_k": 1.0,
                "mean_reciprocal_rank": 1.0,
            }],
        },
        failures=({
            "scope": "case", "identifier": "case-1", "metric": "recall_at_k",
            "actual": 0.5, "minimum": 0.75,
        },),
    )
    report = BenchmarkReport(
        complete=True, schema_version=1, workload_version=1, git_commit=None,
        executed_at_utc="2026-09-18T00:00:00Z", environment=None,
        workload={"version": 1, "seed": 1, "tiers": [100], "chunks_per_note": 3,
                   "changed_fraction": 0.1, "warmup_iterations": 1,
                   "measured_iterations": 1, "queries": []},
        quality=quality, tiers=(),
    )
    assert report_to_dict(report)["quality"] == {
        "passed": False, "evaluation": quality.evaluation,
        "failures": quality.failures,
    }


def test_quality_schema_rejects_absolute_or_traversing_source_paths() -> None:
    report = BenchmarkReport(
        complete=True, schema_version=1, workload_version=1, git_commit=None,
        executed_at_utc="2026-09-18T00:00:00Z", environment=None,
        workload={"version": 1, "seed": 1, "tiers": [100], "chunks_per_note": 3,
                   "changed_fraction": 0.1, "warmup_iterations": 1,
                   "measured_iterations": 1, "queries": []}, quality=QualityBaselineResult(
            passed=True,
            evaluation={"cases": [{"result": {"case_id": "c", "category": "x", "retrieval_mode": "semantic", "k": 1, "retrieved_count": 0, "relevant_count": 0, "metrics": {"precision_at_k": 0, "recall_at_k": 0, "reciprocal_rank": 0}}, "ranking": [{"source_path": "../secret.md", "chunk_index": 0, "rank": 1}]}], "summaries": []},
            failures=(),
        ), tiers=(),
    )
    with pytest.raises(ValueError, match="invalid source_path"):
        report_to_dict(report)


@pytest.mark.parametrize("source_path", ["db.internal/note.md", "localhost/note.md", "192.168.1.2/note.md", "password/Note.md", "traceback/Note.md", "safe/db.internal", "safe/192.168.1.2", "safe/host.internal", "safe/FATAL_connection_refused", "safe/ConnectionError"])
def test_quality_schema_rejects_private_path_segments(source_path: str) -> None:
    report = BenchmarkReport(
        True, 1, 1, None, "2026-09-18T00:00:00Z", None,
        {"version": 1, "seed": 1, "tiers": [100], "chunks_per_note": 3, "changed_fraction": 0.1, "warmup_iterations": 1, "measured_iterations": 1, "queries": []},
        QualityBaselineResult(True, {"cases": [{"result": {"case_id": "c", "category": "x", "retrieval_mode": "semantic", "k": 1, "retrieved_count": 0, "relevant_count": 0, "metrics": {"precision_at_k": 0, "recall_at_k": 0, "reciprocal_rank": 0}}, "ranking": [{"source_path": source_path, "chunk_index": 0, "rank": 1}]}], "summaries": []}, ()),
        (),
    )
    with pytest.raises(ValueError):
        report_to_dict(report)


def test_report_rejects_unsafe_diagnostics_query_text_and_numeric_domains() -> None:
    workload = {"version": 1, "seed": 1, "tiers": [100], "chunks_per_note": 3,
                "changed_fraction": 0.1, "warmup_iterations": 1,
                "measured_iterations": 1, "queries": [{
                    "id": "semantic-unfiltered", "retrieval_mode": "semantic",
                    "query": "postgresql://user:password@host/db",
                    "note_type": None, "topic": None,
                }]}
    report = BenchmarkReport(
        complete=True, schema_version=1, workload_version=1, git_commit=None,
        executed_at_utc="2026-09-18T00:00:00Z", environment=None,
        workload=workload, quality=None, tiers=(),
        failure_stage="not-a-stage", failure_type="FATAL: nope",
    )
    with pytest.raises(ValueError):
        report_to_dict(report)


def test_full_tier_report_with_every_result_model_serializes() -> None:
    timing = TimingDistribution((10, 20), 2, 0.015, 0.02)
    rate = RateMeasurement(2, 1_000_000, 2_000.0)
    indexing = IndexingBenchmarkResult("clean", 2, 2, 0, 0, 0, 0, 2, 6, 6, rate, rate)
    embedding = EmbeddingBenchmarkResult(6, 6, 6, 1_000, rate, rate, rate)
    tier = TierBenchmarkResult(
        2, 6, 6, indexing, embedding, indexing, indexing, embedding,
        (QueryLatencyResult("q1", "semantic", 2, timing),),
        (ModeLatencyResult("semantic", timing),),
    )
    report = BenchmarkReport(
        True, 1, 1, None, "2026-09-18T00:00:00Z", None,
        {"version": 1, "seed": 1, "tiers": [100], "chunks_per_note": 3,
         "changed_fraction": 0.1, "warmup_iterations": 1,
         "measured_iterations": 1, "queries": []}, None, (tier,),
    )
    payload = report_to_dict(report)
    assert payload["tiers"][0]["queries"][0]["timing"]["sample_count"] == 2


@pytest.mark.parametrize("field", ["discovered", "loaded", "result_count"])
def test_full_tier_rejects_bool_in_numeric_fields(field: str) -> None:
    # The complete fixture establishes the typed model shape; each mutation must fail.
    timing = TimingDistribution((10,), 1, 0.01, 0.01)
    rate = RateMeasurement(1, 1_000, 1_000.0)
    indexing = IndexingBenchmarkResult("clean", 1, 1, 0, 0, 0, 0, 1, 3, 3, rate, rate)
    embedding = EmbeddingBenchmarkResult(1, 1, 1, 1_000, rate, rate, rate)
    tier = TierBenchmarkResult(1, 3, 3, indexing, embedding, indexing, indexing, embedding, (QueryLatencyResult("q1", "semantic", 1, timing),), (ModeLatencyResult("semantic", timing),))
    if field == "discovered":
        tier = TierBenchmarkResult(1, 3, 3, IndexingBenchmarkResult("clean", True, 1, 0, 0, 0, 0, 1, 3, 3, rate, rate), embedding, indexing, indexing, embedding, tier.queries, tier.modes)
    elif field == "loaded":
        tier = TierBenchmarkResult(1, 3, 3, indexing, EmbeddingBenchmarkResult(True, 1, 1, 1_000, rate, rate, rate), indexing, indexing, embedding, tier.queries, tier.modes)
    else:
        tier = TierBenchmarkResult(1, 3, 3, indexing, embedding, indexing, indexing, embedding, (QueryLatencyResult("q1", "semantic", True, timing),), tier.modes)
    report = BenchmarkReport(True, 1, 1, None, "2026-09-18T00:00:00Z", None, {"version": 1, "seed": 1, "tiers": [100], "chunks_per_note": 3, "changed_fraction": 0.1, "warmup_iterations": 1, "measured_iterations": 1, "queries": []}, None, (tier,))
    with pytest.raises(ValueError):
        report_to_dict(report)
