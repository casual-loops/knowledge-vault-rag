"""Pure data models and calculations used by benchmark tooling."""

# Validation deliberately reports all malformed public payloads as ValueError.
# ruff's TRY004 recommendation is inappropriate for this serialization API.
# ruff: noqa: TRY004

from __future__ import annotations

import re
from collections.abc import Sequence
from dataclasses import asdict, dataclass
from math import ceil, isfinite
from statistics import median
from typing import Any


@dataclass(frozen=True, slots=True)
class TimingDistribution:
    samples_ns: tuple[int, ...]
    sample_count: int
    median_ms: float
    p95_ms: float


@dataclass(frozen=True, slots=True)
class RateMeasurement:
    count: int
    duration_ns: int
    per_second: float


def summarize_timings(samples_ns: Sequence[int]) -> TimingDistribution:
    samples = tuple(samples_ns)
    if not samples:
        raise ValueError("timing samples must not be empty")
    if any(type(sample) is not int for sample in samples):
        raise ValueError("timing samples must be integers")
    if any(sample < 0 for sample in samples):
        raise ValueError("timing samples must be nonnegative")

    ordered = sorted(samples)
    p95_index = ceil(0.95 * len(ordered)) - 1
    return TimingDistribution(
        samples_ns=samples,
        sample_count=len(samples),
        median_ms=median(ordered) / 1_000_000,
        p95_ms=ordered[p95_index] / 1_000_000,
    )


def calculate_rate(count: int, duration_ns: int) -> RateMeasurement:
    if type(count) is not int or type(duration_ns) is not int:
        raise ValueError("count and duration_ns must be integers")
    if count < 0:
        raise ValueError("count must be nonnegative")
    if duration_ns <= 0:
        raise ValueError("duration_ns must be positive")
    return RateMeasurement(
        count=count,
        duration_ns=duration_ns,
        per_second=count / (duration_ns / 1_000_000_000),
    )


@dataclass(frozen=True, slots=True)
class IndexingBenchmarkResult:
    scenario: str
    discovered: int
    indexed: int
    unchanged: int
    excluded: int
    reactivated: int
    inactivated: int
    observed_documents: int
    observed_chunks: int
    affected_chunks: int
    document_rate: RateMeasurement
    chunk_rate: RateMeasurement


@dataclass(frozen=True, slots=True)
class EmbeddingBenchmarkResult:
    loaded: int
    generated: int
    persisted: int
    load_duration_ns: int
    generation_rate: RateMeasurement
    persistence_rate: RateMeasurement
    total_rate: RateMeasurement


@dataclass(frozen=True, slots=True)
class QueryLatencyResult:
    query_id: str
    retrieval_mode: str
    result_count: int
    timing: TimingDistribution


@dataclass(frozen=True, slots=True)
class ModeLatencyResult:
    retrieval_mode: str
    timing: TimingDistribution


@dataclass(frozen=True, slots=True)
class QualityBaselineResult:
    passed: bool
    evaluation: dict[str, Any]
    failures: tuple[dict[str, Any], ...]


@dataclass(frozen=True, slots=True)
class TierBenchmarkResult:
    note_count: int
    expected_chunks: int
    observed_chunks: int
    clean_indexing: IndexingBenchmarkResult
    initial_embedding: EmbeddingBenchmarkResult
    unchanged_indexing: IndexingBenchmarkResult
    changed_indexing: IndexingBenchmarkResult
    changed_embedding: EmbeddingBenchmarkResult
    queries: tuple[QueryLatencyResult, ...]
    modes: tuple[ModeLatencyResult, ...]


@dataclass(frozen=True, slots=True)
class EnvironmentMetadata:
    python_version: str
    operating_system: str
    logical_cpu_count: int | None
    total_memory_bytes: int | None
    postgresql_version: str
    pgvector_version: str


@dataclass(frozen=True, slots=True)
class BenchmarkReport:
    complete: bool
    schema_version: int
    workload_version: int
    git_commit: str | None
    executed_at_utc: str
    environment: EnvironmentMetadata | None
    workload: dict[str, Any]
    quality: QualityBaselineResult | None
    tiers: tuple[TierBenchmarkResult, ...]
    failure_stage: str | None = None
    failure_type: str | None = None


def report_to_dict(report: BenchmarkReport) -> dict[str, Any]:
    """Validate and serialize the public, privacy-safe report schema."""
    payload = asdict(report)
    _expect_keys(payload, _REPORT_KEYS, "report")
    if not isinstance(payload["complete"], bool) or type(payload["schema_version"]) is not int or type(payload["workload_version"]) is not int:
        raise ValueError("invalid report versions")
    if payload["git_commit"] is not None and (not isinstance(payload["git_commit"], str) or not re.fullmatch(r"[0-9a-f]{7,64}", payload["git_commit"])):
        raise ValueError("invalid git_commit")
    if not isinstance(payload["executed_at_utc"], str) or not _TIMESTAMP.fullmatch(payload["executed_at_utc"]):
        raise ValueError("invalid executed_at_utc")
    if not isinstance(payload["tiers"], (list, tuple)):
        raise ValueError("invalid tiers")
    _validate_tiers(payload["tiers"])
    _validate_environment(payload["environment"])
    _validate_workload(payload["workload"])
    if payload["quality"] is not None:
        _validate_quality(payload["quality"])
    if payload["failure_stage"] is not None and payload["failure_stage"] not in FAILURE_STAGES:
        raise ValueError("invalid failure_stage")
    if payload["failure_type"] is not None:
        _validate_exception_type(payload["failure_type"])
    return payload


_REPORT_KEYS = frozenset({
    "complete", "schema_version", "workload_version", "git_commit",
    "executed_at_utc", "environment", "workload", "quality", "tiers",
    "failure_stage", "failure_type",
})
_WORKLOAD_KEYS = frozenset({
    "version", "seed", "tiers", "chunks_per_note", "changed_fraction",
    "warmup_iterations", "measured_iterations", "queries",
})
_QUERY_KEYS = frozenset({"id", "retrieval_mode", "query", "note_type", "topic"})
_EVALUATION_KEYS = frozenset({"cases", "summaries"})
_CASE_KEYS = frozenset({"result", "ranking"})
_RESULT_KEYS = frozenset({"case_id", "category", "retrieval_mode", "k", "retrieved_count", "relevant_count", "metrics"})
_METRIC_KEYS = frozenset({"precision_at_k", "recall_at_k", "reciprocal_rank"})
_RANK_KEYS = frozenset({"source_path", "chunk_index", "rank"})
_SUMMARY_KEYS = frozenset({"retrieval_mode", "case_count", "mean_precision_at_k", "mean_recall_at_k", "mean_reciprocal_rank"})
_FAILURE_KEYS = frozenset({"scope", "identifier", "metric", "actual", "minimum"})
_TIER_KEYS = frozenset({"note_count", "expected_chunks", "observed_chunks", "clean_indexing", "initial_embedding", "unchanged_indexing", "changed_indexing", "changed_embedding", "queries", "modes"})
_INDEXING_KEYS = frozenset({"scenario", "discovered", "indexed", "unchanged", "excluded", "reactivated", "inactivated", "observed_documents", "observed_chunks", "affected_chunks", "document_rate", "chunk_rate"})
_EMBEDDING_KEYS = frozenset({"loaded", "generated", "persisted", "load_duration_ns", "generation_rate", "persistence_rate", "total_rate"})
_RATE_KEYS = frozenset({"count", "duration_ns", "per_second"})
_TIMING_KEYS = frozenset({"samples_ns", "sample_count", "median_ms", "p95_ms"})
_QUERY_RESULT_KEYS = frozenset({"query_id", "retrieval_mode", "result_count", "timing"})
_MODE_RESULT_KEYS = frozenset({"retrieval_mode", "timing"})
FAILURE_STAGES = frozenset({
    "target_validation", "database_reset", "quality_indexing", "quality_embedding",
    "quality_evaluation", "corpus_generation", "clean_indexing", "initial_embedding",
    "unchanged_indexing", "corpus_mutation", "changed_indexing", "changed_embedding",
    "retrieval_latency", "environment_collection", "git_revision", "serialization",
    "output_write",
})
APPROVED_QUERIES = frozenset({
    "capacity planning for distributed monitoring", "identity access review procedure",
    "quasar marzipan submarine orchard", "monitoring alert threshold", "role assignment",
    "private knowledge retrieval architecture", "operational change validation",
})
_SAFE_LABEL = re.compile(r"^[A-Za-z][A-Za-z0-9_.-]{0,127}$")
_EXCEPTION_TYPE = re.compile(r"^[A-Z][A-Za-z0-9]*(?:\.[A-Za-z][A-Za-z0-9]*)*(?:Error|Exception)$")
_VERSION = re.compile(r"^\d+(?:\.\d+){0,3}$")
_TIMESTAMP = re.compile(r"^\d{4}-\d{2}-\d{2}T\d{2}:\d{2}:\d{2}(?:\.\d+)?Z$")
_HOST = re.compile(r"^(?:localhost|(?:\d{1,3}\.){3}\d{1,3}|(?:[A-Za-z0-9-]+\.)+[A-Za-z]{2,})$", re.IGNORECASE)
_PRIVATE_PATH_MARKERS = ("password", "secret", "credential", "traceback", "exception", "database_url", "postgresql://", "connection", "refused", "fatal", "=", "@")


def _expect_keys(value: Any, expected: frozenset[str], name: str) -> None:
    if not isinstance(value, dict) or set(value) != expected:
        raise ValueError(f"invalid {name} schema")


def _validate_label(value: Any, name: str) -> None:
    if not isinstance(value, str) or not _SAFE_LABEL.fullmatch(value) or _HOST.fullmatch(value):
        raise ValueError(f"invalid {name}")


def _validate_exception_type(value: Any) -> None:
    if not isinstance(value, str) or not _EXCEPTION_TYPE.fullmatch(value):
        raise ValueError("invalid failure_type")


def _validate_relative_path(value: Any, name: str = "source_path") -> None:
    segments = value.split("/") if isinstance(value, str) else []
    host_segments = [segment for index, segment in enumerate(segments) if not (index == len(segments) - 1 and re.fullmatch(r"[A-Za-z0-9_-]+\.md", segment))]
    if (not isinstance(value, str) or not value or "\\" in value or value.startswith("/") or re.match(r"^[A-Za-z]:", value) or ".." in segments or "" in segments or any(_HOST.fullmatch(segment) for segment in host_segments) or any(marker in value.lower() for marker in _PRIVATE_PATH_MARKERS)):
        raise ValueError(f"invalid {name}")


def _finite_number(value: Any, *, minimum: float | None = None, maximum: float | None = None) -> bool:
    if isinstance(value, bool) or not isinstance(value, (int, float)) or not isfinite(value):
        return False
    return (minimum is None or value >= minimum) and (maximum is None or value <= maximum)


def _validate_workload(value: Any) -> None:
    _expect_keys(value, _WORKLOAD_KEYS, "workload")
    if type(value["version"]) is not int or type(value["seed"]) is not int:
        raise ValueError("invalid workload version or seed")
    if not isinstance(value["tiers"], list) or not all(type(item) is int and item > 0 for item in value["tiers"]):
        raise ValueError("invalid workload tiers")
    if type(value["chunks_per_note"]) is not int or value["chunks_per_note"] <= 0 or type(value["warmup_iterations"]) is not int or value["warmup_iterations"] < 0 or type(value["measured_iterations"]) is not int or value["measured_iterations"] <= 0 or not _finite_number(value["changed_fraction"], minimum=0, maximum=1):
        raise ValueError("invalid workload configuration")
    if not isinstance(value["queries"], list):
        raise ValueError("invalid workload queries")
    for query in value["queries"]:
        _expect_keys(query, _QUERY_KEYS, "query")
        _validate_label(query["id"], "query id")
        _validate_label(query["retrieval_mode"], "retrieval mode")
        if not isinstance(query["query"], str) or query["query"] not in APPROVED_QUERIES:
            raise ValueError("invalid query text")
        for key in ("note_type", "topic"):
            if query[key] is not None:
                _validate_label(query[key], key)


def _validate_tiers(values: list[Any] | tuple[Any, ...]) -> None:
    for tier in values:
        _expect_keys(tier, _TIER_KEYS, "tier")
        for key in ("note_count", "expected_chunks", "observed_chunks"):
            if type(tier[key]) is not int or tier[key] < 0:
                raise ValueError(f"invalid {key}")
        for key in ("clean_indexing", "unchanged_indexing", "changed_indexing"):
            _expect_keys(tier[key], _INDEXING_KEYS, "indexing result")
            _validate_label(tier[key]["scenario"], "scenario")
            for counter in _INDEXING_KEYS - {"scenario", "document_rate", "chunk_rate"}:
                if type(tier[key][counter]) is not int or tier[key][counter] < 0:
                    raise ValueError(f"invalid indexing {counter}")
            _validate_rates(tier[key])
        for key in ("initial_embedding", "changed_embedding"):
            _expect_keys(tier[key], _EMBEDDING_KEYS, "embedding result")
            for counter in ("loaded", "generated", "persisted", "load_duration_ns"):
                if type(tier[key][counter]) is not int or tier[key][counter] < 0:
                    raise ValueError(f"invalid embedding {counter}")
            _validate_rates(tier[key])
        if not isinstance(tier["queries"], (list, tuple)) or not isinstance(tier["modes"], (list, tuple)):
            raise ValueError("invalid tier collections")
        for query in tier["queries"]:
            _expect_keys(query, _QUERY_RESULT_KEYS, "query latency")
            _validate_label(query["query_id"], "query_id")
            _validate_label(query["retrieval_mode"], "retrieval_mode")
            if type(query["result_count"]) is not int or query["result_count"] < 0:
                raise ValueError("invalid result_count")
            _validate_timing(query["timing"])
        for mode in tier["modes"]:
            _expect_keys(mode, _MODE_RESULT_KEYS, "mode latency")
            _validate_label(mode["retrieval_mode"], "retrieval_mode")
            _validate_timing(mode["timing"])


def _validate_rates(value: dict[str, Any]) -> None:
    for key in ("document_rate", "chunk_rate", "generation_rate", "persistence_rate", "total_rate"):
        if key in value:
            _expect_keys(value[key], _RATE_KEYS, "rate")
            if type(value[key]["count"]) is not int or value[key]["count"] < 0 or type(value[key]["duration_ns"]) is not int or value[key]["duration_ns"] <= 0 or not _finite_number(value[key]["per_second"], minimum=0):
                raise ValueError("invalid rate")


def _validate_timing(value: Any) -> None:
    _expect_keys(value, _TIMING_KEYS, "timing")
    if not isinstance(value["samples_ns"], (tuple, list)) or not all(type(sample) is int and sample >= 0 for sample in value["samples_ns"]):
        raise ValueError("invalid timing samples")
    if type(value["sample_count"]) is not int or value["sample_count"] != len(value["samples_ns"]):
        raise ValueError("invalid timing count")
    if not _finite_number(value["median_ms"], minimum=0) or not _finite_number(value["p95_ms"], minimum=0):
        raise ValueError("invalid timing statistics")


def _validate_environment(value: Any) -> None:
    if value is None:
        return
    _expect_keys(value, frozenset({"python_version", "operating_system", "logical_cpu_count", "total_memory_bytes", "postgresql_version", "pgvector_version"}), "environment")
    for key in ("python_version", "postgresql_version", "pgvector_version"):
        if not isinstance(value[key], str) or not _VERSION.fullmatch(value[key]):
            raise ValueError(f"invalid {key}")
    if value["operating_system"] not in {"Linux", "Darwin", "Windows"}:
        raise ValueError("invalid operating_system")
    for key in ("logical_cpu_count", "total_memory_bytes"):
        if value[key] is not None and (type(value[key]) is not int or value[key] < 0):
            raise ValueError(f"invalid {key}")


def _validate_quality(value: Any) -> None:
    _expect_keys(value, frozenset({"passed", "evaluation", "failures"}), "quality")
    if not isinstance(value["passed"], bool):
        raise ValueError("invalid quality status")
    evaluation = value["evaluation"]
    _expect_keys(evaluation, _EVALUATION_KEYS, "evaluation")
    if not isinstance(evaluation["cases"], list) or not isinstance(evaluation["summaries"], list):
        raise ValueError("invalid evaluation collections")
    for item in evaluation["cases"]:
        _expect_keys(item, _CASE_KEYS, "evaluation case")
        result = item["result"]
        _expect_keys(result, _RESULT_KEYS, "evaluation result")
        _validate_label(result["case_id"], "case_id")
        _validate_label(result["category"], "category")
        _validate_label(result["retrieval_mode"], "retrieval_mode")
        for key in ("k", "retrieved_count", "relevant_count"):
            if type(result[key]) is not int or result[key] < 0:
                raise ValueError(f"invalid {key}")
        _expect_keys(result["metrics"], _METRIC_KEYS, "metrics")
        for metric in result["metrics"].values():
            if not _finite_number(metric, minimum=0, maximum=1):
                raise ValueError("invalid metric")
        if not isinstance(item["ranking"], list):
            raise ValueError("invalid ranking")
        for rank in item["ranking"]:
            _expect_keys(rank, _RANK_KEYS, "ranking")
            _validate_relative_path(rank["source_path"])
            if not all(type(rank[key]) is int and rank[key] >= 0 for key in ("chunk_index", "rank")):
                raise ValueError("invalid ranking position")
    for summary in evaluation["summaries"]:
        _expect_keys(summary, _SUMMARY_KEYS, "summary")
        _validate_label(summary["retrieval_mode"], "retrieval_mode")
        if type(summary["case_count"]) is not int or summary["case_count"] < 0:
            raise ValueError("invalid case_count")
        for key in ("mean_precision_at_k", "mean_recall_at_k", "mean_reciprocal_rank"):
            if not _finite_number(summary[key], minimum=0, maximum=1):
                raise ValueError(f"invalid {key}")
    if not isinstance(value["failures"], (list, tuple)):
        raise ValueError("invalid threshold failures")
    for failure in value["failures"]:
        _expect_keys(failure, _FAILURE_KEYS, "threshold failure")
        _validate_label(failure["scope"], "failure scope")
        _validate_label(failure["identifier"], "failure identifier")
        _validate_label(failure["metric"], "failure metric")
        if not _finite_number(failure["actual"], minimum=0) or not _finite_number(failure["minimum"], minimum=0):
            raise ValueError("invalid threshold values")
