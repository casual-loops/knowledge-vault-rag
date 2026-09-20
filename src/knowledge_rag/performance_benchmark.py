"""Measured stages used by the synthetic performance benchmark."""

from __future__ import annotations

import json
import os
import subprocess
import tempfile
from collections.abc import Callable
from contextlib import AbstractContextManager, contextmanager
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from shutil import rmtree
from time import perf_counter_ns
from typing import Any

import psycopg
from psycopg import Connection

from knowledge_rag.api.main import run_retrieval
from knowledge_rag.benchmark_corpus import (
    BenchmarkWorkload,
    generate_synthetic_vault,
    load_benchmark_workload,
    modify_synthetic_vault,
    workload_to_dict,
)
from knowledge_rag.benchmark_database import (
    collect_environment_metadata,
    count_benchmark_rows,
    reset_benchmark_tables,
    validate_benchmark_target,
)
from knowledge_rag.benchmark_models import (
    FAILURE_STAGES,
    BenchmarkReport,
    EmbeddingBenchmarkResult,
    IndexingBenchmarkResult,
    ModeLatencyResult,
    QualityBaselineResult,
    QueryLatencyResult,
    TierBenchmarkResult,
    calculate_rate,
    report_to_dict,
    summarize_timings,
)
from knowledge_rag.embedding_pipeline import EmbeddedChunk
from knowledge_rag.embedding_store import (
    load_chunks_missing_embeddings,
    persist_embeddings,
)
from knowledge_rag.embeddings import DeterministicEmbeddingProvider, EmbeddingProvider
from knowledge_rag.evaluation import chunk_identity
from knowledge_rag.evaluation_runner import run_evaluation
from knowledge_rag.evaluation_thresholds import check_thresholds, load_thresholds
from knowledge_rag.ingest.markdown import discover_markdown_files
from knowledge_rag.ingest.persistence import (
    PersistResult,
    persist_note,
    reconcile_inactive_documents,
)
from knowledge_rag.logging_utils import safe_note_reference

Clock = Callable[[], int]
_RETRIEVAL_MODES = ("semantic", "lexical", "hybrid")
_APPROVED_TIERS = (100, 1_000, 10_000)


@dataclass(frozen=True, slots=True)
class BenchmarkRunConfig:
    repository_root: Path
    benchmark_database_url: str
    application_database_url: str
    workload_path: Path
    sample_vault_path: Path
    evaluation_dataset_path: Path
    thresholds_path: Path
    allow_reset: bool
    work_directory: Path | None = None


class BenchmarkStageError(Exception):
    """A public-safe stage failure that deliberately omits exception details."""

    def __init__(self, stage: str, error_type: str) -> None:
        if stage not in FAILURE_STAGES:
            raise ValueError("invalid benchmark stage")
        self.stage = stage
        self.error_type = error_type
        super().__init__(f"{stage}: {error_type}")


def _stage(stage: str, operation: Callable[[], Any]) -> Any:
    try:
        return operation()
    except BenchmarkStageError:
        raise
    except Exception as exc:  # noqa: BLE001 - diagnostics must sanitize every stage failure.
        raise BenchmarkStageError(stage, type(exc).__name__) from None


@contextmanager
def _stage_connection(
    stage: str, factory: Callable[[], AbstractContextManager[Connection]]
):
    manager = _stage(stage, factory)
    connection = _stage(stage, manager.__enter__)
    try:
        yield connection
    except BaseException as primary:
        try:
            _stage(
                stage,
                lambda primary=primary: manager.__exit__(
                    type(primary), primary, primary.__traceback__
                ),
            )
        except BenchmarkStageError:
            pass
        raise
    else:
        _stage(stage, lambda: manager.__exit__(None, None, None))


def _load_approved_workload(path: Path) -> BenchmarkWorkload:
    workload = load_benchmark_workload(path)
    if (
        workload.version != 1
        or workload.seed != 14_069
        or workload.tiers != _APPROVED_TIERS
        or workload.chunks_per_note != 3
        or workload.changed_fraction != 0.1
        or workload.warmup_iterations != 5
        or workload.measured_iterations != 30
    ):
        raise ValueError("invalid approved benchmark workload")
    _validate_approved_retrieval_workload(workload)
    return workload


def _create_owned_work_root(work_directory: Path | None) -> Path:
    if work_directory is not None:
        work_directory.mkdir(parents=True, exist_ok=True)
        return Path(tempfile.mkdtemp(prefix="knowledge-rag-benchmark-", dir=work_directory))
    return Path(tempfile.mkdtemp(prefix="knowledge-rag-benchmark-"))


def _lexical_absolute(path: Path) -> Path:
    return Path(os.path.abspath(path))


def _git_commit(repository_root: Path) -> str:
    return subprocess.run(
        ["git", "rev-parse", "HEAD"],
        cwd=repository_root,
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()


def _timestamp() -> str:
    return datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")  # noqa: UP017


def run_performance_baseline(config: BenchmarkRunConfig) -> BenchmarkReport:
    """Run the complete provider-free benchmark against a validated target."""
    _stage(
        "target_validation",
        lambda: validate_benchmark_target(
            config.application_database_url,
            config.benchmark_database_url,
            config.allow_reset,
        ),
    )
    workload = _stage("target_validation", lambda: _load_approved_workload(config.workload_path))
    provider = DeterministicEmbeddingProvider(dimensions=1536)
    conn_factory = lambda: psycopg.connect(config.benchmark_database_url)

    with _stage_connection("database_reset", conn_factory) as conn:
        _stage("database_reset", lambda: reset_benchmark_tables(conn))
        _stage(
            "quality_indexing",
            lambda: run_indexing_stage(conn, config.sample_vault_path, "clean"),
        )
        _stage("quality_embedding", lambda: run_embedding_stage(conn, provider))

    quality = _stage(
        "quality_evaluation",
        lambda: run_quality_baseline(
            conn_factory=conn_factory,
            provider=provider,
            dataset_path=config.evaluation_dataset_path,
            thresholds_path=config.thresholds_path,
        ),
    )

    work_root = _stage("corpus_generation", lambda: _create_owned_work_root(config.work_directory))
    primary_error: BenchmarkStageError | None = None
    try:
        tiers = _run_tiers(work_root, workload, provider, conn_factory)
    except BenchmarkStageError as error:
        primary_error = error
        raise
    finally:
        try:
            _stage("corpus_generation", lambda: rmtree(work_root))
        except BenchmarkStageError:
            if primary_error is None:
                raise

    environment = _stage(
        "environment_collection",
        lambda: _collect_environment(conn_factory),
    )
    git_commit = _stage("git_revision", lambda: _git_commit(config.repository_root))
    return BenchmarkReport(
        complete=True,
        schema_version=1,
        workload_version=workload.version,
        git_commit=git_commit,
        executed_at_utc=_timestamp(),
        environment=environment,
        workload=workload_to_dict(workload),
        quality=quality,
        tiers=tiers,
    )


def _collect_environment(conn_factory: Callable[[], AbstractContextManager[Connection]]):
    with _stage_connection("environment_collection", conn_factory) as conn:
        return collect_environment_metadata(conn)


def _run_tiers(
    work_root: Path,
    workload: BenchmarkWorkload,
    provider: EmbeddingProvider,
    conn_factory: Callable[[], AbstractContextManager[Connection]],
) -> tuple[TierBenchmarkResult, ...]:
    tiers: list[TierBenchmarkResult] = []
    for note_count in workload.tiers:
        vault_path = work_root / f"tier-{note_count}"
        with _stage_connection("database_reset", conn_factory) as conn:
            _stage("database_reset", lambda: reset_benchmark_tables(conn))
            manifest = _stage(
                "corpus_generation",
                lambda vault_path=vault_path, note_count=note_count: generate_synthetic_vault(
                    vault_path, workload, note_count
                ),
            )
            clean = _stage(
                "clean_indexing",
                lambda vault_path=vault_path: run_indexing_stage(conn, vault_path, "clean"),
            )
            initial_embedding = _stage("initial_embedding", lambda: run_embedding_stage(conn, provider))
            unchanged = _stage(
                "unchanged_indexing",
                lambda vault_path=vault_path: run_indexing_stage(conn, vault_path, "unchanged"),
            )
            _stage(
                "corpus_mutation",
                lambda vault_path=vault_path, manifest=manifest: modify_synthetic_vault(
                    vault_path, manifest, workload.changed_fraction
                ),
            )
            changed = _stage(
                "changed_indexing",
                lambda vault_path=vault_path: run_indexing_stage(
                    conn, vault_path, "changed", workload.chunks_per_note
                ),
            )
            changed_embedding = _stage("changed_embedding", lambda: run_embedding_stage(conn, provider))
        queries, modes = _stage(
            "retrieval_latency",
            lambda: run_retrieval_latency(
                conn_factory=conn_factory, provider=provider, workload=workload
            ),
        )
        tiers.append(
            TierBenchmarkResult(
                note_count=note_count,
                expected_chunks=manifest.chunk_count,
                observed_chunks=clean.observed_chunks,
                clean_indexing=clean,
                initial_embedding=initial_embedding,
                unchanged_indexing=unchanged,
                changed_indexing=changed,
                changed_embedding=changed_embedding,
                queries=queries,
                modes=modes,
            )
        )
    return tuple(tiers)


def make_incomplete_report(
    workload_version: int,
    workload: dict[str, Any],
    stage: str,
    error_type: str,
    git_commit: str | None = None,
) -> BenchmarkReport:
    """Build a diagnostic report containing only public-safe failure fields."""
    if stage not in FAILURE_STAGES:
        raise ValueError("invalid benchmark stage")
    return BenchmarkReport(
        complete=False,
        schema_version=1,
        workload_version=workload_version,
        git_commit=git_commit,
        executed_at_utc=_timestamp(),
        environment=None,
        workload=workload,
        quality=None,
        tiers=(),
        failure_stage=stage,
        failure_type=error_type,
    )


def write_benchmark_report(
    report: BenchmarkReport, output_path: Path, baselines_dir: Path
) -> None:
    """Atomically serialize a report without allowing incomplete canonical baselines."""
    output = _lexical_absolute(output_path)
    baselines = _lexical_absolute(baselines_dir)
    try:
        output.relative_to(baselines)
        is_canonical = True
    except ValueError:
        is_canonical = False
    if is_canonical and not report.complete:
        raise ValueError("incomplete report cannot replace a canonical baseline")

    serialized = json.dumps(report_to_dict(report), indent=2, sort_keys=True) + "\n"
    output.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(
        mode="w", encoding="utf-8", dir=output.parent, prefix=f".{output.name}.", delete=False
    ) as temporary:
        temporary.write(serialized)
        temporary_path = Path(temporary.name)
    temporary_path.replace(output)


def format_benchmark_summary(report: BenchmarkReport) -> str:
    """Return a compact public-safe summary for a completed benchmark report."""
    tier_lines = [
        (
            f"tier {tier.note_count}: chunks={tier.observed_chunks}, "
            f"clean_chunks_per_second={tier.clean_indexing.chunk_rate.per_second:.2f}, "
            f"embedding_per_second={tier.initial_embedding.total_rate.per_second:.2f}"
        )
        for tier in report.tiers
    ]
    quality = "not run" if report.quality is None else str(report.quality.passed).lower()
    return "\n".join((f"benchmark complete={str(report.complete).lower()} quality_passed={quality}", *tier_lines))


def run_retrieval_latency(
    *,
    conn_factory: Callable[[], AbstractContextManager[Connection]],
    provider: EmbeddingProvider,
    workload: BenchmarkWorkload,
    clock_ns: Clock = perf_counter_ns,
    validate_approved_workload: bool = True,
) -> tuple[tuple[QueryLatencyResult, ...], tuple[ModeLatencyResult, ...]]:
    """Measure request-like retrieval latency for every performance query."""
    if validate_approved_workload:
        _validate_approved_retrieval_workload(workload)

    query_results: list[QueryLatencyResult] = []

    for query in workload.queries:
        for _ in range(workload.warmup_iterations):
            with _stage_connection("retrieval_latency", conn_factory) as conn:
                list(
                    run_retrieval(
                        conn=conn,
                        embedding_provider=provider,
                        query=query.query,
                        limit=5,
                        note_type=query.note_type,
                        topic=query.topic,
                        retrieval_mode=query.retrieval_mode,
                    )
                )

        samples: list[int] = []
        result_count = 0
        for _ in range(workload.measured_iterations):
            started_ns = clock_ns()
            with _stage_connection("retrieval_latency", conn_factory) as conn:
                results = list(
                    run_retrieval(
                        conn=conn,
                        embedding_provider=provider,
                        query=query.query,
                        limit=5,
                        note_type=query.note_type,
                        topic=query.topic,
                        retrieval_mode=query.retrieval_mode,
                    )
                )
            samples.append(clock_ns() - started_ns)
            result_count = len(results)

        if len(samples) != workload.measured_iterations:
            raise ValueError("retrieval sample count does not match workload")
        query_results.append(
            QueryLatencyResult(
                query_id=query.query_id,
                retrieval_mode=query.retrieval_mode,
                result_count=result_count,
                timing=summarize_timings(samples),
            )
        )

    mode_results: list[ModeLatencyResult] = []
    for mode in _RETRIEVAL_MODES:
        samples = [
            sample
            for result in query_results
            if result.retrieval_mode == mode
            for sample in result.timing.samples_ns
        ]
        if validate_approved_workload and len(samples) != 3 * workload.measured_iterations:
            raise ValueError("retrieval mode sample count does not match workload")
        if samples:
            mode_results.append(
                ModeLatencyResult(
                    retrieval_mode=mode,
                    timing=summarize_timings(samples),
                )
            )

    if validate_approved_workload and len(mode_results) != len(_RETRIEVAL_MODES):
        raise ValueError("retrieval mode results do not match workload")
    return tuple(query_results), tuple(mode_results)


def _validate_approved_retrieval_workload(workload: BenchmarkWorkload) -> None:
    """Require the production latency workload's three-query mode distribution."""
    if (
        len(workload.queries) != 9
        or len({query.query_id for query in workload.queries}) != len(workload.queries)
        or any(query.retrieval_mode not in _RETRIEVAL_MODES for query in workload.queries)
    ):
        raise ValueError("invalid approved retrieval workload")
    if any(
        sum(query.retrieval_mode == mode for query in workload.queries) != 3
        for mode in _RETRIEVAL_MODES
    ):
        raise ValueError("invalid approved retrieval workload")


def run_quality_baseline(
    *,
    conn_factory: Callable[[], AbstractContextManager[Connection]],
    provider: EmbeddingProvider,
    dataset_path: Path,
    thresholds_path: Path,
    k: int = 5,
) -> QualityBaselineResult:
    """Run judged retrieval evaluation using a fresh connection per case."""

    def retrieval_executor(case, limit):
        with _stage_connection("quality_evaluation", conn_factory) as conn:
            results = list(
                run_retrieval(
                    conn=conn,
                    embedding_provider=provider,
                    query=case.query,
                    limit=limit,
                    note_type=case.filters.get("note_type"),
                    topic=case.filters.get("topic"),
                    retrieval_mode=case.retrieval_mode,
                )
            )
        return [
            chunk_identity(result.source_path, result.chunk_index)
            for result in results
        ]

    evaluation = run_evaluation(
        dataset_path=dataset_path,
        retrieval_executor=retrieval_executor,
        k=k,
    )
    threshold_check = check_thresholds(
        results=evaluation.results,
        summaries=evaluation.summaries,
        thresholds=load_thresholds(thresholds_path),
    )
    return QualityBaselineResult(
        passed=threshold_check.passed,
        evaluation=evaluation.to_dict(),
        failures=tuple(
            {
                "scope": failure.scope,
                "identifier": failure.identifier,
                "metric": failure.metric,
                "actual": failure.actual,
                "minimum": failure.minimum,
            }
            for failure in threshold_check.failures
        ),
    )


def run_indexing_stage(
    conn: Connection,
    vault_path: Path,
    scenario: str,
    chunks_per_note: int | None = None,
    clock_ns: Clock = perf_counter_ns,
) -> IndexingBenchmarkResult:
    """Index one synthetic vault and measure discovery through reconciliation."""
    if scenario not in {"clean", "unchanged", "changed"}:
        raise ValueError("scenario must be clean, unchanged, or changed")
    if scenario == "changed" and (chunks_per_note is None or chunks_per_note <= 0):
        raise ValueError("changed indexing requires chunks_per_note")
    if chunks_per_note is not None and chunks_per_note <= 0:
        raise ValueError("chunks_per_note must be positive")

    started_ns = clock_ns()
    note_paths = discover_markdown_files(vault_path)
    if not note_paths:
        raise ValueError("benchmark requires discovered notes")

    discovered_paths: set[str] = set()
    counts = {result: 0 for result in PersistResult}
    for note_path in note_paths:
        reference = safe_note_reference(vault_path, note_path)
        discovered_paths.add(reference)
        with conn.transaction():
            result = persist_note(conn, vault_path, note_path)
        counts[result] += 1

    with conn.transaction():
        inactivated = reconcile_inactive_documents(conn, discovered_paths)
    duration_ns = clock_ns() - started_ns
    if duration_ns <= 0:
        raise ValueError("indexing duration must be positive")

    observed_documents, observed_chunks = count_benchmark_rows(conn)
    if scenario == "clean":
        affected_chunks = observed_chunks
    elif scenario == "unchanged":
        affected_chunks = 0
    else:
        assert chunks_per_note is not None
        affected_chunks = counts[PersistResult.INDEXED] * chunks_per_note

    return IndexingBenchmarkResult(
        scenario=scenario,
        discovered=len(note_paths),
        indexed=counts[PersistResult.INDEXED],
        unchanged=counts[PersistResult.UNCHANGED],
        excluded=counts[PersistResult.EXCLUDED],
        reactivated=counts[PersistResult.REACTIVATED],
        inactivated=inactivated,
        observed_documents=observed_documents,
        observed_chunks=observed_chunks,
        affected_chunks=affected_chunks,
        document_rate=calculate_rate(len(note_paths), duration_ns),
        chunk_rate=calculate_rate(affected_chunks, duration_ns),
    )


def run_embedding_stage(
    conn: Connection,
    provider: EmbeddingProvider,
    batch_size: int = 100,
    clock_ns: Clock = perf_counter_ns,
) -> EmbeddingBenchmarkResult:
    """Measure loading, local vector generation, and vector persistence."""
    if batch_size <= 0:
        raise ValueError("batch_size must be positive")

    total_started_ns = clock_ns()
    loaded_count = 0
    generated_count = 0
    persisted_count = 0
    load_duration_ns = 0
    generation_duration_ns = 0
    persistence_duration_ns = 0

    while True:
        load_started_ns = clock_ns()
        chunks = load_chunks_missing_embeddings(conn, batch_size)
        load_duration_ns += clock_ns() - load_started_ns
        if not chunks:
            break
        loaded_count += len(chunks)

        generation_started_ns = clock_ns()
        embedded_chunks = [
            EmbeddedChunk(chunk_id=chunk.chunk_id, vector=provider.embed(chunk.content))
            for chunk in chunks
        ]
        generation_duration_ns += clock_ns() - generation_started_ns
        generated_count += len(embedded_chunks)

        persistence_started_ns = clock_ns()
        with conn.transaction():
            persist_embeddings(conn, embedded_chunks)
        persistence_duration_ns += clock_ns() - persistence_started_ns
        persisted_count += len(embedded_chunks)

    total_duration_ns = clock_ns() - total_started_ns
    if loaded_count == 0:
        raise ValueError("embedding workload contains no chunks")
    if (loaded_count, generated_count, persisted_count) != (loaded_count,) * 3:
        raise ValueError("embedding stage counts do not match")
    if load_duration_ns < 0 or generation_duration_ns < 0 or persistence_duration_ns < 0:
        raise ValueError("embedding durations must be nonnegative")
    if generation_duration_ns <= 0 or persistence_duration_ns <= 0 or total_duration_ns <= 0:
        raise ValueError("embedding durations must be positive")

    return EmbeddingBenchmarkResult(
        loaded=loaded_count,
        generated=generated_count,
        persisted=persisted_count,
        load_duration_ns=load_duration_ns,
        generation_rate=calculate_rate(generated_count, generation_duration_ns),
        persistence_rate=calculate_rate(persisted_count, persistence_duration_ns),
        total_rate=calculate_rate(persisted_count, total_duration_ns),
    )
