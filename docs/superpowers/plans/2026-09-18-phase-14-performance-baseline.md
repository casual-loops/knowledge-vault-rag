# Phase 14 Performance Baseline Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (- [ ]) syntax for tracking.

**Goal:** Build a reproducible, privacy-safe pre-refactor benchmark that captures retrieval quality, scaled retrieval latency, indexing throughput, and embedding throughput without using private content or paid providers.

**Architecture:** Preserve the existing judged retrieval evaluation as the quality phase. Add a deterministic synthetic corpus generator and a dedicated performance runner that measures the current persistence, embedding, and retrieval paths at 100, 1,000, and 10,000 notes, then writes one versioned sanitized report.

**Tech Stack:** Python 3.12, psycopg 3, PostgreSQL 17, pgvector 0.8, pytest 8, Ruff, pathlib, dataclasses, hashlib, json, statistics, time.perf_counter_ns.

**Spec:** docs/superpowers/specs/2026-09-18-phase-14-performance-baseline-design.md

## Global Constraints

1. Use only synthetic or deliberately sanitized content.
2. Use the deterministic embedding provider with exactly 1,536 dimensions.
3. Never call a hosted embedding or generation provider.
4. Use corpus tiers of exactly 100, 1,000, and 10,000 notes.
5. Generate exactly three chunks per synthetic note.
6. Modify exactly 10 percent of notes in the changed-note scenario.
7. Use five warm-up iterations and 30 measured iterations for each performance query.
8. Retain raw latency samples and report median and nearest-rank p95.
9. Keep retrieval quality separate from the scaled performance corpus.
10. Do not introduce hard performance thresholds.
11. Record retrieval-quality threshold status without using it as a completion gate because the SHA-256 deterministic provider does not encode semantic similarity.
12. Require BENCHMARK_DATABASE_URL, a database name containing benchmark, a target distinct from the application database, and explicit reset authorization.
13. Never serialize database URLs, hostnames, usernames, IP addresses, ports, database names, filesystem paths, environment variables, note bodies, or raw exception text.
14. The canonical baseline file may be written only by a complete successful run.
15. Use test-driven development and commit after every task.
16. Database-backed tests remain active and must pass in the PostgreSQL and pgvector environment.

---

## File structure

The implementation will create or modify these files:

1. src/knowledge_rag/benchmark_models.py: workload models, result models, timing statistics, throughput calculation, and stable serialization.
2. src/knowledge_rag/benchmark_corpus.py: workload loading, deterministic corpus generation, manifests, and deterministic changed-note mutation.
3. src/knowledge_rag/benchmark_database.py: database-target validation, benchmark table reset, database counts, and sanitized environment collection.
4. src/knowledge_rag/performance_benchmark.py: indexing, embedding, retrieval, quality, and full-run orchestration.
5. scripts/run_performance_baseline.py: command-line parsing, configuration gates, result writing, summary output, and exit status.
6. benchmarks/workload.json: versioned seed, tiers, query workload, and repetition configuration.
7. benchmarks/baselines/phase-14-pre-refactor.json: successful reviewed homelab result, added only in the final task.
8. docs/performance-benchmarking.md: operator procedure, timing boundaries, interpretation, safety, and limitations.
9. README.md: Phase 14 benchmark usage and status.
10. docs/architecture.md: benchmark subsystem architecture.
11. .env.example: empty BENCHMARK_DATABASE_URL setting.
12. .gitignore: disposable benchmark results and generated workloads.
13. tests/test_benchmark_models.py: calculations, validation, and serialization.
14. tests/test_benchmark_corpus.py: deterministic generation and mutation.
15. tests/test_benchmark_database.py: target safety, reset behavior, and environment sanitization.
16. tests/test_performance_benchmark.py: stage orchestration and timing behavior.
17. tests/test_run_performance_baseline.py: command-line gates, output protection, summaries, and failure exit status.
18. tests/test_performance_benchmark_integration.py: PostgreSQL and pgvector lifecycle proof with a deliberately small test workload.

### Task 1: Benchmark result models and timing calculations

**Files:**
- Create: src/knowledge_rag/benchmark_models.py
- Create: tests/test_benchmark_models.py

**Interfaces:**
- Produces: TimingDistribution, RateMeasurement, IndexingBenchmarkResult, EmbeddingBenchmarkResult, QueryLatencyResult, ModeLatencyResult, QualityBaselineResult, TierBenchmarkResult, EnvironmentMetadata, BenchmarkReport.
- Produces: summarize_timings(samples_ns: Sequence[int]) -> TimingDistribution.
- Produces: calculate_rate(count: int, duration_ns: int) -> RateMeasurement.
- Produces: report_to_dict(report: BenchmarkReport) -> dict[str, Any].
- Consumes: only Python standard-library types.

- [ ] **Step 1: Write failing timing and rate tests**

~~~python
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
~~~

- [ ] **Step 2: Run the focused tests and verify failure**

Run:

~~~powershell
python -m pytest tests/test_benchmark_models.py -v
~~~

Expected: collection fails because knowledge_rag.benchmark_models does not exist.

- [ ] **Step 3: Implement timing primitives and explicit result models**

Use nearest-rank p95 so the calculation is stable and easy to audit:

~~~python
from __future__ import annotations

from dataclasses import asdict, dataclass
from math import ceil
from statistics import median
from typing import Any, Sequence


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
    if count < 0:
        raise ValueError("count must be nonnegative")
    if duration_ns <= 0:
        raise ValueError("duration_ns must be positive")

    return RateMeasurement(
        count=count,
        duration_ns=duration_ns,
        per_second=count / (duration_ns / 1_000_000_000),
    )
~~~

Define the remaining frozen dataclasses with these exact fields:

~~~python
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
    return asdict(report)
~~~

- [ ] **Step 4: Add serialization privacy tests**

Construct a complete minimal BenchmarkReport and assert json.dumps(report_to_dict(report), sort_keys=True) contains no keys named database_url, hostname, username, ip_address, filesystem_path, note_body, or exception_text. Assert the payload round-trips through json.loads.

- [ ] **Step 5: Run focused tests**

Run:

~~~powershell
python -m pytest tests/test_benchmark_models.py -v
~~~

Expected: all tests pass.

- [ ] **Step 6: Commit Task 1**

~~~bash
git add src/knowledge_rag/benchmark_models.py tests/test_benchmark_models.py
git commit -m "feat: add benchmark result models"
~~~

### Task 2: Versioned workload and deterministic corpus generation

**Files:**
- Create: benchmarks/workload.json
- Create: src/knowledge_rag/benchmark_corpus.py
- Create: tests/test_benchmark_corpus.py

**Interfaces:**
- Consumes: Timing-independent workload values from benchmarks/workload.json.
- Produces: PerformanceQuery and BenchmarkWorkload.
- Produces: load_benchmark_workload(path: Path) -> BenchmarkWorkload.
- Produces: generate_synthetic_vault(root: Path, workload: BenchmarkWorkload, note_count: int) -> CorpusManifest.
- Produces: modify_synthetic_vault(root: Path, manifest: CorpusManifest, fraction: float) -> tuple[str, ...].
- Produces: CorpusManifest with version, seed, note_count, chunk_count, paths, and content_digest.

- [ ] **Step 1: Add the exact workload definition**

Create benchmarks/workload.json:

~~~json
{
  "version": 1,
  "seed": 14069,
  "tiers": [100, 1000, 10000],
  "chunks_per_note": 3,
  "changed_fraction": 0.1,
  "warmup_iterations": 5,
  "measured_iterations": 30,
  "queries": [
    {
      "id": "semantic-unfiltered",
      "retrieval_mode": "semantic",
      "query": "capacity planning for distributed monitoring",
      "note_type": null,
      "topic": null
    },
    {
      "id": "semantic-filtered",
      "retrieval_mode": "semantic",
      "query": "identity access review procedure",
      "note_type": "reference",
      "topic": "identity"
    },
    {
      "id": "semantic-no-match",
      "retrieval_mode": "semantic",
      "query": "quasar marzipan submarine orchard",
      "note_type": null,
      "topic": null
    },
    {
      "id": "lexical-unfiltered",
      "retrieval_mode": "lexical",
      "query": "monitoring alert threshold",
      "note_type": null,
      "topic": null
    },
    {
      "id": "lexical-filtered",
      "retrieval_mode": "lexical",
      "query": "role assignment",
      "note_type": "reference",
      "topic": "identity"
    },
    {
      "id": "lexical-no-match",
      "retrieval_mode": "lexical",
      "query": "quasar marzipan submarine orchard",
      "note_type": null,
      "topic": null
    },
    {
      "id": "hybrid-unfiltered",
      "retrieval_mode": "hybrid",
      "query": "private knowledge retrieval architecture",
      "note_type": null,
      "topic": null
    },
    {
      "id": "hybrid-filtered",
      "retrieval_mode": "hybrid",
      "query": "operational change validation",
      "note_type": "project",
      "topic": "operations"
    },
    {
      "id": "hybrid-no-match",
      "retrieval_mode": "hybrid",
      "query": "quasar marzipan submarine orchard",
      "note_type": null,
      "topic": null
    }
  ]
}
~~~

- [ ] **Step 2: Write failing workload and corpus tests**

Tests must prove:

1. The workload loads exactly three tiers and nine queries.
2. Query modes contain three semantic, three lexical, and three hybrid entries.
3. Two generation runs with the same seed and tier produce byte-identical files and equal manifests.
4. A 100-note corpus produces 100 Markdown files and exactly 300 chunks when each file is passed through chunk_markdown after load_markdown.
5. The manifest contains only vault-relative paths.
6. Mutation changes exactly 10 files in a 100-note corpus.
7. Mutation preserves three chunks per changed note.
8. Generated files contain no private path, hostname, IP-address, email-address, credential, or external-provider marker.

Use this deterministic assertion:

~~~python
first = generate_synthetic_vault(first_root, workload, 100)
second = generate_synthetic_vault(second_root, workload, 100)

assert first == second
assert [
    path.relative_to(first_root).as_posix()
    for path in sorted(first_root.rglob("*.md"))
] == list(first.paths)

for relative_path in first.paths:
    assert (first_root / relative_path).read_bytes() == (
        second_root / relative_path
    ).read_bytes()
~~~

- [ ] **Step 3: Run focused tests and verify failure**

~~~powershell
python -m pytest tests/test_benchmark_corpus.py -v
~~~

Expected: collection fails because knowledge_rag.benchmark_corpus does not exist.

- [ ] **Step 4: Implement workload loading and deterministic note rendering**

Define these models:

~~~python
@dataclass(frozen=True, slots=True)
class PerformanceQuery:
    query_id: str
    retrieval_mode: str
    query: str
    note_type: str | None
    topic: str | None


@dataclass(frozen=True, slots=True)
class BenchmarkWorkload:
    version: int
    seed: int
    tiers: tuple[int, ...]
    chunks_per_note: int
    changed_fraction: float
    warmup_iterations: int
    measured_iterations: int
    queries: tuple[PerformanceQuery, ...]


@dataclass(frozen=True, slots=True)
class CorpusManifest:
    version: int
    seed: int
    note_count: int
    chunk_count: int
    paths: tuple[str, ...]
    content_digest: str
~~~

Use SHA-256 rather than Python random state to select stable vocabulary:

~~~python
def _selectors(seed: int, index: int) -> bytes:
    return sha256(f"{seed}:{index}".encode("utf-8")).digest()
~~~

Render paths as synthetic/NNNN/note-NNNNN.md. Cycle note_type through reference, project, and study. Cycle topic through monitoring, identity, operations, and knowledge-management. Render YAML frontmatter with ai_access: allowed followed by exactly three headings. Keep every section below 1,800 characters.

Calculate content_digest by feeding each relative path and file bytes to one SHA-256 object in sorted path order.

- [ ] **Step 5: Implement deterministic 10-percent mutation**

Select files whose zero-based sorted position is divisible by 10. Append a stable revision sentence under the third existing heading. Assert the selected count equals int(manifest.note_count * fraction). Return the selected vault-relative paths.

Reject unsupported fractions, missing files, count mismatches, or any mutation that changes the three-chunk invariant.

- [ ] **Step 6: Run focused and existing chunking tests**

~~~powershell
python -m pytest tests/test_benchmark_corpus.py tests/test_chunking.py tests/test_markdown.py -v
~~~

Expected: all tests pass.

- [ ] **Step 7: Commit Task 2**

~~~bash
git add benchmarks/workload.json src/knowledge_rag/benchmark_corpus.py tests/test_benchmark_corpus.py
git commit -m "feat: generate deterministic benchmark corpora"
~~~

### Task 3: Benchmark database safety and sanitized environment metadata

**Files:**
- Create: src/knowledge_rag/benchmark_database.py
- Create: tests/test_benchmark_database.py
- Modify: src/knowledge_rag/config.py
- Modify: .env.example

**Interfaces:**
- Consumes: settings.database_url and settings.benchmark_database_url.
- Produces: validate_benchmark_target(application_url: str, benchmark_url: str | None, allow_reset: bool) -> None.
- Produces: reset_benchmark_tables(conn: Connection) -> None.
- Produces: count_benchmark_rows(conn: Connection) -> tuple[int, int].
- Produces: collect_environment_metadata(conn: Connection) -> EnvironmentMetadata.

- [ ] **Step 1: Write failing database-target safety tests**

Cover these exact cases:

~~~python
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
    benchmark_url,
    allow_reset,
    message,
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
~~~

Add a success test for knowledge_rag_benchmark on the same host and port as knowledge_rag.

- [ ] **Step 2: Run the focused tests and verify failure**

~~~powershell
python -m pytest tests/test_benchmark_database.py -v
~~~

Expected: collection fails because knowledge_rag.benchmark_database does not exist.

- [ ] **Step 3: Add benchmark configuration**

Add to Settings:

~~~python
benchmark_database_url: str | None = None
~~~

Add to .env.example without a value:

~~~text
BENCHMARK_DATABASE_URL=
~~~

- [ ] **Step 4: Implement safe connection-identity comparison**

Use psycopg.conninfo.conninfo_to_dict. Normalize absent ports to 5432 and absent hosts to localhost. Compare host, port, and dbname only. Never include the parsed dictionary or connection string in an exception.

Require benchmark in the lower-cased dbname and allow_reset is True before returning.

- [ ] **Step 5: Implement reset and row counts**

~~~python
def reset_benchmark_tables(conn: Connection) -> None:
    conn.execute(
        """
        TRUNCATE TABLE document_chunks, documents
        RESTART IDENTITY CASCADE;
        """
    )


def count_benchmark_rows(conn: Connection) -> tuple[int, int]:
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
~~~

- [ ] **Step 6: Implement sanitized environment collection**

Query SHOW server_version and the vector extension version. Use platform.python_version(), platform.system(), os.cpu_count(), and Linux os.sysconf values for memory. Return None when the platform does not expose memory safely.

Tests must monkeypatch all platform and operating-system calls and assert EnvironmentMetadata contains only the six approved fields.

- [ ] **Step 7: Run focused tests and configuration regressions**

~~~powershell
python -m pytest tests/test_benchmark_database.py tests/test_embedding_factory.py -v
~~~

Expected: all tests pass.

- [ ] **Step 8: Commit Task 3**

~~~bash
git add src/knowledge_rag/benchmark_database.py src/knowledge_rag/config.py .env.example tests/test_benchmark_database.py
git commit -m "feat: guard benchmark database operations"
~~~

### Task 4: Measured indexing and embedding stages

**Files:**
- Create: src/knowledge_rag/performance_benchmark.py
- Create: tests/test_performance_benchmark.py

**Interfaces:**
- Consumes: discover_markdown_files, persist_note, reconcile_inactive_documents, load_chunks_missing_embeddings, persist_embeddings, DeterministicEmbeddingProvider, count_benchmark_rows, calculate_rate.
- Produces: run_indexing_stage(conn: Connection, vault_path: Path, scenario: str, chunks_per_note: int | None = None, clock_ns: Callable[[], int] = perf_counter_ns) -> IndexingBenchmarkResult.
- Produces: run_embedding_stage(conn: Connection, provider: EmbeddingProvider, batch_size: int = 100, clock_ns: Callable[[], int] = perf_counter_ns) -> EmbeddingBenchmarkResult.

- [ ] **Step 1: Write a failing indexing-stage unit test**

Use a fake connection whose transaction method returns nullcontext. Monkeypatch discovery, persistence, reconciliation, and row counts.

Prove that:

1. All discovered notes are processed.
2. PersistResult counts are preserved.
3. Reconciliation occurs once after all notes.
4. The timer surrounds discovery through reconciliation.
5. A changed indexing result calculates affected_chunks as indexed times chunks_per_note.
6. No note content or exception message appears in the result.

Use a fake clock with explicit values, for example iter((1_000, 2_001)).__next__, so duration is 1,001 nanoseconds.

- [ ] **Step 2: Run the indexing test and verify failure**

~~~powershell
python -m pytest tests/test_performance_benchmark.py::test_run_indexing_stage_records_counts_and_duration -v
~~~

Expected: FAIL because run_indexing_stage is not defined.

- [ ] **Step 3: Implement run_indexing_stage**

The function must:

1. Start the clock immediately before discovery.
2. Process each note in its own conn.transaction().
3. Count every PersistResult value.
4. Build discovered vault-relative paths with safe_note_reference.
5. Reconcile in a separate transaction.
6. Stop the clock immediately after reconciliation.
7. Query observed document and chunk counts after stopping the timer.
8. For clean indexing, set affected_chunks to the increase in observed chunks.
9. For unchanged indexing, require affected_chunks to be zero.
10. For changed indexing, require chunks_per_note and set affected_chunks to indexed count times chunks_per_note.
11. Calculate document throughput from discovered notes.
12. Calculate chunk throughput from affected_chunks, allowing a zero count with a positive duration.

Reject zero discovered notes and any per-note failure so a partial benchmark cannot appear complete.

- [ ] **Step 4: Write failing embedding-stage tests**

Use a fake provider and monkeypatch load_chunks_missing_embeddings and persist_embeddings.

Cover:

1. Batches of 100, 100, 50, and 0 produce loaded, generated, and persisted counts of 250.
2. Load, generation, persistence, and total durations accumulate independently.
3. The provider receives every chunk exactly once.
4. Persistence receives matching EmbeddedChunk identifiers and vectors.
5. A provider exception escapes and no successful result is returned.
6. A zero-vector workload fails rather than calculating a misleading rate.

- [ ] **Step 5: Implement run_embedding_stage**

For each batch:

1. Time loading eligible chunks.
2. Stop when the loaded batch is empty.
3. Time deterministic provider calls and build EmbeddedChunk objects.
4. Time persist_embeddings in a transaction.
5. Accumulate stage durations and counts.

Measure total duration around the complete loop. Build generation, persistence, and total RateMeasurement values from the final count. Require loaded, generated, and persisted counts to match.

- [ ] **Step 6: Run focused tests**

~~~powershell
python -m pytest tests/test_performance_benchmark.py tests/test_embedding_pipeline.py tests/test_embeddings.py -v
~~~

Expected: all tests pass.

- [ ] **Step 7: Commit Task 4**

~~~bash
git add src/knowledge_rag/performance_benchmark.py tests/test_performance_benchmark.py
git commit -m "feat: measure indexing and embedding throughput"
~~~

### Task 5: Retrieval latency and retrieval-quality phases

**Files:**
- Modify: src/knowledge_rag/performance_benchmark.py
- Modify: tests/test_performance_benchmark.py

**Interfaces:**
- Consumes: BenchmarkWorkload, run_retrieval, run_evaluation, load_thresholds, check_thresholds, summarize_timings.
- Produces: run_retrieval_latency(conn_factory: Callable[[], ContextManager[Connection]], provider: EmbeddingProvider, workload: BenchmarkWorkload, clock_ns: Callable[[], int] = perf_counter_ns) -> tuple[tuple[QueryLatencyResult, ...], tuple[ModeLatencyResult, ...]].
- Produces: run_quality_baseline(conn_factory: Callable[[], ContextManager[Connection]], provider: EmbeddingProvider, dataset_path: Path, thresholds_path: Path, k: int = 5) -> QualityBaselineResult.

- [ ] **Step 1: Write failing retrieval-latency tests**

Create a one-query workload with two warm-ups and three measured iterations. Use a fake connection context manager, monkeypatch run_retrieval, and provide clock values that create samples of 10, 20, and 30 nanoseconds.

Assert:

~~~python
queries, modes = run_retrieval_latency(
    conn_factory=fake_connection_factory,
    provider=object(),
    workload=workload,
    clock_ns=fake_clock,
)

assert retrieval_calls == 5
assert queries[0].timing.samples_ns == (10, 20, 30)
assert queries[0].timing.sample_count == 3
assert modes[0].retrieval_mode == "semantic"
assert modes[0].timing.samples_ns == (10, 20, 30)
~~~

Also assert every timed operation acquires and closes its own connection and consumes the returned result list.

- [ ] **Step 2: Run the retrieval test and verify failure**

~~~powershell
python -m pytest tests/test_performance_benchmark.py::test_run_retrieval_latency_excludes_warmups -v
~~~

Expected: FAIL because run_retrieval_latency is not defined.

- [ ] **Step 3: Implement retrieval latency measurement**

Warm-ups execute the complete connection and retrieval path but do not call the measurement clock. Each measured iteration records one nonnegative sample around connection acquisition, run_retrieval, result materialization, and connection close.

Build one QueryLatencyResult per workload query. Build one ModeLatencyResult per mode by concatenating all query samples for that mode in workload order.

Validate that each query has measured_iterations samples and each mode has three times measured_iterations samples for the approved workload.

- [ ] **Step 4: Write failing quality-baseline tests**

Use a temporary evaluation dataset, deterministic retrieval executor, and threshold file. Cover a passing result and a threshold failure.

Assert QualityBaselineResult.evaluation equals EvaluationRun.to_dict(). Convert each threshold failure to this exact public-safe dictionary:

~~~python
{
    "scope": failure.scope,
    "identifier": failure.identifier,
    "metric": failure.metric,
    "actual": failure.actual,
    "minimum": failure.minimum,
}
~~~

- [ ] **Step 5: Implement run_quality_baseline**

The retrieval executor opens a fresh connection for each evaluation case, calls run_retrieval with the case fields, and converts results with chunk_identity. Run the existing evaluation and threshold functions without changing their types or output.

- [ ] **Step 6: Run focused and evaluation regression tests**

~~~powershell
python -m pytest tests/test_performance_benchmark.py tests/test_evaluation.py tests/test_evaluation_runner.py tests/test_evaluation_thresholds.py -v
~~~

Expected: all tests pass.

- [ ] **Step 7: Commit Task 5**

~~~bash
git add src/knowledge_rag/performance_benchmark.py tests/test_performance_benchmark.py
git commit -m "feat: measure retrieval baseline latency"
~~~

### Task 6: Full benchmark orchestration and protected command-line entry point

**Files:**
- Modify: src/knowledge_rag/performance_benchmark.py
- Create: scripts/run_performance_baseline.py
- Create: tests/test_run_performance_baseline.py
- Modify: .gitignore

**Interfaces:**
- Produces: BenchmarkRunConfig.
- Produces: run_performance_baseline(config: BenchmarkRunConfig) -> BenchmarkReport.
- Produces: write_benchmark_report(report: BenchmarkReport, output_path: Path, baselines_dir: Path) -> None.
- Produces: format_benchmark_summary(report: BenchmarkReport) -> str.
- Produces: make_incomplete_report(workload_version: int, workload: dict[str, Any], stage: str, error_type: str, git_commit: str | None = None) -> BenchmarkReport.
- Produces: scripts.run_performance_baseline.main() -> int.

- [ ] **Step 1: Add BenchmarkRunConfig**

Use this exact immutable configuration:

~~~python
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
~~~

- [ ] **Step 2: Write failing orchestration tests**

Monkeypatch database connections and every stage function. Prove this order:

1. Validate target before connecting.
2. Reset tables.
3. Index and embed sample vault.
4. Run quality evaluation and record its metrics and threshold status without treating threshold failure as a benchmark failure.
5. For each tier, reset tables, generate corpus, run clean indexing, run initial embedding, run unchanged indexing, mutate 10 percent, run changed indexing, run changed embedding, run retrieval timing.
6. Collect sanitized environment metadata.
7. Return a complete BenchmarkReport with tiers ordered 100, 1,000, 10,000.
8. Remove temporary generated vaults.
9. Preserve a supplied work_directory.

Add a test showing a failed quality threshold is recorded and later tiers still run. Add separate tests showing a stage exception stops later tiers and raises a typed BenchmarkStageError whose string contains only the stage name and exception type.

- [ ] **Step 3: Implement the full orchestrator**

Instantiate DeterministicEmbeddingProvider(dimensions=1536) directly.

Use psycopg.connect(config.benchmark_database_url) only after validate_benchmark_target succeeds. Use a connection factory that opens the same validated benchmark URL for quality and retrieval operations.

Read the Git commit with this fixed command from repository_root:

~~~python
subprocess.run(
    ["git", "rev-parse", "HEAD"],
    cwd=repository_root,
    check=True,
    capture_output=True,
    text=True,
).stdout.strip()
~~~

Record datetime.now(timezone.utc).isoformat(). Never serialize the repository path or database URL.

- [ ] **Step 4: Write failing report-output tests**

Cover:

1. Complete output can be written beneath benchmarks/baselines.
2. Incomplete output is rejected beneath benchmarks/baselines.
3. Incomplete output may be written to a path outside the baselines directory.
4. JSON uses indent=2, sort_keys=True, and a trailing newline.
5. Formatting prints tier counts and aggregate metrics without connection data or note content.

- [ ] **Step 5: Implement protected report writing**

Resolve output_path and baselines_dir lexically. If output is within baselines_dir and report.complete is False, raise ValueError("incomplete report cannot replace a canonical baseline").

Write through a sibling temporary file and Path.replace so a partial serialization cannot corrupt an existing complete file.

- [ ] **Step 6: Implement command-line parsing and configuration gates**

The script arguments are:

~~~text
--allow-reset
--workload PATH
--output PATH
--work-directory PATH
--diagnostic-output PATH
~~~

Defaults:

~~~text
workload: benchmarks/workload.json
output: benchmarks/results/latest.json
work-directory: omitted, so temporary
diagnostic-output: omitted
~~~

main() must:

1. Load settings.
2. Build BenchmarkRunConfig.
3. Run the benchmark.
4. Write the complete report.
5. Print format_benchmark_summary.
6. Return 0.

On failure, print only benchmark incomplete, the sanitized stage, and exception type. If diagnostic-output is supplied, call make_incomplete_report with environment and quality set to None, tiers set to an empty tuple, and only the sanitized stage and exception type populated. Write that incomplete report outside the canonical baselines directory. Return 1.

The target validation must occur before psycopg.connect, corpus generation, or output replacement.

- [ ] **Step 7: Ignore disposable results and work directories**

Add:

~~~text
benchmarks/results/
benchmarks/work/
~~~

Do not ignore benchmarks/baselines.

- [ ] **Step 8: Run command-line and focused tests**

~~~powershell
python -m pytest tests/test_run_performance_baseline.py tests/test_performance_benchmark.py -v
~~~

Expected: all tests pass.

- [ ] **Step 9: Commit Task 6**

~~~bash
git add src/knowledge_rag/performance_benchmark.py scripts/run_performance_baseline.py tests/test_run_performance_baseline.py .gitignore
git commit -m "feat: orchestrate performance baseline runs"
~~~

### Task 7: PostgreSQL integration coverage and operator documentation

**Files:**
- Create: tests/test_performance_benchmark_integration.py
- Create: docs/performance-benchmarking.md
- Modify: README.md
- Modify: docs/architecture.md

**Interfaces:**
- Consumes: all benchmark interfaces from Tasks 1 through 6.
- Produces: database-backed proof that the benchmark measures the current application behavior.
- Produces: complete public operator and interpretation documentation.

- [ ] **Step 1: Write the database integration fixture**

Create temporary documents and document_chunks tables with VECTOR(1536), following tests/test_production_ingestion.py. Use a 10-note synthetic corpus, three chunks per note, a 10-percent mutation, one warm-up, and two measured iterations.

Keep the connection open so temporary tables shadow permanent tables. Supply a connection factory that returns nullcontext(conn) without closing it.

- [ ] **Step 2: Write lifecycle integration assertions**

The integration test must prove:

1. Clean indexing reports 10 indexed documents and 30 chunks.
2. Initial embedding reports 30 generated and persisted vectors.
3. The unchanged run reports 10 unchanged documents and zero affected chunks.
4. Exactly one changed note is reported after mutation.
5. Exactly three changed chunks are embedded.
6. Active database counts remain 10 documents and 30 chunks.
7. Semantic, lexical, and hybrid queries each record the configured two samples.
8. No result structure contains note bodies, database URLs, or absolute paths.

- [ ] **Step 3: Run the integration test**

~~~powershell
python -m pytest tests/test_performance_benchmark_integration.py -v
~~~

Expected: PASS in the PostgreSQL and pgvector environment. A connection refusal is an environment failure and must not be converted into a skipped test.

- [ ] **Step 4: Write the operator guide**

docs/performance-benchmarking.md must document:

1. Creating a dedicated database whose name contains benchmark.
2. Applying the existing schema to that database.
3. Setting BENCHMARK_DATABASE_URL privately in .env.
4. Why the application and benchmark database identities must differ.
5. The explicit reset flag and exactly which tables are truncated.
6. Quality-phase versus performance-phase behavior.
7. Corpus tiers, fixed seed, three chunks per note, and 10-percent mutation.
8. Timing boundaries, five warm-ups, 30 samples, median, and nearest-rank p95.
9. Deterministic-provider limitations.
10. A normal disposable run.
11. A reviewed canonical baseline run.
12. Comparing a later refactor against the pre-refactor JSON.
13. Cleanup, failure handling, and diagnostic output.
14. Prohibited private and infrastructure data.

Use these commands:

~~~powershell
python scripts/apply_schema.py
python scripts/run_performance_baseline.py --allow-reset
python scripts/run_performance_baseline.py --allow-reset --output benchmarks/baselines/phase-14-pre-refactor.json
~~~

Explain that apply_schema.py uses DATABASE_URL, so the operator must temporarily point DATABASE_URL at the dedicated benchmark database only while applying schema, then restore the normal value before running the benchmark. Do not print either URL.

- [ ] **Step 5: Update project documentation**

Add the benchmark subsystem to the README capability table and repository layout. Add a Phase 14 subsection with the safe run command and link to docs/performance-benchmarking.md.

Add an architecture section that shows:

1. Existing judged sample corpus flows to quality evaluation.
2. Generated tier corpus flows to indexing, embedding, and retrieval timing.
3. Both flow into the sanitized versioned report.
4. The production vault and production database remain outside the subsystem.

- [ ] **Step 6: Run focused documentation and integration checks**

~~~powershell
python -m pytest tests/test_benchmark_models.py tests/test_benchmark_corpus.py tests/test_benchmark_database.py tests/test_performance_benchmark.py tests/test_run_performance_baseline.py tests/test_performance_benchmark_integration.py -v
python -m ruff check src/knowledge_rag/benchmark_models.py src/knowledge_rag/benchmark_corpus.py src/knowledge_rag/benchmark_database.py src/knowledge_rag/performance_benchmark.py scripts/run_performance_baseline.py tests/test_benchmark_models.py tests/test_benchmark_corpus.py tests/test_benchmark_database.py tests/test_performance_benchmark.py tests/test_run_performance_baseline.py tests/test_performance_benchmark_integration.py
~~~

Expected: all tests and Ruff checks pass.

- [ ] **Step 7: Commit Task 7**

~~~bash
git add tests/test_performance_benchmark_integration.py docs/performance-benchmarking.md README.md docs/architecture.md
git commit -m "docs: document performance baseline workflow"
~~~

### Task 8: Full verification and canonical homelab baseline capture

**Files:**
- Create after successful execution: benchmarks/baselines/phase-14-pre-refactor.json
- Modify only if verification identifies a defect: benchmark source, tests, or documentation from earlier tasks.

**Interfaces:**
- Consumes: the complete benchmark command and dedicated homelab benchmark database.
- Produces: the canonical Phase 14 pre-refactor report required by Issue #69.

- [ ] **Step 1: Run the complete automated test suite**

~~~powershell
python -m pytest
~~~

Expected: all tests pass. The exact collected count will be the current main-branch count plus the new benchmark tests.

- [ ] **Step 2: Run changed-file lint**

~~~powershell
python -m ruff check src/knowledge_rag/benchmark_models.py src/knowledge_rag/benchmark_corpus.py src/knowledge_rag/benchmark_database.py src/knowledge_rag/performance_benchmark.py scripts/run_performance_baseline.py tests/test_benchmark_models.py tests/test_benchmark_corpus.py tests/test_benchmark_database.py tests/test_performance_benchmark.py tests/test_run_performance_baseline.py tests/test_performance_benchmark_integration.py
~~~

Expected: no violations.

- [ ] **Step 3: Confirm the private benchmark configuration without printing it**

~~~powershell
$benchmarkEnvFileConfigured = (Test-Path .env) -and (Select-String -Path .env -Pattern '^BENCHMARK_DATABASE_URL=.+
~~~

Expected: no output.

- [ ] **Step 4: Run a disposable full benchmark**

~~~powershell
python scripts/run_performance_baseline.py --allow-reset
~~~

Expected: exit code 0, complete true, quality metrics and threshold status recorded, and all three tiers report expected note and chunk counts.

- [ ] **Step 5: Run the canonical baseline**

~~~powershell
python scripts/run_performance_baseline.py --allow-reset --output benchmarks/baselines/phase-14-pre-refactor.json
~~~

Expected: exit code 0 and a complete canonical report.

- [ ] **Step 6: Validate report structure and counts**

~~~powershell
python -c "import json, pathlib; p=pathlib.Path('benchmarks/baselines/phase-14-pre-refactor.json'); d=json.loads(p.read_text()); assert d['complete'] is True; assert [x['note_count'] for x in d['tiers']] == [100, 1000, 10000]; assert [x['observed_chunks'] for x in d['tiers']] == [300, 3000, 30000]; assert isinstance(d['quality']['passed'], bool); print('baseline structure valid')"
~~~

Expected: baseline structure valid.

- [ ] **Step 7: Scan the report for prohibited data patterns**

Run searches for URL schemes, local path prefixes, private-address prefixes, and credential-shaped keys:

~~~powershell
rg -n "postgresql://|DATABASE_URL|BENCHMARK_DATABASE_URL|hostname|username|password|api[_-]?key|(^|[^0-9])(10\.[0-9]+\.[0-9]+\.[0-9]+|192\.168\.[0-9]+\.[0-9]+|172\.(1[6-9]|2[0-9]|3[01])\.[0-9]+\.[0-9]+)|[A-Za-z]:\\\\|/home/|/Users/" benchmarks/baselines/phase-14-pre-refactor.json
~~~

Expected: no matches.

- [ ] **Step 8: Review measurement plausibility**

Confirm:

1. Every query has 30 raw samples.
2. Every retrieval mode has 90 combined samples per tier.
3. Median and p95 are nonnegative, and p95 is at least the median.
4. Clean and changed indexing durations are positive.
5. Initial embedding counts equal tier chunk counts.
6. Changed embedding counts equal 10 percent of tier chunk counts.
7. The unchanged run creates no chunks or embeddings.
8. Environment fields contain only the six approved values.

If any invariant fails, add a regression test, make the smallest correction, rerun the focused test, rerun the full suite, and regenerate the baseline.

- [ ] **Step 9: Commit the canonical baseline**

~~~bash
git add benchmarks/baselines/phase-14-pre-refactor.json
git commit -m "perf: record Phase 14 pre-refactor baseline"
~~~

- [ ] **Step 10: Final branch verification**

~~~powershell
git status --short
git diff --check main...HEAD
python -m pytest
~~~

Expected: clean status, no whitespace errors, and all tests pass.
 -Quiet)
if (-not $env:BENCHMARK_DATABASE_URL -and -not $benchmarkEnvFileConfigured) {
    throw "BENCHMARK_DATABASE_URL is not configured"
}
~~~

Expected: no output.

- [ ] **Step 4: Run a disposable full benchmark**

~~~powershell
python scripts/run_performance_baseline.py --allow-reset
~~~

Expected: exit code 0, complete true, quality metrics and threshold status recorded, and all three tiers report expected note and chunk counts.

- [ ] **Step 5: Run the canonical baseline**

~~~powershell
python scripts/run_performance_baseline.py --allow-reset --output benchmarks/baselines/phase-14-pre-refactor.json
~~~

Expected: exit code 0 and a complete canonical report.

- [ ] **Step 6: Validate report structure and counts**

~~~powershell
python -c "import json, pathlib; p=pathlib.Path('benchmarks/baselines/phase-14-pre-refactor.json'); d=json.loads(p.read_text()); assert d['complete'] is True; assert [x['note_count'] for x in d['tiers']] == [100, 1000, 10000]; assert [x['observed_chunks'] for x in d['tiers']] == [300, 3000, 30000]; assert isinstance(d['quality']['passed'], bool); print('baseline structure valid')"
~~~

Expected: baseline structure valid.

- [ ] **Step 7: Scan the report for prohibited data patterns**

Run searches for URL schemes, local path prefixes, private-address prefixes, and credential-shaped keys:

~~~powershell
rg -n "postgresql://|DATABASE_URL|BENCHMARK_DATABASE_URL|hostname|username|password|api[_-]?key|(^|[^0-9])(10\.[0-9]+\.[0-9]+\.[0-9]+|192\.168\.[0-9]+\.[0-9]+|172\.(1[6-9]|2[0-9]|3[01])\.[0-9]+\.[0-9]+)|[A-Za-z]:\\\\|/home/|/Users/" benchmarks/baselines/phase-14-pre-refactor.json
~~~

Expected: no matches.

- [ ] **Step 8: Review measurement plausibility**

Confirm:

1. Every query has 30 raw samples.
2. Every retrieval mode has 90 combined samples per tier.
3. Median and p95 are nonnegative, and p95 is at least the median.
4. Clean and changed indexing durations are positive.
5. Initial embedding counts equal tier chunk counts.
6. Changed embedding counts equal 10 percent of tier chunk counts.
7. The unchanged run creates no chunks or embeddings.
8. Environment fields contain only the six approved values.

If any invariant fails, add a regression test, make the smallest correction, rerun the focused test, rerun the full suite, and regenerate the baseline.

- [ ] **Step 9: Commit the canonical baseline**

~~~bash
git add benchmarks/baselines/phase-14-pre-refactor.json
git commit -m "perf: record Phase 14 pre-refactor baseline"
~~~

- [ ] **Step 10: Final branch verification**

~~~powershell
git status --short
git diff --check main...HEAD
python -m pytest
~~~

Expected: clean status, no whitespace errors, and all tests pass.
