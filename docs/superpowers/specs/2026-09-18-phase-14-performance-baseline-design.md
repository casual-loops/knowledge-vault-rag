# Phase 14 Performance Baseline Design

## Context

Issue #69 establishes a reproducible pre-refactor baseline for retrieval quality and application performance. Later Phase 14 changes will use this baseline to determine whether they improve latency and throughput without reducing retrieval quality.

The current repository already has a deterministic retrieval-quality evaluation framework. Its six-note sample vault has explicit relevance judgments, but it is too small to expose meaningful database, ingestion, or retrieval scaling behavior. The performance baseline therefore needs a separate synthetic workload while preserving the existing quality evaluation unchanged.

## Goals

1. Capture the current retrieval-quality results from the existing evaluation suite.
2. Measure semantic, lexical, and hybrid retrieval latency at three controlled corpus sizes.
3. Measure initial indexing, unchanged indexing, changed-note indexing, embedding generation, and embedding persistence throughput.
4. Make the workload reproducible without committing thousands of generated notes.
5. Retain a sanitized, machine-readable pre-refactor result in the repository.
6. Avoid private vault content, identifying infrastructure data, and paid external providers.
7. Provide a stable comparison contract for the remaining Phase 14 refactors.

## Non-goals

1. This issue will not add vector or full-text indexes.
2. This issue will not batch embedding requests or database writes.
3. This issue will not add connection pooling or provider reuse.
4. This issue will not establish hard performance regression thresholds.
5. This issue will not estimate hosted embedding-provider latency or cost.
6. This issue will not use the private production vault or its derived database.
7. This issue will not treat the scaled synthetic corpus as a relevance-quality dataset.

## Approved decisions

1. Benchmarks run against a dedicated synthetic database on the homelab PostgreSQL host.
2. Corpus tiers contain 100, 1,000, and 10,000 notes.
3. The corpus is generated on demand from a fixed seed.
4. Retrieval timing uses warm-up iterations followed by repeated measured iterations.
5. Retrieval reports median and p95 latency.
6. Embedding measurements use the deterministic 1,536-dimension provider.
7. Vector generation and database persistence are measured separately.
8. Performance results are recorded for comparison but do not gate automated tests.
9. Indexing covers a clean run, an unchanged rerun, and a deterministic 10-percent-change rerun.
10. Quality and performance remain separate benchmark phases.

## Architecture

The benchmark subsystem is separate from production operations and from the existing retrieval-quality implementation.

The proposed components are:

1. src/knowledge_rag/benchmark_corpus.py owns deterministic corpus generation, manifest creation, and controlled note modification.
2. src/knowledge_rag/benchmark_models.py owns immutable result models, timing distributions, throughput calculations, and JSON serialization.
3. src/knowledge_rag/performance_benchmark.py owns database preparation, workload execution, stage timing, and aggregate report construction.
4. scripts/run_performance_baseline.py owns argument parsing, safety validation, dependency creation, result writing, summary output, and process exit status.
5. benchmarks/workload.json stores the versioned workload definition, fixed seed, tiers, query workload, warm-up count, and repetition count.
6. benchmarks/baselines/phase-14-pre-refactor.json stores the reviewed pre-refactor result.
7. docs/performance-benchmarking.md documents prerequisites, execution, interpretation, comparison, safety, and limitations.

The benchmark runner calls existing discovery, parsing, persistence, embedding-store, retrieval, evaluation, and threshold functions. It does not create alternative implementations of those behaviors.

## Quality baseline phase

The quality phase uses the existing six-note sample vault and evaluation/retrieval_cases.json.

The runner will:

1. Reset the validated benchmark database tables.
2. Index and embed the sample vault with the deterministic 1,536-dimension provider.
3. Run the existing retrieval evaluation cases.
4. Check the existing thresholds.
5. retain precision at K, recall at K, reciprocal rank, aggregate mode summaries, and ranked results in their existing structure.
6. Record threshold status in the combined baseline report.

The quality dataset remains small and judged. Generated performance notes are never mixed into this phase because the current deterministic embedding provider is hash-based rather than semantically meaningful. Mixing thousands of generated distractors into the judged corpus would make semantic-quality results arbitrary.

A quality-threshold failure makes the benchmark incomplete.

## Synthetic performance corpus

The generator creates independent vaults for the 100-note, 1,000-note, and 10,000-note tiers.

Each note has:

1. A stable, synthetic relative path.
2. Deterministic frontmatter.
3. ai_access set to allowed.
4. A note type and topic selected from controlled public-safe vocabularies.
5. Three Markdown sections whose contents stay below the current chunk-size limit.
6. Exactly three resulting chunks.
7. No private names, infrastructure identifiers, credentials, or production content.

The expected tier sizes are therefore:

| Notes | Chunks |
| ---: | ---: |
| 100 | 300 |
| 1,000 | 3,000 |
| 10,000 | 30,000 |

The manifest records the workload version, seed, tier, expected file count, expected chunk count, metadata distribution, and content digest. Repeating generation with the same inputs must produce byte-identical notes and an identical manifest.

For the changed-note scenario, the generator selects exactly 10 percent of notes through a stable rule derived from note identity. It modifies content without changing the number of chunks. The expected changed counts are 10, 100, and 1,000 notes.

Generated vaults use a temporary directory and are deleted after a successful or failed run unless a troubleshooting option explicitly preserves them. Absolute temporary paths are never stored in the result.

## Performance indexing scenarios

Each tier begins with initialized application tables containing no documents or chunks. Schema setup and corpus generation are outside measured indexing time.

### Clean indexing

The timer includes Markdown discovery, policy evaluation, parsing, chunking, persistence, per-note transactions, and reconciliation. Embedding is excluded and measured separately.

The report records discovered notes, indexed notes, chunks created, duration, documents per second, and chunks per second.

### Unchanged rerun

The same unmodified vault is processed again. The report records unchanged notes, unexpected writes, duration, and documents checked per second. The expected number of new chunks and embeddings is zero.

### Changed-note rerun

Exactly 10 percent of the notes are modified and processed again. The report records changed notes, unchanged notes, chunks rebuilt, duration, documents checked per second, and changed documents per second.

The changed chunks are embedded in a separate measured embedding stage.

## Embedding measurements

The benchmark instantiates DeterministicEmbeddingProvider directly with 1,536 dimensions. It does not call the configured provider factory, and any configured external API key is ignored.

For each embedding workload, the runner measures:

1. Chunk loading duration.
2. Deterministic vector-generation duration.
3. Embedding persistence duration.
4. Total embedding-stage duration.
5. Embeddings generated and persisted.
6. Vectors generated per second.
7. Embeddings persisted per second.
8. End-to-end embeddings completed per second.

The benchmark uses the current single-item provider interface and current persistence behavior. Later batching and bulk-write refactors will run the same workload and report changes against this baseline.

These measurements represent application and database throughput using a local deterministic provider. They do not estimate hosted-service network latency, rate limits, service-side batching, or API cost.

## Retrieval performance workload

The performance query workload contains nine sanitized queries:

1. Three semantic queries.
2. Three lexical queries.
3. Three hybrid queries.

Within each mode, the workload includes one unfiltered query, one metadata-filtered query, and one no-match query.

For every query at every tier, the runner performs five unrecorded warm-up iterations followed by 30 measured iterations. Warm-up is intended to reduce first-use and database-cache noise. It is not presented as a controlled cold-cache benchmark.

Each measured retrieval operation follows the current request-like boundary by acquiring a database connection, running the configured retrieval mode, consuming the result set, and closing the connection. Semantic and hybrid timings include deterministic query-vector generation. Hybrid timings include both semantic and lexical retrieval plus fusion.

Raw latency samples are stored as integer nanoseconds. The report derives median and p95 latency in milliseconds for each query and retrieval mode. Mode summaries retain the sample count so incomplete measurements cannot appear valid.

The benchmark does not claim that the nine performance queries measure relevance. They exist only to exercise stable retrieval paths and filters at increasing corpus sizes.

## Result contract

The combined report uses an explicit schema version and workload version.

Top-level fields include:

1. Completion status.
2. Benchmark schema version.
3. Workload version.
4. Git commit identifier.
5. Execution timestamp in UTC.
6. Sanitized environment metadata.
7. Workload configuration.
8. Quality baseline results and threshold status.
9. Per-tier performance results.
10. Sanitized failure stage and exception type when incomplete.

Sanitized environment metadata includes:

1. Python version.
2. Operating-system family.
3. Logical CPU count.
4. Total memory size.
5. PostgreSQL version.
6. pgvector version.

The report excludes hostnames, usernames, IP addresses, ports, database names, connection strings, filesystem paths, environment variables, and note bodies.

Each tier result contains:

1. Expected and observed document and chunk counts.
2. Clean, unchanged, and changed-note indexing results.
3. Initial and changed-note embedding results.
4. Per-query raw latency samples.
5. Per-query median and p95 latency.
6. Per-mode median and p95 latency.
7. Operation counts and throughput values.

JSON output uses stable key ordering and a documented representation. Measured values are expected to vary, but the schema, field meaning, workload, and calculations remain stable.

## Database safety

The runner requires BENCHMARK_DATABASE_URL and an explicit reset flag before making destructive database changes.

Before reset, it must verify:

1. The benchmark URL is explicitly configured.
2. The normal application database URL and benchmark URL do not resolve to the same host, port, and database identity.
3. The target database name contains benchmark.
4. The explicit reset flag is present.
5. The runner will use DeterministicEmbeddingProvider rather than a hosted provider.

Failure of any check stops execution before a database connection performs a reset.

A reset truncates only the application benchmark tables and restarts their identities. The runner does not drop the database, extensions, schemas, or unrelated objects.

## Failure handling

A stage failure stops the current run. Later tiers do not continue because the report would no longer represent one comparable benchmark execution.

An incomplete run cannot replace the committed baseline path. The runner may write an incomplete diagnostic report to an explicitly supplied temporary output. Diagnostic output contains the failed stage and exception type, not exception text that could expose connection or path details.

Temporary generated vaults are cleaned up unless preservation was explicitly requested for troubleshooting.

The command returns a nonzero exit status when:

1. Safety validation fails.
2. Corpus generation or manifest validation fails.
3. Database preparation fails.
4. Quality thresholds fail.
5. Any indexing, embedding, retrieval, calculation, or serialization stage fails.
6. Expected operation or sample counts do not match observed counts.

## Baseline retention and comparison

The initial successful homelab run will be reviewed before its sanitized JSON is committed as benchmarks/baselines/phase-14-pre-refactor.json.

Issue #69 will not add hard latency or throughput thresholds. Shared-host activity, operating-system scheduling, and PostgreSQL cache state can cause natural timing variation.

Later Phase 14 refactors will run the same workload and compare:

1. Retrieval quality, which must continue to satisfy existing thresholds.
2. Median and p95 latency by mode and tier.
3. Initial, unchanged, and changed-note indexing throughput.
4. Vector-generation, persistence, and total embedding throughput.
5. Expected and observed operation counts.

Performance differences will be interpreted with their environment metadata and raw samples. A future issue may establish regression tolerances after enough repeated evidence exists.

## Verification

### Unit tests

Unit coverage will prove:

1. Corpus generation is byte-identical for identical inputs.
2. Different seeds or tiers produce intentionally different manifests.
3. Each note produces exactly three chunks.
4. Tier counts and metadata distributions match the workload definition.
5. The modification pass changes exactly 10 percent of notes and preserves chunk counts.
6. Database-target validation rejects missing, matching, or ambiguously named targets.
7. Median, p95, duration, and throughput calculations are correct.
8. Empty samples and zero-duration inputs fail safely.
9. Report serialization is stable and omits prohibited fields.
10. Fake clocks make timing tests deterministic.
11. Query workloads contain the expected modes, filter variants, and counts.
12. Incomplete runs cannot overwrite the canonical baseline target.

### PostgreSQL and pgvector integration tests

Integration coverage will use deliberately small test tiers and prove:

1. Clean indexing creates the expected documents and chunks.
2. The unchanged rerun creates no new chunks or embeddings.
3. The changed-note rerun updates only the selected documents.
4. Initial and changed-note embedding stages persist the expected vectors.
5. Semantic, lexical, and hybrid measurements produce the expected sample counts.
6. Database reset affects only the benchmark application tables.
7. Quality evaluation and threshold checking remain functional.

### Manual baseline verification

The canonical homelab run will verify:

1. All three approved tiers complete.
2. Expected and observed counts agree.
3. Quality thresholds pass.
4. No hosted provider is contacted.
5. No private content or infrastructure identifiers appear in the result.
6. The full test suite and changed-file lint checks pass.
7. The resulting JSON can be parsed and summarized again without rerunning the benchmark.

## Documentation

docs/performance-benchmarking.md will explain:

1. Benchmark database prerequisites.
2. Safe database creation and configuration.
3. Workload generation and cleanup.
4. The exact timing boundaries.
5. The quality and performance separation.
6. How to run and interpret a baseline.
7. How to compare a later refactor.
8. Known limitations and sources of timing variability.
9. Privacy and sanitization guarantees.

## Success criteria

Issue #69 is complete when:

1. The existing retrieval-quality baseline is captured.
2. All three retrieval modes have documented median and p95 latency at all three tiers.
3. Clean, unchanged, and 10-percent-change indexing throughput is captured.
4. Vector generation, persistence, and total embedding throughput are captured.
5. The environment and workload are sufficiently documented for comparison.
6. The successful report contains no private content or identifying infrastructure details.
7. The benchmark requires no paid or hosted provider.
8. Automated unit and database integration coverage passes.
9. The canonical Phase 14 pre-refactor result is committed.
