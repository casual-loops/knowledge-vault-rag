# Performance benchmarking

The Phase 14 benchmark captures a reproducible, provider-free baseline for the
current indexing, embedding, and retrieval paths. It is an operator workflow,
not a production maintenance command: use a dedicated disposable PostgreSQL
database and never point it at the application database.

## Prerequisites and safety boundary

- Python dependencies for this repository, PostgreSQL, and the pgvector extension.
- A dedicated database whose name contains `benchmark` (for example,
  `knowledge_rag_benchmark`), created by an authorized database administrator.
- A private `.env` file containing a non-empty `BENCHMARK_DATABASE_URL`. Do not
  commit it; `.env.example` intentionally leaves this variable empty.
- The normal `DATABASE_URL`, which identifies the application database and must
  differ from the benchmark database's normalized `(host, port, database name)`
  identity. A different database name on the same host and port is sufficient;
  different credentials for the same database are not.

Create the dedicated database and enable any site-required privileges/extensions
using the normal administrative process. Do not reuse the production or normal
development database: each benchmark starts by truncating `document_chunks` and
`documents` with `TRUNCATE ... RESTART IDENTITY CASCADE`. The cascade also truncates
any tables with foreign-key dependencies on them, so never attach unrelated data
to this disposable schema. It does not drop the database, extensions, or schema.
The runner refuses to reset unless the target name contains
`benchmark`, the benchmark and application identities differ, and `--allow-reset`
is present.

Apply the existing schema to the dedicated database before its first benchmark.
`apply_schema.py` reads `DATABASE_URL`, not `BENCHMARK_DATABASE_URL`. Temporarily
set `DATABASE_URL` privately to the dedicated benchmark database only for schema
application, run the command, then restore the normal application value before
running the benchmark. Never print either URL.

```powershell
python scripts/apply_schema.py
```

## What the benchmark measures

The quality phase indexes the existing judged six-note sample corpus, embeds it,
runs the retrieval evaluation cases, and records precision at K, recall at K,
reciprocal rank, rankings, summaries, and quality-threshold status. This remains
separate from performance measurements so generated distractor notes never alter
the judged corpus.

The performance phase generates public-safe, deterministic corpus tiers of 100,
1,000, and 10,000 notes. The fixed workload seed `14069` produces three chunks per note.
After clean indexing and initial embedding, an unchanged rerun is measured, then
exactly 10 percent of notes are deterministically changed without altering each
note's three-chunk shape. The changed chunks are embedded in a separate stage.

Embedding always uses the local `DeterministicEmbeddingProvider` with 1,536
dimensions. It does not use the configured provider factory, hosted-provider
credentials, or paid APIs. The reported embedding and retrieval timings therefore
measure local application/database work, not hosted-provider network latency,
rate limits, service-side batching, or cost. Do not treat them as hosted-provider
performance estimates.

Each tier records one clean, unchanged, and changed indexing operation, not 30
repetitions of ingestion. Indexing timing includes file discovery, parsing,
chunking, persistence, and inactive-document reconciliation; observed row-count
queries run afterward. Its document rate counts all discovered notes, while the
chunk rate counts affected chunks (zero for unchanged indexing). Embedding timing
separately records missing-chunk loading, deterministic vector generation, and
vector persistence, plus the total loop including its final empty load. Neither
stage includes corpus generation/mutation or database reset. Transaction and
connection boundaries follow the current runner, not an end-to-end import service.

For each sanitized semantic, lexical, and hybrid query at every tier, the runner
performs five unrecorded warm-ups and 30 measured iterations. Warm-ups reduce
first-use and cache noise; they are not a cold-cache benchmark. A measured
retrieval iteration includes acquiring a connection, executing the selected
retrieval path, consuming results, and closing the connection. Semantic and
hybrid measurements include deterministic query-vector generation; hybrid also
includes lexical retrieval and fusion. Raw samples are nanoseconds, while median
and nearest-rank p95 are reported in milliseconds.
The approved workload has three queries per mode: 30 samples per query and 90
pooled samples per mode. Nearest-rank p95 is sorted sample `ceil(0.95 * N)`
(one-based). Warm-ups apply only to retrieval, not indexing or embedding.

There are no hard latency or throughput performance gates. Quality threshold
status is recorded for comparison, but a threshold failure is not a completion
gate for this provider-free benchmark: the deterministic hash-based embedding is
not a semantic-quality provider. Review status and metrics before using a result
as a baseline.

## Run and outputs

For a normal disposable run, with the private benchmark URL configured and the
normal `DATABASE_URL` restored, run:

```powershell
python scripts/run_performance_baseline.py --allow-reset
```

This writes a disposable, ignored report at `benchmarks/results/latest.json` and
prints a compact sanitized summary. A complete report includes the workload and
schema versions, commit, UTC timestamp, approved environment metadata, quality
status, per-tier counts, indexing and embedding rates, retrieval samples, and
median/p95 summaries. It excludes note bodies, database identities and URLs,
absolute/private paths, arbitrary environment values, and exception details.
Quality rankings retain public sample-vault-relative chunk identifiers.

After reviewing a successful disposable run, capture the canonical pre-refactor
baseline with the exact same environment and workload:

```powershell
python scripts/run_performance_baseline.py --allow-reset --output benchmarks/baselines/phase-14-pre-refactor.json
```

Only a complete report can be written to the canonical baseline directory. Review
the JSON for expected tiers and counts, quality status, sanitized fields, and any
unusual timing before committing it. This task does not create that canonical
baseline; it is created only after a reviewed successful run.

## Interpretation and comparison

Keep the pre-refactor JSON and its environment metadata with the reviewed change.
After a refactor, rerun the same workload and compare it with that JSON by tier
and retrieval mode: median and p95 latency, clean/unchanged/changed indexing
throughput, deterministic vector generation and persistence throughput, operation
counts, quality metrics, and recorded threshold status. Interpret changes with
the raw samples and environment metadata; shared-host scheduling and PostgreSQL
cache state naturally vary between runs. Do not claim a performance regression
or improvement from a single noisy value.

The performance queries exercise stable code paths and filters; they are not
relevance judgments. The judged quality corpus is intentionally small, and the
deterministic provider does not model hosted semantic behavior. The benchmark also
does not model production vault size, private content distribution, live traffic,
or hosted embedding/generation latency.

## Failure handling and cleanup

The runner stops at the first failed stage, does not continue later tiers, returns
a nonzero status, and may write an explicitly requested diagnostic report outside
the canonical baseline directory via `--diagnostic-output benchmarks/results/failure.json`.
Its failure details contain only a sanitized stage and exception type, alongside
the versioned workload and report envelope; partial results are not a baseline.
Check the private configuration, dedicated database,
schema, pgvector installation, and available database capacity without copying
connection strings into logs or issue reports. Generated synthetic work directories
are cleaned up automatically after successful or failed runs (unless cleanup itself
fails). The database can retain partial synthetic state after a failure; fix the
cause and rerun with reset authorization. Existing report files are not proof of
the latest run's success: check the exit status and report timestamp/commit.

When the benchmark is no longer needed, have an authorized operator remove the
dedicated disposable benchmark database using the local database administration
procedure. Never use the benchmark reset command as a substitute for deleting a
database.

Do not place private vault content, real note bodies, credentials, tokens, API
keys, usernames, hostnames, IP addresses, database URLs, filesystem paths, or
infrastructure diagnostics in workload files, reports, commits, screenshots, or
support requests.
