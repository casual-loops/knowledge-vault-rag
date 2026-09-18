# Production Vault Indexing Runbook

This runbook documents the operator workflow for safely enabling and maintaining indexing of a private production vault replica.

The Markdown vault remains the source of truth. PostgreSQL, chunks, embeddings, and retrieval indexes are derived state and can be rebuilt.

## Safety principles

Production indexing follows these rules:

1. The production vault path is supplied at runtime and is never committed to this public repository.
2. Production ingestion requires explicit opt-in.
3. Path exclusions, note-type exclusions, and `ai_access` policy are evaluated before persistence or external provider use.
4. Preflight runs before any database connection or provider initialization.
5. Dry-run performs read-only database inspection and never persists changes or invokes an embedding provider.
6. Real indexing uses the same policy-aware ingestion path validated by automated tests.
7. Operator output contains aggregate counts, safe vault-relative references, and error types only. It must not contain note bodies, sensitive frontmatter, credentials, private absolute paths, or provider payloads.

## Prerequisites

Before enabling production indexing, verify:

* The private vault replica exists on the host where ingestion will run.
* The replica is synchronized and readable by the ingestion process.
* PostgreSQL and pgvector are available and the current schema migrations have been applied.
* The application environment can connect to the intended database.
* Excluded paths and excluded note types are reviewed.
* Notes that may use external providers explicitly declare `ai_access: allowed`.
* Notes that must remain local declare `ai_access: local-only`.
* Notes that must not enter the index declare `ai_access: exclude`, or are covered by an exclusion rule.
* Backups or other recovery controls exist for database state before the first real production run.

Do not place a real production vault path, hostname, credential, token, or private identifier in committed configuration or documentation.

## Production configuration

The production runner reads two dedicated environment variables:

```text
KNOWLEDGE_RAG_PRODUCTION_VAULT_PATH=
KNOWLEDGE_RAG_ENABLE_PRODUCTION_INGESTION=false
```

`KNOWLEDGE_RAG_PRODUCTION_VAULT_PATH` must contain an absolute path to the private vault replica.

`KNOWLEDGE_RAG_ENABLE_PRODUCTION_INGESTION` must be set to `true` before the production runner will accept the configuration.

The public `.env.example` intentionally leaves the real path blank and production ingestion disabled.

The normal privacy configuration is also reused:

```text
EXCLUDED_PATHS=[]
EXCLUDED_NOTE_TYPES=[]
```

Excluded paths must be safe vault-relative paths. Note-type exclusions must contain usable non-empty values.

## ai_access policy

Production privacy behavior is fail-conservative.

Supported values are:

| Value | Local indexing | External embeddings | External generation |
| --- | --- | --- | --- |
| `allowed` | Yes | Yes | Yes |
| `local-only` | Yes | No | No |
| `exclude` | No | No | No |

Missing, malformed, or unknown `ai_access` values resolve to `local-only`.

This fallback permits local indexing but does not permit external provider use.

## Step 1: Run privacy preflight

Run preflight before the first production indexing run and after any meaningful privacy configuration change.

```powershell
python scripts/ingest_production_vault.py --preflight
```

Preflight validates the production configuration and inspects the vault without opening a database connection or initializing an embedding provider.

The report includes:

* total discovered notes
* notes excluded by configured path
* notes excluded by configured note type
* explicit `allowed` declarations
* explicit `local-only` declarations
* explicit `exclude` declarations
* notes using the conservative fallback
* sanitized validation failures

A successful preflight reports:

```text
mode: preflight
passed: true
```

If preflight reports `passed: false`, do not proceed to dry-run or real indexing until the failure is understood and corrected.

## Step 2: Run a dry-run audit

After preflight passes, run:

```powershell
python scripts/ingest_production_vault.py --dry-run
```

Dry-run inspects the current production vault and compares eligible notes with existing database state.

It may execute read-only queries. It does not insert, update, or delete database records, and it does not initialize or call an embedding provider.

The audit classifies notes as:

| Result | Meaning |
| --- | --- |
| `would-index` | A new eligible note would be added |
| `would-update` | Existing indexed state differs, or an inactive note would be reactivated |
| `unchanged` | Existing active content matches the current note |
| `excluded` | Current policy prevents indexing |

The audit also reports whether eligible content is externally permitted or local-only.

Review unexpected update counts, exclusion counts, or external eligibility before continuing.

## Step 3: First production indexing run

Only proceed when preflight passes and the dry-run output is understood.

Run:

```powershell
python scripts/ingest_production_vault.py
```

The production workflow:

1. Discovers Markdown notes under the configured vault replica.
2. Applies path exclusions.
3. Parses Markdown and frontmatter.
4. Applies note-type exclusions.
5. Resolves `ai_access`.
6. Persists eligible new or changed documents and chunks.
7. Skips unchanged content.
8. Reactivates previously inactive documents when the same source returns.
9. Reconciles active database documents that are no longer present in the completed vault scan.
10. Generates missing embeddings only for active documents whose effective `ai_access` is `allowed`.

Each note is handled transactionally. A note-specific failure is reported using a sanitized reference and error type rather than private content.

Reconciliation only completes when the vault scan finishes without note-specific failures.

## Step 4: Validate the first run

Review the operator summary.

Expected fields include:

```text
complete
discovered
indexed
unchanged
excluded
reactivated
inactivated
embedded
reconciliation_completed
```

A successful run reports `complete: true`.

If `complete: false`, inspect the sanitized failure type, correct the underlying condition, and rerun. Do not copy private note content into public issues or logs while troubleshooting.

After the first successful run, validate retrieval with known private test queries through the normal retrieval interface. Private test queries and expected results must remain outside this public repository.

## Incremental indexing

Subsequent production runs use the same command:

```powershell
python scripts/ingest_production_vault.py
```

Incremental behavior is based on stable vault-relative source identity and a content hash that includes note content and metadata.

Expected behavior:

* unchanged active notes are skipped
* changed notes are updated
* newly discovered notes are indexed
* previously inactive unchanged notes can be reactivated without unnecessary re-embedding
* removed or renamed source paths are preserved as inactive records after reconciliation
* new or changed externally eligible chunks receive embeddings
* local-only chunks remain without external embeddings

Run preflight again after changing exclusion settings, privacy conventions, or production configuration.

Use dry-run before a real run when a large vault reorganization or broad policy change is expected.

## Rollback and recovery

The Markdown vault is authoritative. The database index is derived state.

### Failed note processing

A failed note does not require deleting the source note. Correct the parsing, policy, filesystem, or database problem and rerun ingestion.

### Failed embedding

Persisted document and chunk state can remain valid when embedding fails. Correct the provider or connectivity problem and rerun. Missing eligible embeddings are retried.

### Accidental exclusion or policy change

Correct the configuration or note metadata, run preflight, review dry-run, then run normal ingestion again.

### Removed or renamed note

The old source path is retained as inactive rather than destructively deleted. A renamed note is indexed under its new vault-relative path.

### Database loss or corruption

Restore from a validated database backup when appropriate. Because the index is derived from the Markdown vault, the index can also be rebuilt from the private vault replica after schema initialization and privacy validation.

A rebuild must still follow the normal sequence:

```text
preflight
dry-run
production indexing
retrieval validation
```

## Logging and privacy expectations

Production output must remain operationally useful without exposing source content.

Safe output includes:

* aggregate counters
* vault-relative note references
* policy classification counts
* generic exception type names
* completion state

Do not log:

* note bodies
* sensitive frontmatter values
* full provider prompts
* external provider payloads
* absolute private vault paths
* database credentials
* API keys or tokens
* private hostnames or network identifiers

Public bug reports and repository documentation must use synthetic replacements for any production-specific information.

## Operational cadence

A normal maintenance workflow is:

```text
configuration or policy change
        |
        v
preflight
        |
        v
dry-run when impact is material or uncertain
        |
        v
production ingestion
        |
        v
review sanitized summary
        |
        v
validate retrieval when appropriate
```

For routine incremental runs with unchanged privacy configuration, the operator may run normal production ingestion directly. Preflight remains recommended after policy, exclusion, deployment, or configuration changes.

## Public repository boundary

This runbook documents the procedure without publishing environment-specific production information.

Keep the following outside the public repository:

* real vault paths
* private note names when identifying
* hostnames
* IP addresses
* credentials
* tokens
* private evaluation cases
* production database exports
* logs containing private content

Use sanitized examples when documenting failures or demonstrating the workflow.
