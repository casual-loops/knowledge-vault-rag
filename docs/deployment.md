# Deployment Model

## Active homelab deployment

The RAG backend runs in a dedicated Debian 13 Linux container on the homelab virtualization platform.

Current allocation:

- 2 vCPU
- 2 GB RAM
- 512 MB swap
- 32 GB root disk
- No GPU

The container hosts:

- PostgreSQL 17
- pgvector 0.8.0
- the `knowledge_rag` application database and restricted application role
- Python application code
- ingestion and retrieval components
- grounded answer generation components
- the FastAPI query service

Embedding and generation providers are selected through runtime configuration. Deterministic local providers are available for development and testing. Optional external adapters can be enabled when credentials and billing are available.

## Development workstation

The public repository is cloned to the development workstation. A Python virtual environment is used locally for tests and application development.

Authenticated PostgreSQL connectivity from the workstation to the homelab database has been validated over the private LAN.

The local `.env` contains the private database connection string and any optional external-provider credentials and is excluded from Git. Credentials and private network addressing must never be committed.

## Running the FastAPI service

From an activated Python virtual environment in the repository root:

```powershell
uvicorn knowledge_rag.api.main:app --reload
```

The development server listens on localhost by default.

Useful endpoints:

- `GET /health`
- `POST /query`
- `POST /answer`
- `/docs` for generated OpenAPI documentation

The current health check verifies database connectivity. If the database is unavailable, the endpoint returns HTTP 503 with a sanitized response.

## Query behavior

`POST /query` accepts semantic search text and optional retrieval controls:

- `top_k`, default 5, allowed range 1 through 25
- `note_type`
- `topic`

The API delegates to the existing retrieval layer and returns source-aware chunk metadata.

Invalid requests return HTTP 422. Operational failures in database access, embedding-provider setup, or retrieval return HTTP 503 with generic error details.

## Grounded answer behavior

`POST /answer` accepts the same query controls as `POST /query` and reuses the semantic retrieval layer before invoking grounded generation.

The grounded generation pipeline:

1. converts retrieval results into policy-aware source chunks
2. applies generation privacy policy
3. removes `local-only` chunks before external generation
4. builds bounded context
5. invokes the configured generation provider
6. returns the answer with stable citations and the source chunks actually used

Citation metadata includes:

- citation identifier
- vault-relative source path
- title
- heading path
- chunk index

Returned source records also include source content and effective AI access policy.

If no usable source context remains, the endpoint returns an empty answer with empty citation and source lists and does not invoke the generation provider.

Generation-provider failures return HTTP 503 with a generic response. Error bodies must not expose note content, provider credentials, connection details, or raw exception text.

## Provider configuration

Generation is abstracted behind a provider interface.

The deterministic local provider supports development and automated testing without external credentials. The optional OpenAI adapter is selected only when the required runtime configuration is present.

External provider configuration belongs in local runtime configuration such as `.env` and must not be committed.

## Vault data path

The canonical Obsidian vault remains on the workstation. Syncthing replicates it to ZFS-backed homelab storage with staggered file versioning.

The real vault is intentionally not indexed yet. Development continues to use:

```text
examples/sample-vault/
```

The future production ingestion service should consume the homelab replica and should use read-only access wherever practical.

## Database networking

PostgreSQL is configured to listen on its private LAN interface in addition to localhost. Authentication is controlled through `pg_hba.conf` with SCRAM authentication.

Deployment-specific addresses and credentials are intentionally omitted from this public repository.

Production access rules should be limited to trusted clients and required network ranges.

## Portable development option

`docker-compose.yml` remains in the repository as a portable option for contributors or future isolated development environments. It is not required by the active homelab deployment.

## Operational follow-up

Before production vault indexing, complete these infrastructure tasks:

1. Configure scheduled backup protection for the RAG container and database state.
2. Perform and document a restore validation.
3. Add the RAG container to Checkmk.
4. Discover and monitor host resources, PostgreSQL health, and FastAPI health.
5. Validate alerting behavior.
6. Validate external-provider configuration and privacy behavior before enabling live generation against real vault content.

## Next software milestone

Phase 10 is complete. The next software milestone is Phase 11: hybrid lexical and vector search.
