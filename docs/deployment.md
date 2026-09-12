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
- the FastAPI query service

Embedding and language-model inference remain external services during the initial implementation.

## Development workstation

The public repository is cloned to the development workstation. A Python virtual environment is used locally for tests and application development.

Authenticated PostgreSQL connectivity from the workstation to the homelab database has been validated over the private LAN.

The local `.env` contains the private database connection string and is excluded from Git. Credentials and private network addressing must never be committed.

## Running the FastAPI service

From an activated Python virtual environment in the repository root:

```powershell
uvicorn knowledge_rag.api.main:app --reload
```

The development server listens on localhost by default.

Useful endpoints:

- `GET /health`
- `POST /query`
- `/docs` for generated OpenAPI documentation

The current health check verifies database connectivity. If the database is unavailable, the endpoint returns HTTP 503 with a sanitized response.

## Query behavior

`POST /query` accepts semantic search text and optional retrieval controls:

- `top_k`, default 5, allowed range 1 through 25
- `note_type`
- `topic`

The API delegates to the existing retrieval layer and returns source-aware chunk metadata.

Invalid requests return HTTP 422. Operational failures in database access, embedding-provider setup, or retrieval return HTTP 503 without exposing sensitive internal details.

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

## Next software milestone

Phase 9 is complete. The next software milestone is grounded answer generation with citations.
