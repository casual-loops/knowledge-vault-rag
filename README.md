# Knowledge Vault RAG

A public, sanitized implementation of a private-first personal knowledge assistant built around an Obsidian vault.

The private vault remains the authoritative source of truth. This repository contains only the software, architecture, infrastructure documentation, and synthetic sample content required to develop and demonstrate the system safely in public.

## Goals

- Preserve personal knowledge as human-readable Markdown.
- Use Obsidian for authoring, linking, and navigation.
- Synchronize a protected replica of the vault to homelab infrastructure.
- Parse Markdown and structured frontmatter into a rebuildable search index.
- Store embeddings and metadata in PostgreSQL with pgvector.
- Expose retrieval and answer generation through a small FastAPI service.
- Return source references with generated answers.
- Keep private notes, credentials, internal documentation, and identifying infrastructure details out of the public repository.

## High-level architecture

```text
Workstation
├── Obsidian
│   └── Private Markdown vault
├── Private Git repository
└── Python development environment
        |
        +--> Syncthing
        |      |
        |      v
        |   Homelab vault replica
        |
        +--------------------------+
                                   |
                                   v
                         Dedicated RAG container
                         ├── PostgreSQL 17
                         ├── pgvector
                         ├── Markdown ingestion
                         └── FastAPI service
                                   |
                                   v
                         External embedding and LLM APIs
```

## Design principles

1. The Markdown vault is canonical. The vector index is disposable and rebuildable.
2. Private knowledge never belongs in this public repository.
3. Retrieval should be inspectable. Answers should identify their source notes.
4. The first implementation favors understandable components over heavy AI frameworks.
5. Infrastructure should remain lightweight enough for modest homelab hardware.
6. Sanitized sample data should make the public project reproducible without exposing the real vault.
7. Development and production data remain separated. Public development uses only synthetic sample content until ingestion and privacy controls are validated.

## Current stack

| Layer                 | Technology                                                                         | Status                                      |
| --------------------- | ---------------------------------------------------------------------------------- | ------------------------------------------- |
| Knowledge authoring   | Obsidian and Markdown                                                              | Operational                                 |
| Vault version history | Private Git repository                                                             | Operational                                 |
| Vault synchronization | Syncthing                                                                          | Operational                                 |
| Homelab vault storage | ZFS-backed storage                                                                 | Operational                                 |
| Application           | Python                                                                             | Operational development implementation      |
| API                   | FastAPI                                                                            | Retrieval and grounded answer APIs implemented |
| Database              | PostgreSQL 17                                                                      | Operational in homelab                      |
| Vector search         | pgvector 0.8.0                                                                     | Enabled                                     |
| Lexical search        | PostgreSQL full-text search                                                        | Implemented                                 |
| Hybrid ranking        | Reciprocal Rank Fusion                                                             | Implemented                                 |
| Embeddings            | Provider abstraction with deterministic local provider and optional OpenAI adapter | Implemented                                 |
| Generation            | Provider abstraction with deterministic local provider and optional OpenAI adapter | Implemented                                 |
| Production runtime    | Dedicated Debian 13 LXC                                                            | Operational                                 |

## Development model

Application development occurs from a workstation checkout of this public repository. The Python environment connects over the private LAN to the PostgreSQL instance hosted by the dedicated RAG container.

The real private vault is not used for development ingestion yet. Development continues against:

```text
examples/sample-vault/
```

This keeps parser, chunking, database persistence, and privacy behavior testable before any real personal data is indexed.

## Repository layout

```text
.
├── docs/
├── examples/sample-vault/
├── infrastructure/
├── scripts/
├── sql/
├── src/knowledge_rag/
├── tests/
├── .env.example
├── .gitignore
├── docker-compose.yml
└── pyproject.toml
```

Docker Compose remains available as a portable development option, but the active homelab implementation uses PostgreSQL directly on the dedicated Linux container.

## Roadmap

- [x] Phase 0: Private Obsidian vault and initial metadata templates
- [x] Phase 1: Private Git history and synchronized homelab vault replica
- [x] Phase 2: Public repository and sanitized architecture documentation
- [x] Phase 3: Markdown loading, frontmatter parsing, and heading-aware chunking scaffold
- [x] Phase 4: Dedicated Debian RAG container deployed
- [x] Phase 5: PostgreSQL 17 and pgvector 0.8.0 installed and connectivity validated
- [x] Phase 6: Persist parsed sample documents and chunks in PostgreSQL
- [x] Phase 7: Privacy-aware indexing and retrieval policies
- [x] Phase 8: Embedding pipeline and semantic retrieval
- [x] Phase 9: FastAPI query service
- [x] Phase 10: Grounded answer generation with citations
- [x] Phase 11: Hybrid lexical and vector search
- [ ] Phase 12: Retrieval evaluation and regression tests
- [ ] Phase 13: Production vault indexing
- [ ] Phase 14: Web interface and operational hardening

## FastAPI Query Service

The FastAPI service exposes semantic, lexical, and hybrid retrieval together with grounded answer generation through a small JSON API.

### Health Endpoint

```http
GET /health
```

A healthy response:

```json
{
  "status": "ok",
  "database": "ok"
}
```

If a required dependency is unavailable, the service returns HTTP `503` with a generic error response that does not expose connection details or internal exceptions.

### Query Endpoint

```http
POST /query
```

Example hybrid request:

```json
{
  "query": "privacy controls for external AI use",
  "top_k": 5,
  "note_type": "reference",
  "topic": "privacy-demo",
  "retrieval_mode": "hybrid"
}
```

Request fields:

| Field | Required | Description |
| --- | --- | --- |
| `query` | Yes | Retrieval query text. Must not be empty. |
| `top_k` | No | Maximum number of results. Defaults to `5`; allowed range is `1` through `25`. |
| `note_type` | No | Filters results to a specific note type. |
| `topic` | No | Filters results to a specific topic value. |
| `retrieval_mode` | No | Selects `semantic`, `lexical`, or `hybrid`. Defaults to `semantic`. |

Retrieval modes:

- `semantic` uses vector similarity from pgvector.
- `lexical` uses PostgreSQL full-text search over title, heading path, and chunk content.
- `hybrid` combines semantic and lexical rankings with Reciprocal Rank Fusion.

Hybrid search deduplicates the same chunk when it appears in both retrieval lists. Chunk identity is based on source path and chunk index. Ranking is deterministic, including deterministic tie-breaking.

Example response:

```json
{
  "results": [
    {
      "source_path": "50 Privacy/Allowed Note.md",
      "title": "Allowed Note",
      "note_type": "reference",
      "topic": "privacy-demo",
      "ai_access": "allowed",
      "chunk_index": 0,
      "heading_path": "Allowed Note",
      "content": "Synthetic example content.",
      "score": 0.0325,
      "score_type": "hybrid"
    }
  ]
}
```

Returned source metadata includes:

- vault-relative source path
- title
- note type
- topic
- effective AI access policy
- chunk index
- heading path
- chunk content
- retrieval score
- score type

Scores are higher-is-better within each retrieval mode. Semantic, lexical, and hybrid scores have different meanings and should not be compared across modes as if they shared one scale.

Invalid request bodies, including unsupported retrieval modes, return HTTP `422`.

Database, embedding-provider, or retrieval failures return HTTP `503` with a generic service error rather than exposing internal details.

### Grounded Answer Endpoint

```http
POST /answer
```

The grounded answer endpoint performs the selected retrieval mode, applies generation privacy controls, builds bounded context, and returns a generated answer together with citation and source metadata.

Example request:

```json
{
  "query": "privacy controls for external AI use",
  "top_k": 5,
  "note_type": "reference",
  "topic": "privacy-demo",
  "retrieval_mode": "hybrid"
}
```

The request supports the same retrieval controls as `POST /query`.

Citation identifiers are assigned in source order and are independent of the configured generation provider.

Only source chunks actually used for generation are returned. When an external generation provider is selected, `local-only` chunks are removed before prompt construction, citation creation, and provider invocation.

If retrieval produces no usable generation context, the endpoint returns an empty answer with empty citation and source lists.

Invalid request bodies return HTTP `422`.

Retrieval, embedding-provider, generation-provider, database, or generation failures return HTTP `503` with a generic service error that does not expose note content, credentials, connection information, or internal exception text.

## Infrastructure follow-up

The RAG container is operational, but two infrastructure controls remain open before the service is considered production-ready:

- Configure and validate backups for the RAG container and database state.
- Add the RAG container to Checkmk monitoring and validate service discovery and alerting.

## Privacy boundary

The real vault is intentionally excluded from this project. Never commit real journal entries, private bookmarks, employer documentation, credentials, internal hostnames, IP addresses, API keys, tokens, or private source paths.

Use `examples/sample-vault/` for synthetic or deliberately sanitized demonstration content.

The production system is expected to add explicit policy controls so notes can be excluded from indexing or restricted from being sent to external model APIs.

## Status

The infrastructure foundation, hybrid retrieval pipeline, and grounded answer generation pipeline are operational.

The system can parse synthetic Markdown notes, enforce privacy-aware indexing and generation rules, persist documents and chunks in PostgreSQL, generate deterministic development embeddings, store vectors in pgvector, perform semantic and lexical retrieval with metadata filtering, combine rankings with Reciprocal Rank Fusion, and expose retrieval and grounded answer generation through FastAPI.

Semantic, lexical, and hybrid retrieval remain independently selectable. Hybrid retrieval preserves source metadata and privacy state while deduplicating overlapping chunks.

Grounded generation uses a provider abstraction with a deterministic local implementation for development and testing and an optional OpenAI adapter for external generation. Automated tests do not require live external API credentials.

Generated answers return stable citation identifiers together with vault-relative source path, title, heading path, chunk index, source content, and effective AI access policy for the chunks actually used.

External generation applies privacy filtering before prompt construction so `local-only` content is not transmitted to an external provider or exposed through generated-answer citations.

The next software milestone is Phase 12: retrieval evaluation and regression tests.
