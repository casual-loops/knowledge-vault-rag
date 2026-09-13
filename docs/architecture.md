# Architecture

## Purpose

Knowledge Vault RAG is designed as a private-first personal knowledge system. Obsidian and Markdown remain the authoritative knowledge layer. The RAG components index a synchronized replica and can be rebuilt without changing the source vault.

## Logical flow

```text
Workstation
  Obsidian
    |
    v
Private Knowledge Vault
    |
    +--> private Git history
    |
    +--> Syncthing
            |
            v
Homelab replica
    |
    v
Markdown loader
    |
    v
Frontmatter parser
    |
    v
Chunker
    |
    v
Embedding provider
    |
    v
PostgreSQL + pgvector
    |
    v
Semantic retriever
    |
    +--> POST /query
    |
    +--> Grounded generation pipeline
            |
            +--> generation policy filter
            |
            +--> bounded context builder
            |
            +--> generation provider
            |
            +--> citation mapper
                    |
                    v
                POST /answer
```

## FastAPI query flow

The retrieval API validates requests and delegates search to the existing semantic retrieval layer.

```text
Client
  |
  v
POST /query
  |
  v
Pydantic validation
  |
  v
Configured embedding provider
  |
  v
Database connection
  |
  v
semantic_search()
  |
  v
pgvector similarity search
  |
  v
QueryResponse
```

Supported controls are `query`, `top_k`, `note_type`, and `topic`. The result limit defaults to 5 and is bounded from 1 through 25.

Each result includes source path, title, note type, topic, effective AI access policy, chunk index, heading path, chunk content, and vector distance.

## Grounded answer flow

The grounded answer endpoint reuses semantic retrieval, converts retrieved results into policy-aware source chunks, and delegates answer construction to the grounded generation pipeline.

```text
Client
  |
  v
POST /answer
  |
  v
Pydantic validation
  |
  v
Configured embedding provider
  |
  v
semantic_search()
  |
  v
RetrievedChunk conversion
  |
  v
Generation policy filter
  |
  v
Bounded context builder
  |
  v
Configured generation provider
  |
  v
Stable citation mapping
  |
  v
GroundedQueryResponse
```

For external generation, local-only chunks are removed before prompt construction. Local providers can use locally permitted context.

The grounded pipeline returns only source chunks that were actually included in the bounded generation context.

Citation identifiers are assigned in source order as application-level values such as `S1`, `S2`, and `S3`. Citation mapping is independent of provider output format.

Each citation can include vault-relative source path, title, heading path, and chunk index. The generated response also returns the source chunks used, including source content and effective AI access policy.

If no usable generation context remains after retrieval and policy filtering, generation is skipped and the service returns an empty answer with empty citation and source lists.

## Provider model

Embedding and generation providers are selected through application configuration.

Both layers support deterministic local providers for development and automated tests. Optional OpenAI adapters are implemented behind the same provider abstractions.

Provider-specific SDK behavior is kept outside retrieval, grounding, citation, and API response logic.

The generation provider exposes whether it is external so the grounded pipeline can enforce external-use policy without hard-coding a specific vendor.

## Health and error behavior

`GET /health` validates database connectivity in addition to application availability.

A healthy dependency check returns HTTP 200. Database dependency failures return HTTP 503 with a generic response.

Invalid request bodies return HTTP 422. Database, embedding-provider, retrieval, generation-provider, and generation failures return HTTP 503 with generic service errors.

## Trust boundaries

### Private source boundary

The real vault contains personal data and is never committed to this repository. The homelab replica is treated as private data.

### Public development boundary

This repository contains only code, sanitized configuration examples, synthetic notes, and non-identifying architecture documentation.

### External model boundary

Only the minimum retrieved context required for a query should be sent to external embedding or generation APIs. Privacy controls are enforced before indexing and again before externally permitted model use. Notes may be excluded by path, note type, or `ai_access` policy, and `local-only` content must remain inside the local environment.

For grounded generation, external-use filtering occurs before prompt construction, provider invocation, citation creation, and source return mapping.

## Deployment model

The production-oriented runtime is a lightweight Linux container hosting PostgreSQL, pgvector, ingestion components, retrieval logic, grounded generation logic, and FastAPI.

External inference remains optional. Deterministic local providers support development and testing without paid API access, while external providers can be enabled through runtime configuration.

The FastAPI service can run from the development workstation while connecting to the private PostgreSQL service, or later run directly inside the RAG container.

## Source of truth

The vector database is a derived index. If it is lost or corrupted, it should be possible to recreate it entirely from the vault replica.

Generated answers and citations are also derived output. They do not replace the underlying Markdown source notes as the authoritative record.
