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
    v
FastAPI query service
```

## FastAPI query flow

The query API validates requests and delegates search to the existing retrieval layer.

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

## Health and error behavior

`GET /health` validates database connectivity in addition to application availability.

A healthy dependency check returns HTTP 200. Database dependency failures return HTTP 503 with a generic response.

Invalid request bodies return HTTP 422. Database, embedding-provider, and retrieval failures return HTTP 503 without exposing internal exception details.

## Trust boundaries

### Private source boundary

The real vault contains personal data and is never committed to this repository. The homelab replica is treated as private data.

### Public development boundary

This repository contains only code, sanitized configuration examples, synthetic notes, and non-identifying architecture documentation.

### External model boundary

Only the minimum retrieved context required for a query should be sent to external embedding or generation APIs. Privacy controls are enforced before indexing and again before externally permitted model use. Notes may be excluded by path, note type, or `ai_access` policy, and `local-only` content must remain inside the local environment.

## Deployment model

The production-oriented runtime is a lightweight Linux container hosting PostgreSQL, pgvector, ingestion components, and FastAPI. Model inference remains external to avoid local GPU and memory requirements.

The current FastAPI service can run from the development workstation while connecting to the private PostgreSQL service, or later run directly inside the RAG container.

## Source of truth

The vector database is a derived index. If it is lost or corrupted, it should be possible to recreate it entirely from the vault replica.
