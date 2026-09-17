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
PostgreSQL + pgvector
    |
    +--> semantic retriever
    |
    +--> lexical retriever
            |
            v
      reciprocal rank fusion
            |
            v
       hybrid retriever
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

## Retrieval modes

The retrieval layer supports three independently callable modes.

### Semantic retrieval

Semantic retrieval embeds the query and ranks indexed chunks using pgvector similarity.

Semantic result scores are normalized to a higher-is-better representation before they leave the retrieval layer.

### Lexical retrieval

Lexical retrieval uses PostgreSQL full-text search over document title, heading path, and chunk content.

It does not require an embedding provider and can be called independently from vector search.

### Hybrid retrieval

Hybrid retrieval combines the ranked semantic and lexical result lists with Reciprocal Rank Fusion.

RRF uses rank position rather than attempting to compare raw semantic and lexical scores directly. For each chunk, contributions from every list where it appears are summed using the reciprocal-rank formula.

The implementation uses a smoothing constant and deterministic tie-breaking. A chunk appearing in both retrieval paths is deduplicated and receives contributions from both rankings.

Chunk identity is defined by vault-relative source path plus chunk index.

Source metadata and `ai_access` state are preserved through fusion.

## FastAPI query flow

The API validates requests and delegates to the selected retrieval mode.

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
retrieval_mode
  |
  +--> semantic_search()
  |
  +--> lexical_search()
  |
  +--> semantic_search() + lexical_search()
            |
            v
      hybrid_search()
            |
            v
        QueryResponse
```

Supported controls are `query`, `top_k`, `note_type`, `topic`, and `retrieval_mode`. The result limit defaults to 5 and is bounded from 1 through 25.

`retrieval_mode` accepts `semantic`, `lexical`, or `hybrid` and defaults to `semantic`.

Each result includes source path, title, note type, topic, effective AI access policy, chunk index, heading path, chunk content, score, and score type.

Scores are meaningful within their retrieval mode. They are not intended for cross-mode comparison.

## Grounded answer flow

The grounded answer endpoint uses the same retrieval selection before converting results into policy-aware source chunks.

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
Selected retrieval mode
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

## Retrieval evaluation subsystem

Retrieval quality is evaluated against a public-safe synthetic dataset stored in `evaluation/retrieval_cases.json`.

Each case identifies the query, retrieval mode, optional filters, and expected relevant chunks. Relevant chunks use the same stable source-path and chunk-index identity used by the retrieval layer.

The evaluation subsystem calculates precision at K, recall at K, and reciprocal rank for each case. Aggregate summaries are produced by retrieval mode so semantic, lexical, and hybrid behavior can be reviewed independently.

The runner in `scripts/evaluate_retrieval.py` records per-query rankings, writes deterministic JSON output, and prints a concise summary. Regression floors are configured separately in `evaluation/thresholds.json`.

Threshold checks can fail the evaluation command when configured mode-level or case-level minimums are breached. Threshold configuration is intentionally separate from retrieval implementation so baseline changes remain explicit and reviewable.

Unit coverage for the evaluation framework uses deterministic fixtures and does not require paid or nondeterministic providers.

Detailed methodology, metric definitions, threshold guidance, local execution instructions, and known limitations are documented in `docs/retrieval-evaluation.md`.

## Provider model

Embedding and generation providers are selected through application configuration.

Both layers support deterministic local providers for development and automated tests. Optional OpenAI adapters are implemented behind the same provider abstractions.

Provider-specific SDK behavior is kept outside retrieval, grounding, citation, and API response logic.

The generation provider exposes whether it is external so the grounded pipeline can enforce external-use policy without hard-coding a specific vendor.

Lexical retrieval is intentionally independent of the embedding provider.

## Health and error behavior

`GET /health` validates database connectivity in addition to application availability.

A healthy dependency check returns HTTP 200. Database dependency failures return HTTP 503 with a generic response.

Invalid request bodies, including unsupported retrieval modes, return HTTP 422. Database, embedding-provider, retrieval, generation-provider, and generation failures return HTTP 503 with generic service errors.

## Trust boundaries

### Private source boundary

The real vault contains personal data and is never committed to this repository. The homelab replica is treated as private data.

### Public development boundary

This repository contains only code, sanitized configuration examples, synthetic notes, and non-identifying architecture documentation.

### External model boundary

Only the minimum retrieved context required for a query should be sent to external embedding or generation APIs. Privacy controls are enforced before indexing and again before externally permitted model use. Notes may be excluded by path, note type, or `ai_access` policy, and `local-only` content must remain inside the local environment.

Hybrid fusion does not relax privacy state. Fused results retain the same `ai_access` metadata as their source chunks.

For grounded generation, external-use filtering occurs before prompt construction, provider invocation, citation creation, and source return mapping.

Evaluation fixtures in the public repository must remain synthetic and sanitized. Production or private-vault evaluation data must remain outside the public repository and follow the same trust boundary as the source vault.

## Deployment model

The production-oriented runtime is a lightweight Linux container hosting PostgreSQL, pgvector, ingestion components, semantic retrieval, lexical retrieval, hybrid ranking, grounded generation logic, and FastAPI.

External inference remains optional. Deterministic local providers support development and testing without paid API access, while external providers can be enabled through runtime configuration.

The FastAPI service can run from the development workstation while connecting to the private PostgreSQL service, or later run directly inside the RAG container.

## Source of truth

The search index is a derived structure. If it is lost or corrupted, it should be possible to recreate it entirely from the vault replica.

Generated answers and citations are also derived output. They do not replace the underlying Markdown source notes as the authoritative record.
