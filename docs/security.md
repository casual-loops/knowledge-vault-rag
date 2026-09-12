# Security and Privacy Boundary

## Public repository rule

Assume every committed file is permanently public.

Do not commit:

- Personal journal entries or private notes
- Employer or customer documentation
- Credentials, tokens, API keys, cookies, or certificates
- Private IP addresses, internal DNS names, or identifying hostnames
- Real Syncthing device IDs
- Personal browser exports or bookmark collections
- Private repository URLs
- Database dumps from the real system
- The real Obsidian vault or its Syncthing version store

## Development data

Use only `examples/sample-vault/` for tests, screenshots, demonstrations, and reproducible examples.

Sample notes must be synthetic or deliberately sanitized.

The real private vault must not be used during development of ingestion, embeddings, retrieval, or generation behavior until the applicable privacy controls have been validated.

## Secrets

Runtime secrets belong in `.env`, a secret manager, or deployment-specific protected configuration.

`.env.example` documents variable names only.

## Threat model

The primary privacy risk is unintended disclosure of private vault content through indexing, logging, embeddings, retrieval, generation context, public source control, API responses, or external APIs.

The system therefore assumes:

- Markdown notes may contain sensitive personal or technical information.
- Local retrieval does not imply permission to send content to an external provider.
- External embedding and generation providers are separate trust boundaries.
- Configuration mistakes and missing metadata must fail conservatively.
- Application logs must not become an accidental secondary copy of note content.
- API error responses must not expose secrets, connection details, raw exception text, or unintended note content.
- The public repository must never contain real private-vault data.

## Note-level AI access policy

Notes may define an `ai_access` frontmatter property.

Supported values are:

| Value        | Local indexing | External embeddings | External generation |
| ------------ | -------------- | ------------------- | ------------------- |
| `allowed`    | Yes            | Yes                 | Yes                 |
| `local-only` | Yes            | No                  | No                  |
| `exclude`    | No             | No                  | No                  |

Missing, malformed, or unknown values resolve to `local-only`.

External access therefore requires an explicit `allowed` value.

## Configured exclusions

Operators may also exclude content globally by:

- vault-relative path
- note type

Path and type exclusions take precedence over note-level `ai_access`.

A note excluded by configuration does not enter the searchable document index.

## Enforcement model

Privacy is enforced at multiple boundaries.

### Ingestion gate

Before persistence, the system evaluates:

1. configured path exclusions
2. configured note-type exclusions
3. note-level `ai_access`

Notes that fail the ingestion policy are not written to the searchable document or chunk tables.

### External generation gate

Locally indexed content may still include `local-only` notes.

Before retrieved chunks are assembled into context for an external generation model, policy is evaluated again.

Only content explicitly marked `allowed` may cross the external generation boundary.

The embedding pipeline must apply the same external-use policy before sending chunk content to an external embedding provider.

### API boundary

The FastAPI query service exposes retrieval results through explicit response models and validates request data before query execution.

Invalid request bodies are rejected with HTTP `422`.

Operational failures in database connectivity, embedding-provider setup, or retrieval return HTTP `503` with generic error details.

API error responses must not include:

- credentials or tokens
- database connection strings
- internal hostnames or private network details
- raw stack traces or exception text
- note bodies or retrieved context not already part of an authorized successful response

The `/health` endpoint reports dependency status without revealing sensitive connection details.

## Logging policy

Normal application logs may include operational metadata such as:

- vault-relative source path
- document UUID
- indexing result
- chunk count
- timing information
- error type

Normal logs should not include:

- note bodies
- chunk bodies
- retrieved generation context
- sensitive frontmatter values
- credentials or API secrets

## Operator workflow

When creating or importing a note:

1. Decide whether the note may be indexed.
2. Decide whether its contents may leave the local environment.
3. Set `ai_access` explicitly when external use is permitted or indexing must be prohibited.
4. Use configured path or note-type exclusions for broad classes of content that should never be indexed.
5. Treat missing `ai_access` as `local-only`.
6. Validate exclusion rules before enabling ingestion of a real vault.
7. Review external-provider policy before enabling embeddings or generation.
8. Keep the real vault, database contents, and runtime secrets outside the public repository.
9. Verify API and health responses remain sanitized before exposing the service beyond trusted development access.

## Defense in depth

Privacy does not depend on a single metadata field.

The system combines:

- public repository hygiene
- runtime secret separation
- configured path exclusions
- configured note-type exclusions
- note-level privacy policy
- ingestion-time enforcement
- persisted `ai_access` state
- external generation filtering
- API response modeling and sanitized errors
- log-content restrictions

A failure at one layer should not automatically authorize disclosure at another.

## Principle

The assistant is allowed to derive from the vault. It is never the authoritative owner of the vault.

The Markdown vault remains canonical, while indexes and embeddings are derived and rebuildable.
