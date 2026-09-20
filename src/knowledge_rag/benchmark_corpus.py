"""Deterministic, public-safe synthetic corpora for performance benchmarks."""

from __future__ import annotations

import json
from dataclasses import dataclass
from hashlib import sha256
from pathlib import Path, PurePosixPath
from typing import Any

from knowledge_rag.ingest.chunker import chunk_markdown
from knowledge_rag.ingest.markdown import load_markdown

_WORKLOAD_KEYS = frozenset(
    {
        "version",
        "seed",
        "tiers",
        "chunks_per_note",
        "changed_fraction",
        "warmup_iterations",
        "measured_iterations",
        "queries",
    }
)
_QUERY_KEYS = frozenset({"id", "retrieval_mode", "query", "note_type", "topic"})
_NOTE_TYPES = ("reference", "project", "study")
_TOPICS = ("monitoring", "identity", "operations", "knowledge-management")
_MODES = frozenset({"semantic", "lexical", "hybrid"})
_PUBLIC_PHRASES = (
    "capacity planning uses measured service demand and clear review intervals",
    "identity access review records role assignment and approval procedure",
    "operational change validation checks monitoring alert threshold behavior",
    "knowledge retrieval architecture benefits from concise, reusable observations",
    "distributed monitoring improves signal quality through consistent reporting",
    "the documented workflow names an owner, a decision, and a follow-up date",
)


@dataclass(frozen=True, slots=True)
class PerformanceQuery:
    query_id: str
    retrieval_mode: str
    query: str
    note_type: str | None
    topic: str | None


@dataclass(frozen=True, slots=True)
class BenchmarkWorkload:
    version: int
    seed: int
    tiers: tuple[int, ...]
    chunks_per_note: int
    changed_fraction: float
    warmup_iterations: int
    measured_iterations: int
    queries: tuple[PerformanceQuery, ...]


@dataclass(frozen=True, slots=True)
class CorpusManifest:
    version: int
    seed: int
    note_count: int
    chunk_count: int
    paths: tuple[str, ...]
    content_digest: str


def load_benchmark_workload(path: Path) -> BenchmarkWorkload:
    """Load and validate the versioned, timing-independent benchmark workload."""
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise ValueError("unable to load benchmark workload") from exc
    if not isinstance(payload, dict) or set(payload) != _WORKLOAD_KEYS:
        raise ValueError("invalid workload schema")
    if type(payload["version"]) is not int or payload["version"] <= 0:
        raise ValueError("invalid workload version")
    if type(payload["seed"]) is not int:
        raise ValueError("invalid workload seed")
    tiers = payload["tiers"]
    if (
        not isinstance(tiers, list)
        or not tiers
        or any(type(tier) is not int or tier <= 0 for tier in tiers)
        or len(set(tiers)) != len(tiers)
    ):
        raise ValueError("invalid workload tiers")
    if type(payload["chunks_per_note"]) is not int or payload["chunks_per_note"] != 3:
        raise ValueError("invalid chunks_per_note")
    if payload["changed_fraction"] != 0.1:
        raise ValueError("invalid changed_fraction")
    if type(payload["warmup_iterations"]) is not int or payload["warmup_iterations"] < 0:
        raise ValueError("invalid warmup_iterations")
    if type(payload["measured_iterations"]) is not int or payload["measured_iterations"] <= 0:
        raise ValueError("invalid measured_iterations")
    queries = payload["queries"]
    if not isinstance(queries, list) or len(queries) != 9:
        raise ValueError("invalid workload queries")
    parsed_queries: list[PerformanceQuery] = []
    seen_ids: set[str] = set()
    for item in queries:
        if not isinstance(item, dict) or set(item) != _QUERY_KEYS:
            raise ValueError("invalid workload query schema")
        query_id = item["id"]
        mode = item["retrieval_mode"]
        if (
            not isinstance(query_id, str)
            or not query_id
            or query_id in seen_ids
            or not isinstance(mode, str)
            or mode not in _MODES
            or not isinstance(item["query"], str)
            or not item["query"]
        ):
            raise ValueError("invalid workload query")
        if item["note_type"] is not None and not isinstance(item["note_type"], str):
            raise ValueError("invalid workload note_type")
        if item["topic"] is not None and not isinstance(item["topic"], str):
            raise ValueError("invalid workload topic")
        seen_ids.add(query_id)
        parsed_queries.append(
            PerformanceQuery(query_id, mode, item["query"], item["note_type"], item["topic"])
        )
    if any(sum(query.retrieval_mode == mode for query in parsed_queries) != 3 for mode in _MODES):
        raise ValueError("workload must contain three queries per mode")
    return BenchmarkWorkload(
        version=payload["version"],
        seed=payload["seed"],
        tiers=tuple(tiers),
        chunks_per_note=payload["chunks_per_note"],
        changed_fraction=float(payload["changed_fraction"]),
        warmup_iterations=payload["warmup_iterations"],
        measured_iterations=payload["measured_iterations"],
        queries=tuple(parsed_queries),
    )


def generate_synthetic_vault(
    root: Path, workload: BenchmarkWorkload, note_count: int
) -> CorpusManifest:
    """Generate a deterministic vault and return its relative-path manifest."""
    if note_count not in workload.tiers:
        raise ValueError("note_count is not an approved workload tier")
    if note_count <= 0:
        raise ValueError("note_count must be positive")
    root.mkdir(parents=True, exist_ok=True)
    if any(root.iterdir()):
        raise ValueError("target root must be empty or nonempty generation is unsafe")
    paths: list[str] = []
    digest = sha256()
    for index in range(note_count):
        relative = _relative_note_path(index)
        path = root / relative
        path.parent.mkdir(parents=True, exist_ok=True)
        content = _render_note(workload.seed, index)
        encoded = content.encode("utf-8")
        path.write_bytes(encoded)
        paths.append(relative)
        digest.update(relative.encode("utf-8"))
        digest.update(b"\0")
        digest.update(encoded)
    return CorpusManifest(
        version=workload.version,
        seed=workload.seed,
        note_count=note_count,
        chunk_count=note_count * workload.chunks_per_note,
        paths=tuple(paths),
        content_digest=digest.hexdigest(),
    )


def modify_synthetic_vault(
    root: Path, manifest: CorpusManifest, fraction: float
) -> tuple[str, ...]:
    """Apply a stable ten-percent content change without changing chunk counts."""
    if fraction != 0.1:
        raise ValueError("only a ten-percent mutation is supported")
    root_resolved = root.resolve()
    expected = int(manifest.note_count * fraction)
    if len(manifest.paths) != manifest.note_count:
        raise ValueError("manifest does not support the requested mutation")
    paths = _validated_manifest_paths(root, manifest)
    for relative in paths:
        path = root_resolved / relative
        if not path.is_file() or path.is_symlink():
            raise ValueError("manifest path is missing or not a regular file")
    selected = tuple(paths[index] for index in range(0, manifest.note_count, 10))
    if len(selected) != expected:
        raise ValueError("manifest does not support the requested mutation")
    for relative in selected:
        path = root_resolved / relative
        original = path.read_text(encoding="utf-8")
        path.write_text(
            original.rstrip() + "\nRevision note: this synthetic record was reviewed in the controlled rerun.\n",
            encoding="utf-8",
        )
        if len(chunk_markdown(load_markdown(path).content)) != 3:
            raise ValueError("mutation changed the chunk invariant")
    return selected


def _validated_manifest_paths(root: Path, manifest: CorpusManifest) -> tuple[str, ...]:
    """Validate every manifest path before any path is read or modified."""
    root_resolved = root.resolve()
    validated: list[str] = []
    seen: set[str] = set()
    for relative in manifest.paths:
        if not isinstance(relative, str) or not relative or "\\" in relative:
            raise ValueError("invalid manifest path")
        parsed = PurePosixPath(relative)
        if (
            parsed.is_absolute()
            or parsed.as_posix() != relative
            or any(part in {"", ".", ".."} for part in parsed.parts)
            or relative in seen
        ):
            raise ValueError("invalid manifest path")
        candidate = (root_resolved / Path(*parsed.parts)).resolve()
        try:
            candidate.relative_to(root_resolved)
        except ValueError as exc:
            raise ValueError("invalid manifest path") from exc
        seen.add(relative)
        validated.append(relative)
    return tuple(sorted(validated))


def _selectors(seed: int, index: int) -> bytes:
    return sha256(f"{seed}:{index}".encode()).digest()


def _relative_note_path(index: int) -> str:
    return f"synthetic/{index // 1000:04d}/note-{index:05d}.md"


def _render_note(seed: int, index: int) -> str:
    selectors = _selectors(seed, index)
    note_type = _NOTE_TYPES[index % len(_NOTE_TYPES)]
    topic = _TOPICS[index % len(_TOPICS)]
    phrase_a = _PUBLIC_PHRASES[selectors[0] % len(_PUBLIC_PHRASES)]
    phrase_b = _PUBLIC_PHRASES[selectors[1] % len(_PUBLIC_PHRASES)]
    phrase_c = _PUBLIC_PHRASES[selectors[2] % len(_PUBLIC_PHRASES)]
    return (
        "---\n"
        "ai_access: allowed\n"
        f"note_type: {note_type}\n"
        f"topic: {topic}\n"
        "---\n\n"
        f"# Observation {index:05d}\n{phrase_a}.\n\n"
        f"# Procedure {index:05d}\n{phrase_b}.\n\n"
        f"# Validation {index:05d}\n{phrase_c}.\n"
    )


def workload_to_dict(workload: BenchmarkWorkload) -> dict[str, Any]:
    """Return the workload using the exact JSON/report serializer field names."""
    return {
        "version": workload.version,
        "seed": workload.seed,
        "tiers": list(workload.tiers),
        "chunks_per_note": workload.chunks_per_note,
        "changed_fraction": workload.changed_fraction,
        "warmup_iterations": workload.warmup_iterations,
        "measured_iterations": workload.measured_iterations,
        "queries": [
            {
                "id": query.query_id,
                "retrieval_mode": query.retrieval_mode,
                "query": query.query,
                "note_type": query.note_type,
                "topic": query.topic,
            }
            for query in workload.queries
        ],
    }
