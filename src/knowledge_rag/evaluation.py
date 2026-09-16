import json
from pathlib import Path
from typing import Any

from dataclasses import asdict, dataclass


ChunkIdentity = tuple[str, int]


@dataclass(frozen=True, slots=True)
class RetrievalMetrics:
    """Retrieval quality metrics for one ranked result set."""

    precision_at_k: float
    recall_at_k: float
    reciprocal_rank: float

    def to_dict(self) -> dict[str, float]:
        """Return deterministic machine-readable metric output."""

        return asdict(self)


@dataclass(frozen=True, slots=True)
class EvaluationCase:
    """One retrieval evaluation case."""

    case_id: str
    category: str
    query: str
    retrieval_mode: str
    filters: dict[str, str]
    relevant: set[ChunkIdentity]


@dataclass(frozen=True, slots=True)
class EvaluationResult:
    """Metrics for one evaluated retrieval case."""

    case_id: str
    category: str
    retrieval_mode: str
    k: int
    retrieved_count: int
    relevant_count: int
    metrics: RetrievalMetrics

    def to_dict(self) -> dict[str, Any]:
        """Return deterministic machine-readable evaluation output."""

        return {
            "case_id": self.case_id,
            "category": self.category,
            "retrieval_mode": self.retrieval_mode,
            "k": self.k,
            "retrieved_count": self.retrieved_count,
            "relevant_count": self.relevant_count,
            "metrics": self.metrics.to_dict(),
        }


@dataclass(frozen=True, slots=True)
class RetrievalModeSummary:
    """Aggregate retrieval quality for one retrieval mode."""

    retrieval_mode: str
    case_count: int
    mean_precision_at_k: float
    mean_recall_at_k: float
    mean_reciprocal_rank: float

    def to_dict(self) -> dict[str, Any]:
        """Return deterministic machine-readable summary output."""

        return asdict(self)


def summarize_by_retrieval_mode(
    results: list[EvaluationResult],
) -> list[RetrievalModeSummary]:
    """Aggregate evaluation results by retrieval mode."""

    grouped: dict[str, list[EvaluationResult]] = {}

    for result in results:
        grouped.setdefault(
            result.retrieval_mode,
            [],
        ).append(result)

    summaries: list[RetrievalModeSummary] = []

    for retrieval_mode in sorted(grouped):
        mode_results = grouped[retrieval_mode]
        case_count = len(mode_results)

        summaries.append(
            RetrievalModeSummary(
                retrieval_mode=retrieval_mode,
                case_count=case_count,
                mean_precision_at_k=sum(
                    result.metrics.precision_at_k
                    for result in mode_results
                )
                / case_count,
                mean_recall_at_k=sum(
                    result.metrics.recall_at_k
                    for result in mode_results
                )
                / case_count,
                mean_reciprocal_rank=sum(
                    result.metrics.reciprocal_rank
                    for result in mode_results
                )
                / case_count,
            )
        )

    return summaries


def load_evaluation_cases(
    path: Path,
) -> list[EvaluationCase]:
    """Load retrieval evaluation cases from JSON."""

    payload = json.loads(
        path.read_text(
            encoding="utf-8",
        )
    )

    cases: list[EvaluationCase] = []

    for item in payload["cases"]:
        relevant = {
            chunk_identity(
                judgment["source_path"],
                judgment["chunk_index"],
            )
            for judgment in item["relevant_chunks"]
            if judgment["relevance"] > 0
        }

        cases.append(
            EvaluationCase(
                case_id=item["id"],
                category=item["category"],
                query=item["query"],
                retrieval_mode=item["retrieval_mode"],
                filters=item["filters"],
                relevant=relevant,
            )
        )

    return cases


def evaluate_case(
    case: EvaluationCase,
    retrieved: list[ChunkIdentity],
    *,
    k: int,
) -> EvaluationResult:
    """Evaluate retrieved chunks against one judgment case."""

    metrics = evaluate_retrieval(
        retrieved,
        case.relevant,
        k=k,
    )

    return EvaluationResult(
        case_id=case.case_id,
        category=case.category,
        retrieval_mode=case.retrieval_mode,
        k=k,
        retrieved_count=len(retrieved),
        relevant_count=len(case.relevant),
        metrics=metrics,
    )


def chunk_identity(
    source_path: str,
    chunk_index: int,
) -> ChunkIdentity:
    """Return the stable identity for one source chunk."""

    return (
        source_path,
        chunk_index,
    )


def precision_at_k(
    retrieved: list[ChunkIdentity],
    relevant: set[ChunkIdentity],
    *,
    k: int,
) -> float:
    """Return precision among the first k retrieved chunks."""

    if k <= 0:
        raise ValueError("k must be greater than zero")

    top_k = retrieved[:k]

    if not top_k:
        return 0.0

    relevant_retrieved = sum(
        1
        for chunk in top_k
        if chunk in relevant
    )

    return relevant_retrieved / len(top_k)


def recall_at_k(
    retrieved: list[ChunkIdentity],
    relevant: set[ChunkIdentity],
    *,
    k: int,
) -> float:
    """Return recall among the first k retrieved chunks."""

    if k <= 0:
        raise ValueError("k must be greater than zero")

    if not relevant:
        return 1.0 if not retrieved[:k] else 0.0

    relevant_retrieved = sum(
        1
        for chunk in retrieved[:k]
        if chunk in relevant
    )

    return relevant_retrieved / len(relevant)


def reciprocal_rank(
    retrieved: list[ChunkIdentity],
    relevant: set[ChunkIdentity],
) -> float:
    """Return the reciprocal rank of the first relevant chunk."""

    for rank, chunk in enumerate(
        retrieved,
        start=1,
    ):
        if chunk in relevant:
            return 1.0 / rank

    return 0.0


def evaluate_retrieval(
    retrieved: list[ChunkIdentity],
    relevant: set[ChunkIdentity],
    *,
    k: int,
) -> RetrievalMetrics:
    """Evaluate one ranked retrieval result set."""

    return RetrievalMetrics(
        precision_at_k=precision_at_k(
            retrieved,
            relevant,
            k=k,
        ),
        recall_at_k=recall_at_k(
            retrieved,
            relevant,
            k=k,
        ),
        reciprocal_rank=reciprocal_rank(
            retrieved,
            relevant,
        ),
    )