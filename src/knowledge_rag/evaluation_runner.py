import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable

from knowledge_rag.evaluation import (
    ChunkIdentity,
    EvaluationCase,
    EvaluationResult,
    RetrievalModeSummary,
    evaluate_case,
    load_evaluation_cases,
    summarize_by_retrieval_mode,
)


@dataclass(frozen=True, slots=True)
class RankedEvaluationCase:
    """One evaluated case with its ranked retrieved chunks."""

    result: EvaluationResult
    ranking: list[ChunkIdentity]

    def to_dict(self) -> dict[str, Any]:
        """Return machine-readable case output."""

        return {
            "result": self.result.to_dict(),
            "ranking": [
                {
                    "source_path": source_path,
                    "chunk_index": chunk_index,
                    "rank": rank,
                }
                for rank, (
                    source_path,
                    chunk_index,
                ) in enumerate(
                    self.ranking,
                    start=1,
                )
            ],
        }


@dataclass(frozen=True, slots=True)
class EvaluationRun:
    """Complete output from one retrieval evaluation run."""

    cases: list[RankedEvaluationCase]
    summaries: list[RetrievalModeSummary]

    @property
    def results(self) -> list[EvaluationResult]:
        """Return per-case metric results."""

        return [
            case.result
            for case in self.cases
        ]

    def to_dict(self) -> dict[str, Any]:
        """Return deterministic machine-readable evaluation output."""

        return {
            "cases": [
                case.to_dict()
                for case in self.cases
            ],
            "summaries": [
                summary.to_dict()
                for summary in self.summaries
            ],
        }


RetrievalExecutor = Callable[
    [EvaluationCase, int],
    list[ChunkIdentity],
]


def run_evaluation(
    *,
    dataset_path: Path,
    retrieval_executor: RetrievalExecutor,
    k: int = 5,
) -> EvaluationRun:
    """Execute all evaluation cases and aggregate metrics."""

    if not dataset_path.is_file():
        raise FileNotFoundError(
            f"Evaluation dataset not found: {dataset_path}"
        )

    cases = load_evaluation_cases(
        dataset_path,
    )

    if not cases:
        raise ValueError(
            "Evaluation dataset contains no cases"
        )

    evaluated_cases: list[RankedEvaluationCase] = []

    for case in cases:
        retrieved = retrieval_executor(
            case,
            k,
        )

        result = evaluate_case(
            case,
            retrieved,
            k=k,
        )

        evaluated_cases.append(
            RankedEvaluationCase(
                result=result,
                ranking=retrieved,
            )
        )

    results = [
        case.result
        for case in evaluated_cases
    ]

    return EvaluationRun(
        cases=evaluated_cases,
        summaries=summarize_by_retrieval_mode(
            results,
        ),
    )


def write_evaluation_output(
    run: EvaluationRun,
    output_path: Path,
) -> None:
    """Write deterministic machine-readable evaluation output."""

    output_path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    output_path.write_text(
        json.dumps(
            run.to_dict(),
            indent=2,
            sort_keys=True,
        )
        + "\n",
        encoding="utf-8",
    )


def format_evaluation_summary(
    run: EvaluationRun,
) -> str:
    """Return a concise human-readable evaluation summary."""

    lines = [
        "Retrieval evaluation summary",
        f"Cases: {len(run.results)}",
    ]

    for summary in run.summaries:
        lines.append(
            
                f"{summary.retrieval_mode}: "
                f"cases={summary.case_count}, "
                f"precision@k={summary.mean_precision_at_k:.3f}, "
                f"recall@k={summary.mean_recall_at_k:.3f}, "
                f"MRR={summary.mean_reciprocal_rank:.3f}"
            
        )

    return "\n".join(lines)