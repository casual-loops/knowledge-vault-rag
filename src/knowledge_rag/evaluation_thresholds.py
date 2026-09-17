import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from knowledge_rag.evaluation import EvaluationResult, RetrievalModeSummary


@dataclass(frozen=True, slots=True)
class ThresholdFailure:
    """One retrieval evaluation threshold failure."""

    scope: str
    identifier: str
    metric: str
    actual: float
    minimum: float

    def to_dict(self) -> dict[str, Any]:
        return {
            "scope": self.scope,
            "identifier": self.identifier,
            "metric": self.metric,
            "actual": self.actual,
            "minimum": self.minimum,
        }


@dataclass(frozen=True, slots=True)
class ThresholdCheck:
    """Result of evaluating retrieval quality thresholds."""

    passed: bool
    failures: list[ThresholdFailure]

    def to_dict(self) -> dict[str, Any]:
        return {
            "passed": self.passed,
            "failures": [
                failure.to_dict()
                for failure in self.failures
            ],
        }


def load_thresholds(
    path: Path,
) -> dict[str, Any]:
    """Load retrieval regression thresholds."""

    if not path.is_file():
        raise FileNotFoundError(
            f"Threshold configuration not found: {path}"
        )

    return json.loads(
        path.read_text(
            encoding="utf-8",
        )
    )


def check_thresholds(
    *,
    results: list[EvaluationResult],
    summaries: list[RetrievalModeSummary],
    thresholds: dict[str, Any],
) -> ThresholdCheck:
    """Check evaluation metrics against configured regression thresholds."""

    failures: list[ThresholdFailure] = []

    mode_thresholds = thresholds["mode_thresholds"]

    for summary in summaries:
        configured = mode_thresholds.get(
            summary.retrieval_mode,
        )

        if configured is None:
            continue

        metrics = {
            "mean_precision_at_k": summary.mean_precision_at_k,
            "mean_recall_at_k": summary.mean_recall_at_k,
            "mean_reciprocal_rank": summary.mean_reciprocal_rank,
        }

        for metric, actual in metrics.items():
            minimum = configured[metric]

            if actual < minimum:
                failures.append(
                    ThresholdFailure(
                        scope="mode",
                        identifier=summary.retrieval_mode,
                        metric=metric,
                        actual=actual,
                        minimum=minimum,
                    )
                )

    case_thresholds = thresholds["case_thresholds"]

    for result in results:
        case_metrics = {
            "recall_at_k": (
                result.metrics.recall_at_k,
                case_thresholds["minimum_recall_at_k"],
            ),
            "reciprocal_rank": (
                result.metrics.reciprocal_rank,
                case_thresholds["minimum_reciprocal_rank"],
            ),
        }

        for metric, (
            actual,
            minimum,
        ) in case_metrics.items():
            if actual < minimum:
                failures.append(
                    ThresholdFailure(
                        scope="case",
                        identifier=result.case_id,
                        metric=metric,
                        actual=actual,
                        minimum=minimum,
                    )
                )

    return ThresholdCheck(
        passed=not failures,
        failures=failures,
    )