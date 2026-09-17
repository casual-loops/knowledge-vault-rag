import json

import pytest

from knowledge_rag.evaluation import (
    EvaluationResult,
    RetrievalMetrics,
    RetrievalModeSummary,
)
from knowledge_rag.evaluation_thresholds import (
    check_thresholds,
    load_thresholds,
)


def make_result(
    *,
    case_id: str,
    recall_at_k: float,
    reciprocal_rank: float,
) -> EvaluationResult:
    return EvaluationResult(
        case_id=case_id,
        category="synthetic",
        retrieval_mode="semantic",
        k=5,
        retrieved_count=1,
        relevant_count=1,
        metrics=RetrievalMetrics(
            precision_at_k=1.0,
            recall_at_k=recall_at_k,
            reciprocal_rank=reciprocal_rank,
        ),
    )


def make_summary(
    *,
    retrieval_mode: str,
    precision: float,
    recall: float,
    reciprocal_rank: float,
) -> RetrievalModeSummary:
    return RetrievalModeSummary(
        retrieval_mode=retrieval_mode,
        case_count=1,
        mean_precision_at_k=precision,
        mean_recall_at_k=recall,
        mean_reciprocal_rank=reciprocal_rank,
    )


def test_threshold_check_passes_when_metrics_meet_minimums() -> None:
    thresholds = {
        "mode_thresholds": {
            "semantic": {
                "mean_precision_at_k": 0.4,
                "mean_recall_at_k": 0.6,
                "mean_reciprocal_rank": 0.6,
            }
        },
        "case_thresholds": {
            "minimum_recall_at_k": 0.5,
            "minimum_reciprocal_rank": 0.5,
        },
    }

    check = check_thresholds(
        results=[
            make_result(
                case_id="semantic-case",
                recall_at_k=1.0,
                reciprocal_rank=1.0,
            )
        ],
        summaries=[
            make_summary(
                retrieval_mode="semantic",
                precision=0.8,
                recall=1.0,
                reciprocal_rank=1.0,
            )
        ],
        thresholds=thresholds,
    )

    assert check.passed is True
    assert check.failures == []


def test_threshold_check_reports_mode_regression() -> None:
    thresholds = {
        "mode_thresholds": {
            "semantic": {
                "mean_precision_at_k": 0.5,
                "mean_recall_at_k": 0.6,
                "mean_reciprocal_rank": 0.6,
            }
        },
        "case_thresholds": {
            "minimum_recall_at_k": 0.0,
            "minimum_reciprocal_rank": 0.0,
        },
    }

    check = check_thresholds(
        results=[],
        summaries=[
            make_summary(
                retrieval_mode="semantic",
                precision=0.25,
                recall=0.9,
                reciprocal_rank=0.9,
            )
        ],
        thresholds=thresholds,
    )

    assert check.passed is False
    assert len(check.failures) == 1

    failure = check.failures[0]

    assert failure.scope == "mode"
    assert failure.identifier == "semantic"
    assert failure.metric == "mean_precision_at_k"
    assert failure.actual == 0.25
    assert failure.minimum == 0.5


def test_threshold_check_reports_case_regression() -> None:
    thresholds = {
        "mode_thresholds": {},
        "case_thresholds": {
            "minimum_recall_at_k": 0.5,
            "minimum_reciprocal_rank": 0.5,
        },
    }

    check = check_thresholds(
        results=[
            make_result(
                case_id="weak-case",
                recall_at_k=0.25,
                reciprocal_rank=0.0,
            )
        ],
        summaries=[],
        thresholds=thresholds,
    )

    assert check.passed is False

    assert [
        failure.to_dict()
        for failure in check.failures
    ] == [
        {
            "scope": "case",
            "identifier": "weak-case",
            "metric": "recall_at_k",
            "actual": 0.25,
            "minimum": 0.5,
        },
        {
            "scope": "case",
            "identifier": "weak-case",
            "metric": "reciprocal_rank",
            "actual": 0.0,
            "minimum": 0.5,
        },
    ]


def test_threshold_equal_to_minimum_passes() -> None:
    thresholds = {
        "mode_thresholds": {
            "hybrid": {
                "mean_precision_at_k": 0.5,
                "mean_recall_at_k": 0.5,
                "mean_reciprocal_rank": 0.5,
            }
        },
        "case_thresholds": {
            "minimum_recall_at_k": 0.0,
            "minimum_reciprocal_rank": 0.0,
        },
    }

    check = check_thresholds(
        results=[],
        summaries=[
            make_summary(
                retrieval_mode="hybrid",
                precision=0.5,
                recall=0.5,
                reciprocal_rank=0.5,
            )
        ],
        thresholds=thresholds,
    )

    assert check.passed is True


def test_unconfigured_mode_is_ignored() -> None:
    thresholds = {
        "mode_thresholds": {},
        "case_thresholds": {
            "minimum_recall_at_k": 0.0,
            "minimum_reciprocal_rank": 0.0,
        },
    }

    check = check_thresholds(
        results=[],
        summaries=[
            make_summary(
                retrieval_mode="lexical",
                precision=0.0,
                recall=0.0,
                reciprocal_rank=0.0,
            )
        ],
        thresholds=thresholds,
    )

    assert check.passed is True


def test_load_thresholds(tmp_path) -> None:
    path = tmp_path / "thresholds.json"

    path.write_text(
        json.dumps(
            {
                "version": 1,
                "mode_thresholds": {},
                "case_thresholds": {
                    "minimum_recall_at_k": 0.0,
                    "minimum_reciprocal_rank": 0.0,
                },
            }
        ),
        encoding="utf-8",
    )

    thresholds = load_thresholds(
        path,
    )

    assert thresholds["version"] == 1


def test_load_thresholds_fails_when_missing(
    tmp_path,
) -> None:
    path = tmp_path / "missing.json"

    with pytest.raises(
        FileNotFoundError,
        match="Threshold configuration not found",
    ):
        load_thresholds(
            path,
        )