from pathlib import Path

import pytest

from knowledge_rag.evaluation import (
    EvaluationCase,
    evaluate_case,
    evaluate_retrieval,
    load_evaluation_cases,
    precision_at_k,
    recall_at_k,
    reciprocal_rank,
    EvaluationResult,
    RetrievalMetrics,
    summarize_by_retrieval_mode,
)


A = ("A.md", 0)
B = ("B.md", 0)
C = ("C.md", 0)
D = ("D.md", 0)


def test_precision_at_k() -> None:
    retrieved = [
        A,
        B,
        C,
    ]

    relevant = {
        A,
        C,
    }

    assert precision_at_k(
        retrieved,
        relevant,
        k=3,
    ) == pytest.approx(2 / 3)


def test_recall_at_k() -> None:
    retrieved = [
        A,
        B,
        C,
    ]

    relevant = {
        A,
        C,
        D,
    }

    assert recall_at_k(
        retrieved,
        relevant,
        k=3,
    ) == pytest.approx(2 / 3)


def test_reciprocal_rank_first_result() -> None:
    assert reciprocal_rank(
        [A, B, C],
        {A},
    ) == 1.0


def test_reciprocal_rank_later_result() -> None:
    assert reciprocal_rank(
        [A, B, C],
        {C},
    ) == pytest.approx(1 / 3)


def test_reciprocal_rank_returns_zero_without_match() -> None:
    assert reciprocal_rank(
        [A, B],
        {C},
    ) == 0.0


def test_empty_retrieval_returns_zero_precision() -> None:
    assert precision_at_k(
        [],
        {A},
        k=5,
    ) == 0.0


def test_empty_retrieval_returns_zero_recall_when_relevant_exists() -> None:
    assert recall_at_k(
        [],
        {A},
        k=5,
    ) == 0.0


def test_no_result_case_has_perfect_recall_when_nothing_is_returned() -> None:
    assert recall_at_k(
        [],
        set(),
        k=5,
    ) == 1.0


def test_unexpected_result_fails_no_result_recall() -> None:
    assert recall_at_k(
        [A],
        set(),
        k=5,
    ) == 0.0


def test_k_must_be_positive() -> None:
    with pytest.raises(
        ValueError,
        match="k must be greater than zero",
    ):
        precision_at_k(
            [A],
            {A},
            k=0,
        )


def test_evaluate_retrieval_returns_machine_readable_metrics() -> None:
    metrics = evaluate_retrieval(
        [
            A,
            B,
            C,
        ],
        {
            A,
            C,
        },
        k=3,
    )

    assert metrics.to_dict() == {
        "precision_at_k": pytest.approx(2 / 3),
        "recall_at_k": 1.0,
        "reciprocal_rank": 1.0,
    }


def test_load_evaluation_cases() -> None:
    dataset_path = (
        Path(__file__).resolve().parents[1]
        / "evaluation"
        / "retrieval_cases.json"
    )

    cases = load_evaluation_cases(
        dataset_path,
    )

    assert cases

    case_ids = {
        case.case_id
        for case in cases
    }

    assert "exact_azure_rbac" in case_ids
    assert "semantic_least_privilege_scope" in case_ids
    assert "no_result_unrelated_domain" in case_ids


def test_load_evaluation_cases_ignores_zero_relevance_judgments(
    tmp_path,
) -> None:
    dataset = tmp_path / "dataset.json"

    dataset.write_text(
        """
        {
          "version": 1,
          "cases": [
            {
              "id": "example",
              "category": "semantic",
              "query": "example",
              "retrieval_mode": "semantic",
              "filters": {},
              "relevant_chunks": [
                {
                  "source_path": "Relevant.md",
                  "chunk_index": 0,
                  "relevance": 3
                },
                {
                  "source_path": "Irrelevant.md",
                  "chunk_index": 0,
                  "relevance": 0
                }
              ]
            }
          ]
        }
        """,
        encoding="utf-8",
    )

    cases = load_evaluation_cases(
        dataset,
    )

    assert cases[0].relevant == {
        ("Relevant.md", 0),
    }


def test_evaluate_case_returns_machine_readable_result() -> None:
    case = EvaluationCase(
        case_id="example",
        category="semantic",
        query="synthetic query",
        retrieval_mode="semantic",
        filters={},
        relevant={
            A,
            C,
        },
    )

    result = evaluate_case(
        case,
        [
            A,
            B,
            C,
        ],
        k=3,
    )

    assert result.to_dict() == {
        "case_id": "example",
        "category": "semantic",
        "retrieval_mode": "semantic",
        "k": 3,
        "retrieved_count": 3,
        "relevant_count": 2,
        "metrics": {
            "precision_at_k": pytest.approx(2 / 3),
            "recall_at_k": 1.0,
            "reciprocal_rank": 1.0,
        },
    }


def test_evaluation_result_preserves_retrieval_mode() -> None:
    case = EvaluationCase(
        case_id="hybrid-example",
        category="ambiguous",
        query="synthetic query",
        retrieval_mode="hybrid",
        filters={},
        relevant={
            A,
        },
    )

    result = evaluate_case(
        case,
        [A],
        k=5,
    )

    assert result.retrieval_mode == "hybrid"


def test_summarize_by_retrieval_mode() -> None:
    results = [
        EvaluationResult(
            case_id="semantic-1",
            category="semantic",
            retrieval_mode="semantic",
            k=5,
            retrieved_count=3,
            relevant_count=2,
            metrics=RetrievalMetrics(
                precision_at_k=0.5,
                recall_at_k=1.0,
                reciprocal_rank=1.0,
            ),
        ),
        EvaluationResult(
            case_id="semantic-2",
            category="semantic",
            retrieval_mode="semantic",
            k=5,
            retrieved_count=2,
            relevant_count=1,
            metrics=RetrievalMetrics(
                precision_at_k=0.25,
                recall_at_k=0.5,
                reciprocal_rank=0.5,
            ),
        ),
        EvaluationResult(
            case_id="hybrid-1",
            category="ambiguous",
            retrieval_mode="hybrid",
            k=5,
            retrieved_count=3,
            relevant_count=2,
            metrics=RetrievalMetrics(
                precision_at_k=0.75,
                recall_at_k=1.0,
                reciprocal_rank=1.0,
            ),
        ),
    ]

    summaries = summarize_by_retrieval_mode(
        results,
    )

    assert [
        summary.retrieval_mode
        for summary in summaries
    ] == [
        "hybrid",
        "semantic",
    ]

    hybrid = summaries[0]
    semantic = summaries[1]

    assert hybrid.to_dict() == {
        "retrieval_mode": "hybrid",
        "case_count": 1,
        "mean_precision_at_k": 0.75,
        "mean_recall_at_k": 1.0,
        "mean_reciprocal_rank": 1.0,
    }

    assert semantic.to_dict() == {
        "retrieval_mode": "semantic",
        "case_count": 2,
        "mean_precision_at_k": pytest.approx(0.375),
        "mean_recall_at_k": pytest.approx(0.75),
        "mean_reciprocal_rank": pytest.approx(0.75),
    }


def test_summarize_by_retrieval_mode_returns_empty_list() -> None:
    assert summarize_by_retrieval_mode([]) == []


def test_mode_summary_output_is_deterministic() -> None:
    results = [
        EvaluationResult(
            case_id="semantic",
            category="semantic",
            retrieval_mode="semantic",
            k=5,
            retrieved_count=1,
            relevant_count=1,
            metrics=RetrievalMetrics(
                precision_at_k=1.0,
                recall_at_k=1.0,
                reciprocal_rank=1.0,
            ),
        ),
        EvaluationResult(
            case_id="lexical",
            category="exact-match",
            retrieval_mode="lexical",
            k=5,
            retrieved_count=1,
            relevant_count=1,
            metrics=RetrievalMetrics(
                precision_at_k=1.0,
                recall_at_k=1.0,
                reciprocal_rank=1.0,
            ),
        ),
        EvaluationResult(
            case_id="hybrid",
            category="ambiguous",
            retrieval_mode="hybrid",
            k=5,
            retrieved_count=1,
            relevant_count=1,
            metrics=RetrievalMetrics(
                precision_at_k=1.0,
                recall_at_k=1.0,
                reciprocal_rank=1.0,
            ),
        ),
    ]

    summaries = summarize_by_retrieval_mode(
        results,
    )

    assert [
        summary.retrieval_mode
        for summary in summaries
    ] == [
        "hybrid",
        "lexical",
        "semantic",
    ]