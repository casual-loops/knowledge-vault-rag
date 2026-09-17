import json

import pytest

from knowledge_rag.evaluation import (
    EvaluationCase,
)
from knowledge_rag.evaluation_runner import (
    format_evaluation_summary,
    run_evaluation,
    write_evaluation_output,
)


def make_case(
    *,
    case_id: str,
    retrieval_mode: str,
    relevant: set[tuple[str, int]],
) -> EvaluationCase:
    return EvaluationCase(
        case_id=case_id,
        category="synthetic",
        query="synthetic query",
        retrieval_mode=retrieval_mode,
        filters={},
        relevant=relevant,
    )


def test_run_evaluation_executes_all_cases(
    tmp_path,
    monkeypatch,
) -> None:
    dataset = tmp_path / "dataset.json"

    dataset.write_text(
        """
        {
          "version": 1,
          "cases": [
            {
              "id": "semantic-case",
              "category": "semantic",
              "query": "semantic query",
              "retrieval_mode": "semantic",
              "filters": {},
              "relevant_chunks": [
                {
                  "source_path": "A.md",
                  "chunk_index": 0,
                  "relevance": 3
                }
              ]
            },
            {
              "id": "hybrid-case",
              "category": "ambiguous",
              "query": "hybrid query",
              "retrieval_mode": "hybrid",
              "filters": {},
              "relevant_chunks": [
                {
                  "source_path": "B.md",
                  "chunk_index": 0,
                  "relevance": 3
                }
              ]
            }
          ]
        }
        """,
        encoding="utf-8",
    )

    calls: list[tuple[str, int]] = []

    def fake_executor(
        case: EvaluationCase,
        k: int,
    ):
        calls.append(
            (
                case.case_id,
                k,
            )
        )

        if case.case_id == "semantic-case":
            return [
                ("A.md", 0),
            ]

        return [
            ("B.md", 0),
        ]

    run = run_evaluation(
        dataset_path=dataset,
        retrieval_executor=fake_executor,
        k=5,
    )

    assert calls == [
        ("semantic-case", 5),
        ("hybrid-case", 5),
    ]

    assert len(run.results) == 2

    assert [
        summary.retrieval_mode
        for summary in run.summaries
    ] == [
        "hybrid",
        "semantic",
    ]


def test_run_evaluation_fails_when_dataset_is_missing(
    tmp_path,
) -> None:
    dataset = tmp_path / "missing.json"

    with pytest.raises(
        FileNotFoundError,
        match="Evaluation dataset not found",
    ):
        run_evaluation(
            dataset_path=dataset,
            retrieval_executor=lambda case, k: [],
        )


def test_run_evaluation_fails_when_dataset_has_no_cases(
    tmp_path,
) -> None:
    dataset = tmp_path / "dataset.json"

    dataset.write_text(
        """
        {
          "version": 1,
          "cases": []
        }
        """,
        encoding="utf-8",
    )

    with pytest.raises(
        ValueError,
        match="Evaluation dataset contains no cases",
    ):
        run_evaluation(
            dataset_path=dataset,
            retrieval_executor=lambda case, k: [],
        )


def test_run_evaluation_fails_for_malformed_json(
    tmp_path,
) -> None:
    dataset = tmp_path / "dataset.json"

    dataset.write_text(
        "{not valid json",
        encoding="utf-8",
    )

    with pytest.raises(
        json.JSONDecodeError,
    ):
        run_evaluation(
            dataset_path=dataset,
            retrieval_executor=lambda case, k: [],
        )


def test_write_evaluation_output_is_deterministic(
    tmp_path,
) -> None:
    dataset = tmp_path / "dataset.json"
    output = tmp_path / "nested" / "results.json"

    dataset.write_text(
        """
        {
          "version": 1,
          "cases": [
            {
              "id": "semantic-case",
              "category": "semantic",
              "query": "semantic query",
              "retrieval_mode": "semantic",
              "filters": {},
              "relevant_chunks": [
                {
                  "source_path": "A.md",
                  "chunk_index": 0,
                  "relevance": 3
                }
              ]
            }
          ]
        }
        """,
        encoding="utf-8",
    )

    run = run_evaluation(
        dataset_path=dataset,
        retrieval_executor=lambda case, k: [
            ("A.md", 0),
        ],
        k=5,
    )

    write_evaluation_output(
        run,
        output,
    )

    first = output.read_text(
        encoding="utf-8",
    )

    write_evaluation_output(
        run,
        output,
    )

    second = output.read_text(
        encoding="utf-8",
    )

    assert first == second

    payload = json.loads(first)

    assert payload["cases"][0]["result"]["case_id"] == "semantic-case"
    assert payload["cases"][0]["ranking"] == [
        {
            "source_path": "A.md",
            "chunk_index": 0,
            "rank": 1,
        }
    ]
    assert payload["summaries"][0]["retrieval_mode"] == "semantic"


def test_format_evaluation_summary() -> None:

    case = make_case(
        case_id="semantic-case",
        retrieval_mode="semantic",
        relevant={
            ("A.md", 0),
        },
    )

    from knowledge_rag.evaluation import evaluate_case, summarize_by_retrieval_mode
    from knowledge_rag.evaluation_runner import (
        EvaluationRun,
        RankedEvaluationCase,
    )

    result = evaluate_case(
        case,
        [
            ("A.md", 0),
        ],
        k=5,
    )

    run = EvaluationRun(
    cases=[
        RankedEvaluationCase(
            result=result,
            ranking=[
                ("A.md", 0),
            ],
        )
    ],
    summaries=summarize_by_retrieval_mode(
        [
            result,
        ]
    ),
    )

    summary = format_evaluation_summary(
        run,
    )

    assert summary == (
        "Retrieval evaluation summary\n"
        "Cases: 1\n"
        "semantic: cases=1, "
        "precision@k=1.000, "
        "recall@k=1.000, "
        "MRR=1.000"
    )