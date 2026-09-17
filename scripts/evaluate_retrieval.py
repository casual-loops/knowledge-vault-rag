import argparse
from pathlib import Path

from knowledge_rag.api.main import run_retrieval
from knowledge_rag.db import get_connection
from knowledge_rag.embedding_factory import get_embedding_provider
from knowledge_rag.evaluation import (
    EvaluationCase,
    chunk_identity,
)
from knowledge_rag.evaluation_runner import (
    format_evaluation_summary,
    run_evaluation,
    write_evaluation_output,
)


REPO_ROOT = Path(__file__).resolve().parents[1]

DEFAULT_DATASET = (
    REPO_ROOT
    / "evaluation"
    / "retrieval_cases.json"
)

DEFAULT_OUTPUT = (
    REPO_ROOT
    / "evaluation"
    / "results.json"
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run retrieval quality evaluation.",
    )

    parser.add_argument(
        "--dataset",
        type=Path,
        default=DEFAULT_DATASET,
    )

    parser.add_argument(
        "--output",
        type=Path,
        default=DEFAULT_OUTPUT,
    )

    parser.add_argument(
        "--k",
        type=int,
        default=5,
    )

    return parser.parse_args()


def main() -> None:
    args = parse_args()

    embedding_provider = get_embedding_provider()

    def execute_case(
        case: EvaluationCase,
        k: int,
    ):
        with get_connection() as conn:
            results = run_retrieval(
                conn=conn,
                embedding_provider=embedding_provider,
                query=case.query,
                limit=k,
                note_type=case.filters.get("note_type"),
                topic=case.filters.get("topic"),
                retrieval_mode=case.retrieval_mode,
            )

        return [
            chunk_identity(
                result.source_path,
                result.chunk_index,
            )
            for result in results
        ]

    run = run_evaluation(
        dataset_path=args.dataset,
        retrieval_executor=execute_case,
        k=args.k,
    )

    write_evaluation_output(
        run,
        args.output,
    )

    print(
        format_evaluation_summary(
            run,
        )
    )


if __name__ == "__main__":
    main()