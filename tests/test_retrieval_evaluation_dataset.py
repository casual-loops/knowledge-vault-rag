import json
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]
DATASET_PATH = REPO_ROOT / "evaluation" / "retrieval_cases.json"
SAMPLE_VAULT = REPO_ROOT / "examples" / "sample-vault"


def load_dataset() -> dict:
    return json.loads(
        DATASET_PATH.read_text(
            encoding="utf-8",
        )
    )


def test_retrieval_evaluation_dataset_is_valid_json() -> None:
    dataset = load_dataset()

    assert dataset["version"] == 1
    assert dataset["cases"]


def test_relevance_values_are_supported() -> None:
    dataset = load_dataset()

    valid_relevance = {
        0,
        1,
        2,
        3,
    }

    for case in dataset["cases"]:
        for judgment in case["relevant_chunks"]:
            assert judgment["relevance"] in valid_relevance


def test_all_judged_source_paths_exist() -> None:
    dataset = load_dataset()

    for case in dataset["cases"]:
        for judgment in case["relevant_chunks"]:
            source_path = SAMPLE_VAULT / judgment["source_path"]

            assert source_path.is_file(), (
                f"Missing evaluation source: "
                f"{judgment['source_path']}"
            )


def test_case_ids_are_unique() -> None:
    dataset = load_dataset()

    case_ids = [
        case["id"]
        for case in dataset["cases"]
    ]

    assert len(case_ids) == len(set(case_ids))


def test_required_evaluation_categories_are_present() -> None:
    dataset = load_dataset()

    categories = {
        case["category"]
        for case in dataset["cases"]
    }

    assert {
        "exact-match",
        "semantic",
        "ambiguous",
        "filtered",
        "no-result",
    }.issubset(categories)


def test_retrieval_modes_are_supported() -> None:
    dataset = load_dataset()

    valid_modes = {
        "semantic",
        "lexical",
        "hybrid",
    }

    for case in dataset["cases"]:
        assert case["retrieval_mode"] in valid_modes