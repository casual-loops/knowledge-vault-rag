from knowledge_rag.retrieval import (
    SearchResult,
    hybrid_search,
)


def make_result(
    *,
    source_path: str,
    chunk_index: int,
    score: float,
    score_type: str,
    note_type: str = "reference",
    topic: str = "demo",
    ai_access: str = "allowed",
    content: str | None = None,
) -> SearchResult:
    return SearchResult(
        source_path=source_path,
        title=source_path.removesuffix(".md"),
        note_type=note_type,
        topic=topic,
        ai_access=ai_access,
        chunk_index=chunk_index,
        heading_path="Section",
        content=content or f"Synthetic content for {source_path}",
        score=score,
        score_type=score_type,
    )


def test_hybrid_ranking_is_stable_for_known_rankings() -> None:
    semantic = [
        make_result(
            source_path="Conceptual.md",
            chunk_index=0,
            score=0.95,
            score_type="semantic",
        ),
        make_result(
            source_path="Shared.md",
            chunk_index=0,
            score=0.90,
            score_type="semantic",
        ),
        make_result(
            source_path="SemanticOnly.md",
            chunk_index=0,
            score=0.85,
            score_type="semantic",
        ),
    ]

    lexical = [
        make_result(
            source_path="ExactKeyword.md",
            chunk_index=0,
            score=0.80,
            score_type="lexical",
        ),
        make_result(
            source_path="Shared.md",
            chunk_index=0,
            score=0.75,
            score_type="lexical",
        ),
        make_result(
            source_path="LexicalOnly.md",
            chunk_index=0,
            score=0.70,
            score_type="lexical",
        ),
    ]

    results = hybrid_search(
        semantic,
        lexical,
        limit=5,
    )

    assert [result.source_path for result in results] == [
        "Shared.md",
        "Conceptual.md",
        "ExactKeyword.md",
        "LexicalOnly.md",
        "SemanticOnly.md",
    ]


def test_exact_keyword_and_semantic_matches_both_survive_fusion() -> None:
    semantic = [
        make_result(
            source_path="SemanticMatch.md",
            chunk_index=0,
            score=0.92,
            score_type="semantic",
            content="Operating system patches cannot be downloaded.",
        )
    ]

    lexical = [
        make_result(
            source_path="ExactKeyword.md",
            chunk_index=0,
            score=0.88,
            score_type="lexical",
            content="FileWave booster HTTP 403.",
        )
    ]

    results = hybrid_search(
        semantic,
        lexical,
        limit=5,
    )

    assert {result.source_path for result in results} == {
        "SemanticMatch.md",
        "ExactKeyword.md",
    }


def test_duplicate_chunk_is_returned_once() -> None:
    semantic = [
        make_result(
            source_path="Shared.md",
            chunk_index=3,
            score=0.95,
            score_type="semantic",
        )
    ]

    lexical = [
        make_result(
            source_path="Shared.md",
            chunk_index=3,
            score=0.90,
            score_type="lexical",
        )
    ]

    results = hybrid_search(
        semantic,
        lexical,
    )

    assert len(results) == 1
    assert results[0].source_path == "Shared.md"
    assert results[0].chunk_index == 3


def test_privacy_state_is_preserved_after_fusion() -> None:
    semantic = [
        make_result(
            source_path="Local.md",
            chunk_index=0,
            score=0.95,
            score_type="semantic",
            ai_access="local-only",
        )
    ]

    lexical = [
        make_result(
            source_path="Allowed.md",
            chunk_index=0,
            score=0.90,
            score_type="lexical",
            ai_access="allowed",
        )
    ]

    results = hybrid_search(
        semantic,
        lexical,
    )

    privacy = {
        result.source_path: result.ai_access
        for result in results
    }

    assert privacy == {
        "Allowed.md": "allowed",
        "Local.md": "local-only",
    }


def test_filtered_result_sets_fuse_without_reintroducing_filtered_chunks() -> None:
    semantic = [
        make_result(
            source_path="Reference.md",
            chunk_index=0,
            score=0.95,
            score_type="semantic",
            note_type="reference",
            topic="privacy",
        )
    ]

    lexical = [
        make_result(
            source_path="Reference.md",
            chunk_index=0,
            score=0.90,
            score_type="lexical",
            note_type="reference",
            topic="privacy",
        )
    ]

    results = hybrid_search(
        semantic,
        lexical,
    )

    assert len(results) == 1
    assert results[0].note_type == "reference"
    assert results[0].topic == "privacy"


def test_hybrid_regressions_do_not_require_external_providers() -> None:
    semantic = [
        make_result(
            source_path="LocalSemantic.md",
            chunk_index=0,
            score=0.90,
            score_type="semantic",
        )
    ]

    lexical = [
        make_result(
            source_path="LocalLexical.md",
            chunk_index=0,
            score=0.80,
            score_type="lexical",
        )
    ]

    results = hybrid_search(
        semantic,
        lexical,
    )

    assert len(results) == 2