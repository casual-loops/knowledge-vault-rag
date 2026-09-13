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
) -> SearchResult:
    return SearchResult(
        source_path=source_path,
        title=source_path.removesuffix(".md"),
        note_type="reference",
        topic="demo",
        ai_access="allowed",
        chunk_index=chunk_index,
        heading_path="Section",
        content=f"Content for {source_path}",
        score=score,
        score_type=score_type,
    )


def test_hybrid_search_combines_rankings() -> None:
    semantic = [
        make_result(
            source_path="A.md",
            chunk_index=0,
            score=0.95,
            score_type="semantic",
        ),
        make_result(
            source_path="B.md",
            chunk_index=0,
            score=0.90,
            score_type="semantic",
        ),
    ]

    lexical = [
        make_result(
            source_path="B.md",
            chunk_index=0,
            score=0.80,
            score_type="lexical",
        ),
        make_result(
            source_path="C.md",
            chunk_index=0,
            score=0.70,
            score_type="lexical",
        ),
    ]

    results = hybrid_search(
        semantic,
        lexical,
        limit=3,
    )

    assert [result.source_path for result in results] == [
        "B.md",
        "A.md",
        "C.md",
    ]


def test_hybrid_search_deduplicates_overlapping_chunks() -> None:
    semantic = [
        make_result(
            source_path="Shared.md",
            chunk_index=2,
            score=0.95,
            score_type="semantic",
        )
    ]

    lexical = [
        make_result(
            source_path="Shared.md",
            chunk_index=2,
            score=0.75,
            score_type="lexical",
        )
    ]

    results = hybrid_search(
        semantic,
        lexical,
    )

    assert len(results) == 1
    assert results[0].source_path == "Shared.md"
    assert results[0].chunk_index == 2


def test_hybrid_search_breaks_ties_deterministically() -> None:
    semantic = [
        make_result(
            source_path="B.md",
            chunk_index=0,
            score=0.9,
            score_type="semantic",
        ),
        make_result(
            source_path="A.md",
            chunk_index=0,
            score=0.8,
            score_type="semantic",
        ),
    ]

    lexical = []

    results = hybrid_search(
        semantic,
        lexical,
        rrf_k=60,
    )

    assert [result.source_path for result in results] == [
        "B.md",
        "A.md",
    ]


def test_hybrid_search_preserves_privacy_state() -> None:
    semantic = [
        SearchResult(
            source_path="Local.md",
            title="Local",
            note_type="reference",
            topic="demo",
            ai_access="local-only",
            chunk_index=0,
            heading_path="Local",
            content="Local content.",
            score=0.9,
            score_type="semantic",
        )
    ]

    results = hybrid_search(
        semantic,
        [],
    )

    assert results[0].ai_access == "local-only"


def test_hybrid_search_respects_limit() -> None:
    semantic = [
        make_result(
            source_path="A.md",
            chunk_index=0,
            score=0.9,
            score_type="semantic",
        ),
        make_result(
            source_path="B.md",
            chunk_index=0,
            score=0.8,
            score_type="semantic",
        ),
    ]

    results = hybrid_search(
        semantic,
        [],
        limit=1,
    )

    assert len(results) == 1


def test_hybrid_search_returns_empty_list_for_empty_inputs() -> None:
    assert hybrid_search([], []) == []


def test_hybrid_search_breaks_equal_fused_scores_deterministically() -> None:
    semantic = [
        make_result(
            source_path="B.md",
            chunk_index=0,
            score=0.9,
            score_type="semantic",
        )
    ]

    lexical = [
        make_result(
            source_path="A.md",
            chunk_index=0,
            score=0.8,
            score_type="lexical",
        )
    ]

    results = hybrid_search(
        semantic,
        lexical,
    )

    assert [result.source_path for result in results] == [
        "A.md",
        "B.md",
    ]