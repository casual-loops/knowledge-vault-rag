from typing import Any

from knowledge_rag.retrieval import lexical_search


class FakeCursor:
    def __init__(self, rows):
        self._rows = rows

    def fetchall(self):
        return self._rows


class FakeConnection:
    def __init__(self, rows):
        self.rows = rows
        self.executed_sql: str | None = None
        self.executed_params: tuple | None = None

    def execute(self, sql, params):
        self.executed_sql = sql
        self.executed_params = params
        return FakeCursor(self.rows)


def test_lexical_search_returns_ranked_results() -> None:
    conn: Any = FakeConnection(
        [
            (
                "Reference.md",
                "Reference",
                "reference",
                "demo",
                "allowed",
                0,
                "Reference",
                "Synthetic lexical match.",
                0.75,
            )
        ]
    )

    results = lexical_search(
        conn,
        query="lexical match",
        limit=5,
    )

    assert len(results) == 1

    result = results[0]

    assert result.source_path == "Reference.md"
    assert result.title == "Reference"
    assert result.note_type == "reference"
    assert result.topic == "demo"
    assert result.ai_access == "allowed"
    assert result.chunk_index == 0
    assert result.heading_path == "Reference"
    assert result.content == "Synthetic lexical match."
    assert result.score == 0.75
    assert result.score_type == "lexical"


def test_lexical_search_passes_filters() -> None:
    conn: Any = FakeConnection([])

    lexical_search(
        conn,
        query="synthetic",
        limit=7,
        note_type="reference",
        topic="demo",
    )

    assert conn.executed_params == (
        "synthetic",
        "synthetic",
        "reference",
        "reference",
        "demo",
        "demo",
        "demo",
        7,
    )


def test_lexical_search_uses_postgres_full_text_search() -> None:
    conn: Any = FakeConnection([])

    lexical_search(
        conn,
        query="synthetic",
    )

    assert conn.executed_sql is not None
    assert "to_tsvector" in conn.executed_sql
    assert "plainto_tsquery" in conn.executed_sql
    assert "ts_rank_cd" in conn.executed_sql


def test_lexical_search_returns_empty_list_when_no_matches() -> None:
    conn: Any = FakeConnection([])

    results = lexical_search(
        conn,
        query="no match",
    )

    assert results == []


def test_lexical_search_is_independent_of_embedding_provider() -> None:
    conn: Any = FakeConnection([])

    results = lexical_search(
        conn,
        query="synthetic",
    )

    assert results == []