from typing import Any, Literal

from pydantic import BaseModel, Field


class QueryRequest(BaseModel):
    """Request body for retrieval."""

    query: str = Field(min_length=1)
    top_k: int = Field(default=5, ge=1, le=25)
    note_type: str | None = Field(default=None, min_length=1)
    topic: str | None = Field(default=None, min_length=1)
    retrieval_mode: Literal[
        "semantic",
        "lexical",
        "hybrid",
    ] = "semantic"


class QueryResult(BaseModel):
    """One retrieval result returned by the API."""

    source_path: str
    title: str
    note_type: str | None
    topic: Any
    ai_access: str
    chunk_index: int
    heading_path: str | None
    content: str
    score: float
    score_type: str


class QueryResponse(BaseModel):
    """Response body for semantic retrieval."""

    results: list[QueryResult]


class GroundedQueryRequest(BaseModel):
    """Request body for grounded answer generation."""

    query: str = Field(min_length=1)
    top_k: int = Field(default=5, ge=1, le=25)
    note_type: str | None = Field(default=None, min_length=1)
    topic: str | None = Field(default=None, min_length=1)
    retrieval_mode: Literal[
        "semantic",
        "lexical",
        "hybrid",
    ] = "semantic"

class CitationResult(BaseModel):
    """Citation metadata returned with a grounded answer."""

    citation_id: str
    source_path: str
    title: str | None
    heading_path: str | None
    chunk_index: int | None


class GroundedSourceResult(BaseModel):
    """One source chunk used for grounded answer generation."""

    source_path: str
    title: str | None
    heading_path: str | None
    chunk_index: int | None
    content: str
    ai_access: str


class GroundedQueryResponse(BaseModel):
    """Response body for grounded answer generation."""

    answer: str
    citations: list[CitationResult]
    sources: list[GroundedSourceResult]