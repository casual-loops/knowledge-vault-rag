from typing import Any

from pydantic import BaseModel, Field


class QueryRequest(BaseModel):
    """Request body for semantic retrieval."""

    query: str = Field(min_length=1)
    top_k: int = Field(default=5, ge=1, le=25)
    note_type: str | None = Field(default=None, min_length=1)
    topic: str | None = Field(default=None, min_length=1)


class QueryResult(BaseModel):
    """One semantic retrieval result returned by the API."""

    source_path: str
    title: str
    note_type: str | None
    topic: Any
    ai_access: str
    chunk_index: int
    heading_path: str | None
    content: str
    distance: float


class QueryResponse(BaseModel):
    """Response body for semantic retrieval."""

    results: list[QueryResult]