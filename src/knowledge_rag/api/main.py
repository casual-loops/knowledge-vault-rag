from fastapi import FastAPI

from knowledge_rag.api.models import (
    QueryRequest,
    QueryResponse,
    QueryResult,
)
from knowledge_rag.db import get_connection
from knowledge_rag.embedding_factory import get_embedding_provider
from knowledge_rag.retrieval import semantic_search


app = FastAPI(
    title="Knowledge Vault RAG",
    version="0.1.0",
)


@app.get("/health")
def health() -> dict[str, str]:
    return {"status": "ok"}


@app.post(
    "/query",
    response_model=QueryResponse,
)
def query(request: QueryRequest) -> QueryResponse:
    """Run semantic retrieval for a query string."""

    provider = get_embedding_provider()

    with get_connection() as conn:
        results = semantic_search(
            conn,
            provider,
            query=request.query,
        )

    return QueryResponse(
        results=[
            QueryResult(
                source_path=result.source_path,
                title=result.title,
                note_type=result.note_type,
                topic=result.topic,
                ai_access=result.ai_access,
                chunk_index=result.chunk_index,
                heading_path=result.heading_path,
                content=result.content,
                distance=result.distance,
            )
            for result in results
        ]
    )