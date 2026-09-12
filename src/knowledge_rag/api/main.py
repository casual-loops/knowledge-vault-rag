from fastapi import FastAPI, HTTPException, status

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
    try:
        with get_connection() as conn:
            conn.execute("SELECT 1;")

        return {
            "status": "ok",
            "database": "ok",
        }

    except Exception:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Service dependency unavailable.",
        )


@app.post(
    "/query",
    response_model=QueryResponse,
)
def query(request: QueryRequest) -> QueryResponse:
    """Run semantic retrieval for a query string."""

    try:
        provider = get_embedding_provider()

        with get_connection() as conn:
            results = semantic_search(
                conn,
                provider,
                query=request.query,
                limit=request.top_k,
                note_type=request.note_type,
                topic=request.topic,
            )

    except Exception:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Query service temporarily unavailable.",
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