from fastapi import FastAPI, HTTPException, status

from knowledge_rag.api.models import (
    CitationResult,
    GroundedQueryRequest,
    GroundedQueryResponse,
    GroundedSourceResult,
    QueryRequest,
    QueryResponse,
    QueryResult,
)
from knowledge_rag.db import get_connection
from knowledge_rag.embedding_factory import get_embedding_provider
from knowledge_rag.generation_factory import get_generation_provider
from knowledge_rag.grounded_generation import generate_grounded_answer
from knowledge_rag.retrieval import (
    hybrid_search,
    lexical_search,
    semantic_search,
)
from knowledge_rag.retrieval_policy import RetrievedChunk


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
            results = run_retrieval(
                conn=conn,
                embedding_provider=provider,
                query=request.query,
                limit=request.top_k,
                note_type=request.note_type,
                topic=request.topic,
                retrieval_mode=request.retrieval_mode,
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
                score=result.score,
                score_type=result.score_type,
            )
            for result in results
        ]
    )


@app.post(
    "/answer",
    response_model=GroundedQueryResponse,
)
def answer(request: GroundedQueryRequest) -> GroundedQueryResponse:
    """Run semantic retrieval and generate a grounded answer."""

    try:
        embedding_provider = get_embedding_provider()
        generation_provider = get_generation_provider()

        with get_connection() as conn:
            results = run_retrieval(
                conn=conn,
                embedding_provider=embedding_provider,
                query=request.query,
                limit=request.top_k,
                note_type=request.note_type,
                topic=request.topic,
                retrieval_mode=request.retrieval_mode,
            )

        chunks = [
            RetrievedChunk(
                source_path=result.source_path,
                content=result.content,
                ai_access=result.ai_access,
                metadata={
                    "title": result.title,
                    "heading_path": result.heading_path,
                    "chunk_index": result.chunk_index,
                },
            )
            for result in results
        ]

        grounded = generate_grounded_answer(
            query=request.query,
            chunks=chunks,
            provider=generation_provider,
        )

    except Exception:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Answer service temporarily unavailable.",
        )

    return GroundedQueryResponse(
        answer=grounded.answer,
        citations=[
            CitationResult(
                citation_id=citation.citation_id,
                source_path=citation.source_path,
                title=citation.title,
                heading_path=citation.heading_path,
                chunk_index=citation.chunk_index,
            )
            for citation in grounded.citations
        ],
        sources=[
            GroundedSourceResult(
                source_path=source.source_path,
                title=source.metadata.get("title"),
                heading_path=source.metadata.get("heading_path"),
                chunk_index=source.metadata.get("chunk_index"),
                content=source.content,
                ai_access=source.ai_access,
            )
            for source in grounded.sources
        ],
    )


def run_retrieval(
    *,
    conn,
    embedding_provider,
    query: str,
    limit: int,
    note_type: str | None,
    topic: str | None,
    retrieval_mode: str,
):
    """Run the requested retrieval strategy."""

    if retrieval_mode == "semantic":
        return semantic_search(
            conn,
            embedding_provider,
            query=query,
            limit=limit,
            note_type=note_type,
            topic=topic,
        )

    if retrieval_mode == "lexical":
        return lexical_search(
            conn,
            query=query,
            limit=limit,
            note_type=note_type,
            topic=topic,
        )

    semantic_results = semantic_search(
        conn,
        embedding_provider,
        query=query,
        limit=limit,
        note_type=note_type,
        topic=topic,
    )

    lexical_results = lexical_search(
        conn,
        query=query,
        limit=limit,
        note_type=note_type,
        topic=topic,
    )

    return hybrid_search(
        semantic_results,
        lexical_results,
        limit=limit,
    )