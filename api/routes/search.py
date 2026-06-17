from fastapi import APIRouter, HTTPException
from api.schemas import SearchResponse, CompareResponse, SearchResult
from api.main import retriever

router = APIRouter()


def format_results(results: list) -> list[SearchResult]:
    """convert result from retriever to SearchResult schema."""

    formatted = []
    for result in results:
        text = (result.get("text") or result.get("abstract", ""))[:300]
        score = result.get("score", 0.0)

        # BM25 scores can be higher than 1, so we normalize them to be between 0 and 1
        # Semantic scores are already between 0 and 1, so we don't need to normalize them
        # we convert them to float to avoid serialization issues with Decimal
        formatted.append(
            SearchResult(
                id=str(result["_id"]),
                title=result.get("title", ""),
                text=text,
                score=float(score),
            )
        )
    return formatted


@router.get("/search", response_model=SearchResponse)
async def search(q: str, method: str = "hybrid", top_k: int = 10):
    if not q or not q.strip():
        raise HTTPException(status_code=400, detail="Query cannot be empty")

    method = method.lower().strip()

    if method == "bm25":
        results = retriever.bm25_retriever.search(q, top_k)
    elif method == "semantic":
        results = retriever.semantic_retriever.search(q, top_k)
    elif method == "hybrid":
        results = retriever.search(q, top_k)
    else:
        raise HTTPException(
            status_code=400,
            detail=f"Invalid method '{method}'. Must be one of: bm25, semantic, hybrid",
        )

    if not results:
        raise HTTPException(status_code=404, detail="No results found for this query")

    return SearchResponse(
        query=q,
        method=method,
        total=len(results),
        results=format_results(results),
    )


@router.get("/compare", response_model=CompareResponse)
async def compare(q: str, top_k: int = 5):
    if not q or not q.strip():
        raise HTTPException(status_code=400, detail="Query cannot be empty")

    bm25_results = retriever.bm25_retriever.search(q, top_k)
    semantic_results = retriever.semantic_retriever.search(q, top_k)
    hybrid_results = retriever.search(q, top_k)

    return CompareResponse(
        query=q,
        bm25=format_results(bm25_results),
        semantic=format_results(semantic_results),
        hybrid=format_results(hybrid_results),
    )
