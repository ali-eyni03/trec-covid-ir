from pydantic import BaseModel


class SearchResult(BaseModel):
    id: str
    title: str
    text: str
    score: float


class SearchResponse(BaseModel):
    query: str
    method: str
    total: int
    results: list[SearchResult]


class CompareResponse(BaseModel):
    query: str
    bm25: list[SearchResult]
    semantic: list[SearchResult]
    hybrid: list[SearchResult]
