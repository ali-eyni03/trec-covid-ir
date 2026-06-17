from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from contextlib import asynccontextmanager
from core.hybrid_retriever import HybridRetriever

retriever = HybridRetriever()


@asynccontextmanager
async def lifespan(app: FastAPI):
    retriever.bm25_retriever.load_index("data/processed/bm25_index.pkl")
    retriever.semantic_retriever.load_index("data/processed/semantic_index.faiss")
    yield


app = FastAPI(
    title="TREC-COVID Information Retrieval",
    description="Hybrid IR system combining BM25 and Semantic Search",
    version="1.0.0",
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

from api.routes.search import router

app.include_router(router, prefix="/api")
