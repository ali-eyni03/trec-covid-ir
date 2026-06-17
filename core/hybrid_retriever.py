from .bm25_retriever import BM25Retriever
from .semantic_retriever import SemanticRetriever


class HybridRetriever:
    def __init__(self):
        self.bm25_retriever = BM25Retriever()
        self.semantic_retriever = SemanticRetriever()

    def search(self, query, top_k=5):
        bm25_result = self.bm25_retriever.search(query, top_k * 2)
        semantic_result = self.semantic_retriever.search(query, top_k * 2)

        # Normalize BM25 scores
        # If all BM25 scores are zero (query has no matching terms in the corpus),
        # max_bm25 becomes zero and would cause a division by zero → use max(max_bm25, 1e-9)
        max_bm25 = max((r["score"] for r in bm25_result), default=0)
        max_bm25 = max(max_bm25, 1e-9)  # avoid division by zero

        scores = {}

        for result in bm25_result:
            doc_id = result["_id"]
            scores[doc_id] = {
                "bm25_score": (result["score"] / max_bm25) * 0.3,
                "semantic_score": 0.0,
                "title": result["title"],
                # BM25 snippet is always a str (it is truncated in bm25_retriever.py)
                "text": result.get("text", ""),
            }

        for result in semantic_result:
            doc_id = result["_id"]
            # semantic text should always have a fallback
            text = result.get("text") or result.get("abstract", "")

            if doc_id in scores:
                scores[doc_id]["semantic_score"] = result["score"] * 0.7
                # If the BM25 text is shorter than the semantic text (e.g., 150 vs 300 chars),
                # prefer the longer semantic text
                if len(scores[doc_id]["text"]) < len(text):
                    scores[doc_id]["text"] = text
            else:
                scores[doc_id] = {
                    "bm25_score": 0.0,
                    "semantic_score": result["score"] * 0.7,
                    "title": result["title"],
                    "text": text,
                }

        # Final score (sorted)
        combined = [
            {
                "_id": doc_id,
                "title": s["title"],
                "text": s["text"] or "", 
                "score": s["bm25_score"] + s["semantic_score"],
                "bm25_score": round(s["bm25_score"], 4),
                "semantic_score": round(s["semantic_score"], 4),
            }
            for doc_id, s in scores.items()
        ]
        combined.sort(key=lambda x: x["score"], reverse=True)
        return combined[:top_k]
